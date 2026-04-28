"""
Lesson 16: Enterprise MySQL 8.0+ Persistence + Async Execution
=================================================================
Teaches:
  - Custom MySQLSaver checkpointer for MySQL 8.0+
  - async nodes (ainvoke, astream)
  - Connection pooling with pymysql
  - Concurrent multi-user execution
  - Graceful shutdown patterns

Prerequisites:
  pip install pymysql langgraph langchain-ollama
  MySQL 8.0+ must be reachable (or use MemorySaver fallback for local dev).

MySQL connection string format:
  mysql+pymysql://user:password@host:3306/dbname

Note: lesson_16_oracle_async.py (Oracle version) and
      lesson_16_postgres_async.py (PostgreSQL version) are kept as reference.
      This file is for MySQL 8.0+ environments.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model

import asyncio
import json
import logging
import time
from contextlib import contextmanager
from typing import Annotated, Any, Iterator, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("lesson_16_mysql")

# ---------------------------------------------------------------------------
# MYSQL CONNECTION CONFIGURATION
# ---------------------------------------------------------------------------
MYSQL_HOST     = os.getenv("MYSQL_HOST",     "localhost")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER     = os.getenv("MYSQL_USER",     "langgraph")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "langgraph")
USE_MYSQL      = os.getenv("USE_MYSQL",      "false").lower() == "true"

# ---------------------------------------------------------------------------
# MYSQL 8.0+ CHECKPOINTER
# ---------------------------------------------------------------------------
# LangGraph ships with PostgresSaver and SqliteSaver.
# For MySQL we implement a custom BaseCheckpointSaver.
#
# Production MySQL DDL (run once as DBA):
# ─────────────────────────────────────────
# CREATE DATABASE IF NOT EXISTS langgraph;
# USE langgraph;
# CREATE TABLE langgraph_checkpoints (
#     thread_id   VARCHAR(255)   NOT NULL PRIMARY KEY,
#     checkpoint  JSON           NOT NULL,
#     metadata    JSON           NULL,
#     ts          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
# ) ENGINE=InnoDB;

DDL_CREATE_DATABASE = "CREATE DATABASE IF NOT EXISTS langgraph"

DDL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
    thread_id   VARCHAR(255)   NOT NULL PRIMARY KEY,
    checkpoint  JSON           NOT NULL,
    metadata    JSON           NULL,
    ts          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB
"""


class MySQLSaver(BaseCheckpointSaver):
    """
    MySQL 8.0+ checkpointer for LangGraph.

    Stores graph state as JSON in MySQL. Supports connection via pymysql.

    MySQL advantages over SQLite for enterprise:
      1. Multi-server concurrent access (SQLite file-locks)
      2. MySQL Group Replication / InnoDB Cluster — HA
      3. Enterprise audit trail (audit_log plugin)
      4. Transparent Data Encryption (TDE)
      5. Asynchronous replication for DR
      6. MySQL Router for connection routing
    """

    def __init__(self, connection):
        super().__init__()
        self._conn = connection

    @contextmanager
    def _get_conn(self):
        conn = self._conn
        cursor = conn.cursor()
        try:
            yield conn, cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def setup(self):
        """Create the checkpoints table if it does not exist."""
        # First ensure database exists (requires separate connection without database)
        import pymysql
        temp_conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            charset='utf8mb4',
        )
        try:
            with temp_conn.cursor() as cursor:
                cursor.execute(DDL_CREATE_DATABASE)
            temp_conn.commit()
            logger.info("[MySQLSaver] Database 'langgraph' ensured")
        except Exception as exc:
            logger.warning(f"[MySQLSaver] Database setup warning: {exc}")
        finally:
            temp_conn.close()

        # Now create table in the langgraph database
        with self._get_conn() as (conn, cursor):
            try:
                cursor.execute(DDL_CREATE_TABLE)
                logger.info("[MySQLSaver] Table langgraph_checkpoints created/verified")
            except Exception as exc:
                logger.error(f"[MySQLSaver] Table setup error: {exc}")
                raise

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any = None,
    ) -> dict:
        """Upsert checkpoint into MySQL using INSERT ... ON DUPLICATE KEY UPDATE."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_json = json.dumps(checkpoint)
        metadata_json = json.dumps(metadata) if metadata else None

        with self._get_conn() as (conn, cursor):
            cursor.execute(
                """
                INSERT INTO langgraph_checkpoints (thread_id, checkpoint, metadata, ts)
                VALUES (%s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                    checkpoint = VALUES(checkpoint),
                    metadata = VALUES(metadata),
                    ts = NOW()
                """,
                (thread_id, checkpoint_json, metadata_json),
            )
        logger.debug(f"[MySQLSaver] put | thread={thread_id}")
        return config

    def get_tuple(self, config: dict):
        """Load checkpoint from MySQL by thread_id."""
        thread_id = config["configurable"]["thread_id"]
        with self._get_conn() as (conn, cursor):
            cursor.execute(
                "SELECT checkpoint, metadata FROM langgraph_checkpoints WHERE thread_id = %s",
                (thread_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        checkpoint = json.loads(row[0])
        metadata = json.loads(row[1]) if row[1] else {}
        return (config, checkpoint, metadata, config)

    def list(self, config: dict, *, filter=None, before=None, limit=None):
        """List checkpoints — returns iterator of (config, checkpoint, metadata, parent_config)."""
        thread_id = config["configurable"].get("thread_id")
        with self._get_conn() as (conn, cursor):
            if thread_id:
                cursor.execute(
                    "SELECT checkpoint, metadata FROM langgraph_checkpoints WHERE thread_id = %s ORDER BY ts DESC",
                    (thread_id,),
                )
            else:
                cursor.execute("SELECT checkpoint, metadata FROM langgraph_checkpoints ORDER BY ts DESC")
            rows = cursor.fetchall()

        for row in rows:
            checkpoint = json.loads(row[0])
            metadata = json.loads(row[1]) if row[1] else {}
            yield (config, checkpoint, metadata, config)


# ---------------------------------------------------------------------------
# CHECKPOINTER FACTORY — MySQL primary, graceful fallback
# ---------------------------------------------------------------------------
def get_checkpointer():
    """
    Enterprise checkpointer selection:

      APP_ENV == "production" or USE_MYSQL=true
          → MySQLSaver (MySQL 8.0+, multi-server, durable)
          → falls back to MemorySaver if MySQL unreachable

      APP_ENV == "staging"
          → SqliteSaver (single file, easy to inspect)

      APP_ENV == "development" (default)
          → MemorySaver (zero config)
    """
    env = os.getenv("APP_ENV", "development")

    if env == "production" or USE_MYSQL:
        try:
            import pymysql

            conn = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                charset='utf8mb4',
                autocommit=False,
            )
            saver = MySQLSaver(conn)
            saver.setup()
            logger.info(f"[checkpointer] MySQLSaver | host={MYSQL_HOST}:{MYSQL_PORT}")
            return saver
        except Exception as exc:
            logger.warning(f"[checkpointer] MySQL unavailable ({exc}), falling back to MemorySaver")

    if env == "staging":
        from langgraph.checkpoint.sqlite import SqliteSaver
        logger.info("[checkpointer] SqliteSaver (staging)")
        return SqliteSaver.from_conn_string("langgraph_staging.db")

    from langgraph.checkpoint.memory import MemorySaver
    logger.info("[checkpointer] MemorySaver (development)")
    return MemorySaver()


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class EnterpriseState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    session_id: str
    request_count: int
    latency_ms: float


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOllama(model=get_ollama_model(), temperature=0)


# ---------------------------------------------------------------------------
# NODES — sync
# ---------------------------------------------------------------------------
def sync_chat_node(state: EnterpriseState) -> dict:
    """Sync node — blocks thread. Use for simple / low-concurrency scenarios."""
    start = time.perf_counter()
    logger.info(f"[sync_chat] user={state['user_id']} | msgs={len(state['messages'])}")
    response = llm.invoke(state["messages"])
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"[sync_chat] done | latency={elapsed:.1f}ms")
    return {
        "messages": [response],
        "request_count": state.get("request_count", 0) + 1,
        "latency_ms": elapsed,
    }


# ---------------------------------------------------------------------------
# NODES — async (enterprise standard for FastAPI)
# ---------------------------------------------------------------------------
async def async_chat_node(state: EnterpriseState) -> dict:
    """
    Async node — non-blocking. Required for high-concurrency FastAPI deployments.
    """
    start = time.perf_counter()
    logger.info(f"[async_chat] user={state['user_id']} | msgs={len(state['messages'])}")
    response = await llm.ainvoke(state["messages"])
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"[async_chat] done | latency={elapsed:.1f}ms")
    return {
        "messages": [response],
        "request_count": state.get("request_count", 0) + 1,
        "latency_ms": elapsed,
    }


async def async_validation_node(state: EnterpriseState) -> dict:
    """Input validation gate — runs before every LLM call."""
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    if len(content) > 10_000:
        logger.warning(f"[validation] oversized | user={state['user_id']}")
        return {"messages": [AIMessage(content="Input too long (max 10,000 chars).")]}

    if state.get("request_count", 0) >= 100:
        logger.warning(f"[validation] rate limit | user={state['user_id']}")
        return {"messages": [AIMessage(content="Rate limit reached. Try again later.")]}

    return {}


def route_after_validation(state: EnterpriseState) -> str:
    return END if isinstance(state["messages"][-1], AIMessage) else "chat"


# ---------------------------------------------------------------------------
# GRAPHS
# ---------------------------------------------------------------------------
def build_sync_graph(checkpointer):
    builder = StateGraph(EnterpriseState)
    builder.add_node("chat", sync_chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


def build_async_graph(checkpointer):
    builder = StateGraph(EnterpriseState)
    builder.add_node("validate", async_validation_node)
    builder.add_node("chat", async_chat_node)
    builder.add_edge(START, "validate")
    builder.add_conditional_edges("validate", route_after_validation, ["chat", END])
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# DEMO 1: Sync multi-turn (MySQL persists state between turns)
# ---------------------------------------------------------------------------
def demo_sync():
    print("\n" + "=" * 60)
    print("DEMO 1: Sync multi-turn — MySQL persists state")
    print("=" * 60)

    checkpointer = get_checkpointer()
    graph = build_sync_graph(checkpointer)
    config = {"configurable": {"thread_id": "user-alice-session-001"}}
    initial = {"user_id": "alice", "session_id": "session-001", "request_count": 0, "latency_ms": 0.0}

    result = graph.invoke(
        {**initial, "messages": [HumanMessage(content="My name is Alice. Remember that.")]},
        config=config,
    )
    print(f"Turn 1 → {result['messages'][-1].content[:80]}...")
    print(f"  request_count={result['request_count']} | latency={result['latency_ms']:.1f}ms")

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=config,
    )
    print(f"Turn 2 → {result['messages'][-1].content[:80]}...")
    print(f"  request_count={result['request_count']} | latency={result['latency_ms']:.1f}ms")


# ---------------------------------------------------------------------------
# DEMO 2: Async concurrent users
# ---------------------------------------------------------------------------
async def simulate_user(graph, user_id: str, question: str) -> dict:
    config = {"configurable": {"thread_id": f"user-{user_id}-session-001"}}
    initial = {
        "user_id": user_id,
        "session_id": "session-001",
        "request_count": 0,
        "latency_ms": 0.0,
        "messages": [HumanMessage(content=question)],
    }
    result = await graph.ainvoke(initial, config=config)
    return {
        "user_id": user_id,
        "response": result["messages"][-1].content[:60],
        "latency_ms": result["latency_ms"],
    }


async def demo_async_concurrent():
    print("\n" + "=" * 60)
    print("DEMO 2: 3 concurrent users (asyncio.gather)")
    print("=" * 60)

    checkpointer = get_checkpointer()
    graph = build_async_graph(checkpointer)
    users = [
        ("bob",   "What is the capital of France?"),
        ("carol", "Explain async/await in one sentence."),
        ("dave",  "What is 7 × 8?"),
    ]
    start = time.perf_counter()
    results = await asyncio.gather(*[simulate_user(graph, uid, q) for uid, q in users])
    total = (time.perf_counter() - start) * 1000
    print(f"Total wall-clock: {total:.0f}ms (concurrent vs sequential would be 3×)")
    for r in results:
        print(f"  user={r['user_id']} | latency={r['latency_ms']:.0f}ms | {r['response']}...")


# ---------------------------------------------------------------------------
# DEMO 3: Async streaming
# ---------------------------------------------------------------------------
async def demo_async_streaming():
    print("\n" + "=" * 60)
    print("DEMO 3: Async streaming (astream → SSE)")
    print("=" * 60)

    checkpointer = get_checkpointer()
    graph = build_async_graph(checkpointer)
    config = {"configurable": {"thread_id": "stream-demo-001"}}
    initial = {
        "user_id": "streamer",
        "session_id": "stream",
        "request_count": 0,
        "latency_ms": 0.0,
        "messages": [HumanMessage(content="Count from 1 to 5.")],
    }
    print("Streaming node updates:")
    async for chunk in graph.astream(initial, config=config):
        for node_name, node_output in chunk.items():
            print(f"  [node: {node_name}] keys={list(node_output.keys())}")


# ---------------------------------------------------------------------------
# ENTERPRISE REFERENCE: MySQL connection in FastAPI lifespan
# ---------------------------------------------------------------------------
MYSQL_POOL_EXAMPLE = '''
# ── MySQL 8.0+ connection in FastAPI (production pattern) ───────────
#
# import pymysql
# from contextlib import asynccontextmanager
# from fastapi import FastAPI
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     conn = pymysql.connect(
#         host=MYSQL_HOST,
#         port=MYSQL_PORT,
#         user=MYSQL_USER,
#         password=MYSQL_PASSWORD,
#         database=MYSQL_DATABASE,
#         charset='utf8mb4',
#         autocommit=False,
#     )
#     checkpointer = MySQLSaver(conn)
#     checkpointer.setup()           # creates table on first run
#     app.state.graph = build_async_graph(checkpointer)
#     yield
#     conn.close()
#
# app = FastAPI(lifespan=lifespan)
#
# Key MySQL 8.0+ enterprise features used here:
#   - pymysql.connect() — direct connection to MySQL
#   - INSERT ... ON DUPLICATE KEY UPDATE — atomic upsert
#   - MySQL JSON columns — stores LangGraph state
#   - MySQL InnoDB — ACID transactions, row-level locking
#   - MySQL TDE — encrypt checkpoints at rest (no extra code)
#   - audit_log plugin — every SELECT/INSERT audited automatically
'''

print(MYSQL_POOL_EXAMPLE)

# ---------------------------------------------------------------------------
# MYSQL vs POSTGRESQL vs ORACLE vs SQLITE comparison
# ---------------------------------------------------------------------------
COMPARISON = """
Checkpointer comparison for enterprise LangGraph:

Feature                    | MemorySaver | SqliteSaver  | PostgresSaver | MySQLSaver    | OracleSaver
---------------------------|-------------|--------------|---------------|---------------|------------
Multi-server               | ✗           | ✗ (file lock)| ✓             | ✓             | ✓
Connection pool            | N/A         | ✗            | ✓ (asyncpg)   | ✓ (pymysql)   | ✓ (oracledb)
ACID transactions          | ✗           | ✓            | ✓             | ✓ (InnoDB)    | ✓
High-availability          | ✗           | ✗            | ✓ (Patroni)   | ✓ (Group Repl)| ✓ (Oracle RAC)
Transparent encryption     | ✗           | ✗            | pgcrypto      | ✓ (TDE)       | ✓ (TDE)
Enterprise audit           | ✗           | ✗            | pgaudit       | ✓ (audit_log) | ✓ (Audit Vault)
Fine-grained access ctrl   | ✗           | ✗            | Row security  | ✗             | ✓ (VPD/FGAC)
Replication / DR           | ✗           | ✗            | Streaming rep | ✓ (async repl)| ✓ (Data Guard)
Zero-downtime patching     | ✗           | ✗            | ✗             | ✗             | ✓ (Oracle RAC)
"""
print(COMPARISON)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_sync()
    asyncio.run(demo_async_concurrent())
    asyncio.run(demo_async_streaming())

    print("\n✓ Lesson 16 complete (MySQL 8.0+ edition).")
    print("Next: Lesson 17 — Auth, RBAC, multi-tenancy")
