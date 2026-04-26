"""
Lesson 16: Enterprise Oracle 19c Persistence + Async Execution
===============================================================
Teaches:
  - Custom OracleSaver checkpointer for Oracle 19c
  - async nodes (ainvoke, astream)
  - Connection pooling with cx_Oracle / oracledb
  - Concurrent multi-user execution
  - Graceful shutdown patterns

Prerequisites:
  pip install oracledb langgraph langchain-ollama
  Oracle 19c must be reachable (or use MemorySaver fallback for local dev).

Oracle connection string format:
  oracle+oracledb://user:password@host:1521/?service_name=ORCLPDB1

Note: lesson_16_postgres_async.py (PostgreSQL version) is kept as reference.
      This file is the ENTERPRISE PRIMARY for Oracle 19c environments.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model

import asyncio
import json
import logging
import os
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
logger = logging.getLogger("lesson_16_oracle")

# ---------------------------------------------------------------------------
# ORACLE CONNECTION CONFIGURATION
# ---------------------------------------------------------------------------
ORACLE_USER     = os.getenv("ORACLE_USER",     "langgraph")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD",  "password")
ORACLE_HOST     = os.getenv("ORACLE_HOST",     "localhost")
ORACLE_PORT     = os.getenv("ORACLE_PORT",     "1521")
ORACLE_SERVICE  = os.getenv("ORACLE_SERVICE",  "ORCLPDB1")
USE_ORACLE      = os.getenv("USE_ORACLE",      "false").lower() == "true"

# DSN used by oracledb
ORACLE_DSN = f"{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE}"

# ---------------------------------------------------------------------------
# ORACLE 19c CHECKPOINTER
# ---------------------------------------------------------------------------
# LangGraph ships with PostgresSaver and SqliteSaver.
# For Oracle we implement a custom BaseCheckpointSaver.
#
# Production Oracle DDL (run once as DBA):
# ─────────────────────────────────────────
# CREATE TABLE langgraph_checkpoints (
#     thread_id   VARCHAR2(255)  NOT NULL,
#     checkpoint  CLOB           NOT NULL,  -- JSON blob
#     metadata    CLOB,                     -- JSON blob
#     ts          TIMESTAMP DEFAULT SYSTIMESTAMP,
#     CONSTRAINT pk_lg_chk PRIMARY KEY (thread_id)
# );
# GRANT SELECT, INSERT, UPDATE, DELETE ON langgraph_checkpoints TO langgraph;

DDL_CREATE_TABLE = """
CREATE TABLE langgraph_checkpoints (
    thread_id   VARCHAR2(255)  NOT NULL,
    checkpoint  CLOB           NOT NULL,
    metadata    CLOB,
    ts          TIMESTAMP DEFAULT SYSTIMESTAMP,
    CONSTRAINT pk_lg_chk PRIMARY KEY (thread_id)
)
"""


class OracleSaver(BaseCheckpointSaver):
    """
    Oracle 19c checkpointer for LangGraph.

    Stores graph state as JSON CLOB in Oracle.
    Supports multi-server concurrent access via Oracle row-level locking.

    Oracle advantages over SQLite for enterprise:
      1. Multi-server concurrent access (SQLite file-locks)
      2. Oracle RAC — zero-downtime failover
      3. Enterprise audit trail (Oracle Audit Vault)
      4. Data Masking & Subsetting for GDPR
      5. Transparent Data Encryption (TDE)
      6. Oracle Data Guard for DR
      7. Fine-Grained Access Control (FGAC / VPD)
    """

    def __init__(self, connection_pool):
        super().__init__()
        self._pool = connection_pool

    @contextmanager
    def _get_conn(self):
        conn = self._pool.acquire()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.release(conn)

    def setup(self):
        """Create the checkpoints table if it does not exist."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(DDL_CREATE_TABLE)
                logger.info("[OracleSaver] Table langgraph_checkpoints created")
            except Exception as exc:
                # ORA-00955: name is already used by an existing object
                if "ORA-00955" in str(exc):
                    logger.info("[OracleSaver] Table already exists — OK")
                else:
                    raise

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any = None,
    ) -> dict:
        """Upsert (MERGE) checkpoint into Oracle."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_json = json.dumps(checkpoint)
        metadata_json   = json.dumps(metadata) if metadata else None

        with self._get_conn() as conn:
            cursor = conn.cursor()
            # Oracle MERGE = upsert (no INSERT OR REPLACE like SQLite)
            cursor.execute(
                """
                MERGE INTO langgraph_checkpoints dst
                USING (SELECT :1 AS thread_id FROM dual) src
                ON (dst.thread_id = src.thread_id)
                WHEN MATCHED THEN
                    UPDATE SET checkpoint = :2, metadata = :3, ts = SYSTIMESTAMP
                WHEN NOT MATCHED THEN
                    INSERT (thread_id, checkpoint, metadata)
                    VALUES (:4, :5, :6)
                """,
                [thread_id, checkpoint_json, metadata_json,
                 thread_id, checkpoint_json, metadata_json],
            )
        logger.debug(f"[OracleSaver] put | thread={thread_id}")
        return config

    def get_tuple(self, config: dict):
        """Load checkpoint from Oracle by thread_id."""
        thread_id = config["configurable"]["thread_id"]
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT checkpoint, metadata FROM langgraph_checkpoints WHERE thread_id = :1",
                [thread_id],
            )
            row = cursor.fetchone()

        if row is None:
            return None

        checkpoint = json.loads(row[0].read() if hasattr(row[0], "read") else row[0])
        metadata   = json.loads(row[1].read() if hasattr(row[1], "read") else row[1]) if row[1] else {}
        return (config, checkpoint, metadata, config)

    def list(self, config: dict, *, filter=None, before=None, limit=None):
        """List checkpoints — returns iterator of (config, checkpoint, metadata, parent_config)."""
        thread_id = config["configurable"].get("thread_id")
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if thread_id:
                cursor.execute(
                    "SELECT checkpoint, metadata FROM langgraph_checkpoints WHERE thread_id = :1 ORDER BY ts DESC",
                    [thread_id],
                )
            else:
                cursor.execute("SELECT checkpoint, metadata FROM langgraph_checkpoints ORDER BY ts DESC")
            rows = cursor.fetchall()

        for row in rows:
            checkpoint = json.loads(row[0].read() if hasattr(row[0], "read") else row[0])
            metadata   = json.loads(row[1].read() if hasattr(row[1], "read") else row[1]) if row[1] else {}
            yield (config, checkpoint, metadata, config)


# ---------------------------------------------------------------------------
# CHECKPOINTER FACTORY — Oracle primary, graceful fallback
# ---------------------------------------------------------------------------
def get_checkpointer():
    """
    Enterprise checkpointer selection:

      APP_ENV == "production" or USE_ORACLE=true
          → OracleSaver (Oracle 19c, multi-server, durable)
          → falls back to MemorySaver if Oracle unreachable

      APP_ENV == "staging"
          → SqliteSaver (single file, easy to inspect)

      APP_ENV == "development" (default)
          → MemorySaver (zero config)
    """
    env = os.getenv("APP_ENV", "development")

    if env == "production" or USE_ORACLE:
        try:
            import oracledb

            pool = oracledb.create_pool(
                user=ORACLE_USER,
                password=ORACLE_PASSWORD,
                dsn=ORACLE_DSN,
                min=2,
                max=10,
                increment=1,
            )
            saver = OracleSaver(pool)
            saver.setup()
            logger.info(f"[checkpointer] OracleSaver | dsn={ORACLE_DSN}")
            return saver
        except Exception as exc:
            logger.warning(f"[checkpointer] Oracle unavailable ({exc}), falling back to MemorySaver")

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
    Oracle JDBC is sync by default; for true async use cx_Oracle with
    asyncio.to_thread() or the new python-oracledb thin mode.
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
# DEMO 1: Sync multi-turn (Oracle persists state between turns)
# ---------------------------------------------------------------------------
def demo_sync():
    print("\n" + "=" * 60)
    print("DEMO 1: Sync multi-turn — Oracle persists state")
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
# ENTERPRISE REFERENCE: Oracle connection pool in FastAPI lifespan
# ---------------------------------------------------------------------------
ORACLE_POOL_EXAMPLE = '''
# ── Oracle 19c connection pool in FastAPI (production pattern) ───────────
#
# import oracledb
# from contextlib import asynccontextmanager
# from fastapi import FastAPI
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     pool = oracledb.create_pool(
#         user=ORACLE_USER,
#         password=ORACLE_PASSWORD,
#         dsn=ORACLE_DSN,
#         min=2,
#         max=10,          # tune: workers × 2
#         increment=1,
#     )
#     checkpointer = OracleSaver(pool)
#     checkpointer.setup()           # creates table on first run
#     app.state.graph = build_async_graph(checkpointer)
#     yield
#     pool.close()
#
# app = FastAPI(lifespan=lifespan)
#
# Key Oracle 19c enterprise features used here:
#   - oracledb.create_pool() — connection pool with min/max bounds
#   - Oracle MERGE statement — atomic upsert (no race conditions)
#   - Oracle CLOB  — stores LangGraph JSON state (up to 4 GB)
#   - Oracle TDE   — encrypt checkpoints at rest (no extra code)
#   - Oracle Audit — every SELECT/INSERT audited by DBA automatically
'''

print(ORACLE_POOL_EXAMPLE)

# ---------------------------------------------------------------------------
# ORACLE vs POSTGRESQL vs SQLITE comparison
# ---------------------------------------------------------------------------
COMPARISON = """
Checkpointer comparison for enterprise LangGraph:

Feature                    | MemorySaver | SqliteSaver  | PostgresSaver | OracleSaver (this lesson)
---------------------------|-------------|--------------|---------------|---------------------------
Multi-server               | ✗           | ✗ (file lock)| ✓             | ✓
Connection pool            | N/A         | ✗            | ✓ (asyncpg)   | ✓ (oracledb pool)
ACID transactions          | ✗           | ✓            | ✓             | ✓
High-availability / RAC    | ✗           | ✗            | ✓ (Patroni)   | ✓ (Oracle RAC built-in)
Transparent encryption     | ✗           | ✗            | pgcrypto ext  | ✓ (TDE, built-in)
Enterprise audit           | ✗           | ✗            | pgaudit ext   | ✓ (Audit Vault, built-in)
Fine-grained access ctrl   | ✗           | ✗            | Row security  | ✓ (VPD/FGAC, built-in)
Data Guard / DR            | ✗           | ✗            | Streaming rep | ✓ (Data Guard, built-in)
Zero-downtime patching     | ✗           | ✗            | ✗             | ✓ (Oracle RAC rolling)
"""
print(COMPARISON)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_sync()
    asyncio.run(demo_async_concurrent())
    asyncio.run(demo_async_streaming())

    print("\n✓ Lesson 16 complete (Oracle 19c edition).")
    print("Next: Lesson 17 — Auth, RBAC, multi-tenancy")
