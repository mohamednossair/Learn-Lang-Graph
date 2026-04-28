"""
Lesson 16: Enterprise PostgreSQL Persistence + Async Execution
==============================================================
Teaches:
  - PostgresSaver for multi-server shared checkpointing
  - async nodes (ainvoke, astream)
  - Connection pooling with asyncpg
  - Concurrent multi-user execution
  - Graceful shutdown patterns

Prerequisites: pip install psycopg2-binary asyncpg langgraph-checkpoint-postgres
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model

import asyncio
import logging
import os
import time
from typing import Annotated, AsyncIterator

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("lesson_16")

# ---------------------------------------------------------------------------
# DATABASE CONFIGURATION
# ---------------------------------------------------------------------------
# In production this comes from environment variables / secrets manager.
# For local demo we use SQLite fallback so the lesson works without Postgres.
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://postgres:password@localhost:5432/langgraph_demo",
)
USE_POSTGRES = os.getenv("USE_POSTGRES", "false").lower() == "true"


def get_checkpointer():
    """
    Return the appropriate checkpointer based on environment.

    Enterprise rule:
      - Local dev  → MemorySaver (zero config)
      - Staging    → SqliteSaver (single file, easy reset)
      - Production → PostgresSaver (multi-server, durable, pooled)
    """
    env = os.getenv("APP_ENV", "development")

    if env == "production" or USE_POSTGRES:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            logger.info("Using PostgresSaver (production mode)")
            # sync version for demo; async version shown in async section below
            return PostgresSaver.from_conn_string(POSTGRES_URL)
        except Exception as exc:
            logger.warning(f"Postgres unavailable ({exc}), falling back to SQLite")

    if env == "staging":
        from langgraph.checkpoint.sqlite import SqliteSaver

        logger.info("Using SqliteSaver (staging mode)")
        return SqliteSaver.from_conn_string("langgraph_staging.db")

    from langgraph.checkpoint.memory import MemorySaver

    logger.info("Using MemorySaver (development mode)")
    return MemorySaver()


# ---------------------------------------------------------------------------
# STATE DEFINITION
# ---------------------------------------------------------------------------
class EnterpriseState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    session_id: str
    request_count: int          # tracks usage for rate limiting / billing
    latency_ms: float           # tracks per-turn latency for SLA monitoring


# ---------------------------------------------------------------------------
# LLM SETUP
# ---------------------------------------------------------------------------
llm = ChatOllama(model=get_ollama_model(), temperature=0)


# ---------------------------------------------------------------------------
# NODES — sync versions (for comparison)
# ---------------------------------------------------------------------------
def sync_chat_node(state: EnterpriseState) -> dict:
    """Standard synchronous node — blocks the thread while waiting for LLM."""
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
# NODES — async versions (enterprise standard)
# ---------------------------------------------------------------------------
async def async_chat_node(state: EnterpriseState) -> dict:
    """
    Async node — does NOT block the event loop.
    FastAPI + uvicorn can serve other requests while this awaits the LLM.

    Key enterprise benefit: with 4 uvicorn workers and async nodes,
    you can handle hundreds of concurrent users without thread exhaustion.
    """
    start = time.perf_counter()
    logger.info(f"[async_chat] user={state['user_id']} | msgs={len(state['messages'])}")

    # ainvoke() is the async counterpart of invoke()
    response = await llm.ainvoke(state["messages"])

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"[async_chat] done | latency={elapsed:.1f}ms")

    return {
        "messages": [response],
        "request_count": state.get("request_count", 0) + 1,
        "latency_ms": elapsed,
    }


async def async_validation_node(state: EnterpriseState) -> dict:
    """
    Validate input before sending to LLM.
    Enterprise pattern: validate at graph boundary, not inside nodes.
    """
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    if len(content) > 10_000:
        logger.warning(f"[validation] oversized input from user={state['user_id']}")
        return {
            "messages": [
                AIMessage(content="Input too long. Please keep messages under 10,000 characters.")
            ]
        }

    if state.get("request_count", 0) >= 100:
        logger.warning(f"[validation] rate limit exceeded for user={state['user_id']}")
        return {
            "messages": [
                AIMessage(content="Rate limit reached. Please try again later.")
            ]
        }

    return {}   # no changes — pass through


def route_after_validation(state: EnterpriseState) -> str:
    """Route to LLM if validation passed, else END."""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        return END      # validation already replied with an error
    return "chat"


# ---------------------------------------------------------------------------
# BUILD GRAPHS
# ---------------------------------------------------------------------------
def build_sync_graph(checkpointer):
    """Synchronous graph — simpler, fine for low-concurrency scenarios."""
    builder = StateGraph(EnterpriseState)
    builder.add_node("chat", sync_chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


def build_async_graph(checkpointer):
    """
    Async graph — recommended for production FastAPI deployments.
    All nodes are async; graph.ainvoke() / graph.astream() are non-blocking.
    """
    builder = StateGraph(EnterpriseState)
    builder.add_node("validate", async_validation_node)
    builder.add_node("chat", async_chat_node)
    builder.add_edge(START, "validate")
    builder.add_conditional_edges("validate", route_after_validation, ["chat", END])
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# DEMO: SYNC MULTI-TURN
# ---------------------------------------------------------------------------
def demo_sync():
    """Shows multi-turn conversation persisted across invocations."""
    print("\n" + "=" * 60)
    print("DEMO 1: Sync multi-turn with persistent memory")
    print("=" * 60)

    checkpointer = get_checkpointer()
    graph = build_sync_graph(checkpointer)

    # thread_id = unique session identifier (per user per session)
    config = {"configurable": {"thread_id": "user-alice-session-001"}}
    initial_state = {"user_id": "alice", "session_id": "session-001", "request_count": 0, "latency_ms": 0.0}

    # Turn 1
    result = graph.invoke(
        {**initial_state, "messages": [HumanMessage(content="My name is Alice. Remember that.")]},
        config=config,
    )
    print(f"Turn 1 → {result['messages'][-1].content[:80]}...")
    print(f"  request_count={result['request_count']} | latency={result['latency_ms']:.1f}ms")

    # Turn 2 — state is loaded from checkpointer automatically
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=config,
    )
    print(f"Turn 2 → {result['messages'][-1].content[:80]}...")
    print(f"  request_count={result['request_count']} | latency={result['latency_ms']:.1f}ms")


# ---------------------------------------------------------------------------
# DEMO: ASYNC CONCURRENT USERS
# ---------------------------------------------------------------------------
async def simulate_user(graph, user_id: str, question: str) -> dict:
    """Simulates one user sending a message — runs concurrently with others."""
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
    """
    Fires 3 user requests concurrently using asyncio.gather().

    Enterprise insight:
      - Sync graph.invoke() → 3 requests take 3 × latency (sequential)
      - Async graph.ainvoke() → 3 requests take ~1 × latency (concurrent)
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Async concurrent users (asyncio.gather)")
    print("=" * 60)

    checkpointer = get_checkpointer()
    graph = build_async_graph(checkpointer)

    users = [
        ("bob",   "What is the capital of France?"),
        ("carol", "Explain async/await in one sentence."),
        ("dave",  "What is 7 × 8?"),
    ]

    start = time.perf_counter()

    # All 3 run concurrently — total time ≈ max(individual times)
    tasks = [simulate_user(graph, uid, q) for uid, q in users]
    results = await asyncio.gather(*tasks)

    total = (time.perf_counter() - start) * 1000
    print(f"Total wall-clock time for 3 concurrent users: {total:.0f}ms")
    for r in results:
        print(f"  user={r['user_id']} | latency={r['latency_ms']:.0f}ms | reply={r['response']}...")


# ---------------------------------------------------------------------------
# DEMO: ASYNC STREAMING
# ---------------------------------------------------------------------------
async def demo_async_streaming():
    """
    astream() yields graph snapshots as they happen.
    Enterprise use: pipe tokens to client via Server-Sent Events (SSE)
    for real-time UI responses.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Async streaming (astream)")
    print("=" * 60)

    checkpointer = get_checkpointer()
    graph = build_async_graph(checkpointer)
    config = {"configurable": {"thread_id": "stream-demo-001"}}

    initial = {
        "user_id": "streamer",
        "session_id": "session-stream",
        "request_count": 0,
        "latency_ms": 0.0,
        "messages": [HumanMessage(content="Count from 1 to 5 slowly.")],
    }

    print("Streaming node updates:")
    async for chunk in graph.astream(initial, config=config):
        for node_name, node_output in chunk.items():
            print(f"  [node: {node_name}] keys updated: {list(node_output.keys())}")


# ---------------------------------------------------------------------------
# ENTERPRISE PATTERN: PostgresSaver pool (reference, requires real Postgres)
# ---------------------------------------------------------------------------
POSTGRES_POOL_EXAMPLE = '''
# ── How to use PostgresSaver with async connection pool ──────────────────
#
# from psycopg_pool import AsyncConnectionPool
# from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
#
# async def lifespan(app: FastAPI):
#     async with AsyncConnectionPool(
#         conninfo=POSTGRES_URL,
#         min_size=2,
#         max_size=10,        # enterprise: tune per expected concurrency
#         kwargs={"autocommit": True},
#     ) as pool:
#         checkpointer = AsyncPostgresSaver(pool)
#         await checkpointer.setup()   # creates tables on first run
#         app.state.graph = build_async_graph(checkpointer)
#         yield
#         # pool is closed automatically on exit
#
# Key benefits over SqliteSaver:
#   1. Multiple API server instances share the SAME state
#   2. Concurrent reads/writes with no file locking
#   3. Built-in connection pooling (no per-request connections)
#   4. ACID transactions — no partial state writes
#   5. Easy backup, point-in-time recovery, read replicas
'''

print(POSTGRES_POOL_EXAMPLE)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Demo 1: sync (always works)
    demo_sync()

    # Demo 2 + 3: async
    asyncio.run(demo_async_concurrent())
    asyncio.run(demo_async_streaming())

    print("\n✓ Lesson 16 complete.")
    print("Next: Lesson 17 — Auth, RBAC, multi-tenancy")
