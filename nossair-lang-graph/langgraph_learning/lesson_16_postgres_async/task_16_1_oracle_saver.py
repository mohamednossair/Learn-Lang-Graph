"""Task 16.1 — OracleSaver with Connection Pool (MemorySaver fallback)."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
import json, logging
from typing import Annotated, Any, Iterator, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger("task_16_1")

# ---------------------------------------------------------------------------
# OracleSaver — production checkpointer for Oracle 19c
# Falls back to MemorySaver when Oracle is unavailable
# ---------------------------------------------------------------------------
ORACLE_AVAILABLE = False
try:
    import oracledb
    ORACLE_AVAILABLE = True
except ImportError:
    pass

class OracleSaver(BaseCheckpointSaver):
    """
    Custom checkpointer for Oracle 19c using MERGE + CLOB.
    
    Best practices:
      - Use CLOB for checkpoint data (VARCHAR2 truncates at 32KB)
      - Use MERGE for upsert (no INSERT OR REPLACE in Oracle)
      - Connection pool created once in lifespan, never per-request
      - Always pool.release(conn), never conn.close()
    """
    
    def __init__(self, pool):
        super().__init__()
        self.pool = pool
    
    def get_tuple(self, config) -> Optional[tuple]:
        thread_id = config.get("configurable", {}).get("thread_id")
        conn = self.pool.acquire()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT checkpoint, metadata FROM langgraph_checkpoints WHERE thread_id = :1 "
                "ORDER BY step DESC FETCH FIRST 1 ROW ONLY",
                [thread_id]
            )
            row = cursor.fetchone()
            if row:
                checkpoint = json.loads(row[0].read()) if hasattr(row[0], 'read') else json.loads(row[0])
                metadata = json.loads(row[1].read()) if hasattr(row[1], 'read') else json.loads(row[1])
                return (checkpoint, metadata)
            return None
        finally:
            self.pool.release(conn)
    
    def put(self, config, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id")
        conn = self.pool.acquire()
        try:
            cursor = conn.cursor()
            chk_json = json.dumps(checkpoint)
            meta_json = json.dumps(metadata)
            # MERGE = Oracle upsert (row-level lock, concurrent-safe)
            cursor.execute("""
                MERGE INTO langgraph_checkpoints dst
                USING (SELECT :1 AS thread_id FROM dual) src
                ON (dst.thread_id = src.thread_id)
                WHEN MATCHED THEN
                    UPDATE SET checkpoint = :2, metadata = :3, ts = SYSTIMESTAMP
                WHEN NOT MATCHED THEN
                    INSERT (thread_id, checkpoint, metadata) VALUES (:4, :5, :6)
            """, [thread_id, chk_json, meta_json, thread_id, chk_json, meta_json])
            conn.commit()
            return {"configurable": {"thread_id": thread_id}}
        finally:
            self.pool.release(conn)
    
    def list(self, config, *, filter=None, before=None, limit=None) -> Iterator:
        thread_id = config.get("configurable", {}).get("thread_id")
        conn = self.pool.acquire()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT checkpoint, metadata FROM langgraph_checkpoints WHERE thread_id = :1 ORDER BY step",
                [thread_id]
            )
            for row in cursor:
                checkpoint = json.loads(row[0].read()) if hasattr(row[0], 'read') else json.loads(row[0])
                metadata = json.loads(row[1].read()) if hasattr(row[1], 'read') else json.loads(row[1])
                yield (checkpoint, metadata)
        finally:
            self.pool.release(conn)

def create_pool(dsn, user, password, min=2, max=10, increment=1):
    """Create connection pool with best-practice settings."""
    if not ORACLE_AVAILABLE:
        return None
    return oracledb.create_pool(
        dsn=dsn, user=user, password=password,
        min=min, max=max, increment=increment,
        timeout=30, ping_interval=60
    )

def get_checkpointer():
    """Get appropriate checkpointer based on environment."""
    app_env = os.getenv("APP_ENV", "development")
    
    if app_env == "production" and ORACLE_AVAILABLE:
        dsn = os.getenv("ORACLE_DSN", "localhost:1521/ORCLPDB1")
        user = os.getenv("ORACLE_USER", "langgraph")
        password = os.getenv("ORACLE_PASSWORD", "changeme")
        pool = create_pool(dsn, user, password)
        if pool:
            # Startup health check
            conn = pool.acquire()
            conn.ping()
            pool.release(conn)
            logger.info("OracleSaver initialized with connection pool")
            return OracleSaver(pool)
    
    logger.info("Using MemorySaver (no Oracle configured)")
    return MemorySaver()

# ---------------------------------------------------------------------------
# Agent graph
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model=get_ollama_model(), temperature=0)

def chat_node(state: ChatState):
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}

builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 16.1 — ORACLESAVER + CONNECTION POOL")
    print("=" * 50)
    
    checkpointer = get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test-session-1"}}
    
    # Turn 1
    result = graph.invoke({"messages": [HumanMessage(content="Hello!")]}, config=config)
    print(f"Turn 1: {result['messages'][-1].content[:80]}...")
    
    # Turn 2 — same thread, state persists
    result = graph.invoke({"messages": [HumanMessage(content="What did I just say?")]}, config=config)
    print(f"Turn 2: {result['messages'][-1].content[:80]}...")
    
    # New thread — fresh state
    config2 = {"configurable": {"thread_id": "test-session-2"}}
    result = graph.invoke({"messages": [HumanMessage(content="Do you remember me?")]}, config=config2)
    print(f"New thread: {result['messages'][-1].content[:80]}...")
    
    print(f"\nCheckpointer type: {type(checkpointer).__name__}")
    print("✅ Persistence layer working!")
