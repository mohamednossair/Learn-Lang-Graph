"""Task 18.2 — Trace ID + Health Checks."""
import sys, os, uuid, time, logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger("task_18_2")

# ---------------------------------------------------------------------------
# Trace ID — generate at API boundary, propagate through state
# ---------------------------------------------------------------------------
class TraceState(TypedDict):
    messages: Annotated[list, add_messages]
    trace_id: str
    tenant_id: str

llm = ChatOllama(model=get_ollama_model(), temperature=0)

def chat_with_trace(state: TraceState) -> dict:
    trace = state.get("trace_id", "no-trace")[:8]
    logger.info(f"[chat] trace={trace} | processing message")
    resp = llm.invoke(state["messages"])
    logger.info(f"[chat] trace={trace} | DONE")
    return {"messages": [resp]}

builder = StateGraph(TraceState)
builder.add_node("chat", chat_with_trace)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
trace_graph = builder.compile()

# ---------------------------------------------------------------------------
# Health checks — liveness vs readiness
# ---------------------------------------------------------------------------
def liveness_check():
    """Is process alive? (restart if fails)"""
    # Never check external services here — DB outage should NOT restart pods
    return {"status": "alive"}

def readiness_check(llm_available=True, db_available=True):
    """Is service ready for traffic? (remove from LB if fails)"""
    checks = {
        "llm": llm_available,
        "database": db_available,
    }
    all_ok = all(checks.values())
    return {"status": "ready" if all_ok else "not_ready", "checks": checks}

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 18.2 — TRACE ID + HEALTH CHECKS")
    print("=" * 50)
    
    # Trace ID test
    print("\n--- Trace ID propagation ---")
    trace_id = str(uuid.uuid4())
    result = trace_graph.invoke({
        "messages": [HumanMessage(content="Hello")],
        "trace_id": trace_id,
        "tenant_id": "acme-corp"
    })
    print(f"  trace_id: {trace_id[:8]}...")
    print(f"  Response: {result['messages'][-1].content[:60]}...")
    
    # Health checks
    print("\n--- Health checks ---")
    print(f"  Liveness: {liveness_check()}")
    print(f"  Readiness (all OK): {readiness_check()}")
    print(f"  Readiness (DB down): {readiness_check(db_available=False)}")
    
    print("\n--- Kubernetes probe design ---")
    print("  livenessProbe:  /health/live  (process alive only)")
    print("  readinessProbe: /health/ready (dependencies reachable)")
    print("  Rule: liveness never depends on external services")
    
    print("\n✅ Tracing + health checks working!")
