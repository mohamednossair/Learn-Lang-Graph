"""Task 18.1 — Prometheus Metrics Agent."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
import time, logging
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

PROMETHEUS_AVAILABLE = False
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger("task_18_1")

# ---------------------------------------------------------------------------
# Metric definitions — use Histogram for latency (not Summary!)
# ---------------------------------------------------------------------------
if PROMETHEUS_AVAILABLE:
    REQUEST_TOTAL = Counter("langgraph_requests_total", "Total requests", ["tenant_id", "status"])
    ERROR_TOTAL = Counter("langgraph_errors_total", "Total errors", ["error_type"])
    LATENCY = Histogram("langgraph_request_latency_seconds", "Request latency",
                        ["tenant_id"], buckets=[0.5, 1, 2, 5, 10, 30])
    ACTIVE_SESSIONS = Gauge("langgraph_active_sessions", "Active sessions", ["tenant_id"])
    TOKENS_TOTAL = Counter("langgraph_tokens_total", "Token usage", ["tenant_id", "direction"])

class MetricsState(TypedDict):
    messages: Annotated[list, add_messages]
    tenant_id: str
    start_time: float

llm = ChatOllama(model=get_ollama_model(), temperature=0)

def chat_with_metrics(state: MetricsState) -> dict:
    tenant_id = state.get("tenant_id", "default")
    start = state.get("start_time", time.time())
    
    try:
        resp = llm.invoke(state["messages"])
        if PROMETHEUS_AVAILABLE:
            LATENCY.labels(tenant_id=tenant_id).observe(time.time() - start)
            REQUEST_TOTAL.labels(tenant_id=tenant_id, status="success").inc()
        return {"messages": [resp]}
    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            ERROR_TOTAL.labels(error_type=type(e).__name__).inc()
            REQUEST_TOTAL.labels(tenant_id=tenant_id, status="error").inc()
        return {"messages": [AIMessage(content=f"Error: {e}")]}

builder = StateGraph(MetricsState)
builder.add_node("chat", chat_with_metrics)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
metrics_graph = builder.compile()

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 18.1 — PROMETHEUS METRICS AGENT")
    print("=" * 50)
    
    if PROMETHEUS_AVAILABLE:
        start_http_server(9095)
        print("Metrics server on http://localhost:9095/metrics")
    else:
        print("pip install prometheus-client for metrics endpoint")
    
    # Simulate requests
    for i in range(3):
        result = metrics_graph.invoke({
            "messages": [HumanMessage(content=f"Question {i+1}")],
            "tenant_id": "acme-corp",
            "start_time": time.time()
        })
        print(f"  Q{i+1}: {result['messages'][-1].content[:60]}...")
    
    if PROMETHEUS_AVAILABLE:
        print("\nKey PromQL queries:")
        print("  rate(langgraph_requests_total[5m])")
        print("  histogram_quantile(0.99, rate(langgraph_request_latency_seconds_bucket[5m]))")
        print("  rate(langgraph_errors_total[5m]) / rate(langgraph_requests_total[5m])")
    
    print("\n✅ Metrics instrumentation working!")
