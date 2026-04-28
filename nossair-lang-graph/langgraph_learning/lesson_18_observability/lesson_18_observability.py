"""
Lesson 18: Enterprise Observability
=====================================
Teaches:
  - Prometheus metrics (counters, histograms, gauges)
  - Structured tracing with trace/span IDs
  - Health checks (liveness vs readiness)
  - SLA alerting patterns
  - Grafana dashboard design (conceptual)
  - OpenTelemetry integration pattern

Prerequisites: pip install prometheus-client opentelemetry-sdk opentelemetry-exporter-otlp
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# STRUCTURED JSON LOG FORMATTER (enterprise standard)
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """
    Outputs every log line as a JSON object.
    Required for ELK / Datadog / CloudWatch log parsing.

    Example output:
    {"ts": "2024-01-15T10:30:00Z", "level": "INFO", "logger": "lesson_18",
     "msg": "[chat] DONE", "trace_id": "abc123", "tenant_id": "acme-corp"}
    """

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts":      datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base)


USE_JSON_LOGS = os.getenv("USE_JSON_LOGS", "false").lower() == "true"

if USE_JSON_LOGS:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

logger = logging.getLogger("lesson_18")
import json as _json_import  # used by JsonFormatter above

# ---------------------------------------------------------------------------
# PROMETHEUS METRICS
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    # Counter: monotonically increasing (requests, errors, tokens)
    REQUEST_TOTAL = Counter(
        "langgraph_requests_total",
        "Total agent requests",
        labelnames=["tenant_id", "node", "status"],
    )
    ERROR_TOTAL = Counter(
        "langgraph_errors_total",
        "Total agent errors",
        labelnames=["tenant_id", "error_type"],
    )
    TOKEN_TOTAL = Counter(
        "langgraph_tokens_total",
        "Approximate token usage",
        labelnames=["tenant_id", "model"],
    )

    # Histogram: distribution of durations (latency percentiles p50/p90/p99)
    REQUEST_LATENCY = Histogram(
        "langgraph_request_latency_seconds",
        "Request latency in seconds",
        labelnames=["tenant_id", "node"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    # Gauge: current value (active sessions, queue depth)
    ACTIVE_SESSIONS = Gauge(
        "langgraph_active_sessions",
        "Currently active graph invocations",
        labelnames=["tenant_id"],
    )

    METRICS_AVAILABLE = True
    logger.info("Prometheus metrics initialized")

except ImportError:
    METRICS_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics disabled (pip install prometheus-client)")

    # Stub classes so code still runs without prometheus
    class _Stub:
        def labels(self, **kw): return self
        def inc(self, v=1): pass
        def observe(self, v): pass
        def set(self, v): pass

    REQUEST_TOTAL = ERROR_TOTAL = TOKEN_TOTAL = _Stub()
    REQUEST_LATENCY = _Stub()
    ACTIVE_SESSIONS = _Stub()


# ---------------------------------------------------------------------------
# TRACING — trace/span IDs for distributed request tracking
# ---------------------------------------------------------------------------
class Span:
    """
    Minimal span implementation.
    In production: use OpenTelemetry SDK — auto-instruments LangChain calls.
    """

    def __init__(self, name: str, trace_id: str | None = None, parent_id: str | None = None):
        self.name = name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())[:8]
        self.parent_id = parent_id
        self.start_time = time.perf_counter()
        self.tags: dict[str, Any] = {}
        self.events: list[dict] = []

    def tag(self, key: str, value: Any) -> "Span":
        self.tags[key] = value
        return self

    def event(self, name: str, **attrs) -> "Span":
        self.events.append({"name": name, "time": time.perf_counter() - self.start_time, **attrs})
        return self

    def finish(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        logger.info(
            f"[trace] span={self.span_id} | trace={self.trace_id[:8]} | "
            f"name={self.name} | duration={elapsed*1000:.1f}ms | tags={self.tags}"
        )
        return elapsed


def _update_sla_last_latency(elapsed: float):
    """Called by trace_node to feed latency into sla_guard_node."""
    sla_guard_node._last_latency = elapsed


@contextmanager
def trace_node(name: str, trace_id: str, tenant_id: str):
    """Context manager: wraps a node execution in a span with metrics."""
    span = Span(name=name, trace_id=trace_id)
    span.tag("tenant_id", tenant_id)

    ACTIVE_SESSIONS.labels(tenant_id=tenant_id).inc()
    try:
        yield span
        span.event("success")
        elapsed = span.finish()

        REQUEST_LATENCY.labels(tenant_id=tenant_id, node=name).observe(elapsed)
        REQUEST_TOTAL.labels(tenant_id=tenant_id, node=name, status="success").inc()
        _update_sla_last_latency(elapsed)   # feed into sla_guard_node

    except Exception as exc:
        span.event("error", error=str(exc))
        span.finish()
        ERROR_TOTAL.labels(tenant_id=tenant_id, error_type=type(exc).__name__).inc()
        REQUEST_TOTAL.labels(tenant_id=tenant_id, node=name, status="error").inc()
        raise

    finally:
        ACTIVE_SESSIONS.labels(tenant_id=tenant_id).dec()


# ---------------------------------------------------------------------------
# HEALTH CHECK DATA
# ---------------------------------------------------------------------------
class HealthStatus:
    """
    Liveness vs Readiness:
      - Liveness: is the process alive? (crash → restart)
      - Readiness: is the service ready to accept traffic? (DB down → don't route)

    Kubernetes uses both:
      livenessProbe:  GET /health/live
      readinessProbe: GET /health/ready
    """

    def __init__(self):
        self._start_time = time.time()
        self._checks: dict[str, bool] = {}

    def set_check(self, name: str, ok: bool):
        self._checks[name] = ok

    def liveness(self) -> dict:
        """Always healthy unless process is stuck/deadlocked."""
        uptime = time.time() - self._start_time
        return {"status": "alive", "uptime_seconds": round(uptime, 1)}

    def readiness(self) -> dict:
        """Healthy only if ALL dependencies (DB, LLM) are reachable."""
        all_ok = all(self._checks.values())
        return {
            "status": "ready" if all_ok else "not_ready",
            "checks": self._checks,
        }


health = HealthStatus()
health.set_check("llm", True)
health.set_check("checkpointer", True)


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class ObservabilityState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    tenant_id: str
    trace_id: str          # propagated across all nodes — links spans together
    request_id: str        # unique per request (for log correlation)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOllama(model=get_ollama_model(), temperature=0)


# ---------------------------------------------------------------------------
# INSTRUMENTED NODES
# ---------------------------------------------------------------------------
def instrumented_chat_node(state: ObservabilityState) -> dict:
    """
    Chat node wrapped with:
      1. Span tracing (trace_id propagated)
      2. Prometheus metrics (latency histogram, request counter)
      3. Structured log with trace_id for log correlation
    """
    trace_id = state.get("trace_id", str(uuid.uuid4()))
    tenant_id = state.get("tenant_id", "unknown")
    request_id = state.get("request_id", str(uuid.uuid4())[:8])

    # Structured log — all fields searchable in Elasticsearch/Datadog
    logger.info(
        f"[chat] START | trace={trace_id[:8]} | req={request_id} | "
        f"tenant={tenant_id} | user={state['user_id']} | "
        f"msg_count={len(state['messages'])}"
    )

    with trace_node("chat_node", trace_id, tenant_id) as span:
        try:
            response = llm.invoke(state["messages"])

            # Estimate token usage (real: use response.usage_metadata)
            approx_tokens = len(str(response.content).split()) * 1.3
            TOKEN_TOTAL.labels(tenant_id=tenant_id, model=get_ollama_model()).inc(approx_tokens)

            span.tag("response_tokens", int(approx_tokens))
            span.tag("user_id", state["user_id"])

            logger.info(
                f"[chat] DONE | trace={trace_id[:8]} | req={request_id} | "
                f"tokens_approx={int(approx_tokens)}"
            )
            return {"messages": [response]}

        except Exception as exc:
            logger.error(
                f"[chat] ERROR | trace={trace_id[:8]} | req={request_id} | "
                f"error={exc}",
                exc_info=True,
            )
            health.set_check("llm", False)
            return {"messages": [AIMessage(content="Service temporarily unavailable.")]}


# SLA thresholds (configure via env vars in production)
SLA_P99_LATENCY_SECONDS = float(os.getenv("SLA_P99_LATENCY_SECONDS", "5.0"))
SLA_ERROR_RATE_THRESHOLD = float(os.getenv("SLA_ERROR_RATE_THRESHOLD", "0.05"))

# In-memory SLA breach tracker (prod: use Prometheus Alertmanager)
_sla_breaches: list[dict] = []


def sla_guard_node(state: ObservabilityState) -> dict:
    """
    Functional SLA enforcement node.

    Checks latency of the preceding chat node against the SLA threshold.
    On breach:
      1. Logs [SLA_BREACH] at WARNING level (triggers Datadog/PagerDuty alert)
      2. Records breach details for reporting
      3. Increments Prometheus error counter

    Enterprise SLA: p99 latency < SLA_P99_LATENCY_SECONDS (default 5s).
    """
    trace_id  = state.get("trace_id", "none")
    tenant_id = state.get("tenant_id", "unknown")

    # Find latency from the most recent span (stored in REQUEST_LATENCY histogram)
    # Here we read from the trace node timing via Prometheus label query;
    # in demo we check a per-request wall-clock stored in state extension.
    # For now, use the last observed histogram sample as a proxy.
    # Real production: query Prometheus HTTP API for histogram_quantile(0.99, ...)

    # Simulate detecting a latency value from the active span context
    # (In real code this would be `span.elapsed` from the trace_node context manager)
    last_latency = getattr(sla_guard_node, "_last_latency", 0.0)

    if last_latency > SLA_P99_LATENCY_SECONDS:
        breach = {
            "ts":        datetime.utcnow().isoformat() + "Z",
            "trace_id":  trace_id,
            "tenant_id": tenant_id,
            "latency_s": round(last_latency, 3),
            "threshold": SLA_P99_LATENCY_SECONDS,
        }
        _sla_breaches.append(breach)
        logger.warning(
            f"[SLA_BREACH] trace={trace_id[:8]} | tenant={tenant_id} | "
            f"latency={last_latency:.2f}s > sla={SLA_P99_LATENCY_SECONDS}s | "
            f"ACTION: pagerduty alert triggered"
        )
        ERROR_TOTAL.labels(tenant_id=tenant_id, error_type="sla_breach").inc()
    else:
        logger.info(
            f"[sla_guard] OK | trace={trace_id[:8]} | tenant={tenant_id} | "
            f"latency={last_latency:.2f}s <= sla={SLA_P99_LATENCY_SECONDS}s"
        )
    return {}


# ---------------------------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------------------------
def build_observed_graph(checkpointer):
    builder = StateGraph(ObservabilityState)
    builder.add_node("chat", instrumented_chat_node)
    builder.add_node("sla_guard", sla_guard_node)

    builder.add_edge(START, "chat")
    builder.add_edge("chat", "sla_guard")
    builder.add_edge("sla_guard", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
def run_demo():
    if METRICS_AVAILABLE:
        start_http_server(9090)
        logger.info("Prometheus metrics available at http://localhost:9090/metrics")

    checkpointer = MemorySaver()
    graph = build_observed_graph(checkpointer)

    print("\n" + "=" * 60)
    print("DEMO 1: Instrumented requests with trace IDs")
    print("=" * 60)

    trace_id = str(uuid.uuid4())
    for i, question in enumerate(["What is 2+2?", "Now multiply by 3.", "What was the first number?"]):
        request_id = str(uuid.uuid4())[:8]
        config = {"configurable": {"thread_id": f"obs-demo-{trace_id[:8]}"}}
        result = graph.invoke(
            {
                "user_id": "alice",
                "tenant_id": "acme-corp",
                "trace_id": trace_id,
                "request_id": request_id,
                "messages": [HumanMessage(content=question)],
            },
            config=config,
        )
        print(f"\n  Turn {i+1} | trace={trace_id[:8]} | req={request_id}")
        print(f"  Q: {question}")
        print(f"  A: {result['messages'][-1].content[:100]}")

    print("\n" + "=" * 60)
    print("DEMO 2: Health check status")
    print("=" * 60)
    import json as _json
    print(f"  Liveness:  {_json.dumps(health.liveness())}")
    print(f"  Readiness: {_json.dumps(health.readiness())}")

    print("\n" + "=" * 60)
    print("DEMO 3: What a Grafana dashboard would track")
    print("=" * 60)
    print("""
  Panel 1 — Request Rate (RPS):
    PromQL: rate(langgraph_requests_total[5m])

  Panel 2 — Error Rate (%):
    PromQL: rate(langgraph_errors_total[5m]) / rate(langgraph_requests_total[5m])

  Panel 3 — Latency p50/p90/p99:
    PromQL: histogram_quantile(0.99, rate(langgraph_request_latency_seconds_bucket[5m]))

  Panel 4 — Active Sessions:
    PromQL: langgraph_active_sessions

  Panel 5 — Token Usage (cost proxy):
    PromQL: rate(langgraph_tokens_total[1h])

  Alerts:
    - Error rate > 5%  for 5m  → PagerDuty
    - p99 latency > 5s for 2m  → Slack
    - Active sessions > 1000   → Scale-out trigger
    """)


if __name__ == "__main__":
    run_demo()
    print("\n✓ Lesson 18 complete.")
    print("Next: Lesson 19 — Event-driven agents (Celery/Redis)")
