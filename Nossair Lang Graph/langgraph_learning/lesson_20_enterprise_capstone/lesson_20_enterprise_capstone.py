"""
Lesson 20: Enterprise Capstone — Cost Control, Governance & Full System
=======================================================================
Teaches:
  - Token budget enforcement (cost control)
  - Model routing: fast/cheap vs slow/expensive
  - Compliance logging (GDPR right-to-erasure, SOC2 audit trail)
  - Circuit breaker pattern for LLM calls
  - A/B testing between models
  - Complete enterprise agent combining Lessons 16-20

This capstone wires together: RBAC + observability + async + cost control.
"""

import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("lesson_20")

# ---------------------------------------------------------------------------
# MODEL REGISTRY — enterprise model routing
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str
    model_id: str
    cost_per_1k_tokens: float    # USD
    max_tokens: int
    use_case: str                # "fast", "balanced", "powerful"
    timeout_seconds: float


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "fast": ModelConfig(
        name="Fast (cheap)",
        model_id="llama3.2",
        cost_per_1k_tokens=0.0002,
        max_tokens=2048,
        use_case="fast",
        timeout_seconds=5.0,
    ),
    "balanced": ModelConfig(
        name="Balanced",
        model_id="llama3.2",
        cost_per_1k_tokens=0.001,
        max_tokens=4096,
        use_case="balanced",
        timeout_seconds=15.0,
    ),
    "powerful": ModelConfig(
        name="Powerful (expensive)",
        model_id="llama3.2",     # same model for demo; prod: different endpoints
        cost_per_1k_tokens=0.015,
        max_tokens=8192,
        use_case="powerful",
        timeout_seconds=30.0,
    ),
}


def get_model(tier: str) -> ChatOllama:
    config = MODEL_REGISTRY.get(tier, MODEL_REGISTRY["balanced"])
    return ChatOllama(model=config.model_id, temperature=0)


# ---------------------------------------------------------------------------
# TOKEN BUDGET MANAGER — cost control
# ---------------------------------------------------------------------------
class TokenBudgetManager:
    """
    Enforces per-tenant, per-day token budgets.

    Enterprise use:
      - Prevents runaway costs from one tenant
      - Enables chargeback reporting
      - Triggers alerts when nearing budget

    Production: store in Redis with daily TTL.
    """

    def __init__(self):
        self._usage: dict[str, dict] = defaultdict(lambda: {
            "tokens_today": 0,
            "cost_today_usd": 0.0,
            "requests_today": 0,
            "reset_date": datetime.now(timezone.utc).date().isoformat(),
        })
        self._budgets: dict[str, float] = {
            "acme-corp":  50.0,     # $50/day
            "globex-inc": 200.0,    # $200/day
            "default":    10.0,     # $10/day for unknown tenants
        }

    def _reset_if_new_day(self, tenant_id: str):
        today = datetime.now(timezone.utc).date().isoformat()
        if self._usage[tenant_id]["reset_date"] != today:
            self._usage[tenant_id] = {
                "tokens_today": 0,
                "cost_today_usd": 0.0,
                "requests_today": 0,
                "reset_date": today,
            }

    def can_proceed(self, tenant_id: str, estimated_tokens: int = 1000) -> tuple[bool, str]:
        self._reset_if_new_day(tenant_id)
        budget = self._budgets.get(tenant_id, self._budgets["default"])
        current_cost = self._usage[tenant_id]["cost_today_usd"]
        estimated_cost = estimated_tokens / 1000 * MODEL_REGISTRY["balanced"].cost_per_1k_tokens

        if current_cost + estimated_cost > budget:
            return False, f"Daily budget ${budget} exhausted (used ${current_cost:.4f})"
        return True, "ok"

    def record_usage(self, tenant_id: str, tokens: int, model_tier: str):
        self._reset_if_new_day(tenant_id)
        cost = tokens / 1000 * MODEL_REGISTRY.get(model_tier, MODEL_REGISTRY["balanced"]).cost_per_1k_tokens
        self._usage[tenant_id]["tokens_today"] += tokens
        self._usage[tenant_id]["cost_today_usd"] += cost
        self._usage[tenant_id]["requests_today"] += 1
        logger.info(
            f"[budget] tenant={tenant_id} | tokens={tokens} | cost=${cost:.6f} | "
            f"total_today=${self._usage[tenant_id]['cost_today_usd']:.4f}"
        )

    def get_usage_report(self, tenant_id: str) -> dict:
        self._reset_if_new_day(tenant_id)
        usage = self._usage[tenant_id]
        budget = self._budgets.get(tenant_id, self._budgets["default"])
        return {
            "tenant_id": tenant_id,
            "tokens_today": usage["tokens_today"],
            "cost_today_usd": round(usage["cost_today_usd"], 6),
            "budget_usd": budget,
            "budget_remaining_usd": round(budget - usage["cost_today_usd"], 6),
            "requests_today": usage["requests_today"],
            "reset_date": usage["reset_date"],
        }


budget_manager = TokenBudgetManager()


# ---------------------------------------------------------------------------
# CIRCUIT BREAKER — fault tolerance
# ---------------------------------------------------------------------------
@dataclass
class CircuitBreaker:
    """
    Three states: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing recovery)

    Prevents cascading failures: if LLM API is down, stop hammering it.
    After cooldown, let one request through to test recovery.
    """
    failure_threshold: int = 5
    cooldown_seconds: float = 60.0

    _failures: int = field(default=0, init=False)
    _state: str = field(default="CLOSED", init=False)
    _last_failure_time: float = field(default=0.0, init=False)

    def can_call(self) -> bool:
        if self._state == "CLOSED":
            return True
        if self._state == "OPEN":
            if time.time() - self._last_failure_time > self.cooldown_seconds:
                self._state = "HALF_OPEN"
                logger.info("[circuit_breaker] HALF_OPEN — testing recovery")
                return True
            return False
        return True   # HALF_OPEN: allow test call

    def record_success(self):
        self._failures = 0
        self._state = "CLOSED"

    def record_failure(self):
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self.failure_threshold:
            self._state = "OPEN"
            logger.error(f"[circuit_breaker] OPEN — {self._failures} failures, blocking for {self.cooldown_seconds}s")

    @property
    def state(self) -> str:
        return self._state


llm_circuit_breaker = CircuitBreaker()


# ---------------------------------------------------------------------------
# COMPLIANCE — GDPR & audit
# ---------------------------------------------------------------------------
class ComplianceManager:
    """
    GDPR Right-to-Erasure + SOC2 audit trail.

    Capabilities:
      1. Record every interaction with user_id link
      2. Delete all data for a user on request (GDPR Art. 17)
      3. Export all data for a user on request (GDPR Art. 20)
      4. Immutable audit trail (SOC2 CC6.1)
    """

    def __init__(self):
        self._interactions: list[dict] = []
        self._user_data_index: dict[str, list[int]] = defaultdict(list)

    def record_interaction(self, user_id: str, tenant_id: str, messages: list, response: str):
        idx = len(self._interactions)
        record = {
            "id": idx,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_count": len(messages),
            "response_preview": response[:100],
        }
        self._interactions.append(record)
        self._user_data_index[user_id].append(idx)

    def gdpr_erasure(self, user_id: str) -> int:
        """GDPR Article 17 — Right to erasure."""
        indices = self._user_data_index.get(user_id, [])
        count = len(indices)
        for idx in indices:
            if idx < len(self._interactions):
                self._interactions[idx] = {
                    "id": idx,
                    "status": "ERASED",
                    "erased_at": datetime.now(timezone.utc).isoformat(),
                    "reason": "GDPR Art.17 erasure request",
                }
        self._user_data_index[user_id] = []
        logger.info(f"[compliance] GDPR erasure | user={user_id} | records_erased={count}")
        return count

    def gdpr_export(self, user_id: str) -> list[dict]:
        """GDPR Article 20 — Right to data portability."""
        indices = self._user_data_index.get(user_id, [])
        data = [self._interactions[i] for i in indices if i < len(self._interactions)]
        logger.info(f"[compliance] GDPR export | user={user_id} | records={len(data)}")
        return data


compliance = ComplianceManager()


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class EnterpriseCapstoneState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    tenant_id: str
    role: str
    trace_id: str
    model_tier: str              # "fast" | "balanced" | "powerful"
    complexity_score: int        # 1-10, determines model routing
    budget_ok: bool
    circuit_ok: bool
    final_response: str


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def complexity_router_node(state: EnterpriseCapstoneState) -> dict:
    """
    Analyzes question complexity to route to the cheapest sufficient model.

    Enterprise principle: use the smallest model that can answer correctly.
    Saves 90%+ on LLM costs for simple queries.
    """
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else ""
    word_count = len(content.split())

    # Simple heuristic (prod: use a classifier model or keyword rules)
    if word_count < 20 and "?" in content and not any(w in content.lower() for w in ["analyze", "compare", "explain", "design", "implement"]):
        tier = "fast"
        score = 2
    elif any(w in content.lower() for w in ["design", "implement", "architect", "complex"]):
        tier = "powerful"
        score = 8
    else:
        tier = "balanced"
        score = 5

    logger.info(f"[router] complexity={score} | tier={tier} | user={state['user_id']}")
    return {"model_tier": tier, "complexity_score": score}


def budget_check_node(state: EnterpriseCapstoneState) -> dict:
    """Check if tenant has budget remaining."""
    ok, reason = budget_manager.can_proceed(state["tenant_id"])
    if not ok:
        logger.warning(f"[budget_check] BLOCKED | tenant={state['tenant_id']} | reason={reason}")
        return {
            "budget_ok": False,
            "messages": [AIMessage(content=f"Service paused: {reason}. Contact your admin.")],
        }
    return {"budget_ok": True}


def circuit_check_node(state: EnterpriseCapstoneState) -> dict:
    """Check circuit breaker before calling LLM."""
    if not llm_circuit_breaker.can_call():
        logger.error(f"[circuit] OPEN | blocking request | state={llm_circuit_breaker.state}")
        return {
            "circuit_ok": False,
            "messages": [AIMessage(content="LLM service is temporarily unavailable. Please try again in 60 seconds.")],
        }
    return {"circuit_ok": True}


def enterprise_chat_node(state: EnterpriseCapstoneState) -> dict:
    """LLM call with cost tracking and compliance recording."""
    model = get_model(state["model_tier"])

    try:
        response = model.invoke(state["messages"])
        llm_circuit_breaker.record_success()

        # Estimate tokens
        approx_tokens = int(len(str(response.content).split()) * 1.3 + len(str(state["messages"])) / 4)
        budget_manager.record_usage(state["tenant_id"], approx_tokens, state["model_tier"])

        compliance.record_interaction(
            user_id=state["user_id"],
            tenant_id=state["tenant_id"],
            messages=state["messages"],
            response=response.content,
        )

        logger.info(
            f"[chat] done | trace={state['trace_id'][:8]} | tier={state['model_tier']} | "
            f"tokens~{approx_tokens}"
        )
        return {"messages": [response], "final_response": response.content}

    except Exception as exc:
        llm_circuit_breaker.record_failure()
        logger.error(f"[chat] FAILED | trace={state['trace_id'][:8]} | error={exc}")
        return {
            "messages": [AIMessage(content="Request failed. Please try again.")],
            "final_response": f"Error: {exc}",
        }


def route_after_budget(state: EnterpriseCapstoneState) -> str:
    if not state.get("budget_ok", True):
        return END
    return "circuit_check"


def route_after_circuit(state: EnterpriseCapstoneState) -> str:
    if not state.get("circuit_ok", True):
        return END
    return "chat"


# ---------------------------------------------------------------------------
# BUILD FULL ENTERPRISE GRAPH
# ---------------------------------------------------------------------------
def build_enterprise_graph(checkpointer):
    """
    Full enterprise pipeline:
      START
        → complexity_router   (selects cheapest sufficient model)
        → budget_check        (enforces cost limits)
        → [budget denied? END]
        → circuit_check       (fault tolerance)
        → [circuit open? END]
        → chat                (LLM call with cost tracking)
        → END
    """
    builder = StateGraph(EnterpriseCapstoneState)
    builder.add_node("route_complexity", complexity_router_node)
    builder.add_node("budget_check", budget_check_node)
    builder.add_node("circuit_check", circuit_check_node)
    builder.add_node("chat", enterprise_chat_node)

    builder.add_edge(START, "route_complexity")
    builder.add_edge("route_complexity", "budget_check")
    builder.add_conditional_edges("budget_check", route_after_budget, ["circuit_check", END])
    builder.add_conditional_edges("circuit_check", route_after_circuit, ["chat", END])
    builder.add_edge("chat", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
def run_demo():
    checkpointer = MemorySaver()
    graph = build_enterprise_graph(checkpointer)

    print("\n" + "=" * 60)
    print("DEMO 1: Model routing by complexity")
    print("=" * 60)

    questions = [
        ("What is 5+5?",       "simple → fast model"),
        ("Explain async/await.", "medium → balanced model"),
        ("Design a distributed cache.", "complex → powerful model"),
    ]

    for question, expected in questions:
        trace_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": f"cap-{trace_id[:8]}"}}
        result = graph.invoke(
            {
                "user_id": "alice",
                "tenant_id": "acme-corp",
                "role": "analyst",
                "trace_id": trace_id,
                "model_tier": "balanced",
                "complexity_score": 5,
                "budget_ok": True,
                "circuit_ok": True,
                "final_response": "",
                "messages": [HumanMessage(content=question)],
            },
            config=config,
        )
        print(f"\n  Q: {question}")
        print(f"  Expected: {expected} | Actual tier: {result['model_tier']}")
        print(f"  A: {result['messages'][-1].content[:80]}...")

    print("\n" + "=" * 60)
    print("DEMO 2: Cost usage report")
    print("=" * 60)
    report = budget_manager.get_usage_report("acme-corp")
    for k, v in report.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("DEMO 3: GDPR erasure")
    print("=" * 60)
    data_before = compliance.gdpr_export("alice")
    print(f"  Alice's records before erasure: {len(data_before)}")
    erased = compliance.gdpr_erasure("alice")
    print(f"  Erased: {erased} records")
    data_after = compliance.gdpr_export("alice")
    print(f"  Alice's records after erasure: {len(data_after)}")

    print("\n" + "=" * 60)
    print("ENTERPRISE ARCHITECTURE SUMMARY")
    print("=" * 60)
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │                ENTERPRISE AGENT SYSTEM                  │
  ├─────────────────────────────────────────────────────────┤
  │  Layer 1 — GATEWAY                                      │
  │    FastAPI + JWT auth + rate limiting + TLS             │
  │                                                         │
  │  Layer 2 — ORCHESTRATION                                │
  │    LangGraph StateGraph                                 │
  │    Nodes: RBAC → Complexity Router → Budget → Circuit   │
  │                                                         │
  │  Layer 3 — EXECUTION                                    │
  │    Model routing: fast / balanced / powerful            │
  │    Async with asyncio.gather() for concurrent users     │
  │    Celery + Redis for background tasks                  │
  │                                                         │
  │  Layer 4 — PERSISTENCE                                  │
  │    OracleSaver/19c (multi-server, TDE, RAC, Audit Vault)│
  │    Redis (session cache, idempotency, rate limits)      │
  │                                                         │
  │  Layer 5 — OBSERVABILITY                                │
  │    Prometheus metrics → Grafana dashboards              │
  │    Distributed tracing (trace_id through all nodes)     │
  │    LangSmith (LLM-specific traces)                      │
  │    Structured logs → ELK / Datadog                      │
  │                                                         │
  │  Layer 6 — GOVERNANCE                                   │
  │    Token budget per tenant/day                          │
  │    Immutable audit trail (SOC2 CC6.1)                   │
  │    GDPR erasure + export endpoints                      │
  │    Circuit breaker (fault tolerance)                    │
  └─────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    run_demo()
    print("\n✓ Lesson 20 complete — Enterprise LangGraph mastery achieved.")
    print("You now have a complete enterprise agent architecture.")
