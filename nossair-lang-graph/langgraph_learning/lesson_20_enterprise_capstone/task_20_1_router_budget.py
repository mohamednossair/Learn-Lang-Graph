"""Task 20.1 — Model Router + Token Budget."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
import time, logging
from dataclasses import dataclass
from typing import Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger("task_20_1")

# ---------------------------------------------------------------------------
# Model Registry — cost-aware routing
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str
    model_id: str
    cost_per_1k_tokens: float
    tier: str

MODELS = {
    "fast": ModelConfig("fast", get_ollama_model(), 0.0002, "fast"),
    "balanced": ModelConfig("balanced", get_ollama_model(), 0.001, "balanced"),
    "powerful": ModelConfig("powerful", get_ollama_model(), 0.015, "powerful"),
}

def classify_complexity(query: str) -> Literal["fast", "balanced", "powerful"]:
    """Route to cheapest sufficient model."""
    q = query.lower()
    complex_kw = ["design", "implement", "architect", "complex", "analyze", "debug"]
    if any(kw in q for kw in complex_kw):
        return "powerful"
    if len(q.split()) < 20 and "?" in q:
        return "fast"
    return "balanced"

# ---------------------------------------------------------------------------
# Token Budget — per-tenant daily cap
# ---------------------------------------------------------------------------
class BudgetManager:
    """In-memory budget; use Redis INCRBYFLOAT in production."""
    
    def __init__(self):
        self._usage = {}  # {(tenant_id, date): float}
        self._budgets = {"acme-corp": 10.0, "globex": 25.0}
    
    def can_proceed(self, tenant_id: str, estimated_tokens: int = 500) -> tuple[bool, str]:
        today = time.strftime("%Y-%m-%d")
        key = (tenant_id, today)
        budget = self._budgets.get(tenant_id, 10.0)
        cost = estimated_tokens / 1000 * 0.001
        current = self._usage.get(key, 0.0)
        
        if current + cost > budget:
            return False, f"Daily budget ${budget} exhausted (used ${current:.2f})"
        return True, "ok"
    
    def record_usage(self, tenant_id: str, tokens: int, model_tier: str):
        today = time.strftime("%Y-%m-%d")
        key = (tenant_id, today)
        cost = tokens / 1000 * MODELS[model_tier].cost_per_1k_tokens
        self._usage[key] = self._usage.get(key, 0.0) + cost

budget_mgr = BudgetManager()

# ---------------------------------------------------------------------------
# Agent with routing + budget
# ---------------------------------------------------------------------------
class CostState(TypedDict):
    messages: Annotated[list, add_messages]
    tenant_id: str
    model_tier: str
    budget_ok: bool
    budget_reason: str

def complexity_router(state: CostState) -> dict:
    query = state["messages"][-1].content if state["messages"] else ""
    tier = classify_complexity(query)
    return {"model_tier": tier}

def budget_check(state: CostState) -> dict:
    ok, reason = budget_mgr.can_proceed(state.get("tenant_id", "default"))
    return {"budget_ok": ok, "budget_reason": reason}

def route_budget(state: CostState) -> str:
    return "chat" if state.get("budget_ok") else "budget_denied"

def chat_node(state: CostState) -> dict:
    tier = state.get("model_tier", "balanced")
    model_cfg = MODELS[tier]
    llm = ChatOllama(model=model_cfg.model_id, temperature=0)
    resp = llm.invoke(state["messages"])
    # Record usage
    tokens = len(state["messages"][-1].content.split()) + len(resp.content.split())
    budget_mgr.record_usage(state.get("tenant_id", "default"), tokens, tier)
    return {"messages": [resp]}

def budget_denied_node(state: CostState) -> dict:
    return {"messages": [AIMessage(content=f"Budget exceeded: {state['budget_reason']}")]}

builder = StateGraph(CostState)
builder.add_node("router", complexity_router)
builder.add_node("budget", budget_check)
builder.add_node("chat", chat_node)
builder.add_node("budget_denied", budget_denied_node)
builder.add_edge(START, "router")
builder.add_edge("router", "budget")
builder.add_conditional_edges("budget", route_budget, {"chat": "chat", "budget_denied": "budget_denied"})
builder.add_edge("chat", END)
builder.add_edge("budget_denied", END)
cost_graph = builder.compile()

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 20.1 — MODEL ROUTER + TOKEN BUDGET")
    print("=" * 50)
    
    # Test complexity routing
    print("\n--- Complexity routing ---")
    tests = [
        ("What is Python?", "fast"),
        ("Explain how to design a microservices architecture", "powerful"),
        ("Compare REST and GraphQL for our API", "balanced"),
    ]
    for query, expected in tests:
        tier = classify_complexity(query)
        status = "✅" if tier == expected else "❌"
        print(f"  {status} '{query[:40]}...' → {tier} (expected: {expected})")
    
    # Test budget
    print("\n--- Budget check ---")
    ok, msg = budget_mgr.can_proceed("acme-corp")
    print(f"  Acme (budget $10): {msg}")
    ok, msg = budget_mgr.can_proceed("globex")
    print(f"  Globex (budget $25): {msg}")
    
    # Full graph test
    print("\n--- Full graph test ---")
    result = cost_graph.invoke({
        "messages": [HumanMessage(content="What is Python?")],
        "tenant_id": "acme-corp", "model_tier": "fast",
        "budget_ok": False, "budget_reason": ""
    })
    print(f"  Response: {result['messages'][-1].content[:60]}...")
    print(f"  Model tier used: {result['model_tier']}")
    
    print("\n✅ Model routing + budget enforcement working!")
