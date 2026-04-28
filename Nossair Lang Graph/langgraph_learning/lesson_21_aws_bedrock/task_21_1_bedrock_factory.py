"""Task 21.1 — Bedrock LLM Factory + Cost Tracker."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
import logging, time
from typing import Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger("task_21_1")

# ---------------------------------------------------------------------------
# LLM Factory — feature flag pattern for Ollama → Bedrock migration
# ---------------------------------------------------------------------------
BEDROCK_AVAILABLE = False
try:
    from langchain_aws import ChatBedrockConverse
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    pass

def get_llm(provider: str = None, model_tier: str = "fast"):
    """Factory: returns ChatOllama or ChatBedrockConverse based on env var.
    
    Zero-downtime migration pattern:
      1. Deploy with LLM_PROVIDER=ollama (zero change)
      2. Shadow traffic: log both responses
      3. Canary: 5% to bedrock
      4. Roll to 100%
      5. Rollback = flip env var back
    """
    provider = provider or os.getenv("LLM_PROVIDER", "ollama")
    
    if provider == "bedrock" and BEDROCK_AVAILABLE:
        model_ids = {
            "fast": "anthropic.claude-3-haiku-20240307-v1:0",
            "balanced": "anthropic.claude-3-sonnet-20240229-v1:0",
            "powerful": "anthropic.claude-3-opus-20240229-v1:0",
        }
        model_id = model_ids.get(model_tier, model_ids["fast"])
        region = os.getenv("AWS_REGION", "us-east-1")
        
        # boto3 credential chain: env vars → profile → SSO → IAM role
        # NEVER hardcode AWS_ACCESS_KEY_ID in code
        client = boto3.client("bedrock-runtime", region_name=region)
        return ChatBedrockConverse(
            model=model_id,
            client=client,
            max_tokens=1024,
            temperature=0.1,
        )
    
    # Fallback to Ollama
    return __import__("langchain_ollama", fromlist=["ChatOllama"]).ChatOllama(
        model=get_ollama_model(), temperature=0
    )

# ---------------------------------------------------------------------------
# Cost Tracker — extract token usage from response_metadata
# ---------------------------------------------------------------------------
class CostTracker:
    """Track per-tenant Bedrock costs. Use Redis INCRBYFLOAT in production."""
    
    PRICING = {
        "anthropic.claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "anthropic.claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-opus": {"input": 0.015, "output": 0.075},
        "ollama": {"input": 0.0, "output": 0.0},
    }
    
    def __init__(self):
        self._usage = {}  # {(tenant_id, date): {"tokens_in": int, "tokens_out": int, "cost": float}}
    
    def record(self, tenant_id: str, model_id: str, input_tokens: int, output_tokens: int):
        today = time.strftime("%Y-%m-%d")
        key = (tenant_id, today)
        
        # Find pricing tier
        pricing = self.PRICING.get("ollama", {"input": 0, "output": 0})
        for model_prefix, prices in self.PRICING.items():
            if model_prefix in model_id:
                pricing = prices
                break
        
        cost = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])
        
        if key not in self._usage:
            self._usage[key] = {"tokens_in": 0, "tokens_out": 0, "cost": 0.0}
        self._usage[key]["tokens_in"] += input_tokens
        self._usage[key]["tokens_out"] += output_tokens
        self._usage[key]["cost"] += cost
    
    def get_usage(self, tenant_id: str) -> dict:
        today = time.strftime("%Y-%m-%d")
        return self._usage.get((tenant_id, today), {"tokens_in": 0, "tokens_out": 0, "cost": 0.0})

cost_tracker = CostTracker()

# ---------------------------------------------------------------------------
# Agent with LLM factory + cost tracking
# ---------------------------------------------------------------------------
class BedrockState(TypedDict):
    messages: Annotated[list, add_messages]
    tenant_id: str
    model_tier: str
    provider: str

def chat_node(state: BedrockState) -> dict:
    provider = state.get("provider", os.getenv("LLM_PROVIDER", "ollama"))
    tier = state.get("model_tier", "fast")
    llm = get_llm(provider=provider, model_tier=tier)
    
    resp = llm.invoke(state["messages"])
    
    # Extract token usage from response_metadata
    usage = resp.response_metadata.get("usage", {})
    input_tokens = usage.get("input_tokens", len(state["messages"][-1].content.split()))
    output_tokens = usage.get("output_tokens", len(resp.content.split()))
    
    model_id = getattr(llm, "model_id", getattr(llm, "model", "ollama"))
    cost_tracker.record(state.get("tenant_id", "default"), str(model_id), input_tokens, output_tokens)
    
    return {"messages": [resp]}

builder = StateGraph(BedrockState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
bedrock_graph = builder.compile()

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 21.1 — BEDROCK LLM FACTORY + COST TRACKER")
    print("=" * 50)
    
    # Test LLM factory
    print("\n--- LLM Factory ---")
    provider = os.getenv("LLM_PROVIDER", "ollama")
    print(f"  Provider: {provider}")
    print(f"  Bedrock available: {BEDROCK_AVAILABLE}")
    
    llm = get_llm()
    print(f"  LLM type: {type(llm).__name__}")
    
    # Test cost tracker
    print("\n--- Cost Tracker ---")
    cost_tracker.record("acme-corp", "anthropic.claude-3-haiku-20240307", 500, 200)
    cost_tracker.record("acme-corp", "anthropic.claude-3-sonnet-20240229", 1000, 500)
    usage = cost_tracker.get_usage("acme-corp")
    print(f"  Acme usage: {usage}")
    
    # Test graph
    print("\n--- Graph test ---")
    result = bedrock_graph.invoke({
        "messages": [HumanMessage(content="Hello!")],
        "tenant_id": "acme-corp", "model_tier": "fast", "provider": "ollama"
    })
    print(f"  Response: {result['messages'][-1].content[:60]}...")
    print(f"  Usage: {cost_tracker.get_usage('acme-corp')}")
    
    print("\n--- Migration pattern ---")
    print("  1. LLM_PROVIDER=ollama (current)")
    print("  2. LLM_PROVIDER=bedrock (after migration)")
    print("  3. Rollback = flip env var back")
    print("\n✅ Bedrock factory + cost tracking working!")
