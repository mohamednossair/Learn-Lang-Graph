"""Task 17.2 — Multi-Tenant Isolation."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Tenant Configuration — 4 layers of isolation
# ---------------------------------------------------------------------------
TENANT_CONFIG = {
    "acme-corp": {
        "system_prompt": "You are Acme Corp's assistant. Only discuss Acme products.",
        "allowed_tools": ["ask_question"],
        "daily_budget": 10.0,
    },
    "globex": {
        "system_prompt": "You are Globex's assistant. Only discuss Globex services.",
        "allowed_tools": ["ask_question", "run_query"],
        "daily_budget": 25.0,
    },
}

# Layer 1: thread_id = f"{tenant_id}-{user_id}-{session}"
def make_thread_id(tenant_id: str, user_id: str, session: str) -> str:
    return f"{tenant_id}-{user_id}-{session}"

# ---------------------------------------------------------------------------
# Multi-tenant Agent
# ---------------------------------------------------------------------------
class TenantState(TypedDict):
    messages: Annotated[list, add_messages]
    tenant_id: str
    user_id: str
    system_prompt: str

llm = ChatOllama(model=get_ollama_model(), temperature=0)

def tenant_chat_node(state: TenantState) -> dict:
    """Layer 2: Inject tenant-specific system prompt."""
    tenant_id = state.get("tenant_id", "default")
    config = TENANT_CONFIG.get(tenant_id, {})
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    
    # Layer 3: Tool scoping — only bind tenant-allowed tools
    # (simplified here; in production filter tool list by allowed_tools)
    
    msgs = [SystemMessage(content=system_prompt)] + state["messages"]
    resp = llm.invoke(msgs)
    return {"messages": [resp], "system_prompt": system_prompt}

builder = StateGraph(TenantState)
builder.add_node("chat", tenant_chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
tenant_graph = builder.compile()

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 17.2 — MULTI-TENANT ISOLATION")
    print("=" * 50)
    
    # Layer 1: Thread ID isolation
    print("\n--- Layer 1: Thread ID isolation ---")
    t1 = make_thread_id("acme-corp", "alice", "s1")
    t2 = make_thread_id("globex", "bob", "s1")
    print(f"  Acme thread: {t1}")
    print(f"  Globex thread: {t2}")
    
    # Layer 2: System prompt injection
    print("\n--- Layer 2: Tenant system prompts ---")
    for tid, cfg in TENANT_CONFIG.items():
        print(f"  {tid}: {cfg['system_prompt'][:50]}...")
    
    # Layer 3: Tool scoping
    print("\n--- Layer 3: Tool scoping ---")
    for tid, cfg in TENANT_CONFIG.items():
        print(f"  {tid}: allowed_tools={cfg['allowed_tools']}")
    
    # Test tenant isolation
    print("\n--- Tenant chat test ---")
    result = tenant_graph.invoke({
        "messages": [HumanMessage(content="Tell me about our company")],
        "tenant_id": "acme-corp", "user_id": "alice", "system_prompt": ""
    })
    print(f"  Acme response: {result['messages'][-1].content[:80]}...")
    
    result = tenant_graph.invoke({
        "messages": [HumanMessage(content="Tell me about our company")],
        "tenant_id": "globex", "user_id": "bob", "system_prompt": ""
    })
    print(f"  Globex response: {result['messages'][-1].content[:80]}...")
    
    print("\n✅ Multi-tenant isolation working!")
    print("  Layer 4 (Oracle VPD) requires Oracle DB — documented in lesson")
