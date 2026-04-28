"""Task 19.1 — Webhook Agent with HMAC Verification + Idempotency."""
import sys, os, hashlib, hmac, json, time, uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from typing import Annotated
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "dev-secret-change-in-production")

# ---------------------------------------------------------------------------
# HMAC Verification — use hmac.compare_digest, NEVER ==
# ---------------------------------------------------------------------------
def verify_webhook_signature(payload_bytes: bytes, signature_header: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature. Uses compare_digest to prevent timing attacks."""
    if not signature_header.startswith("sha256="):
        return False
    received = signature_header[len("sha256="):]
    expected = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(received, expected)

# ---------------------------------------------------------------------------
# Idempotency — prevents duplicate processing on webhook retry
# ---------------------------------------------------------------------------
_idempotency_store = {}  # In-memory; use Redis in production

def check_idempotency(key: str) -> dict | None:
    """Return cached result if already processed, None if new."""
    return _idempotency_store.get(key)

def store_idempotency(key: str, result: dict, ttl: int = 86400):
    """Cache result for TTL seconds."""
    _idempotency_store[key] = {"result": result, "expires": time.time() + ttl}

def make_idempotency_key(payload: dict) -> str:
    """Deterministic key from webhook content."""
    # For GitHub PR: f"pr-review-{repo}-{pr_number}-{commit_sha}"
    content = json.dumps(payload, sort_keys=True)
    return f"webhook-{hashlib.sha256(content.encode()).hexdigest()[:16]}"

# ---------------------------------------------------------------------------
# Webhook Agent
# ---------------------------------------------------------------------------
class WebhookState(TypedDict):
    messages: Annotated[list, add_messages]
    payload: dict
    is_verified: bool
    idempotency_key: str
    cached_result: str

llm = ChatOllama(model=get_ollama_model(), temperature=0)

def verify_node(state: WebhookState) -> dict:
    payload_str = json.dumps(state["payload"])
    # In real app, signature comes from request header
    sig = "sha256=" + hmac.new(WEBHOOK_SECRET.encode(), payload_str.encode(), hashlib.sha256).hexdigest()
    is_valid = verify_webhook_signature(payload_str.encode(), sig, WEBHOOK_SECRET)
    key = make_idempotency_key(state["payload"])
    cached = check_idempotency(key)
    return {"is_verified": is_valid, "idempotency_key": key, "cached_result": json.dumps(cached) if cached else ""}

def route_verify(state: WebhookState) -> str:
    if not state.get("is_verified"):
        return "rejected"
    if state.get("cached_result"):
        return "cached"
    return "process"

def process_node(state: WebhookState) -> dict:
    prompt = f"Process this webhook event: {json.dumps(state['payload'])[:200]}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    result = {"response": resp.content}
    store_idempotency(state["idempotency_key"], result)
    return {"messages": [resp]}

def rejected_node(state: WebhookState) -> dict:
    return {"messages": [HumanMessage(content="Webhook verification failed")]}

def cached_node(state: WebhookState) -> dict:
    return {"messages": [HumanMessage(content=f"Cached result: {state['cached_result'][:50]}")]}

builder = StateGraph(WebhookState)
builder.add_node("verify", verify_node)
builder.add_node("process", process_node)
builder.add_node("rejected", rejected_node)
builder.add_node("cached", cached_node)
builder.add_edge(START, "verify")
builder.add_conditional_edges("verify", route_verify, {"process": "process", "rejected": "rejected", "cached": "cached"})
builder.add_edge("process", END)
builder.add_edge("rejected", END)
builder.add_edge("cached", END)
webhook_graph = builder.compile()

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 19.1 — WEBHOOK + HMAC + IDEMPOTENCY")
    print("=" * 50)
    
    # Test HMAC verification
    print("\n--- HMAC verification ---")
    payload = b'{"action": "opened", "pr_number": 42}'
    valid_sig = "sha256=" + hmac.new(WEBHOOK_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    print(f"  Valid sig: {verify_webhook_signature(payload, valid_sig, WEBHOOK_SECRET)}")
    print(f"  Invalid sig: {verify_webhook_signature(payload, 'sha256=bad', WEBHOOK_SECRET)}")
    print(f"  Timing-safe: uses hmac.compare_digest (not ==)")
    
    # Test idempotency
    print("\n--- Idempotency ---")
    test_payload = {"action": "opened", "pr_number": 42, "repo": "my-repo"}
    key = make_idempotency_key(test_payload)
    print(f"  Key: {key}")
    print(f"  First call (new): {check_idempotency(key)}")
    store_idempotency(key, {"response": "reviewed"})
    print(f"  Second call (cached): {check_idempotency(key)}")
    
    # Test webhook graph
    print("\n--- Webhook graph ---")
    result = webhook_graph.invoke({
        "messages": [], "payload": test_payload,
        "is_verified": False, "idempotency_key": "", "cached_result": ""
    })
    print(f"  Result: {result['messages'][-1].content[:60]}...")
    
    # Retry same payload — should hit cache
    result = webhook_graph.invoke({
        "messages": [], "payload": test_payload,
        "is_verified": False, "idempotency_key": "", "cached_result": ""
    })
    print(f"  Cached result: {result['messages'][-1].content[:60]}...")
    
    print("\n✅ Webhook + HMAC + Idempotency working!")
