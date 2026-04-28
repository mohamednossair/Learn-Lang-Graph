"""Task 17.1 — JWT Auth + RBAC Agent."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
import hashlib, hmac, json, time, logging
from datetime import datetime, timedelta, timezone
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger("task_17_1")

# ---------------------------------------------------------------------------
# JWT Implementation (simplified — use python-jose in production)
# ---------------------------------------------------------------------------
import base64

SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * padding)

def create_token(payload: dict, expires_min: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    payload["exp"] = int((datetime.now(timezone.utc) + timedelta(minutes=expires_min)).timestamp())
    payload["iat"] = int(datetime.now(timezone.utc).timestamp())
    header = _b64url_encode(json.dumps({"alg": ALGORITHM, "typ": "JWT"}).encode())
    body = _b64url_encode(json.dumps(payload).encode())
    sig = hmac.new(SECRET_KEY.encode(), f"{header}.{body}".encode(), hashlib.sha256).hexdigest()
    return f"{header}.{body}.{sig}"

def decode_token(token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token format")
    # Verify signature
    expected_sig = hmac.new(SECRET_KEY.encode(), f"{parts[0]}.{parts[1]}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(parts[2], expected_sig):
        raise ValueError("Invalid signature")
    payload = json.loads(_b64url_decode(parts[1]))
    if payload.get("exp", 0) < datetime.now(timezone.utc).timestamp():
        raise ValueError("Token expired")
    return payload

# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------
ROLE_PERMISSIONS = {
    "viewer":  {"ask_question", "view_history"},
    "analyst": {"ask_question", "view_history", "run_query", "export_data"},
    "admin":   {"ask_question", "view_history", "run_query", "export_data", "delete_history", "manage_users"},
}

def has_permission(role: str, action: str) -> bool:
    perms = ROLE_PERMISSIONS.get(role, set())
    return action in perms or "*" in perms

# ---------------------------------------------------------------------------
# Agent with Auth + RBAC
# ---------------------------------------------------------------------------
class AuthState(TypedDict):
    messages: Annotated[list, add_messages]
    user_role: str
    action: str
    is_authorized: bool
    denial_reason: str

llm = ChatOllama(model=get_ollama_model(), temperature=0)

def auth_node(state: AuthState) -> dict:
    """Central auth check — always first node."""
    role = state.get("user_role", "viewer")
    action = state.get("action", "ask_question")
    if has_permission(role, action):
        return {"is_authorized": True, "denial_reason": ""}
    return {"is_authorized": False, "denial_reason": f"Role '{role}' lacks permission: {action}"}

def route_auth(state: AuthState) -> str:
    return "chat" if state.get("is_authorized") else "denied"

def chat_node(state: AuthState) -> dict:
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}

def denied_node(state: AuthState) -> dict:
    return {"messages": [AIMessage(content=f"Access denied: {state['denial_reason']}")]}

builder = StateGraph(AuthState)
builder.add_node("auth", auth_node)
builder.add_node("chat", chat_node)
builder.add_node("denied", denied_node)
builder.add_edge(START, "auth")
builder.add_conditional_edges("auth", route_auth, {"chat": "chat", "denied": "denied"})
builder.add_edge("chat", END)
builder.add_edge("denied", END)
auth_graph = builder.compile()

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 17.1 — JWT AUTH + RBAC AGENT")
    print("=" * 50)
    
    # Create tokens
    viewer_token = create_token({"sub": "alice", "role": "viewer", "tenant_id": "acme"})
    analyst_token = create_token({"sub": "bob", "role": "analyst", "tenant_id": "acme"})
    admin_token = create_token({"sub": "carol", "role": "admin", "tenant_id": "acme"})
    
    # Decode and test
    print("\n--- Token decoding ---")
    for name, token in [("viewer", viewer_token), ("analyst", analyst_token), ("admin", admin_token)]:
        claims = decode_token(token)
        print(f"  {name}: sub={claims['sub']}, role={claims['role']}, exp={claims['exp']}")
    
    # RBAC tests
    print("\n--- RBAC permission checks ---")
    tests = [
        ("viewer", "ask_question", True),
        ("viewer", "delete_history", False),
        ("analyst", "run_query", True),
        ("analyst", "manage_users", False),
        ("admin", "delete_history", True),
    ]
    for role, action, expected in tests:
        result = has_permission(role, action)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {role}.{action} = {result}")
    
    # Graph tests
    print("\n--- Graph auth routing ---")
    result = auth_graph.invoke({
        "messages": [HumanMessage(content="Hello")],
        "user_role": "viewer", "action": "ask_question",
        "is_authorized": False, "denial_reason": ""
    })
    print(f"  Viewer ask_question: {result['messages'][-1].content[:60]}...")
    
    result = auth_graph.invoke({
        "messages": [HumanMessage(content="Delete my history")],
        "user_role": "viewer", "action": "delete_history",
        "is_authorized": False, "denial_reason": ""
    })
    print(f"  Viewer delete_history: {result['messages'][-1].content[:60]}...")
    
    print("\n✅ JWT + RBAC working!")
