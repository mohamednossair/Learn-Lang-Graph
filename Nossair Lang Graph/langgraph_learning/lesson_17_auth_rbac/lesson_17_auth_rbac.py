"""
Lesson 17: Enterprise Auth, RBAC & Multi-Tenancy
=================================================
Teaches:
  - JWT-based authentication for agent APIs
  - Role-Based Access Control (RBAC) inside agent nodes
  - Multi-tenant isolation: data, state, tools per tenant
  - Audit logging for compliance (SOC2, HIPAA, GDPR)
  - Secrets management pattern

Prerequisites: pip install python-jose[cryptography] passlib[bcrypt]
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated

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
logger = logging.getLogger("lesson_17")

# ---------------------------------------------------------------------------
# ROLES & PERMISSIONS
# ---------------------------------------------------------------------------
# Enterprise RBAC: roles map to allowed actions
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "viewer":    {"ask_question", "view_history"},
    "analyst":   {"ask_question", "view_history", "run_query", "export_data"},
    "admin":     {"ask_question", "view_history", "run_query", "export_data",
                  "delete_history", "manage_users", "view_audit_log"},
    "superuser": {"*"},   # all permissions
}


def has_permission(role: str, action: str) -> bool:
    """Check if a role has permission for an action."""
    perms = ROLE_PERMISSIONS.get(role, set())
    return "*" in perms or action in perms


# ---------------------------------------------------------------------------
# JWT — real implementation with python-jose
# ---------------------------------------------------------------------------
JWT_SECRET      = os.getenv("JWT_SECRET", "change-me-in-production-use-32-chars-min")
JWT_ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES",  "30"))
REFRESH_TOKEN_EXPIRE_MINUTES = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", "10080"))  # 7 days

try:
    from jose import JWTError, jwt as jose_jwt
    JWT_AVAILABLE = True
    logger.info("python-jose available — using real JWT")
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("python-jose not installed — using insecure mock JWT (pip install python-jose[cryptography])")


def create_access_token(user_id: str, tenant_id: str, role: str) -> str:
    """
    Create a signed JWT access token.

    Payload (claims):
      sub        — subject (user_id)
      tenant_id  — which tenant this user belongs to
      role       — RBAC role
      exp        — expiry (now + ACCESS_TOKEN_EXPIRE_MINUTES)
      iat        — issued at
      type       — "access" (distinguishes from refresh tokens)
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub":       user_id,
        "tenant_id": tenant_id,
        "role":      role,
        "iat":       now,
        "exp":       now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "type":      "access",
    }
    if JWT_AVAILABLE:
        return jose_jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    # Insecure fallback (dev only)
    import base64
    encoded = base64.b64encode(json.dumps({**payload, "iat": now.isoformat(), "exp": (now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)).isoformat()}).encode()).decode()
    sig = hashlib.sha256(f"{encoded}:{JWT_SECRET}".encode()).hexdigest()[:16]
    return f"mock.{encoded}.{sig}"


def create_refresh_token(user_id: str, tenant_id: str) -> str:
    """
    Create a long-lived refresh token.
    Refresh tokens only carry sub + tenant_id (no role — role is re-fetched on refresh).
    Store the hash of this token in DB to enable single-use / revocation.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub":       user_id,
        "tenant_id": tenant_id,
        "iat":       now,
        "exp":       now + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES),
        "type":      "refresh",
    }
    if JWT_AVAILABLE:
        return jose_jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return f"refresh-mock-{user_id}-{tenant_id}"


def decode_token(token: str) -> dict[str, str]:
    """
    Decode and VERIFY a JWT token.

    Raises PermissionError on:
      - Invalid signature       (token tampered)
      - Expired token           (exp claim in the past)
      - Wrong algorithm         (algorithm confusion attack)

    Enterprise notes:
      - Never catch JWTError silently — always re-raise as PermissionError
      - Always verify algorithm explicitly (prevents alg=none attack)
      - Log failed decode attempts with IP for security monitoring
    """
    if JWT_AVAILABLE:
        try:
            payload = jose_jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return {
                "user_id":   payload["sub"],
                "tenant_id": payload["tenant_id"],
                "role":      payload.get("role", "viewer"),
                "token_type": payload.get("type", "access"),
            }
        except jose_jwt.ExpiredSignatureError:
            raise PermissionError("Token expired — please refresh")
        except JWTError as exc:
            raise PermissionError(f"Invalid token: {exc}") from exc
    # Insecure mock fallback
    import base64
    try:
        parts = token.split(".")
        raw = json.loads(base64.b64decode(parts[1]).decode())
        return {"user_id": raw.get("sub", ""), "tenant_id": raw.get("tenant_id", ""),
                "role": raw.get("role", "viewer"), "token_type": raw.get("type", "access")}
    except Exception as exc:
        raise PermissionError(f"Invalid mock token: {exc}") from exc


def extract_token_from_header(authorization: str) -> str:
    """
    Extract Bearer token from HTTP Authorization header.
    FastAPI usage: Authorization: Bearer <token>

    Example FastAPI dependency:
        from fastapi import Header, Depends
        async def get_current_user(authorization: str = Header(...)):
            token = extract_token_from_header(authorization)
            return decode_token(token)
    """
    if not authorization.startswith("Bearer "):
        raise PermissionError("Authorization header must start with 'Bearer '")
    return authorization[len("Bearer "):]


# ---------------------------------------------------------------------------
# AUDIT LOGGER — compliance-grade (SOC2, HIPAA, GDPR)
# ---------------------------------------------------------------------------
class AuditLogger:
    """
    Structured audit log: every sensitive action recorded with:
      - WHO  (user_id, tenant_id, role)
      - WHAT (action, resource)
      - WHEN (ISO timestamp)
      - RESULT (success/denied/error)

    In production: ship to SIEM (Splunk, Elastic, Datadog)
    """

    def __init__(self):
        self._log: list[dict] = []

    def record(
        self,
        user_id: str,
        tenant_id: str,
        role: str,
        action: str,
        resource: str,
        result: str,
        detail: str = "",
    ):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "tenant_id": tenant_id,
            "role": role,
            "action": action,
            "resource": resource,
            "result": result,
            "detail": detail,
        }
        self._log.append(entry)
        log_level = logging.WARNING if result == "DENIED" else logging.INFO
        logger.log(log_level, f"[AUDIT] {json.dumps(entry)}")

    def get_log(self, tenant_id: str | None = None) -> list[dict]:
        if tenant_id:
            return [e for e in self._log if e["tenant_id"] == tenant_id]
        return list(self._log)


audit = AuditLogger()


# ---------------------------------------------------------------------------
# TENANT CONFIGURATION (multi-tenancy)
# ---------------------------------------------------------------------------
# Each tenant has its own:
#   - system prompt (branding / rules)
#   - allowed tools
#   - data namespace
TENANT_CONFIG: dict[str, dict] = {
    "acme-corp": {
        "system_prompt": "You are Acme Corp's AI assistant. Only discuss Acme products.",
        "allowed_tools": ["search_products", "track_order"],
        "max_requests_per_hour": 500,
        "data_prefix": "acme",       # all their state stored under this prefix
    },
    "globex-inc": {
        "system_prompt": "You are Globex's financial AI. Never share client PII.",
        "allowed_tools": ["run_query", "generate_report"],
        "max_requests_per_hour": 1000,
        "data_prefix": "globex",
    },
}


def get_tenant_config(tenant_id: str) -> dict:
    config = TENANT_CONFIG.get(tenant_id)
    if not config:
        raise ValueError(f"Unknown tenant: {tenant_id}")
    return config


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class RBACState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    tenant_id: str
    role: str
    action_requested: str
    auth_result: str           # "allowed" | "denied"
    audit_trail: list[str]     # human-readable audit entries for this session


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOllama(model="llama3.2", temperature=0)


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def auth_node(state: RBACState) -> dict:
    """
    Gate node — runs BEFORE any real work.
    Checks RBAC permission and records audit event.

    Enterprise pattern: ALL entry points go through this node.
    Never check permissions inside business nodes.
    """
    user_id = state["user_id"]
    tenant_id = state["tenant_id"]
    role = state["role"]
    action = state["action_requested"]

    allowed = has_permission(role, action)
    result = "allowed" if allowed else "denied"

    audit.record(
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        action=action,
        resource="agent",
        result=result.upper(),
    )

    trail_entry = f"{datetime.now(timezone.utc).isoformat()} | {user_id} | {role} | {action} → {result.upper()}"

    if not allowed:
        logger.warning(f"[auth] DENIED | user={user_id} | action={action} | role={role}")
        return {
            "auth_result": "denied",
            "audit_trail": state.get("audit_trail", []) + [trail_entry],
            "messages": [AIMessage(content=f"Access denied. Your role '{role}' cannot perform '{action}'.")],
        }

    logger.info(f"[auth] ALLOWED | user={user_id} | action={action}")
    return {
        "auth_result": "allowed",
        "audit_trail": state.get("audit_trail", []) + [trail_entry],
    }


def tenant_chat_node(state: RBACState) -> dict:
    """
    Chat node that enforces tenant isolation:
      1. Loads this tenant's system prompt
      2. Prepends it to messages so LLM stays in-scope
      3. Never sees another tenant's messages
    """
    tenant_id = state["tenant_id"]
    config = get_tenant_config(tenant_id)

    system_msg = SystemMessage(content=config["system_prompt"])
    all_messages = [system_msg] + list(state["messages"])

    response = llm.invoke(all_messages)

    logger.info(f"[tenant_chat] tenant={tenant_id} | user={state['user_id']}")
    return {"messages": [response]}


def access_denied_node(state: RBACState) -> dict:
    """Terminal node for denied requests — already replied in auth_node."""
    logger.info(f"[denied] user={state['user_id']} request blocked")
    return {}


def route_after_auth(state: RBACState) -> str:
    if state.get("auth_result") == "denied":
        return "denied"
    return "rate_limit"


# ---------------------------------------------------------------------------
# RATE LIMITING NODE
# ---------------------------------------------------------------------------
_rate_counters: dict[str, list[float]] = {}   # user_id → list of request timestamps


def rate_limit_node(state: RBACState) -> dict:
    """
    Sliding-window rate limiter: max 50 requests per user per hour.

    Enterprise pattern:
      - In-memory here (demo). Production: use Redis INCR with TTL.
      - Per-user and per-tenant limits are independent.
      - Rate limit response must NOT reveal system internals.
    """
    user_id = state["user_id"]
    tenant_id = state["tenant_id"]
    now = time.time()
    window = 3600   # 1 hour sliding window
    max_requests = TENANT_CONFIG.get(tenant_id, {}).get("max_requests_per_hour", 50)

    # Prune timestamps outside the window
    timestamps = _rate_counters.get(user_id, [])
    timestamps = [t for t in timestamps if now - t < window]
    timestamps.append(now)
    _rate_counters[user_id] = timestamps

    if len(timestamps) > max_requests:
        logger.warning(f"[rate_limit] EXCEEDED | user={user_id} | count={len(timestamps)}/{max_requests}")
        audit.record(user_id, tenant_id, state["role"], "rate_limit", "agent", "DENIED",
                     f"{len(timestamps)} requests in last hour (limit {max_requests})")
        return {
            "auth_result": "denied",
            "messages": [AIMessage(content=f"Rate limit exceeded ({max_requests} req/hr). Please wait before retrying.")],
        }

    logger.info(f"[rate_limit] OK | user={user_id} | {len(timestamps)}/{max_requests} requests this hour")
    return {}


def route_after_rate_limit(state: RBACState) -> str:
    if state.get("auth_result") == "denied":
        return "denied"
    return "chat"


# ---------------------------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------------------------
def build_rbac_graph(checkpointer):
    """
    Full RBAC pipeline:
      START → auth (RBAC check) → rate_limit (sliding window) → chat → END
                     ↓ denied                    ↓ denied
                    END                           END
    """
    builder = StateGraph(RBACState)
    builder.add_node("auth",       auth_node)
    builder.add_node("rate_limit", rate_limit_node)
    builder.add_node("chat",       tenant_chat_node)
    builder.add_node("denied",     access_denied_node)

    builder.add_edge(START, "auth")
    builder.add_conditional_edges("auth",       route_after_auth,       ["rate_limit", "denied"])
    builder.add_conditional_edges("rate_limit", route_after_rate_limit,  ["chat",       "denied"])
    builder.add_edge("chat",   END)
    builder.add_edge("denied", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
def run_demo():
    checkpointer = MemorySaver()
    graph = build_rbac_graph(checkpointer)

    # --- Real JWT demo ---
    print("\n" + "=" * 60)
    print("DEMO 0: JWT token creation and verification")
    print("=" * 60)
    token = create_access_token("alice", "acme-corp", "analyst")
    print(f"  Access token (first 60 chars): {token[:60]}...")
    claims = decode_token(token)
    print(f"  Decoded claims: {claims}")
    refresh = create_refresh_token("alice", "acme-corp")
    print(f"  Refresh token (first 40 chars): {refresh[:40]}...")
    header = f"Bearer {token}"
    extracted = extract_token_from_header(header)
    print(f"  Extracted from header: {extracted[:30]}... (matches token: {extracted == token})")

    print("\n" + "=" * 60)
    print("DEMO 1: Tenant isolation — same question, different tenants")
    print("=" * 60)

    for tenant, user, role in [
        ("acme-corp",  "alice", "analyst"),
        ("globex-inc", "bob",   "viewer"),
    ]:
        config = {"configurable": {"thread_id": f"{tenant}-{user}-001"}}
        result = graph.invoke(
            {
                "user_id": user,
                "tenant_id": tenant,
                "role": role,
                "action_requested": "ask_question",
                "auth_result": "",
                "audit_trail": [],
                "messages": [HumanMessage(content="Tell me about your main services.")],
            },
            config=config,
        )
        print(f"\n  tenant={tenant} | user={user} | role={role}")
        print(f"  reply={result['messages'][-1].content[:100]}...")

    print("\n" + "=" * 60)
    print("DEMO 2: RBAC enforcement — viewer tries admin action")
    print("=" * 60)

    config = {"configurable": {"thread_id": "acme-corp-charlie-001"}}
    result = graph.invoke(
        {
            "user_id": "charlie",
            "tenant_id": "acme-corp",
            "role": "viewer",
            "action_requested": "delete_history",   # viewer doesn't have this
            "auth_result": "",
            "audit_trail": [],
            "messages": [HumanMessage(content="Delete all history please.")],
        },
        config=config,
    )
    print(f"  reply={result['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("DEMO 3: Audit trail for compliance")
    print("=" * 60)
    log = audit.get_log(tenant_id="acme-corp")
    print(f"  Audit events for acme-corp: {len(log)}")
    for entry in log:
        print(f"    {entry['timestamp']} | {entry['user_id']} | {entry['action']} → {entry['result']}")


if __name__ == "__main__":
    run_demo()

    print("\n✓ Lesson 17 complete.")
    print("Next: Lesson 18 — Observability (Prometheus + structured tracing)")
