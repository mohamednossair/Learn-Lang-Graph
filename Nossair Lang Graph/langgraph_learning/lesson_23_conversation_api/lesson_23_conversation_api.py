"""
Lesson 23: Conversation Management API Layer
=============================================
Teaches:
  - Building the Chatbot API (Conversation Management Layer from the architecture)
  - Session creation, lookup, and routing to the correct specialist agent
  - Tying together: JWT auth (L17) + S3 memory (L22) + agent orchestration (L5/L11)
  - FastAPI endpoints: /chat, /sessions, /history, /upload
  - Usage limitations enforcement per user/tenant
  - Multi-agent routing: data_analysis / db_retrieval / general
  - Streaming responses via Server-Sent Events (SSE)
  - Conversation history retrieval from S3

Architecture role (from High Level Architecture diagram):
    Front End
        └── Conversation Management Layer / Chatbot API  ← THIS LESSON
               └── EC2 FastAPI Server
                      └── Agent Orchestration (routes to specialist agents)

Prerequisites:
    pip install fastapi uvicorn python-jose[cryptography] boto3

Run locally:
    uvicorn lesson_23_conversation_api:app --reload --port 8023
"""

import json
import logging
import os
import time
import uuid
from typing import Annotated, Any, AsyncGenerator, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("lesson_23")

# ===========================================================================
# SECTION 1 — CONFIGURATION
# ===========================================================================

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "my-langgraph-bucket")
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "dev-secret-do-not-use-in-production")
JWT_ALGORITHM = "HS256"

MAX_MESSAGES_PER_SESSION = int(os.getenv("MAX_MESSAGES_PER_SESSION", "100"))
MAX_SESSIONS_PER_TENANT = int(os.getenv("MAX_SESSIONS_PER_TENANT", "50"))
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))

# ===========================================================================
# SECTION 2 — SESSION STORE (in-memory for demo; use Redis in production)
# ===========================================================================
#
# In production (Lesson 19 pattern): replace with Redis
#   - session_store = Redis(host=REDIS_HOST, decode_responses=True)
#   - session_store.setex(session_id, TTL_SECONDS, json.dumps(session_data))
#
# In this lesson: dict-based store to focus on the API layer logic

_session_store: dict[str, dict] = {}
_rate_limit_store: dict[str, list] = {}


def create_session(tenant_id: str, user_id: str, agent_type: str = "general") -> dict:
    """
    Create a new conversation session.

    Returns:
        Session dict with session_id, thread_id, agent routing, timestamps.
    """
    session_id = str(uuid.uuid4())
    thread_id = f"{tenant_id}_{user_id}_{session_id[:8]}"

    session = {
        "session_id": session_id,
        "thread_id": thread_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "agent_type": agent_type,
        "created_at": time.time(),
        "last_active": time.time(),
        "message_count": 0,
        "status": "active",
    }
    _session_store[session_id] = session
    logger.info("Session created: %s (tenant=%s, agent=%s)", session_id, tenant_id, agent_type)
    return session


def get_session(session_id: str) -> Optional[dict]:
    """Retrieve a session by ID. Returns None if not found."""
    return _session_store.get(session_id)


def update_session_activity(session_id: str) -> None:
    """Update last_active timestamp and increment message count."""
    if session_id in _session_store:
        _session_store[session_id]["last_active"] = time.time()
        _session_store[session_id]["message_count"] += 1


def list_sessions_for_tenant(tenant_id: str) -> list[dict]:
    """List all active sessions for a tenant."""
    return [s for s in _session_store.values() if s["tenant_id"] == tenant_id]


# ===========================================================================
# SECTION 3 — JWT AUTH (simplified from Lesson 17)
# ===========================================================================
#
# Full JWT implementation is in Lesson 17. Here we use a lightweight version
# focused on the API gateway role — verifying tokens and extracting user context.

def verify_jwt_token(token: str) -> dict:
    """
    Verify a JWT token and return the decoded payload.
    In production, use the full Lesson 17 JWT implementation.
    """
    try:
        from jose import JWTError, jwt
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def create_demo_token(user_id: str, tenant_id: str, role: str = "user") -> str:
    """Create a demo JWT for testing. Use Lesson 17 for full token management."""
    try:
        from jose import jwt
        payload = {
            "sub": user_id,
            "tenant_id": tenant_id,
            "role": role,
            "exp": int(time.time()) + 3600,
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    except ImportError:
        logger.warning("python-jose not installed. Returning mock token.")
        return f"mock_token_{user_id}_{tenant_id}"


# ===========================================================================
# SECTION 4 — USAGE LIMITATIONS
# ===========================================================================
#
# Maps to 'Usage Limitations' box in the architecture diagram.
# Enforces: per-session message limits, per-tenant session limits, rate limiting.

def check_usage_limits(session: dict, tenant_id: str, user_id: str) -> None:
    """
    Enforce all usage limits. Raises HTTPException if any limit is exceeded.

    Limits checked:
      1. Session message count (MAX_MESSAGES_PER_SESSION)
      2. Tenant active session count (MAX_SESSIONS_PER_TENANT)
      3. Per-user rate limit (RATE_LIMIT_RPM — requests per minute)
    """
    if session["message_count"] >= MAX_MESSAGES_PER_SESSION:
        raise HTTPException(
            status_code=429,
            detail=f"Session limit reached: {MAX_MESSAGES_PER_SESSION} messages per session.",
        )

    tenant_sessions = list_sessions_for_tenant(tenant_id)
    if len(tenant_sessions) >= MAX_SESSIONS_PER_TENANT:
        raise HTTPException(
            status_code=429,
            detail=f"Tenant session limit reached: {MAX_SESSIONS_PER_TENANT} active sessions.",
        )

    now = time.time()
    window_start = now - 60.0
    user_requests = _rate_limit_store.get(user_id, [])
    user_requests = [t for t in user_requests if t > window_start]
    if len(user_requests) >= RATE_LIMIT_RPM:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_RPM} requests/minute.",
        )
    user_requests.append(now)
    _rate_limit_store[user_id] = user_requests


# ===========================================================================
# SECTION 5 — AGENT ORCHESTRATION (ROUTING)
# ===========================================================================
#
# Maps to 'Agent Orchestration' box in the architecture diagram.
# Routes messages to the correct specialist agent based on session.agent_type
# and message content keywords.

class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    tenant_id: str
    thread_id: str
    agent_type: str
    response: Optional[str]


def get_llm():
    """Bedrock if available, Ollama fallback — same pattern as all lessons."""
    try:
        import boto3
        from langchain_aws import ChatBedrockConverse
        model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
        session = boto3.Session(region_name=AWS_REGION)
        client = session.client("bedrock-runtime")
        client.list_foundation_models(byOutputModality="TEXT")
        return ChatBedrockConverse(model_id=model_id, client=client)
    except Exception:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2")


def route_to_agent(state: OrchestratorState) -> Literal["data_analysis_agent", "db_retrieval_agent", "general_agent"]:
    """
    Routing function for the orchestrator.
    Uses the session's agent_type to select the specialist.
    Falls back to keyword detection in the last message.
    """
    agent_type = state.get("agent_type", "general")

    if agent_type == "data_analysis":
        return "data_analysis_agent"
    elif agent_type == "db_retrieval":
        return "db_retrieval_agent"

    last_msg = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            last_msg = m.content.lower()
            break

    if any(kw in last_msg for kw in ["analyze", "data", "csv", "chart", "trend", "metric"]):
        return "data_analysis_agent"
    elif any(kw in last_msg for kw in ["query", "database", "sql", "record", "table", "find"]):
        return "db_retrieval_agent"
    else:
        return "general_agent"


def data_analysis_agent_node(state: OrchestratorState) -> dict:
    """Data Analysis Agent — specialist for document/CSV/metric analysis."""
    llm = get_llm()
    system = SystemMessage(content=(
        "You are a Data Analysis Agent. You help users analyze data, identify trends, "
        "interpret metrics, and generate insights from documents and datasets."
    ))
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    logger.info("DataAnalysisAgent responded (%d chars)", len(response.content))
    return {"response": response.content, "messages": [response]}


def db_retrieval_agent_node(state: OrchestratorState) -> dict:
    """DB Retrieval Agent — specialist for structured data queries."""
    llm = get_llm()
    system = SystemMessage(content=(
        "You are a Database Retrieval Agent. You help users query structured data, "
        "write SQL-like queries, and retrieve records from databases. "
        "Always explain what query you would run and what results it would return."
    ))
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    logger.info("DBRetrievalAgent responded (%d chars)", len(response.content))
    return {"response": response.content, "messages": [response]}


def general_agent_node(state: OrchestratorState) -> dict:
    """General Agent — handles all other conversations."""
    llm = get_llm()
    system = SystemMessage(content=(
        "You are a helpful assistant. Answer clearly, concisely, and accurately."
    ))
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    logger.info("GeneralAgent responded (%d chars)", len(response.content))
    return {"response": response.content, "messages": [response]}


def build_orchestrator_graph() -> StateGraph:
    """
    Build the Agent Orchestration graph.
    Matches the architecture: one entry point, three specialist agents.
    """
    graph = StateGraph(OrchestratorState)

    graph.add_node("data_analysis_agent", data_analysis_agent_node)
    graph.add_node("db_retrieval_agent", db_retrieval_agent_node)
    graph.add_node("general_agent", general_agent_node)

    graph.add_conditional_edges(
        START,
        route_to_agent,
        {
            "data_analysis_agent": "data_analysis_agent",
            "db_retrieval_agent": "db_retrieval_agent",
            "general_agent": "general_agent",
        },
    )
    graph.add_edge("data_analysis_agent", END)
    graph.add_edge("db_retrieval_agent", END)
    graph.add_edge("general_agent", END)

    return graph.compile(checkpointer=MemorySaver())


ORCHESTRATOR = build_orchestrator_graph()


# ===========================================================================
# SECTION 6 — FASTAPI APPLICATION (CHATBOT API)
# ===========================================================================
#
# This is the Conversation Management Layer from the architecture.
# It is the single entry point between the Front End and all agent logic.

app = FastAPI(
    title="Chatbot API — Conversation Management Layer",
    description="LangGraph multi-agent conversation API (Lesson 23)",
    version="1.0.0",
)


# --- Pydantic request/response models ---

class SessionCreateRequest(BaseModel):
    user_id: str
    tenant_id: str
    agent_type: str = "general"


class SessionCreateResponse(BaseModel):
    session_id: str
    thread_id: str
    agent_type: str
    token: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    token: str


class ChatResponse(BaseModel):
    session_id: str
    thread_id: str
    agent_used: str
    response: str
    message_count: int


class HistoryResponse(BaseModel):
    session_id: str
    thread_id: str
    messages: list[dict]


# --- Endpoints ---

@app.post("/sessions", response_model=SessionCreateResponse, tags=["Session Management"])
async def create_session_endpoint(request: SessionCreateRequest):
    """
    Create a new conversation session.
    Returns a session_id, thread_id, and JWT token for subsequent requests.

    Maps to: 'Session & Memory Management' in the architecture diagram.
    """
    existing = list_sessions_for_tenant(request.tenant_id)
    if len(existing) >= MAX_SESSIONS_PER_TENANT:
        raise HTTPException(429, f"Tenant session limit reached: {MAX_SESSIONS_PER_TENANT}")

    session = create_session(request.tenant_id, request.user_id, request.agent_type)
    token = create_demo_token(request.user_id, request.tenant_id)

    return SessionCreateResponse(
        session_id=session["session_id"],
        thread_id=session["thread_id"],
        agent_type=session["agent_type"],
        token=token,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Conversation"])
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint. Routes message to the correct specialist agent.

    Flow:
      1. Verify JWT
      2. Load session
      3. Check usage limits
      4. Route to agent via orchestrator graph
      5. Update session state
      6. Return response

    Maps to: 'Conversation Management Layer' + 'Agent Orchestration' in architecture.
    """
    user_data = verify_jwt_token(request.token)
    tenant_id = user_data.get("tenant_id", "unknown")
    user_id = user_data.get("sub", "unknown")

    session = get_session(request.session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {request.session_id}")

    check_usage_limits(session, tenant_id, user_id)

    config = {"configurable": {"thread_id": session["thread_id"]}}
    state_input = {
        "messages": [HumanMessage(content=request.message)],
        "tenant_id": tenant_id,
        "thread_id": session["thread_id"],
        "agent_type": session["agent_type"],
        "response": None,
    }

    result = ORCHESTRATOR.invoke(state_input, config)

    update_session_activity(request.session_id)

    agent_used = route_to_agent(state_input).replace("_agent", "").replace("_", " ").title()

    return ChatResponse(
        session_id=request.session_id,
        thread_id=session["thread_id"],
        agent_used=agent_used,
        response=result.get("response", ""),
        message_count=session["message_count"],
    )


@app.get("/sessions/{session_id}/history", response_model=HistoryResponse, tags=["Session Management"])
async def get_history_endpoint(session_id: str, token: str):
    """
    Retrieve conversation history for a session.
    In production: also loads from S3 snapshot (Lesson 22) for cross-server access.
    """
    verify_jwt_token(token)
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")

    config = {"configurable": {"thread_id": session["thread_id"]}}
    state = ORCHESTRATOR.get_state(config)

    messages_raw = state.values.get("messages", []) if state.values else []
    messages = [
        {"type": type(m).__name__, "content": m.content}
        for m in messages_raw
    ]

    return HistoryResponse(
        session_id=session_id,
        thread_id=session["thread_id"],
        messages=messages,
    )


@app.get("/sessions", tags=["Session Management"])
async def list_sessions_endpoint(tenant_id: str, token: str):
    """List all active sessions for a tenant. Requires valid JWT."""
    user_data = verify_jwt_token(token)
    if user_data.get("tenant_id") != tenant_id and user_data.get("role") != "admin":
        raise HTTPException(403, "Cannot list sessions for another tenant")

    sessions = list_sessions_for_tenant(tenant_id)
    return {"tenant_id": tenant_id, "session_count": len(sessions), "sessions": sessions}


@app.post("/chat/stream", tags=["Conversation"])
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    The frontend receives tokens as they arrive rather than waiting for the full response.

    Pattern: yield each token chunk as a 'data: ...\n\n' SSE event.
    Frontend: new EventSource('/chat/stream') with onmessage handler.
    """
    user_data = verify_jwt_token(request.token)
    tenant_id = user_data.get("tenant_id", "unknown")
    user_id = user_data.get("sub", "unknown")

    session = get_session(request.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    check_usage_limits(session, tenant_id, user_id)

    async def token_generator() -> AsyncGenerator[str, None]:
        llm = get_llm()
        config = {"configurable": {"thread_id": session["thread_id"]}}
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=request.message),
        ]
        try:
            for chunk in llm.stream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"
            update_session_activity(request.session_id)

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.post("/upload/{session_id}", tags=["Document Management"])
async def upload_document_endpoint(session_id: str, token: str, file: UploadFile = File(...)):
    """
    Upload a document (CSV, PDF, JSON) to S3 for the Data Analysis Agent.
    Returns a presigned URL so the frontend can confirm the upload.

    Integrates with Lesson 22 S3 helper functions.
    """
    user_data = verify_jwt_token(token)
    tenant_id = user_data.get("tenant_id", "unknown")

    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    file_bytes = await file.read()
    doc_id = str(uuid.uuid4())[:8]

    try:
        from lesson_22_aws_s3.lesson_22_aws_s3 import upload_document, get_document_url
        s3_key = upload_document(tenant_id, doc_id, file.filename, file_bytes)
        presigned_url = get_document_url(tenant_id, doc_id, file.filename)
    except ImportError:
        s3_key = f"documents/{tenant_id}/{doc_id}/{file.filename}"
        presigned_url = f"https://simulated-s3.example.com/{s3_key}"
        logger.warning("lesson_22 not importable — simulating S3 upload")

    return {
        "session_id": session_id,
        "doc_id": doc_id,
        "filename": file.filename,
        "s3_key": s3_key,
        "presigned_url": presigned_url,
        "size_bytes": len(file_bytes),
    }


@app.get("/health", tags=["Ops"])
async def health_check():
    """Health endpoint for load balancer and EC2 auto-scaling checks."""
    return {
        "status": "healthy",
        "active_sessions": len(_session_store),
        "service": "conversation-management-api",
        "version": "1.0.0",
    }


# ===========================================================================
# SECTION 7 — CONVERSATION MANAGEMENT PATTERNS
# ===========================================================================
#
# KEY PATTERNS this lesson teaches:
#
# 1. SESSION ISOLATION
#    Each session gets a unique thread_id → LangGraph MemorySaver keeps
#    state isolated per thread. No cross-session contamination.
#
# 2. AGENT ROUTING
#    The Conversation Management Layer decides WHICH agent handles each message.
#    This is different from the agent deciding its own routing (ReAct in L4).
#    Here: routing is based on session metadata + keyword detection.
#
# 3. USAGE LIMITS ENFORCEMENT
#    Three layers: message count per session, sessions per tenant, RPM per user.
#    All checked BEFORE the LLM is invoked → prevents runaway costs.
#
# 4. TOKEN STREAMING (SSE)
#    Frontend gets tokens incrementally. Critical for UX with slow LLMs.
#    SSE is simpler than WebSockets for unidirectional server→client streams.
#
# 5. CONVERSATION CONTINUITY
#    thread_id is stored in the session → same thread_id passed to LangGraph
#    on every request → MemorySaver (or OracleSaver from L16) restores history.


# ===========================================================================
# SECTION 8 — DEMO (without running the FastAPI server)
# ===========================================================================

def run_demo():
    print("\n" + "=" * 65)
    print("LESSON 23 — CONVERSATION MANAGEMENT API DEMO")
    print("=" * 65)

    print("\n--- Demo 1: Session Creation & Routing ---")
    session_data = create_session("tenant_acme", "user_alice", agent_type="general")
    print(f"Session created: {session_data['session_id']}")
    print(f"Thread ID: {session_data['thread_id']}")

    print("\n--- Demo 2: Agent Routing Logic ---")
    test_cases = [
        ("general", "Hello, how are you?"),
        ("general", "Analyze the revenue trend in my CSV data"),
        ("general", "Query the database for all users with status active"),
    ]
    for agent_type, message in test_cases:
        state = {
            "messages": [HumanMessage(content=message)],
            "agent_type": agent_type,
            "tenant_id": "tenant_acme",
            "thread_id": "demo",
            "response": None,
        }
        route = route_to_agent(state)
        print(f"  Message: '{message[:50]}' → {route}")

    print("\n--- Demo 3: Usage Limits ---")
    session_data["message_count"] = MAX_MESSAGES_PER_SESSION
    try:
        check_usage_limits(session_data, "tenant_acme", "user_alice")
    except HTTPException as e:
        print(f"Limit enforced: HTTP {e.status_code} — {e.detail}")

    print("\n--- Demo 4: Orchestrator Graph (direct invoke) ---")
    config = {"configurable": {"thread_id": "demo_lesson23"}}
    result = ORCHESTRATOR.invoke(
        {
            "messages": [HumanMessage(content="What is LangGraph used for?")],
            "tenant_id": "tenant_acme",
            "thread_id": "demo_lesson23",
            "agent_type": "general",
            "response": None,
        },
        config,
    )
    print(f"Response preview: {result.get('response', '')[:200]}...")

    print("\n--- Demo 5: JWT Token ---")
    token = create_demo_token("user_alice", "tenant_acme", role="user")
    print(f"JWT token (first 60 chars): {token[:60]}...")
    decoded = verify_jwt_token(token)
    print(f"Decoded: sub={decoded.get('sub')}, tenant={decoded.get('tenant_id')}, role={decoded.get('role')}")

    print("\n" + "=" * 65)
    print("KEY TAKEAWAYS:")
    print("  1. Session layer sits BETWEEN frontend and agents — enforces all limits")
    print("  2. thread_id = session key → LangGraph memory is per-session")
    print("  3. Agent routing is keyword + session-type based — no LLM call needed to route")
    print("  4. SSE streaming: yield 'data: {...}\\n\\n' chunks to frontend")
    print("  5. /health endpoint is required for EC2 load balancer target group checks")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
    print("\nTo run the FastAPI server:")
    print("  uvicorn lesson_23_conversation_api.lesson_23_conversation_api:app --reload --port 8023")
    print("\nTest endpoints:")
    print("  POST /sessions          — create session")
    print("  POST /chat              — send message")
    print("  GET  /sessions/{id}/history — get conversation history")
    print("  POST /chat/stream       — streaming SSE chat")
    print("  GET  /health            — health check")
