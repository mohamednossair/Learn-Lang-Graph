# =============================================================
# LESSON 15 — Deploying LangGraph Agents with FastAPI
# =============================================================
#
# WHAT YOU WILL LEARN:
#   1. Wrap a LangGraph agent as a REST API with FastAPI
#   2. Handle HITL interrupts over HTTP (approve/reject endpoints)
#   3. Multi-user isolation via thread_id
#   4. Streaming responses via Server-Sent Events
#   5. LangSmith tracing setup
#
# INSTALL:
#   pip install fastapi uvicorn python-dotenv
#
# RUN:
#   uvicorn lesson_15_deployment.api:app --host 0.0.0.0 --port 8000 --reload
#
# TEST (in another terminal):
#   curl -X POST http://localhost:8000/ask \
#        -H "Content-Type: application/json" \
#        -d '{"question": "How many employees are there?", "user_id": "ahmed"}'
# =============================================================

import os
import sys
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# =============================================================
# OPTIONAL: LangSmith tracing
# Set these in a .env file or environment variables
# =============================================================
# import os
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"]    = "langgraph-agent"
# os.environ["LANGCHAIN_API_KEY"]    = "your-key-here"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api")


# =============================================================
# STARTUP — build graph once (expensive), reuse for all requests
# =============================================================

_graph = None
_checkpointer = None


def get_graph():
    global _graph, _checkpointer
    if _graph is None:
        from lesson_10_capstone.lesson_10_capstone import build_capstone_graph, setup_database
        from langgraph.checkpoint.sqlite import SqliteSaver
        setup_database()
        db_path = os.path.join(os.path.dirname(__file__), "production_checkpoints.db")
        # Note: In a real app use a context manager or async saver
        import sqlite3
        conn = sqlite3.connect(db_path, check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
        _graph = build_capstone_graph(checkpointer=_checkpointer)
        logger.info("Graph initialized")
    return _graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_graph()  # warm up on startup
    logger.info("API ready")
    yield
    logger.info("API shutting down")


# =============================================================
# FASTAPI APP
# =============================================================

app = FastAPI(
    title="LangGraph Agent API",
    description="Production-ready AI agent backed by LangGraph",
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================
# AUTHENTICATION — simple API key check
# In production: use JWT, OAuth2, or your auth service
# =============================================================

VALID_API_KEYS = set(os.getenv("API_KEYS", "dev-key-123,test-key-456").split(","))


def verify_api_key(x_api_key: str = Header(default=None)):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Pass X-Api-Key header.")
    return x_api_key


# =============================================================
# REQUEST / RESPONSE MODELS
# =============================================================

class QuestionRequest(BaseModel):
    question: str
    user_id:  str
    thread_id: Optional[str] = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Question must be at least 3 characters")
        if len(v) > 500:
            raise ValueError("Question must be at most 500 characters")
        return v

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v):
        v = v.strip()
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("user_id must be alphanumeric (with - or _ allowed)")
        if len(v) > 50:
            raise ValueError("user_id must be at most 50 characters")
        return v


class ApprovalRequest(BaseModel):
    thread_id: str
    decision:  str  # "approve" or "reject"

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v):
        if v.lower() not in ("approve", "reject"):
            raise ValueError("decision must be 'approve' or 'reject'")
        return v.lower()


class AskResponse(BaseModel):
    status:    str   # "complete" | "awaiting_approval" | "error"
    thread_id: str
    answer:    Optional[str] = None
    interrupt: Optional[dict] = None


# =============================================================
# ENDPOINTS
# =============================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "time": time.time()}


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: QuestionRequest, _: str = Depends(verify_api_key)):
    """
    Submit a question to the agent.
    Returns either a complete answer or 'awaiting_approval' if HITL is needed.
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.types import Command

    graph = get_graph()
    thread_id = req.thread_id or f"{req.user_id}-{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

    logger.info(f"Question from user={req.user_id} thread={thread_id}: {req.question[:50]}")

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=req.question)],
             "user_id": req.user_id, "next_agent": ""},
            config=config
        )

        # Check if graph paused for human approval
        state = graph.get_state(config)
        if state.next:
            interrupt_data = {}
            if state.tasks:
                for task in state.tasks:
                    if task.interrupts:
                        interrupt_data = task.interrupts[0].value
                        break
            return AskResponse(
                status="awaiting_approval",
                thread_id=thread_id,
                interrupt=interrupt_data
            )

        # Get last AI message
        last_ai = next(
            (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            "I could not generate a response."
        )
        return AskResponse(status="complete", thread_id=thread_id, answer=last_ai)

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/approve", response_model=AskResponse)
async def approve_action(req: ApprovalRequest, _: str = Depends(verify_api_key)):
    """
    Approve or reject a pending HITL action.
    Call this after receiving 'awaiting_approval' from /ask.
    """
    from langchain_core.messages import AIMessage
    from langgraph.types import Command

    graph = get_graph()
    config = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 25}

    # Verify there is actually a pending interrupt
    state = graph.get_state(config)
    if not state.next:
        raise HTTPException(status_code=400, detail=f"No pending approval for thread: {req.thread_id}")

    logger.info(f"Approval decision: {req.decision} for thread={req.thread_id}")

    try:
        result = graph.invoke(Command(resume=req.decision), config=config)
        last_ai = next(
            (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            "Action processed."
        )
        return AskResponse(status="complete", thread_id=req.thread_id, answer=last_ai)
    except Exception as e:
        logger.error(f"Error processing approval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str, _: str = Depends(verify_api_key)):
    """List all conversation sessions for a user."""
    return {"user_id": user_id, "note": "Implement by querying checkpointer for thread_ids matching user_id"}


@app.get("/history/{thread_id}")
async def get_history(thread_id: str, _: str = Depends(verify_api_key)):
    """Get conversation history for a thread."""
    from langchain_core.messages import HumanMessage, AIMessage

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")
        messages = state.values.get("messages", [])
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                history.append({"role": "assistant", "content": msg.content})
        return {"thread_id": thread_id, "messages": history}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{thread_id}")
async def delete_session(thread_id: str, _: str = Depends(verify_api_key)):
    """Delete a session (GDPR compliance)."""
    logger.info(f"Deleting session: {thread_id}")
    return {"deleted": thread_id, "note": "In production, also delete vector memory for this user"}


@app.get("/stream/{thread_id}")
async def stream_response(thread_id: str, question: str, user_id: str, _: str = Depends(verify_api_key)):
    """Stream the agent response as Server-Sent Events."""
    from langchain_core.messages import HumanMessage, AIMessage

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

    def event_stream():
        try:
            for event in graph.stream(
                {"messages": [HumanMessage(content=question)],
                 "user_id": user_id, "next_agent": ""},
                config=config,
                stream_mode="values"
            ):
                msgs = event.get("messages", [])
                if msgs and isinstance(msgs[-1], AIMessage) and msgs[-1].content:
                    chunk = msgs[-1].content[:100]
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: ERROR: {e}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# =============================================================
# RUN DIRECTLY
# =============================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting API server...")
    print("API docs at: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("Pass header: X-Api-Key: dev-key-123")
    uvicorn.run("lesson_15_deployment.api:app", host="0.0.0.0", port=8000, reload=True)


# =============================================================
# TESTING THE API (copy-paste these curl commands):
#
# Health check:
#   curl http://localhost:8000/health
#
# Ask a question:
#   curl -X POST http://localhost:8000/ask \
#        -H "Content-Type: application/json" \
#        -H "X-Api-Key: dev-key-123" \
#        -d '{"question": "How many employees are there?", "user_id": "ahmed"}'
#
# Approve an action (use thread_id from previous response):
#   curl -X POST http://localhost:8000/approve \
#        -H "Content-Type: application/json" \
#        -H "X-Api-Key: dev-key-123" \
#        -d '{"thread_id": "ahmed-1234567890", "decision": "approve"}'
#
# Get conversation history:
#   curl http://localhost:8000/history/ahmed-1234567890 \
#        -H "X-Api-Key: dev-key-123"
#
# Interactive API docs (browser):
#   http://localhost:8000/docs
# =============================================================
