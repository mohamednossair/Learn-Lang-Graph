"""Task 21.2 — Streaming + Guardrails."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
import logging
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger("task_21_2")

# ---------------------------------------------------------------------------
# Streaming — accumulate chunks, token usage in FINAL chunk only
# ---------------------------------------------------------------------------
class StreamState(TypedDict):
    messages: Annotated[list, add_messages]
    full_response: str
    total_tokens: int

llm = ChatOllama(model=get_ollama_model(), temperature=0)

def streaming_chat_node(state: StreamState) -> dict:
    """Stream LLM response, accumulate content, extract usage from final chunk."""
    chunks = []
    total_tokens = 0
    
    for chunk in llm.stream(state["messages"]):
        chunks.append(chunk.content)
        # Token usage is ONLY in the final chunk's response_metadata
        usage = chunk.response_metadata.get("usage", {})
        if usage:
            total_tokens = usage.get("output_tokens", usage.get("total_tokens", 0))
    
    full_response = "".join(chunks)
    return {
        "messages": [AIMessage(content=full_response)],
        "full_response": full_response,
        "total_tokens": total_tokens or len(full_response.split())
    }

builder = StateGraph(StreamState)
builder.add_node("chat", streaming_chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
stream_graph = builder.compile()

# ---------------------------------------------------------------------------
# Guardrails — content filtering pattern
# ---------------------------------------------------------------------------
BLOCKED_TOPICS = ["violence", "illegal", "harmful"]
PII_PATTERNS = ["SSN", "credit card", "password"]

def check_guardrails(content: str) -> dict:
    """Simulate Bedrock Guardrails content filter.
    
    Production Bedrock Guardrails:
    - Applied server-side before/after model
    - Cannot be bypassed by prompt injection
    - Logged in CloudTrail for compliance
    - Cost: ~$0.75 per 1000 text units
    """
    content_lower = content.lower()
    
    # Topic denial
    for topic in BLOCKED_TOPICS:
        if topic in content_lower:
            return {"action": "INTERVENED", "reason": f"Blocked topic: {topic}"}
    
    # PII detection (mask in production)
    for pii in PII_PATTERNS:
        if pii.lower() in content_lower:
            return {"action": "MASKED", "reason": f"PII detected: {pii}"}
    
    return {"action": "ALLOWED", "reason": ""}

# ---------------------------------------------------------------------------
# FastAPI SSE streaming pattern (conceptual)
# ---------------------------------------------------------------------------
SSE_PATTERN = """
# FastAPI Server-Sent Events pattern:
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.thread_id}}
        ):
            if "chat" in chunk:
                yield f"data: {chunk['chat']['messages'][-1].content}\\n\\n"
        yield "data: [DONE]\\n\\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
"""

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 21.2 — STREAMING + GUARDRAILS")
    print("=" * 50)
    
    # Test streaming
    print("\n--- Streaming ---")
    result = stream_graph.invoke({
        "messages": [HumanMessage(content="Explain Python in 2 sentences")],
        "full_response": "", "total_tokens": 0
    })
    print(f"  Response: {result['full_response'][:100]}...")
    print(f"  Tokens: {result['total_tokens']}")
    
    # Test guardrails
    print("\n--- Guardrails ---")
    tests = [
        "What is Python?",
        "How to make violence?",
        "My SSN is 123-45-6789",
    ]
    for text in tests:
        result = check_guardrails(text)
        print(f"  '{text[:30]}...' → {result}")
    
    # SSE pattern
    print("\n--- FastAPI SSE pattern ---")
    print(SSE_PATTERN.strip()[:200] + "...")
    
    print("\n--- Guardrail best practices ---")
    print("  ✓ Customer-facing: enable Guardrails (block harmful, mask PII)")
    print("  ✓ Check guardrailAction == 'INTERVENED' in response")
    print("  ✗ Internal dev tools: unnecessary overhead")
    print("  ✗ Don't rely on prompt engineering alone for safety")
    
    print("\n✅ Streaming + Guardrails working!")
