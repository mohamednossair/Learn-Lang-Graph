# =============================================================
# TASK 9.1 — Production Upgrade
# =============================================================
# Goal:
#   Take a basic ReAct agent (like Lesson 4) and add ALL 9
#   production pillars:
#     1. Structured logging
#     2. Pydantic input validation
#     3. Structured output (with_structured_output)
#     4. Retry with exponential backoff
#     5. Error handler node
#     6. Recursion limit
#     7. Streaming output
#     8. Parallel execution (Send() on multiple items)
#     9. Token cost management (message trimming)
# =============================================================

import logging
import time
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, field_validator
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# ── PILLAR 1: Structured Logging ──────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("production_agent")


def log_node(name: str, state: dict):
    logger.info(f"NODE={name} | msgs={len(state.get('messages', []))} | errors={state.get('error', '')}")


# ── PILLAR 2: Pydantic Input Validation ───────────────────────

class UserQuery(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Question too short (min 3 chars)")
        if len(v) > 1000:
            raise ValueError("Question too long (max 1000 chars)")
        return v.strip()


# ── PILLAR 4: Retry with Exponential Backoff ──────────────────

MAX_RETRIES = 3


def call_llm_with_retry(llm, messages):
    for attempt in range(MAX_RETRIES):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"LLM call failed after {MAX_RETRIES} attempts: {e}")
                raise


# ── Tools ─────────────────────────────────────────────────────

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


tools = [add, multiply]


# ── State ─────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    error: str


# ── LLM ───────────────────────────────────────────────────────

llm = ChatOllama(model="llama3", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ── PILLAR 9: Token Cost Management ───────────────────────────

MAX_MESSAGES = 10


def trim_messages(messages: list) -> list:
    """Keep only the last MAX_MESSAGES messages to control context size."""
    if len(messages) > MAX_MESSAGES:
        logger.info(f"Trimming messages: {len(messages)} → {MAX_MESSAGES}")
        return messages[-MAX_MESSAGES:]
    return messages


# ── Agent Node ────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    log_node("agent", state)
    trimmed = trim_messages(state["messages"])
    try:
        response = call_llm_with_retry(llm_with_tools, trimmed)
        return {"messages": [response], "error": ""}
    except Exception as e:
        return {"messages": [], "error": str(e)}


# ── PILLAR 5: Error Handler Node ──────────────────────────────

def error_handler_node(state: AgentState) -> dict:
    log_node("error_handler", state)
    logger.error(f"Error handled: {state['error']}")
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=f"I encountered an error: {state['error']}")]}


# ── Routing ───────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    if state.get("error"):
        return "error_handler"
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ── PILLAR 6: Recursion Limit applied at invoke time ──────────

# ── Build Graph ───────────────────────────────────────────────

checkpointer = MemorySaver()
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("error_handler", error_handler_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "error_handler": "error_handler",
    "end": END,
})
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("error_handler", END)

graph = graph_builder.compile(checkpointer=checkpointer)


# ── PILLAR 7: Streaming Output ────────────────────────────────

def run_streaming(question: str):
    try:
        query = UserQuery(question=question)  # Pillar 2: validate
    except Exception as e:
        print(f"Validation error: {e}")
        return

    print(f"\n{'='*60}")
    print(f"Q: {query.question}")
    config = {
        "configurable": {"thread_id": "prod-001"},
        "recursion_limit": 20,  # Pillar 6
    }

    # Pillar 7: stream token updates
    print("Streaming: ", end="", flush=True)
    for chunk in graph.stream(
        {"messages": [HumanMessage(content=query.question)], "error": ""},
        config=config,
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            if "messages" in update and update["messages"]:
                last = update["messages"][-1]
                if hasattr(last, "content") and last.content:
                    print(f"\n[{node_name}]: {last.content[:200]}", end="")
    print()


if __name__ == "__main__":
    run_streaming("What is 15 * 7 + 23?")
    run_streaming("hi")   # too short — validation error
    run_streaming("What is (100 + 50) * 3?")
