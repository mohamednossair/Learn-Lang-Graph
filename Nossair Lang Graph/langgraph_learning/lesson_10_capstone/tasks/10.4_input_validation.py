# =============================================================
# TASK 10.4 — Input Validation Layer
# =============================================================
# Goal:
#   Add Pydantic validation before every invoke().
#   Rules:
#     - question: 3–500 chars
#     - user_id: alphanumeric + hyphens, max 20 chars
#   Invalid input never enters the graph.
#   Returns a structured error response immediately.
# =============================================================

import re
import sqlite3
import os
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, field_validator, model_validator
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "capstone.db")
llm = ChatOllama(model="llama3.2", temperature=0)


# ── STEP 1: Pydantic Validation Models ────────────────────────

class AgentRequest(BaseModel):
    question: str
    user_id: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError(f"Question too short: {len(v)} chars (min 3)")
        if len(v) > 500:
            raise ValueError(f"Question too long: {len(v)} chars (max 500)")
        return v

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        v = v.strip()
        if len(v) > 20:
            raise ValueError(f"user_id too long: {len(v)} chars (max 20)")
        if not re.match(r'^[a-zA-Z0-9\-]+$', v):
            raise ValueError(f"user_id must be alphanumeric + hyphens only: '{v}'")
        return v


class AgentResponse(BaseModel):
    answer: str
    user_id: str
    valid: bool
    error: str = ""


# ── STEP 2: State ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    error: str


# ── STEP 3: Tools ─────────────────────────────────────────────

@tool
def run_sql(query: str) -> str:
    """Execute a SELECT query. READ-ONLY."""
    if not query.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT allowed."
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(query)
        rows = c.fetchmany(10)
        cols = [d[0] for d in c.description]
        conn.close()
        if not rows:
            return "No rows."
        return " | ".join(cols) + "\n" + "\n".join([" | ".join(str(v) for v in r) for r in rows])
    except Exception as e:
        return f"SQL ERROR: {str(e)}"


@tool
def list_tables() -> str:
    """List all tables in the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    conn.close()
    return "Tables: " + ", ".join(tables)


tools = [list_tables, run_sql]
llm_with_tools = llm.bind_tools(tools)


# ── STEP 4: Graph Nodes ───────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    system = SystemMessage(content="You are a helpful database assistant.")
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ── STEP 5: Build Graph ───────────────────────────────────────

checkpointer = MemorySaver()
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=checkpointer)


# ── STEP 6: Validated Entry Point ────────────────────────────

def ask(question: str, user_id: str) -> AgentResponse:
    """
    Validated entry point. Validates inputs BEFORE invoking the graph.
    Invalid inputs return an AgentResponse with valid=False immediately.
    """
    try:
        req = AgentRequest(question=question, user_id=user_id)
    except Exception as e:
        return AgentResponse(
            answer="",
            user_id=user_id,
            valid=False,
            error=str(e),
        )

    config = {"configurable": {"thread_id": req.user_id}, "recursion_limit": 15}
    result = graph.invoke(
        {"messages": [HumanMessage(content=req.question)], "user_id": req.user_id, "error": ""},
        config=config,
    )
    return AgentResponse(
        answer=result["messages"][-1].content,
        user_id=req.user_id,
        valid=True,
    )


# ── STEP 7: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # (question, user_id, expect_valid)
        ("How many employees are there?", "alice-001", True),
        ("hi", "alice-001", False),          # too short
        ("x" * 501, "alice-001", False),     # too long
        ("What is total revenue?", "bob", True),
        ("What tables exist?", "invalid user!", False),  # bad user_id chars
        ("show data", "a" * 21, False),      # user_id too long
    ]

    print("=" * 65)
    print("Input Validation Layer Test")
    print("=" * 65)

    for question, user_id, expected_valid in test_cases:
        response = ask(question, user_id)
        status = "✓" if response.valid == expected_valid else "✗ UNEXPECTED"
        print(f"\n{status} | valid={response.valid} | user='{user_id}'")
        print(f"   Q: '{question[:60]}'")
        if response.valid:
            print(f"   A: {response.answer[:100]}")
        else:
            print(f"   Error: {response.error}")
