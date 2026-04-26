# =============================================================
# TASK 10.1 — Add HR Specialist to Capstone
# =============================================================
# Goal:
#   Extend the capstone supervisor with a new hr_agent that handles:
#     - hire dates and employee tenure
#     - department transfers
#     - org chart queries (who reports to whom)
#   Update supervisor routing to include "hr" as a valid target.
#
# Prerequisites: Run lesson_10_capstone.py first to create capstone.db
# =============================================================

import sqlite3
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_ollama_model


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "capstone.db")
llm = ChatOllama(model=get_ollama_model(), temperature=0)


# ── STEP 1: State ─────────────────────────────────────────────

class CapstoneState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    user_id: str


# ── STEP 2: HR Tools ──────────────────────────────────────────

@tool
def list_tables() -> str:
    """List all tables in the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    conn.close()
    return "Tables: " + ", ".join(tables)


@tool
def describe_table(table_name: str) -> str:
    """Get columns for a table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    cols = c.fetchall()
    conn.close()
    return ", ".join([f"{col[1]}({col[2]})" for col in cols])


@tool
def run_sql(query: str) -> str:
    """Execute a SELECT query. READ-ONLY."""
    if not query.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT allowed."
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(query)
        rows = c.fetchmany(15)
        cols = [d[0] for d in c.description]
        conn.close()
        if not rows:
            return "No rows."
        header = " | ".join(cols)
        return header + "\n" + "\n".join([" | ".join(str(v) for v in r) for r in rows])
    except Exception as e:
        return f"SQL ERROR: {str(e)}"


hr_tools = [list_tables, describe_table, run_sql]
hr_llm = llm.bind_tools(hr_tools)


# ── STEP 3: HR Agent ──────────────────────────────────────────

def hr_agent_node(state: CapstoneState) -> dict:
    system = SystemMessage(content=(
        "You are an HR specialist. Answer questions about: "
        "hire dates, employee tenure, department transfers, org chart. "
        "Always inspect schema before writing SQL."
    ))
    response = hr_llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def hr_should_continue(state: CapstoneState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "hr_tools"
    return "supervisor"


# ── STEP 4: Supervisor Node ───────────────────────────────────

SUPERVISOR_PROMPT = """You are a business intelligence supervisor.
Route to the correct specialist:
- db_agent: SQL data queries, sales, revenue, employee data
- hr_agent: hire dates, tenure, transfers, org chart, HR questions
- FINISH: task complete

Current conversation context is in the messages above.
Respond with ONLY one word: db_agent, hr_agent, or FINISH"""


def supervisor_node(state: CapstoneState) -> dict:
    response = llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"])
    decision = response.content.strip().lower()
    if decision not in {"db_agent", "hr_agent", "finish"}:
        decision = "db_agent"
    print(f"[supervisor] → {decision}")
    return {"next": decision}


def route_supervisor(state: CapstoneState) -> str:
    return state["next"] if state["next"] != "finish" else "__end__"


# ── STEP 5: DB Agent (simplified from capstone) ───────────────

db_llm = llm.bind_tools([list_tables, describe_table, run_sql])


def db_agent_node(state: CapstoneState) -> dict:
    system = SystemMessage(content="You are a database analyst. Always inspect schema first.")
    response = db_llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def db_should_continue(state: CapstoneState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "db_tools"
    return "supervisor"


# ── STEP 6: Build Graph ───────────────────────────────────────

checkpointer = MemorySaver()
builder = StateGraph(CapstoneState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("db_agent", db_agent_node)
builder.add_node("db_tools", ToolNode([list_tables, describe_table, run_sql]))
builder.add_node("hr_agent", hr_agent_node)
builder.add_node("hr_tools", ToolNode(hr_tools))

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_supervisor, {
    "db_agent": "db_agent",
    "hr_agent": "hr_agent",
    "__end__": END,
})
builder.add_conditional_edges("db_agent", db_should_continue, {
    "db_tools": "db_tools", "supervisor": "supervisor"
})
builder.add_edge("db_tools", "db_agent")
builder.add_conditional_edges("hr_agent", hr_should_continue, {
    "hr_tools": "hr_tools", "supervisor": "supervisor"
})
builder.add_edge("hr_tools", "hr_agent")

graph = builder.compile(checkpointer=checkpointer)


def run(question: str, user_id: str = "user-001"):
    config = {"configurable": {"thread_id": user_id}, "recursion_limit": 20}
    print(f"\n{'='*65}")
    print(f"Q: {question}")
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)], "next": "", "user_id": user_id},
        config=config,
    )
    print(f"A: {result['messages'][-1].content[:300]}")


if __name__ == "__main__":
    run("What is the total sales revenue?")
    run("Who has worked at the company the longest?", user_id="user-002")
    run("How many employees are in each department?", user_id="user-003")
