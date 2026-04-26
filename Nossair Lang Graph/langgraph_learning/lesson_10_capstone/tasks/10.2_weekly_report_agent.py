# =============================================================
# TASK 10.2 — Weekly Report Agent
# =============================================================
# Goal:
#   Add a report_agent to the capstone that:
#     - Queries top metrics from the database
#     - Generates a structured markdown report
#     - Saves to reports/weekly_YYYY-MM-DD.md
#   Supervisor routes "generate report" questions to report_agent.
# =============================================================

import sqlite3
import os
from datetime import date
from typing import Annotated
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
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
llm = ChatOllama(model=get_ollama_model(), temperature=0.2)


class ReportState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    report_path: str


# ── Tools ─────────────────────────────────────────────────────

@tool
def run_sql(query: str) -> str:
    """Execute a SELECT query against capstone.db. READ-ONLY."""
    if not query.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT allowed."
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(query)
        rows = c.fetchmany(20)
        cols = [d[0] for d in c.description]
        conn.close()
        if not rows:
            return "No rows."
        header = " | ".join(cols)
        return header + "\n" + "\n".join([" | ".join(str(v) for v in r) for r in rows])
    except Exception as e:
        return f"SQL ERROR: {str(e)}"


@tool
def list_tables() -> str:
    """List all tables in capstone.db."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    conn.close()
    return "Tables: " + ", ".join(tables)


@tool
def describe_table(table_name: str) -> str:
    """Get column info for a table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    cols = c.fetchall()
    conn.close()
    return ", ".join([f"{col[1]}({col[2]})" for col in cols])


@tool
def save_report(content: str) -> str:
    """Save the markdown report to reports/weekly_YYYY-MM-DD.md."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = f"weekly_{date.today().isoformat()}.md"
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Report saved: {path}"


tools = [list_tables, describe_table, run_sql, save_report]
llm_with_tools = llm.bind_tools(tools)


# ── Supervisor Node ───────────────────────────────────────────

SUPERVISOR_PROMPT = """You are a business supervisor.
Route to: report_agent (for any report generation request) or FINISH.
Respond with ONLY: report_agent or FINISH"""


def supervisor_node(state: ReportState) -> dict:
    response = llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"])
    decision = response.content.strip().lower()
    if "report" in decision:
        decision = "report_agent"
    else:
        decision = "finish"
    return {"next": decision}


def route_supervisor(state: ReportState) -> str:
    return state["next"] if state["next"] != "finish" else "__end__"


# ── Report Agent Node ─────────────────────────────────────────

REPORT_SYSTEM = SystemMessage(content=(
    "You are a business report writer. Generate a weekly markdown report with:\n"
    "# Weekly Business Report — {date}\n"
    "## 1. Employee Summary\n"
    "## 2. Sales Performance\n"
    "## 3. Top Performers\n"
    "Query each section from the database, then call save_report with the full markdown content."
))


def report_agent_node(state: ReportState) -> dict:
    response = llm_with_tools.invoke([REPORT_SYSTEM] + state["messages"])
    return {"messages": [response]}


def report_should_continue(state: ReportState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "supervisor"


# ── Build Graph ───────────────────────────────────────────────

checkpointer = MemorySaver()
builder = StateGraph(ReportState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("report_agent", report_agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_supervisor, {
    "report_agent": "report_agent",
    "__end__": END,
})
builder.add_conditional_edges("report_agent", report_should_continue, {
    "tools": "tools",
    "supervisor": "supervisor",
})
builder.add_edge("tools", "report_agent")
graph = builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    print("=" * 65)
    print("Weekly Report Agent")
    print("=" * 65)
    config = {"configurable": {"thread_id": "report-001"}, "recursion_limit": 25}
    result = graph.invoke(
        {"messages": [HumanMessage(content="Generate the weekly business report")],
         "next": "", "report_path": ""},
        config=config,
    )
    print(f"\nAgent: {result['messages'][-1].content}")
