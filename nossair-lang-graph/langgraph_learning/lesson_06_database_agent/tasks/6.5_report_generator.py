# =============================================================
# TASK 6.5 — Report Generator Agent
# =============================================================
# Goal:
#   Agent queries key metrics from company.db, composes a
#   markdown report, and saves it using a save_report tool.
#   Output: reports/weekly_YYYY-MM-DD.md
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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_ollama_model


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "company.db")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


# ── STEP 1: Tools ─────────────────────────────────────────────

@tool
def list_tables() -> str:
    """List all tables in company.db."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return "Tables: " + ", ".join(tables)


@tool
def describe_table(table_name: str) -> str:
    """Get column names and types for a table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = cursor.fetchall()
        conn.close()
        return "Columns: " + ", ".join([f"{c[1]}({c[2]})" for c in cols])
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def run_sql(query: str) -> str:
    """Execute a SELECT query. READ-ONLY."""
    if not query.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT is allowed."
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchmany(20)
        cols = [d[0] for d in cursor.description]
        conn.close()
        if not rows:
            return "No rows returned."
        header = " | ".join(cols)
        lines = [header, "-" * len(header)]
        lines += [" | ".join(str(v) for v in row) for row in rows]
        return "\n".join(lines)
    except Exception as e:
        return f"SQL ERROR: {str(e)}"


@tool
def save_report(content: str) -> str:
    """Save a markdown report to reports/weekly_YYYY-MM-DD.md.
    Pass the full markdown content as the content argument."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = f"weekly_{date.today().isoformat()}.md"
    filepath = os.path.join(REPORTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [tool:save_report] Saved to {filepath}")
    return f"Report saved to: {filepath}"


tools = [list_tables, describe_table, run_sql, save_report]


# ── STEP 2: Agent ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model=get_ollama_model(), temperature=0.2)
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(content=(
    "You are a business analyst. Generate a weekly markdown report with these sections:\n"
    "1. Employee Summary (total count, by department)\n"
    "2. Top 3 Earners\n"
    "3. Sales Summary (total revenue, top performer)\n"
    "Query the database for each section, then call save_report with the full markdown."
))


def agent_node(state: AgentState) -> dict:
    messages = [SYSTEM] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()


if __name__ == "__main__":
    print("=" * 65)
    print("Report Generator Agent")
    print("=" * 65)
    result = graph.invoke({
        "messages": [HumanMessage(content="Generate the weekly business report and save it.")]
    })
    print(f"\nAgent: {result['messages'][-1].content}")
