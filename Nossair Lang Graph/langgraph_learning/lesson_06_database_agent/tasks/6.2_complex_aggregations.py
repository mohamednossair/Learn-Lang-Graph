# =============================================================
# TASK 6.2 — Complex Aggregations
# =============================================================
# Goal:
#   Ask the database agent complex multi-table aggregation questions:
#   1. "Top 3 departments by average salary with headcount"
#   2. "Employee with highest revenue-to-salary ratio"
#   3. "Monthly sales trend for 2024"
#
# This task uses company.db from lesson_06_database_agent.
# Run lesson_06_database_agent.py first to create the database.
# =============================================================

import sqlite3
import os
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


# ── Tools (same as lesson 6 — copy or import) ─────────────────

@tool
def list_tables() -> str:
    """List all tables in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return "Tables: " + ", ".join(tables)


@tool
def describe_table(table_name: str) -> str:
    """Get columns and 3 sample rows for a table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = cursor.fetchall()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        rows = cursor.fetchall()
        conn.close()
        schema = "\n".join([f"  {c[1]} ({c[2]})" for c in cols])
        samples = "\n".join([str(r) for r in rows])
        return f"Table: {table_name}\nColumns:\n{schema}\nSamples:\n{samples}"
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
        rows = cursor.fetchmany(30)
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


tools = [list_tables, describe_table, run_sql]


# ── Agent Setup ───────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model=get_ollama_model(), temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(content=(
    "You are a database analyst expert in SQL aggregations. "
    "Always inspect schema before writing SQL. Use GROUP BY, "
    "ORDER BY, HAVING, and JOINs as needed. LIMIT to 10 rows."
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


def run(question: str):
    print(f"\n{'='*65}")
    print(f"Q: {question}")
    print("=" * 65)
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    print(f"\nA: {result['messages'][-1].content}")


if __name__ == "__main__":
    run("Show top 3 departments by average salary with headcount for each")
    run("Which employee has the highest revenue-to-salary ratio?")
    run("Show monthly sales trend for 2024")
