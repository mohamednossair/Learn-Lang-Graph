# =============================================================
# TASK 6.3 — Error Recovery (Self-Healing SQL Agent)
# =============================================================
# Goal:
#   Ask: "Show me employee performance scores" (column doesn't exist).
#   The agent should:
#     1. Generate SQL → get SQL ERROR
#     2. Re-inspect schema (describe_table)
#     3. Honestly respond that the column/data doesn't exist
#
# This tests the self-healing loop: error → schema re-check → honest answer
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


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "company.db")


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
    """Get the exact column names and types for a table. ALWAYS call this before writing SQL."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = cursor.fetchall()
        conn.close()
        return "Columns in {}: {}".format(
            table_name,
            ", ".join([f"{c[1]} ({c[2]})" for c in cols])
        )
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def run_sql(query: str) -> str:
    """Execute a SELECT query. Returns rows or an SQL ERROR message."""
    if not query.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT is allowed."
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchmany(10)
        cols = [d[0] for d in cursor.description]
        conn.close()
        if not rows:
            return "Query returned no rows."
        header = " | ".join(cols)
        lines = [header, "-" * len(header)]
        lines += [" | ".join(str(v) for v in row) for row in rows]
        return "\n".join(lines)
    except sqlite3.Error as e:
        return f"SQL ERROR: {str(e)}"


tools = [list_tables, describe_table, run_sql]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(content=(
    "You are a database analyst. When a SQL query fails: "
    "1. Read the error message carefully. "
    "2. Re-inspect the schema with describe_table. "
    "3. If the data truly doesn't exist in any table, say so honestly. "
    "Never invent columns or data."
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
    # These questions reference columns/data that don't exist
    run("Show me employee performance scores")
    run("What is the NPS rating for each department?")
    run("Show me the last_login date for each employee")
