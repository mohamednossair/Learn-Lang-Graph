# =============================================================
# TASK 6.1 — Extend Schema
# =============================================================
# Goal:
#   Add two new tables to company.db:
#     projects(id, name, department_id, budget, status, start_date)
#     project_assignments(employee_id, project_id, role, hours_per_week)
#   Insert 5 projects and 10 assignments.
#   Then ask the agent questions that require JOINs across tables.
#
# Prerequisites: Run lesson_06_database_agent.py first to create company.db
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


# ── STEP 1: Database Path ─────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "company.db")


# ── STEP 2: Create New Tables and Seed Data ───────────────────

def setup_extended_schema():
    """Add projects and project_assignments tables to company.db."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # TODO: CREATE TABLE projects if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER,
            budget REAL,
            status TEXT,
            start_date TEXT
        )
    """)

    # TODO: CREATE TABLE project_assignments if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS project_assignments (
            employee_id INTEGER,
            project_id INTEGER,
            role TEXT,
            hours_per_week INTEGER,
            PRIMARY KEY (employee_id, project_id)
        )
    """)

    # TODO: INSERT 5 projects (check if already inserted first)
    cursor.execute("SELECT COUNT(*) FROM projects")
    if cursor.fetchone()[0] == 0:
        projects = [
            (1, "AI Chatbot Platform", 1, 150000, "active", "2024-01-15"),
            (2, "Sales Dashboard", 2, 80000, "active", "2024-02-01"),
            (3, "Data Warehouse Migration", 1, 200000, "planning", "2024-03-01"),
            (4, "Mobile App v2", 3, 120000, "active", "2024-01-01"),
            (5, "Security Audit", 1, 50000, "completed", "2023-11-01"),
        ]
        cursor.executemany(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)", projects
        )

    # TODO: INSERT 10 project_assignments
    cursor.execute("SELECT COUNT(*) FROM project_assignments")
    if cursor.fetchone()[0] == 0:
        assignments = [
            (1, 1, "Tech Lead", 20),
            (2, 1, "Developer", 30),
            (3, 2, "Analyst", 25),
            (1, 3, "Architect", 15),
            (4, 3, "Developer", 35),
            (5, 4, "Designer", 20),
            (2, 4, "Developer", 25),
            (3, 5, "Security Analyst", 40),
            (6, 2, "Manager", 10),
            (4, 1, "QA", 20),
        ]
        cursor.executemany(
            "INSERT INTO project_assignments VALUES (?, ?, ?, ?)", assignments
        )

    conn.commit()
    conn.close()
    print("✓ Extended schema created and seeded")


# ── STEP 3: Database Tools ────────────────────────────────────

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
    """Get column names and types for a table, plus 2 sample rows."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = cursor.fetchall()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
        rows = cursor.fetchall()
        conn.close()
        schema = "\n".join([f"  {c[1]} ({c[2]})" for c in cols])
        samples = "\n".join([str(r) for r in rows])
        return f"Table: {table_name}\nColumns:\n{schema}\nSample rows:\n{samples}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def run_sql(query: str) -> str:
    """Execute a SELECT query and return results. READ-ONLY — only SELECT is allowed."""
    if not query.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT queries are allowed."
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchmany(20)
        cols = [d[0] for d in cursor.description]
        conn.close()
        if not rows:
            return "Query returned no rows."
        header = " | ".join(cols)
        lines = [header, "-" * len(header)]
        lines += [" | ".join(str(v) for v in row) for row in rows]
        return "\n".join(lines)
    except Exception as e:
        return f"SQL ERROR: {str(e)}"


tools = [list_tables, describe_table, run_sql]


# ── STEP 4: Agent Setup ───────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="llama3", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(content=(
    "You are a database analyst. Always call list_tables first, "
    "then describe_table before writing any SQL. Never guess column names."
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


# ── STEP 5: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()


# ── STEP 6: Test ──────────────────────────────────────────────

def run(question: str):
    print(f"\n{'='*65}")
    print(f"Q: {question}")
    print("=" * 65)
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    print(f"\nA: {result['messages'][-1].content}")


if __name__ == "__main__":
    setup_extended_schema()
    run("Which employees work on more than 2 projects?")
    run("What is the total active project budget?")
    run("Which project has the most assigned employees?")
