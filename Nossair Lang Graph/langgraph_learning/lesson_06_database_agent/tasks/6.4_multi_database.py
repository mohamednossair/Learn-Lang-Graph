# =============================================================
# TASK 6.4 — Multi-Database Agent
# =============================================================
# Goal:
#   Build an agent that queries TWO separate SQLite databases:
#     company.db  — employees, departments, sales
#     inventory.db — products, stock, orders
#   Agent detects which DB the question is about before querying.
#
# This task creates inventory.db from scratch.
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


COMPANY_DB = os.path.join(os.path.dirname(__file__), "..", "company.db")
INVENTORY_DB = os.path.join(os.path.dirname(__file__), "inventory.db")


# ── STEP 1: Create inventory.db ───────────────────────────────

def setup_inventory_db():
    conn = sqlite3.connect(INVENTORY_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            unit_price REAL,
            stock_quantity INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            quantity INTEGER,
            order_date TEXT,
            status TEXT
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?)", [
            (1, "Laptop Pro 15", "Electronics", 1299.99, 45),
            (2, "Wireless Mouse", "Electronics", 29.99, 200),
            (3, "Standing Desk", "Furniture", 599.99, 12),
            (4, "Office Chair", "Furniture", 349.99, 8),
            (5, "USB-C Hub", "Electronics", 49.99, 150),
        ])
        cursor.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", [
            (1, 1, 3, "2024-01-10", "delivered"),
            (2, 2, 10, "2024-01-15", "delivered"),
            (3, 3, 2, "2024-02-01", "pending"),
            (4, 1, 1, "2024-02-10", "delivered"),
            (5, 5, 5, "2024-02-20", "shipped"),
        ])
    conn.commit()
    conn.close()
    print("✓ inventory.db created")


# ── STEP 2: Tools with DB selector ───────────────────────────

def _get_conn(db: str) -> sqlite3.Connection:
    if db == "inventory":
        return sqlite3.connect(INVENTORY_DB)
    return sqlite3.connect(COMPANY_DB)


@tool
def list_tables(database: str) -> str:
    """List all tables in a database. Use database='company' or database='inventory'."""
    conn = _get_conn(database)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return f"Tables in {database}: " + ", ".join(tables)


@tool
def describe_table(database: str, table_name: str) -> str:
    """Get columns for a table in a specific database (company or inventory)."""
    try:
        conn = _get_conn(database)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = cursor.fetchall()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
        rows = cursor.fetchall()
        conn.close()
        schema = ", ".join([f"{c[1]}({c[2]})" for c in cols])
        samples = " | ".join([str(r) for r in rows])
        return f"{database}.{table_name}: {schema}\nSamples: {samples}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def run_sql(database: str, query: str) -> str:
    """Run a SELECT query on a database. database must be 'company' or 'inventory'."""
    if not query.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT is allowed."
    try:
        conn = _get_conn(database)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchmany(15)
        cols = [d[0] for d in cursor.description]
        conn.close()
        if not rows:
            return "No rows."
        header = " | ".join(cols)
        lines = [header, "-" * len(header)]
        lines += [" | ".join(str(v) for v in row) for row in rows]
        return "\n".join(lines)
    except Exception as e:
        return f"SQL ERROR: {str(e)}"


tools = [list_tables, describe_table, run_sql]


# ── STEP 3: Agent ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(content=(
    "You have access to two databases: 'company' (employees/sales) and 'inventory' (products/orders). "
    "Always pass the correct database name to tools. Inspect schema before querying."
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
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    print(f"A: {result['messages'][-1].content}")


if __name__ == "__main__":
    setup_inventory_db()
    run("What products have less than 15 items in stock?")
    run("How many employees are in the Engineering department?")
    run("What is the total value of pending orders in inventory?")
