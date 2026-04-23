# =============================================================
# LESSON 6 — Database Agent (Text-to-SQL with SQLite)
# =============================================================
#
# WHAT YOU WILL LEARN:
#   - Build a real SQLite database from scratch
#   - Tools that query a database (list_tables, describe_table, run_sql)
#   - How the LLM reasons about schema before writing SQL
#   - Safe SQL execution (read-only guard)
#   - How to connect an agent to ANY database
#
# AGENT FLOW:
#   User question (natural language)
#        ↓
#   [agent] → understands question, looks at schema
#        ↓
#   [tools] → list_tables → describe_table → run_sql
#        ↓
#   [agent] → interprets results, forms natural language answer
#        ↓
#   Final answer to user
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


# =============================================================
# PART 1 — BUILD THE SAMPLE DATABASE
# =============================================================

DB_PATH = os.path.join(os.path.dirname(__file__), "company.db")


def build_sample_database():
    """Create a sample company database with realistic data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.executescript("""
        DROP TABLE IF EXISTS employees;
        DROP TABLE IF EXISTS departments;
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS products;

        CREATE TABLE departments (
            id          INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            location    TEXT,
            budget      REAL
        );

        CREATE TABLE employees (
            id            INTEGER PRIMARY KEY,
            name          TEXT NOT NULL,
            department_id INTEGER REFERENCES departments(id),
            salary        REAL,
            hire_date     TEXT,
            job_title     TEXT
        );

        CREATE TABLE products (
            id       INTEGER PRIMARY KEY,
            name     TEXT NOT NULL,
            category TEXT,
            price    REAL,
            stock    INTEGER
        );

        CREATE TABLE sales (
            id          INTEGER PRIMARY KEY,
            employee_id INTEGER REFERENCES employees(id),
            product_id  INTEGER REFERENCES products(id),
            quantity    INTEGER,
            sale_date   TEXT,
            total       REAL
        );

        -- Departments
        INSERT INTO departments VALUES (1, 'Engineering',  'New York',  500000);
        INSERT INTO departments VALUES (2, 'Sales',        'Chicago',   300000);
        INSERT INTO departments VALUES (3, 'Marketing',    'Los Angeles',200000);
        INSERT INTO departments VALUES (4, 'HR',           'New York',  150000);

        -- Employees
        INSERT INTO employees VALUES (1,  'Alice Johnson',  1, 95000,  '2020-03-15', 'Senior Engineer');
        INSERT INTO employees VALUES (2,  'Bob Smith',      1, 85000,  '2021-07-01', 'Engineer');
        INSERT INTO employees VALUES (3,  'Carol White',    2, 72000,  '2019-11-20', 'Sales Manager');
        INSERT INTO employees VALUES (4,  'David Brown',    2, 65000,  '2022-01-10', 'Sales Rep');
        INSERT INTO employees VALUES (5,  'Eve Davis',      2, 67000,  '2021-05-15', 'Sales Rep');
        INSERT INTO employees VALUES (6,  'Frank Miller',   3, 78000,  '2020-08-30', 'Marketing Lead');
        INSERT INTO employees VALUES (7,  'Grace Wilson',   4, 70000,  '2018-04-12', 'HR Manager');
        INSERT INTO employees VALUES (8,  'Hank Taylor',    1, 90000,  '2019-09-05', 'Lead Engineer');
        INSERT INTO employees VALUES (9,  'Iris Anderson',  3, 72000,  '2022-03-22', 'Marketing Analyst');
        INSERT INTO employees VALUES (10, 'Jack Martinez',  2, 63000,  '2023-01-15', 'Sales Rep');

        -- Products
        INSERT INTO products VALUES (1, 'Laptop Pro',    'Electronics', 1299.99, 50);
        INSERT INTO products VALUES (2, 'Wireless Mouse','Electronics',   29.99, 200);
        INSERT INTO products VALUES (3, 'Standing Desk', 'Furniture',    499.99, 30);
        INSERT INTO products VALUES (4, 'Monitor 4K',    'Electronics',  599.99, 75);
        INSERT INTO products VALUES (5, 'Ergonomic Chair','Furniture',   349.99, 40);

        -- Sales
        INSERT INTO sales VALUES (1,  3, 1, 5,  '2024-01-15', 6499.95);
        INSERT INTO sales VALUES (2,  4, 2, 20, '2024-01-20',  599.80);
        INSERT INTO sales VALUES (3,  5, 4, 8,  '2024-02-05', 4799.92);
        INSERT INTO sales VALUES (4,  3, 3, 3,  '2024-02-10', 1499.97);
        INSERT INTO sales VALUES (5,  4, 5, 10, '2024-03-01', 3499.90);
        INSERT INTO sales VALUES (6,  5, 1, 7,  '2024-03-15', 9099.93);
        INSERT INTO sales VALUES (7, 10, 2, 50, '2024-04-01', 1499.50);
        INSERT INTO sales VALUES (8,  3, 4, 4,  '2024-04-20', 2399.96);
        INSERT INTO sales VALUES (9,  4, 1, 3,  '2024-05-05', 3899.97);
        INSERT INTO sales VALUES (10, 5, 3, 5,  '2024-05-10', 2499.95);
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Sample database created at: {DB_PATH}")


# =============================================================
# PART 2 — DATABASE TOOLS
# =============================================================

def get_db_connection():
    return sqlite3.connect(DB_PATH)


@tool
def list_tables() -> str:
    """List all tables available in the database. Always call this first to understand the database structure."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return f"Available tables: {', '.join(tables)}"


@tool
def describe_table(table_name: str) -> str:
    """Get the schema (columns, types) of a specific table. Use this before writing SQL to understand column names."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        if not columns:
            return f"Table '{table_name}' not found."
        result = f"Table: {table_name}\nColumns:\n"
        for col in columns:
            result += f"  - {col[1]} ({col[2]})\n"
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 2;")
        sample = cursor.fetchall()
        result += f"\nSample rows (first 2):\n"
        for row in sample:
            result += f"  {row}\n"
        return result
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()


@tool
def run_sql(query: str) -> str:
    """
    Execute a SQL SELECT query on the database and return the results.
    IMPORTANT: Only SELECT queries are allowed. Never use INSERT, UPDATE, DELETE, DROP.
    Always use JOINs when data spans multiple tables.
    Limit results to 20 rows unless the user asks for more.
    """
    # Safety guard — only allow SELECT
    query_clean = query.strip().upper()
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
    for word in forbidden:
        if query_clean.startswith(word):
            return f"ERROR: Only SELECT queries are allowed. '{word}' is not permitted."

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        if not rows:
            return "Query returned no results."

        # Format as a readable table
        result = f"Columns: {', '.join(columns)}\n"
        result += f"Rows ({len(rows)} total):\n"
        for row in rows:
            result += "  " + " | ".join(str(v) for v in row) + "\n"
        return result
    except sqlite3.Error as e:
        return f"SQL Error: {e}"
    finally:
        conn.close()


@tool
def get_table_relationships() -> str:
    """Get the foreign key relationships between tables to understand how to JOIN them."""
    conn = get_db_connection()
    cursor = conn.cursor()
    relationships = []
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    for table in tables:
        cursor.execute(f"PRAGMA foreign_key_list({table});")
        fks = cursor.fetchall()
        for fk in fks:
            relationships.append(f"{table}.{fk[3]} → {fk[2]}.{fk[4]}")
    conn.close()
    if relationships:
        return "Foreign key relationships:\n" + "\n".join(f"  {r}" for r in relationships)
    return "No foreign key relationships found."


# =============================================================
# PART 3 — THE DATABASE AGENT
# =============================================================

db_tools = [list_tables, describe_table, get_table_relationships, run_sql]


class DBAgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """You are an expert SQL database analyst. Your job is to answer questions about data in a SQLite database.

STRATEGY — always follow this order:
1. Call list_tables() to see available tables
2. Call describe_table() for relevant tables to understand columns
3. If needed, call get_table_relationships() to understand JOINs
4. Write and execute a SQL SELECT query with run_sql()
5. Interpret the results and give a clear natural language answer

RULES:
- Never guess column names — always check schema first
- Use proper JOINs when combining tables
- Always add LIMIT 20 unless user asks for all rows
- If a query fails, fix it and try again
- Give clear, human-readable answers, not just raw data"""


llm = ChatOllama(model="llama3", temperature=0)
llm_with_tools = llm.bind_tools(db_tools)


def agent_node(state: DBAgentState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: DBAgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


graph_builder = StateGraph(DBAgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(db_tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()


def ask_db(question: str) -> str:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content
    print(f"\nAnswer:\n{answer}")
    return answer


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    build_sample_database()

    ask_db("How many employees are in each department?")
    ask_db("Who are the top 3 highest-paid employees?")
    ask_db("What is the total sales revenue per employee? Show names.")
    ask_db("Which product has the highest total quantity sold?")
    ask_db("What is the average salary by department?")


# =============================================================
# EXERCISE:
#   1. Add a new table called "projects" with columns:
#      id, name, department_id, budget, status
#   2. Insert 3 sample projects
#   3. Create a @tool called "get_project_summary" that returns
#      all projects grouped by department
#   4. Ask the agent: "Which department has the most projects?"
# =============================================================
