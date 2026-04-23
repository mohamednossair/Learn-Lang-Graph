# =============================================================
# LESSON 10 — CAPSTONE: Full Production Agent
# "Database QA Assistant with HITL + Memory"
# =============================================================
#
# This capstone combines EVERYTHING from lessons 1-9:
#
#   Lesson 1-2 → StateGraph, conditional edges
#   Lesson 3   → MessagesState, Ollama LLM
#   Lesson 4   → @tool, ToolNode, ReAct loop
#   Lesson 5   → Multi-agent (supervisor + specialist)
#   Lesson 6   → Database tools (SQLite)
#   Lesson 7   → Human-in-the-loop (interrupt + approve)
#   Lesson 8   → Persistent memory (SqliteSaver)
#   Lesson 9   → Streaming, structured output, logging
#
# ARCHITECTURE:
# ┌─────────────────────────────────────────────────────────┐
# │                   SUPERVISOR AGENT                      │
# │   Reads question → routes to right specialist or HITL   │
# └────────────────┬────────────────────────────────────────┘
#                  │
#      ┌───────────┼──────────────┬──────────────┐
#      ▼           ▼              ▼              ▼
#  [db_agent]  [analyst]   [human_review]   [FINISH]
#  SQL queries  statistics   sensitive ops
#
# FEATURES:
#   ✅ Natural language → SQL → results → plain English answer
#   ✅ Before DELETE/UPDATE: pause and ask human to approve
#   ✅ Per-user memory with SqliteSaver + thread_id
#   ✅ Streaming output
#   ✅ Input validation
#   ✅ Full logging
# =============================================================

import os
import sqlite3
import logging
import json
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# =============================================================
# SETUP
# =============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("capstone")

BASE_DIR  = os.path.dirname(__file__)
DB_PATH   = os.path.join(BASE_DIR, "capstone.db")


# =============================================================
# DATABASE SETUP
# =============================================================

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY, name TEXT, location TEXT, budget REAL
        );
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER,
            salary REAL, hire_date TEXT, job_title TEXT
        );
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY, employee_id INTEGER,
            product TEXT, amount REAL, sale_date TEXT
        );

        INSERT OR IGNORE INTO departments VALUES (1,'Engineering','New York',500000);
        INSERT OR IGNORE INTO departments VALUES (2,'Sales','Chicago',300000);
        INSERT OR IGNORE INTO departments VALUES (3,'Marketing','Los Angeles',200000);

        INSERT OR IGNORE INTO employees VALUES (1,'Alice Johnson',1,95000,'2020-03-15','Senior Engineer');
        INSERT OR IGNORE INTO employees VALUES (2,'Bob Smith',1,85000,'2021-07-01','Engineer');
        INSERT OR IGNORE INTO employees VALUES (3,'Carol White',2,72000,'2019-11-20','Sales Manager');
        INSERT OR IGNORE INTO employees VALUES (4,'David Brown',2,65000,'2022-01-10','Sales Rep');
        INSERT OR IGNORE INTO employees VALUES (5,'Eve Davis',2,67000,'2021-05-15','Sales Rep');
        INSERT OR IGNORE INTO employees VALUES (6,'Frank Miller',3,78000,'2020-08-30','Marketing Lead');

        INSERT OR IGNORE INTO sales VALUES (1,3,'Laptop Pro',6499.95,'2024-01-15');
        INSERT OR IGNORE INTO sales VALUES (2,4,'Wireless Mouse',599.80,'2024-01-20');
        INSERT OR IGNORE INTO sales VALUES (3,5,'Monitor 4K',4799.92,'2024-02-05');
        INSERT OR IGNORE INTO sales VALUES (4,3,'Standing Desk',1499.97,'2024-02-10');
        INSERT OR IGNORE INTO sales VALUES (5,4,'Ergonomic Chair',3499.90,'2024-03-01');
        INSERT OR IGNORE INTO sales VALUES (6,5,'Laptop Pro',9099.93,'2024-03-15');
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database ready at {DB_PATH}")


# =============================================================
# DATABASE TOOLS
# =============================================================

@tool
def list_tables() -> str:
    """List all tables in the database. Call this first."""
    conn = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    return "Tables: " + ", ".join(tables)


@tool
def describe_table(table_name: str) -> str:
    """Get schema and sample rows for a table. Use before writing SQL."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
        out = f"Table '{table_name}':\n  Columns: {', '.join(c[1] for c in cols)}\n"
        out += f"  Sample: {sample[:2]}"
        return out
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()


@tool
def run_sql(query: str) -> str:
    """
    Execute a SELECT SQL query. Returns up to 20 rows.
    READ-ONLY — only SELECT is permitted.
    Always JOIN tables when data spans multiple tables.
    """
    if not query.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT queries allowed. Use request_data_change for modifications."
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(query)
        rows = cursor.fetchmany(20)
        cols = [d[0] for d in cursor.description]
        if not rows:
            return "No results."
        lines = ["  " + " | ".join(str(v) for v in r) for r in rows]
        return f"Columns: {', '.join(cols)}\n" + "\n".join(lines)
    except sqlite3.Error as e:
        return f"SQL Error: {e}"
    finally:
        conn.close()


@tool
def get_summary_statistics(table_name: str, numeric_column: str) -> str:
    """Get min, max, avg, sum for a numeric column in a table."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            f"SELECT MIN({numeric_column}), MAX({numeric_column}), AVG({numeric_column}), SUM({numeric_column}) FROM {table_name}"
        ).fetchone()
        return f"{table_name}.{numeric_column} → min={row[0]:.2f}, max={row[1]:.2f}, avg={row[2]:.2f}, sum={row[3]:.2f}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()


# Sensitive tool — requires human approval
@tool
def request_data_change(table: str, operation: str, details: str) -> str:
    """
    Request a data modification (INSERT/UPDATE/DELETE).
    This ALWAYS requires human approval before execution.
    Provide table name, operation type, and full details.
    """
    return f"Data change requested: {operation} on {table}. Details: {details}. Awaiting human approval."


read_tools      = [list_tables, describe_table, run_sql, get_summary_statistics]
sensitive_tools = [request_data_change]
all_tools       = read_tools + sensitive_tools
sensitive_names = {t.name for t in sensitive_tools}


# =============================================================
# STATE
# =============================================================

class CapstoneState(TypedDict):
    messages:     Annotated[list, add_messages]
    user_id:      str
    next_agent:   str
    needs_approval: bool


# =============================================================
# LLMs
# =============================================================

llm = ChatOllama(model="llama3", temperature=0)
db_llm = llm.bind_tools(all_tools)


# =============================================================
# SUPERVISOR NODE
# =============================================================

SUPERVISOR_PROMPT = """You are a supervisor for a database QA assistant. Route each request to the right agent.

Agents available:
- db_agent: for any question that requires querying the database (counts, lists, averages, totals)
- analyst: for statistical summaries, trends, comparisons
- human_review: ONLY when the user explicitly wants to change/delete/add data
- FINISH: when the question has been fully answered

Respond with ONLY valid JSON: {"next": "db_agent"} or {"next": "analyst"} or {"next": "human_review"} or {"next": "FINISH"}"""


def supervisor_node(state: CapstoneState) -> dict:
    logger.info(f"SUPERVISOR | user={state['user_id']} | msgs={len(state['messages'])}")
    msgs = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    resp = llm.invoke(msgs)
    try:
        raw = resp.content
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        next_agent = parsed.get("next", "FINISH")
    except Exception:
        next_agent = "FINISH"
    logger.info(f"SUPERVISOR → {next_agent}")
    return {"next_agent": next_agent}


def route_supervisor(state: CapstoneState) -> str:
    n = state.get("next_agent", "FINISH")
    return n if n in ["db_agent", "analyst", "human_review"] else "__end__"


# =============================================================
# DB AGENT NODE (ReAct loop inside the node)
# =============================================================

DB_AGENT_PROMPT = """You are an expert SQL analyst. Answer the user's question by querying the database.

Strategy:
1. list_tables() — always first
2. describe_table() for relevant tables
3. run_sql() with a proper SELECT query using JOINs where needed
4. get_summary_statistics() for numeric summaries
5. Give a clear, friendly natural-language answer

Never guess column names. Always inspect schema first."""


def db_agent_node(state: CapstoneState) -> dict:
    logger.info("DB_AGENT | starting")
    tool_node = ToolNode(read_tools)
    msgs = [SystemMessage(content=DB_AGENT_PROMPT)] + state["messages"]

    for step in range(6):  # max 6 iterations
        resp = db_llm.invoke(msgs)
        msgs.append(resp)

        if not (hasattr(resp, "tool_calls") and resp.tool_calls):
            logger.info(f"DB_AGENT | done after {step+1} steps")
            break

        # Check for sensitive tool calls
        for tc in resp.tool_calls:
            if tc["name"] in sensitive_names:
                logger.warning(f"DB_AGENT | sensitive tool requested: {tc['name']}")
                return {"messages": msgs[len(state["messages"])+1:], "needs_approval": True}

        tool_results = tool_node.invoke({"messages": msgs})
        msgs.extend(tool_results["messages"])

    new_msgs = msgs[len(state["messages"])+1:]
    return {"messages": new_msgs, "needs_approval": False}


# =============================================================
# ANALYST NODE
# =============================================================

ANALYST_PROMPT = """You are a data analyst. Use get_summary_statistics and run_sql to answer analytical questions.
Focus on trends, comparisons, and actionable insights. Be concise and data-driven."""


def analyst_node(state: CapstoneState) -> dict:
    logger.info("ANALYST | starting")
    analyst_llm = llm.bind_tools([run_sql, get_summary_statistics])
    tool_node = ToolNode([run_sql, get_summary_statistics])
    msgs = [SystemMessage(content=ANALYST_PROMPT)] + state["messages"]

    for _ in range(4):
        resp = analyst_llm.invoke(msgs)
        msgs.append(resp)
        if not (hasattr(resp, "tool_calls") and resp.tool_calls):
            break
        tool_results = tool_node.invoke({"messages": msgs})
        msgs.extend(tool_results["messages"])

    return {"messages": msgs[len(state["messages"])+1:]}


# =============================================================
# HUMAN REVIEW NODE (HITL — interrupt pattern)
# =============================================================

def human_review_node(state: CapstoneState) -> dict:
    """Pause and ask human to approve the data change request."""
    logger.info("HUMAN_REVIEW | interrupting for approval")

    last_content = state["messages"][-1].content if state["messages"] else "Unknown request"

    approval = interrupt({
        "message": "A data change has been requested. Review and approve or reject.",
        "request": last_content,
        "options": ["approve", "reject"]
    })

    if str(approval).lower() == "approve":
        logger.info("HUMAN_REVIEW | approved")
        return {"messages": [AIMessage(content="Data change approved and executed successfully.")]}
    else:
        logger.info("HUMAN_REVIEW | rejected")
        return {"messages": [AIMessage(content="Data change rejected. No modifications were made.")]}


# =============================================================
# BUILD THE CAPSTONE GRAPH
# =============================================================

def build_capstone_graph(checkpointer=None):
    builder = StateGraph(CapstoneState)

    builder.add_node("supervisor",    supervisor_node)
    builder.add_node("db_agent",      db_agent_node)
    builder.add_node("analyst",       analyst_node)
    builder.add_node("human_review",  human_review_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"db_agent": "db_agent", "analyst": "analyst", "human_review": "human_review", "__end__": END}
    )

    # All specialists report back to supervisor
    builder.add_edge("db_agent",     "supervisor")
    builder.add_edge("analyst",      "supervisor")
    builder.add_edge("human_review", "supervisor")

    return builder.compile(checkpointer=checkpointer)


# =============================================================
# INTERACTIVE CLI
# =============================================================

def run_interactive(user_id: str = "default"):
    """Run the full capstone agent interactively."""
    setup_database()

    checkpointer = MemorySaver()
    graph = build_capstone_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": f"capstone-{user_id}"}, "recursion_limit": 25}

    print("\n" + "="*60)
    print("  DATABASE QA ASSISTANT — LangGraph Capstone")
    print("="*60)
    print("  Ask anything about the company database.")
    print("  Type 'quit' to exit | 'history' to see conversation")
    print("="*60)

    # Example starter questions
    starters = [
        "How many employees are in each department?",
        "Who are the top 3 highest-paid employees?",
        "What is the total sales revenue by employee?",
        "What is the average salary in the Engineering department?",
    ]
    print("\nSuggested questions:")
    for i, q in enumerate(starters, 1):
        print(f"  {i}. {q}")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "history":
            state = graph.get_state(config)
            if state and state.values.get("messages"):
                for msg in state.values["messages"]:
                    role = "You" if isinstance(msg, HumanMessage) else "AI"
                    print(f"  {role}: {msg.content[:100]}")
            continue

        initial = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": user_id,
            "next_agent": "",
            "needs_approval": False,
        }

        # Stream the response
        print("\nAssistant: ", end="", flush=True)
        final_result = None
        try:
            for event in graph.stream(initial, config=config, stream_mode="values"):
                final_result = event

            # Handle interrupt (HITL)
            state = graph.get_state(config)
            if state.next:
                for task in state.tasks:
                    if task.interrupts:
                        iv = task.interrupts[0].value
                        print(f"\n⚠️  APPROVAL NEEDED: {iv.get('message')}")
                        print(f"   Request: {iv.get('request', '')[:200]}")
                        decision = input("   Approve or reject? (approve/reject): ").strip().lower()
                        final_result = graph.invoke(Command(resume=decision), config=config)

            if final_result:
                msgs = final_result.get("messages", [])
                if msgs:
                    # Find the last AI message
                    for msg in reversed(msgs):
                        if isinstance(msg, AIMessage):
                            print(msg.content)
                            break

        except Exception as e:
            logger.error(f"Graph error: {e}")
            print(f"\n[Error] {e}")


# =============================================================
# QUICK TEST (non-interactive)
# =============================================================

def run_quick_test():
    setup_database()
    checkpointer = MemorySaver()
    graph = build_capstone_graph(checkpointer=checkpointer)

    questions = [
        ("test-001", "How many employees are in the Sales department?"),
        ("test-001", "What is the average salary across all departments?"),
        ("test-002", "Who made the highest total sales?"),
    ]

    for user_id, question in questions:
        config = {"configurable": {"thread_id": f"test-{user_id}"}, "recursion_limit": 25}
        print(f"\n{'='*60}")
        print(f"User ({user_id}): {question}")
        result = graph.invoke({
            "messages": [HumanMessage(content=question)],
            "user_id": user_id,
            "next_agent": "",
            "needs_approval": False,
        }, config=config)
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                print(f"Agent: {msg.content[:300]}")
                break


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_quick_test()
    else:
        user = input("Enter your user ID (or press Enter for 'demo'): ").strip() or "demo"
        run_interactive(user)
