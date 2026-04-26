# =============================================================
# TASK 10.3 — Upgrade to SqliteSaver
# =============================================================
# Goal:
#   Replace MemorySaver with SqliteSaver in the capstone agent.
#   Verify persistence across restarts.
#   Add CLI flags:
#     --clear          : delete the checkpoint database
#     --list-sessions  : show all active sessions
#
# Usage:
#   python 10.3_sqlite_saver_upgrade.py
#   python 10.3_sqlite_saver_upgrade.py --clear
#   python 10.3_sqlite_saver_upgrade.py --list-sessions
# =============================================================

import sys
import os
import sqlite3
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_ollama_model


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "capstone.db")
CHECKPOINT_DB = os.path.join(os.path.dirname(__file__), "capstone_checkpoints.db")
llm = ChatOllama(model=get_ollama_model(), temperature=0)


# ── STEP 1: State ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


# ── STEP 2: Database Tools ────────────────────────────────────

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
        return f"ERROR: {str(e)}"


@tool
def list_tables() -> str:
    """List all tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    conn.close()
    return "Tables: " + ", ".join(tables)


tools = [list_tables, run_sql]
llm_with_tools = llm.bind_tools(tools)


# ── STEP 3: Agent Node ────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    system = SystemMessage(content="You are a helpful database assistant.")
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ── STEP 4: CLI Utilities ─────────────────────────────────────

def clear_checkpoints():
    if os.path.exists(CHECKPOINT_DB):
        os.remove(CHECKPOINT_DB)
        print(f"✓ Deleted checkpoint database: {CHECKPOINT_DB}")
    else:
        print("No checkpoint database found.")


def list_sessions():
    if not os.path.exists(CHECKPOINT_DB):
        print("No checkpoint database found. Run the agent first.")
        return
    conn = sqlite3.connect(CHECKPOINT_DB)
    c = conn.cursor()
    try:
        c.execute("SELECT DISTINCT thread_id, MAX(ts) as last_active FROM checkpoints GROUP BY thread_id")
        rows = c.fetchall()
        if not rows:
            print("No active sessions.")
        else:
            print(f"Active sessions ({len(rows)}):")
            for thread_id, ts in rows:
                print(f"  - {thread_id} (last: {ts[:19] if ts else 'unknown'})")
    except sqlite3.OperationalError as e:
        print(f"Error reading sessions: {e}")
    finally:
        conn.close()


# ── STEP 5: Build and Run ─────────────────────────────────────

def build_graph(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


def run_conversation(user_id: str, questions: list):
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": user_id}, "recursion_limit": 15}

        for q in questions:
            print(f"\n[{user_id}] Q: {q}")
            result = graph.invoke(
                {"messages": [HumanMessage(content=q)], "user_id": user_id},
                config=config,
            )
            print(f"[{user_id}] A: {result['messages'][-1].content[:200]}")


if __name__ == "__main__":
    if "--clear" in sys.argv:
        clear_checkpoints()
        sys.exit(0)

    if "--list-sessions" in sys.argv:
        list_sessions()
        sys.exit(0)

    print("=" * 60)
    print(f"SqliteSaver Persistence Demo")
    print(f"Checkpoint DB: {CHECKPOINT_DB}")
    print("=" * 60)

    run_conversation("alice", [
        "What tables are in the database?",
        "How many employees are there?",
    ])

    run_conversation("bob", [
        "What is the total sales revenue?",
    ])

    print("\n" + "=" * 60)
    print("Sessions saved. Run with --list-sessions to verify persistence.")
    print("Run with --clear to delete all sessions.")
