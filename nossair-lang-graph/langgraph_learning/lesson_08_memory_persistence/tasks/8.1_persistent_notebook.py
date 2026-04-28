# =============================================================
# TASK 8.1 — Persistent Notebook
# =============================================================
# Goal:
#   Build a note-taking agent backed by SqliteSaver.
#   Commands:
#     "add note: X"  → appends to notes list in state
#     "show notes"   → lists all notes (no LLM needed)
#     "search: X"    → finds notes containing keyword X
#   Restart Python and verify notes persist across restarts.
#
# State: {messages, notes: list, tags: dict}
# =============================================================

import sys
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


# ── STEP 1: State ─────────────────────────────────────────────

def append_notes(existing: list, new: list) -> list:
    return existing + new


def merge_tags(existing: dict, new: dict) -> dict:
    merged = dict(existing)
    merged.update(new)
    return merged


class NotebookState(TypedDict):
    messages: Annotated[list, add_messages]
    notes: Annotated[list, append_notes]
    tags: Annotated[dict, merge_tags]
    output: str


# ── STEP 2: Command Parser Node ───────────────────────────────
# TODO:
#   Parse the last message and dispatch:
#     "add note: X"  → append X to notes, set output = "Note added."
#     "show notes"   → set output = formatted list of all notes
#     "search: X"    → set output = notes containing X (case-insensitive)
#     anything else  → set output = "Unknown command."

def parse_command(state: NotebookState) -> dict:
    from langchain_core.messages import HumanMessage
    last = state["messages"][-1]
    text = last.content.strip() if hasattr(last, "content") else ""
    text_lower = text.lower()

    if text_lower.startswith("add note:"):
        note = text[9:].strip()
        # TODO: return notes append + output message
        pass

    elif text_lower == "show notes":
        notes = state["notes"]
        if not notes:
            return {"output": "No notes yet."}
        # TODO: return formatted numbered list
        pass

    elif text_lower.startswith("search:"):
        keyword = text[7:].strip().lower()
        matches = [n for n in state["notes"] if keyword in n.lower()]
        if not matches:
            return {"output": f"No notes found for '{keyword}'."}
        # TODO: return formatted match list
        pass

    else:
        return {"output": f"Unknown command: '{text}'. Try: 'add note: X', 'show notes', 'search: X'"}


# ── STEP 3: Build Graph ───────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "notebook.db")

with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    graph_builder = StateGraph(NotebookState)
    graph_builder.add_node("parse", parse_command)
    graph_builder.add_edge(START, "parse")
    graph_builder.add_edge("parse", END)
    graph = graph_builder.compile(checkpointer=checkpointer)

    # ── STEP 4: Run ───────────────────────────────────────────

    USER_ID = "user-alice"
    config = {"configurable": {"thread_id": USER_ID}}

    def run(command: str):
        from langchain_core.messages import HumanMessage
        result = graph.invoke(
            {"messages": [HumanMessage(content=command)], "notes": [], "tags": {}, "output": ""},
            config=config,
        )
        print(f"  > {result['output']}")
        return result

    if __name__ == "__main__":
        print("=" * 55)
        print(f"Persistent Notebook — thread: {USER_ID}")
        print(f"Database: {DB_PATH}")
        print("=" * 55)

        commands = [
            "add note: LangGraph uses StateGraph to model workflows",
            "add note: Nodes are Python functions that update state",
            "add note: SqliteSaver persists state to disk",
            "show notes",
            "search: state",
            "add note: add_messages reducer appends to message list",
            "show notes",
        ]

        for cmd in commands:
            print(f"\n$ {cmd}")
            run(cmd)

        print("\n✓ Restart this script to verify notes persisted!")
