# =============================================================
# TASK 8.4 — Session Cleanup
# =============================================================
# Goal:
#   Implement cleanup_old_sessions() that:
#     1. Reads all threads from SqliteSaver
#     2. Deletes threads older than 30 days
#     3. Prints count of cleaned sessions
#
# Also implement list_sessions_with_ages() for visibility.
# =============================================================

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


# ── STEP 1: State ─────────────────────────────────────────────

class SessionState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


# ── STEP 2: Build Minimal Graph ───────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")


def echo_node(state: SessionState) -> dict:
    return {}


with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    graph_builder = StateGraph(SessionState)
    graph_builder.add_node("echo", echo_node)
    graph_builder.add_edge(START, "echo")
    graph_builder.add_edge("echo", END)
    graph = graph_builder.compile(checkpointer=checkpointer)


    # ── STEP 3: Session Utilities ─────────────────────────────

    def list_sessions_with_ages() -> list[dict]:
        """Return all sessions with their last activity timestamp and age in days."""
        # TODO:
        #   Query: SELECT thread_id, MAX(ts) as last_active FROM checkpoints GROUP BY thread_id
        #   Parse ts (ISO format), compute age = (now - last_active).days
        #   Return list of {"thread_id": ..., "last_active": ..., "age_days": ...}
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT thread_id, MAX(ts) as last_active FROM checkpoints GROUP BY thread_id"
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            conn.close()
            return []
        conn.close()

        sessions = []
        now = datetime.now(timezone.utc)
        for thread_id, ts_str in rows:
            try:
                last_active = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if last_active.tzinfo is None:
                    last_active = last_active.replace(tzinfo=timezone.utc)
                age_days = (now - last_active).days
                sessions.append({"thread_id": thread_id, "last_active": ts_str, "age_days": age_days})
            except Exception:
                sessions.append({"thread_id": thread_id, "last_active": ts_str, "age_days": -1})
        return sessions


    def cleanup_old_sessions(max_age_days: int = 30) -> int:
        """Delete all checkpoints for threads older than max_age_days. Returns count deleted."""
        sessions = list_sessions_with_ages()
        old_sessions = [s for s in sessions if s["age_days"] >= max_age_days]

        if not old_sessions:
            print(f"[cleanup] No sessions older than {max_age_days} days found.")
            return 0

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        deleted = 0
        for s in old_sessions:
            # TODO: DELETE FROM checkpoints WHERE thread_id = ?
            # TODO: also delete from checkpoint_writes if table exists
            try:
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (s["thread_id"],))
                deleted += cursor.rowcount
                print(f"[cleanup] Deleted thread '{s['thread_id']}' (age: {s['age_days']} days)")
            except Exception as e:
                print(f"[cleanup] Error deleting {s['thread_id']}: {e}")
        conn.commit()
        conn.close()
        print(f"[cleanup] Total deleted: {deleted} checkpoint rows from {len(old_sessions)} threads")
        return len(old_sessions)


    # ── STEP 4: Seed Some Sessions ────────────────────────────

    def seed_sessions():
        from langchain_core.messages import HumanMessage
        users = ["alice", "bob", "carol", "dave"]
        for user in users:
            config = {"configurable": {"thread_id": f"user-{user}"}}
            graph.invoke(
                {"messages": [HumanMessage(content=f"Hello from {user}")], "user_id": user},
                config=config,
            )
        print(f"[seed] Created {len(users)} sessions")


    # ── STEP 5: Test ──────────────────────────────────────────

    if __name__ == "__main__":
        print("=" * 60)
        print("Session Cleanup Tool")
        print("=" * 60)

        # Create some sessions
        seed_sessions()

        # List all with ages
        print("\nAll sessions:")
        for s in list_sessions_with_ages():
            print(f"  {s['thread_id']}: {s['age_days']} days old (last: {s['last_active'][:19]})")

        # Cleanup old sessions (use 0 days for demo — deletes all)
        print(f"\nRunning cleanup (threshold: 0 days for demo):")
        count = cleanup_old_sessions(max_age_days=0)
        print(f"\nCleaned {count} sessions")

        # Verify
        print("\nSessions after cleanup:")
        remaining = list_sessions_with_ages()
        print(f"  Remaining: {len(remaining)}")
