# =============================================================
# TASK 8.2 — Multi-User Profile System
# =============================================================
# Goal:
#   Each user has their own thread_id and profile stored in state.
#   Chatbot stores name, language, preferences across turns.
#   Implement list_all_sessions() reading all threads from checkpointer.
#
# State: {messages, profile: dict}
# =============================================================

import os
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_ollama_model


# ── STEP 1: State ─────────────────────────────────────────────

def merge_profile(existing: dict, new: dict) -> dict:
    """Reducer: merge profile updates into existing profile."""
    merged = dict(existing)
    merged.update(new)
    return merged


class ProfileState(TypedDict):
    messages: Annotated[list, add_messages]
    profile: Annotated[dict, merge_profile]   # {name, language, preferences: []}


# ── STEP 2: LLM ───────────────────────────────────────────────

llm = ChatOllama(model=get_ollama_model(), temperature=0.3)


# ── STEP 3: Profile Extractor Node ───────────────────────────
# TODO:
#   1. Build a system prompt that shows the current profile
#   2. Ask LLM to detect if the user shared: name, language, or preferences
#   3. Update profile dict with any new information found
#   4. Generate a reply using the updated profile

def profile_node(state: ProfileState) -> dict:
    profile = state["profile"]
    system = SystemMessage(content=(
        f"You are a personalized assistant. Current user profile: {json.dumps(profile)}.\n"
        "If the user mentions their name, preferred language, or preferences, extract and remember them.\n"
        "Respond helpfully and use the user's name if you know it."
    ))

    response = llm.invoke([system] + state["messages"])

    # TODO: parse response or user message to extract profile updates
    # Simple approach: scan latest human message for "my name is X", "I prefer X", "I speak X"
    new_profile = {}
    last_human = state["messages"][-1].content.lower() if state["messages"] else ""

    # TODO: implement extraction logic and return profile updates
    return {"messages": [response], "profile": new_profile}


# ── STEP 4: Build Graph ───────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "profiles.db")


def list_all_sessions(checkpointer) -> list[str]:
    """Return all thread IDs from the checkpointer."""
    # TODO: query the SQLite DB to get distinct thread_ids
    # Hint: checkpointer.conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
    pass


with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    graph_builder = StateGraph(ProfileState)
    graph_builder.add_node("profile", profile_node)
    graph_builder.add_edge(START, "profile")
    graph_builder.add_edge("profile", END)
    graph = graph_builder.compile(checkpointer=checkpointer)

    def chat(user_id: str, message: str):
        config = {"configurable": {"thread_id": user_id}}
        result = graph.invoke(
            {"messages": [HumanMessage(content=message)], "profile": {}},
            config=config,
        )
        return result["messages"][-1].content, result["profile"]

    if __name__ == "__main__":
        print("=" * 60)
        print("Multi-User Profile System")
        print("=" * 60)

        conversations = [
            ("user-alice", "Hi! My name is Alice and I speak English. I love Python."),
            ("user-alice", "What are the best Python libraries for data science?"),
            ("user-bob", "Hello. I'm Bob, I prefer Arabic, and I like machine learning."),
            ("user-bob", "Can you recommend some ML resources?"),
            ("user-alice", "Thanks! What about LangGraph specifically?"),
        ]

        for user_id, message in conversations:
            print(f"\n[{user_id}] You: {message[:60]}")
            reply, profile = chat(user_id, message)
            print(f"[{user_id}] Bot: {reply[:120]}")
            print(f"[{user_id}] Profile: {profile}")

        print(f"\n{'='*60}")
        print("All active sessions:")
        sessions = list_all_sessions(checkpointer)
        for s in (sessions or ["(implement list_all_sessions to see this)"]):
            print(f"  - {s}")
