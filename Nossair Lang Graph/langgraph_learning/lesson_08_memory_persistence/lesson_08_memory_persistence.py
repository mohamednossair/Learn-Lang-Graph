# =============================================================
# LESSON 8 — Persistent Memory & Checkpointers
# =============================================================
#
# WHAT YOU WILL LEARN:
#   - MemorySaver  — in-memory checkpointer (dev/testing)
#   - SqliteSaver  — disk-based checkpointer (production-like)
#   - thread_id    — identifies a conversation session
#   - Resuming a conversation after restart
#   - Cross-session memory (remembering user facts)
#   - Viewing and inspecting saved state history
#
# KEY INSIGHT:
#   Without a checkpointer → graph has NO memory between calls
#   With a checkpointer    → graph saves state after every step
#   thread_id              → separate "channels" (like separate users)
# =============================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model

import os
import sqlite3
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


# =============================================================
# PART 1 — Without Checkpointer (No Memory)
# =============================================================

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model=get_ollama_model(), temperature=0.7)


def chatbot_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_graph(checkpointer=None):
    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=checkpointer)


def demo_no_memory():
    """WITHOUT checkpointer — each call is stateless."""
    print("\n" + "="*60)
    print("PART 1 — WITHOUT Checkpointer (Stateless)")
    print("="*60)

    graph = build_graph()  # no checkpointer

    # Turn 1
    r1 = graph.invoke({"messages": [HumanMessage(content="My name is Ahmed.")]})
    print(f"Turn 1 — AI: {r1['messages'][-1].content[:100]}")

    # Turn 2 — SAME graph, but it has no memory of turn 1
    r2 = graph.invoke({"messages": [HumanMessage(content="What is my name?")]})
    print(f"Turn 2 — AI: {r2['messages'][-1].content[:100]}")
    print("⚠️  The AI doesn't remember because there's no checkpointer!")


# =============================================================
# PART 2 — MemorySaver (In-Memory, Lost on Restart)
# =============================================================

def demo_memory_saver():
    """WITH MemorySaver — state persists within a Python session."""
    print("\n" + "="*60)
    print("PART 2 — WITH MemorySaver (In-Memory Persistence)")
    print("="*60)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    # thread_id identifies this conversation session
    config = {"configurable": {"thread_id": "user-ahmed-001"}}

    # Turn 1
    r1 = graph.invoke(
        {"messages": [HumanMessage(content="My name is Ahmed and I love Python.")]},
        config=config
    )
    print(f"Turn 1 — AI: {r1['messages'][-1].content[:120]}")

    # Turn 2 — graph automatically loads previous state from checkpointer
    r2 = graph.invoke(
        {"messages": [HumanMessage(content="What is my name and what do I love?")]},
        config=config
    )
    print(f"Turn 2 — AI: {r2['messages'][-1].content[:120]}")
    print("✅ AI remembers from turn 1!")

    # Different thread_id = different user = no memory of Ahmed
    config2 = {"configurable": {"thread_id": "user-bob-002"}}
    r3 = graph.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=config2
    )
    print(f"\nDifferent thread — AI: {r3['messages'][-1].content[:100]}")
    print("✅ Different user has separate, isolated memory!")


# =============================================================
# PART 3 — SqliteSaver (Persistent Across Restarts)
# =============================================================

CHECKPOINT_DB = os.path.join(os.path.dirname(__file__), "checkpoints.db")


def demo_sqlite_saver():
    """WITH SqliteSaver — state persists across Python restarts."""
    print("\n" + "="*60)
    print("PART 3 — WITH SqliteSaver (Disk Persistence)")
    print("="*60)

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "persistent-session-1"}}

        # Check if this session has history already
        history = list(graph.get_state_history(config))
        if history:
            print(f"Found existing session with {len(history)} checkpoints!")
            print("Continuing from where we left off...")
        else:
            print("New session — starting fresh")
            graph.invoke(
                {"messages": [HumanMessage(content="Remember: My favorite color is blue and I work as a data engineer.")]},
                config=config
            )

        # This turn will remember the previous context even after restart
        result = graph.invoke(
            {"messages": [HumanMessage(content="What do you remember about me?")]},
            config=config
        )
        print(f"AI: {result['messages'][-1].content[:200]}")
        print(f"\n✅ Data saved to: {CHECKPOINT_DB}")
        print("   Restart Python and run this again — memory persists!")


# =============================================================
# PART 4 — Inspecting State History
# =============================================================

def demo_state_history():
    """View the full history of a conversation."""
    print("\n" + "="*60)
    print("PART 4 — Inspecting State History")
    print("="*60)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "history-demo"}}

    messages_to_send = [
        "My name is Sara.",
        "I am a software engineer.",
        "What do you know about me?",
    ]

    for msg in messages_to_send:
        graph.invoke({"messages": [HumanMessage(content=msg)]}, config=config)

    # View the full state at current point
    current_state = graph.get_state(config)
    print(f"\nCurrent state has {len(current_state.values['messages'])} messages:")
    for i, msg in enumerate(current_state.values["messages"]):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{i+1}] {role}: {msg.content[:80]}")

    # View all checkpoints in history
    print(f"\nState history checkpoints:")
    for i, snapshot in enumerate(graph.get_state_history(config)):
        n_msgs = len(snapshot.values.get("messages", []))
        print(f"  Checkpoint {i}: {n_msgs} messages | next={snapshot.next}")


# =============================================================
# PART 5 — Cross-Session User Profile Memory
# =============================================================
# Pattern: Store user facts in a separate "profile" state field
# that persists and grows across conversations.

class ProfileState(TypedDict):
    messages:     Annotated[list, add_messages]
    user_profile: dict    # persistent facts about the user


def profile_node(state: ProfileState) -> dict:
    profile = state.get("user_profile", {})
    profile_str = "\n".join(f"  - {k}: {v}" for k, v in profile.items()) if profile else "  (empty)"

    system = SystemMessage(content=f"""You are a personalized assistant.
Known facts about this user:
{profile_str}

If the user tells you something about themselves, extract and remember it.
Always use what you know to personalize your responses.""")

    response = llm.invoke([system] + state["messages"])

    # Simple fact extraction (in production, use structured output)
    new_profile = dict(profile)
    content = state["messages"][-1].content.lower()
    if "my name is" in content:
        name = content.split("my name is")[1].split(".")[0].strip().title()
        new_profile["name"] = name
    if "i work" in content or "i am a" in content:
        new_profile["job"] = "software professional"
    if "i love" in content or "i like" in content:
        hobby = content.split("i love" if "i love" in content else "i like")[1].split(".")[0].strip()
        new_profile["interests"] = hobby

    return {"messages": [response], "user_profile": new_profile}


def demo_user_profile():
    print("\n" + "="*60)
    print("PART 5 — Cross-Session User Profile Memory")
    print("="*60)

    checkpointer = MemorySaver()
    builder = StateGraph(ProfileState)
    builder.add_node("chatbot", profile_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "profile-session"}}

    conversations = [
        "Hi! My name is Ahmed. I love machine learning.",
        "I work as a data engineer at a tech company.",
        "Can you suggest something I might enjoy learning next?",
    ]

    for turn in conversations:
        print(f"\nYou: {turn}")
        result = graph.invoke({"messages": [HumanMessage(content=turn)], "user_profile": {}}, config=config)
        state = graph.get_state(config)
        print(f"AI: {result['messages'][-1].content[:150]}")
        print(f"Profile: {state.values.get('user_profile', {})}")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    demo_no_memory()
    demo_memory_saver()
    demo_sqlite_saver()
    demo_state_history()
    demo_user_profile()


# =============================================================
# EXERCISE:
#   1. Build a "notebook" assistant that:
#      a. Saves user notes in a state field called "notes" (list)
#      b. Persists notes using SqliteSaver
#      c. Can answer "What notes do I have?" by reading from state
#   2. Test: add 3 notes, restart Python, ask "show my notes"
# =============================================================
