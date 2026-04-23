# =============================================================
# TASK 3.4 — Message Trimmer (Production Pattern)
# =============================================================
# Goal:
#   Implement production-grade message trimming:
#   When messages > 10:
#     1. Take first 8 messages
#     2. Ask the LLM to summarize them into ONE SystemMessage
#     3. Keep last 2 messages verbatim
#     4. Replace history with: [summary_system_msg] + last_2
#
# This prevents unbounded context growth in production chatbots.
#
# State: {messages, summary_count}
# =============================================================

from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ── STEP 1: State ─────────────────────────────────────────────

class TrimmerState(TypedDict):
    messages: Annotated[list, add_messages]
    summary_count: int   # how many times we have summarized


# ── STEP 2: LLM ───────────────────────────────────────────────

llm = ChatOllama(model="llama3", temperature=0.3)


# ── STEP 3: Trim Helper ───────────────────────────────────────
# TODO: Implement summarize_and_trim(messages: list) -> list
#   - Takes a list of messages
#   - Formats the first 8 into text
#   - Calls LLM: "Summarize this conversation in 2-3 sentences: {text}"
#   - Returns [SystemMessage(f"[Summary of earlier conversation]: {summary}")]
#             + messages[-2:]

def summarize_and_trim(messages: list) -> list:
    pass


# ── STEP 4: Chatbot Node ──────────────────────────────────────
# TODO:
#   1. If len(messages) > 10: call summarize_and_trim, increment summary_count
#   2. Call LLM with (possibly trimmed) messages
#   3. Return {"messages": [response], "summary_count": ...}
#
# IMPORTANT: when returning trimmed messages as a replacement,
# you need to use a different approach than add_messages reducer.
# See hint below.
#
# HINT: The add_messages reducer APPENDS. To REPLACE the list,
# you need to return the full replacement wrapped so LangGraph
# treats it as a replacement. One approach: manage the list
# outside state and only push the new AI response back through
# the graph, holding the trimmed history in a local variable.
# For simplicity here: store trimmed history in state by
# returning the new list as the "messages" key — LangGraph will
# APPEND the new AI msg, but you can pre-clear by returning
# a RemoveMessage list. The simplest approach for this task:
# use a separate "history" list managed by the chat() function,
# trim there before invoking the graph.

def chatbot_node(state: TrimmerState) -> dict:
    pass


# ── STEP 5: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(TrimmerState)

# TODO: add node, edges, compile

graph = graph_builder.compile()


# ── STEP 6: Chat Helper with External Trim ────────────────────
# Manages history outside graph to allow full replacement on trim

def chat(user_input: str, history: list, summary_count: int):
    history.append(HumanMessage(content=user_input))

    # TODO: if len(history) > 10 → call summarize_and_trim and replace history

    result = graph.invoke({"messages": history, "summary_count": summary_count})
    new_msg = result["messages"][-1]
    history.append(new_msg)
    return new_msg.content, history, result["summary_count"]


# ── STEP 7: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Message Trimmer — will trim after 10 messages")
    print("=" * 60)

    history = []
    summary_count = 0

    questions = [
        "What is Python?",
        "What are lists in Python?",
        "What are tuples?",
        "What are sets?",
        "What are dictionaries?",
        "How do I sort a list?",
        "How do I filter a list?",
        "What is a lambda function?",
        "What is a generator?",
        "What is a decorator?",
        "What is the GIL?",   # turn 11 — should trigger trim
        "What is asyncio?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n[Turn {i}] You: {q}")
        reply, history, summary_count = chat(q, history, summary_count)
        print(f"History len: {len(history)} | Summaries: {summary_count}")
        print(f"Bot: {reply[:200]}")
