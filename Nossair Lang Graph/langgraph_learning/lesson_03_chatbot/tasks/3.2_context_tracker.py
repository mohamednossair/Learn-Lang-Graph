# =============================================================
# TASK 3.2 — Context Tracker
# =============================================================
# Goal:
#   Extend the chatbot with extra state fields:
#     topic: str       — auto-detected topic of conversation
#     turn_count: int  — how many turns have happened
#   At turn 5, automatically inject a SystemMessage asking
#   the LLM to summarize the conversation so far.
#
# State: {messages, topic, turn_count}
# =============================================================

from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ── STEP 1: State ─────────────────────────────────────────────

class TrackerState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str       # detected topic (update each turn)
    turn_count: int  # increment each turn


# ── STEP 2: LLM ───────────────────────────────────────────────

llm = ChatOllama(model="llama3.2", temperature=0.5)

SYSTEM_PROMPT = (
    "You are a senior Python tutor with 10 years of experience. "
    "Rules: (1) Always provide a working code example for every answer. "
    "(2) At the start of every response, rate the user's question "
    "on specificity from 1 (very vague) to 5 (very specific). "
    "Format: [Specificity: X/5]. (3) Keep answers concise and practical."
)


# ── STEP 3: Context Node ──────────────────────────────────────

def context_node(state: TrackerState) -> dict:
    turn_count = state.get("turn_count", 0) + 1
    
    # Detect topic from latest HumanMessage
    latest_human = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_human = msg
            break
    
    if latest_human:
        words = latest_human.content.split()[:5]
        topic = " ".join(words)
    else:
        topic = state.get("topic", "")
    
    # Build messages list
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    messages = [system_message] + state["messages"]
    
    # At turn 5, inject summary request
    if turn_count == 5:
        summary_message = SystemMessage(
            content="We are at turn 5. Please briefly summarize what we've discussed so far before answering the next question."
        )
        messages.append(summary_message)
    
    response = llm.invoke(messages)
    return {"messages": [response], "topic": topic, "turn_count": turn_count}


# ── STEP 4: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(TrackerState)


graph_builder.add_node("context", context_node)

graph_builder.add_edge(START, "context")
graph_builder.add_edge("context", END)
graph = graph_builder.compile()


# ── STEP 5: Chat Helper ───────────────────────────────────────

def chat(user_input: str, history: list, topic: str, turn_count: int):
    history.append(HumanMessage(content=user_input))
    result = graph.invoke({
        "messages": history,
        "topic": topic,
        "turn_count": turn_count,
    })
    return result["messages"][-1].content, result["messages"], result["topic"], result["turn_count"]


# ── STEP 6: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Context Tracker Bot — 6 turns, watch turn 5 summary")
    print("=" * 60)

    history = []
    topic = ""
    turn_count = 0

    questions = [
        "Tell me about Python lists",
        "How are lists different from tuples?",
        "When should I use a set instead?",
        "What about dictionaries?",
        "Can you compare all four data structures?",
        "Which one is fastest for lookups?",
    ]

    for q in questions:
        print(f"\n[Turn {turn_count + 1}] You: {q}")
        reply, history, topic, turn_count = chat(q, history, topic, turn_count)
        print(f"Topic: {topic} | Turns: {turn_count}")
        print(f"Bot: {reply}")
