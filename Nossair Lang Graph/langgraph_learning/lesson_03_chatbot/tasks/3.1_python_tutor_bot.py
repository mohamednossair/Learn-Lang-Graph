# =============================================================
# TASK 3.1 — Python Tutor Bot
# =============================================================
# Goal:
#   Build a chatbot that acts as a senior Python tutor.
#   System prompt: "You are a senior Python tutor. Always give
#   code examples. Rate user questions 1-5 on specificity."
#
# State: {messages: Annotated[list, add_messages]}
# Uses: ChatOllama (llama3), MessagesState pattern
# =============================================================

from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ── STEP 1: State ─────────────────────────────────────────────

class TutorState(TypedDict):
    messages: Annotated[list, add_messages]


# ── STEP 2: LLM Setup ─────────────────────────────────────────

llm = ChatOllama(model="llama3", temperature=0.5)

SYSTEM_PROMPT = (
    "You are a senior Python tutor with 10 years of experience. "
    "Rules: (1) Always provide a working code example for every answer. "
    "(2) At the start of every response, rate the user's question "
    "on specificity from 1 (very vague) to 5 (very specific). "
    "Format: [Specificity: X/5]. (3) Keep answers concise and practical."
)


# ── STEP 3: Tutor Node ────────────────────────────────────────
# TODO:
#   - Prepend a SystemMessage with SYSTEM_PROMPT to the messages
#   - Call the LLM with the full message list
#   - Return {"messages": [response]}

def tutor_node(state: TutorState) -> dict:
    pass


# ── STEP 4: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(TutorState)

# TODO: add node, edges, compile

graph = graph_builder.compile()


# ── STEP 5: Chat Helper ───────────────────────────────────────

def chat(user_input: str, history: list) -> tuple[str, list]:
    history.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages": history})
    updated = result["messages"]
    return updated[-1].content, updated


# ── STEP 6: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Python Tutor Bot — type 'quit' to exit")
    print("=" * 60)

    history = []
    test_questions = [
        "what is python",
        "how do I use list comprehensions to filter even numbers?",
        "explain decorators with a real example of memoization",
    ]

    for q in test_questions:
        print(f"\nYou: {q}")
        reply, history = chat(q, history)
        print(f"Tutor: {reply[:400]}")
        print("-" * 60)
