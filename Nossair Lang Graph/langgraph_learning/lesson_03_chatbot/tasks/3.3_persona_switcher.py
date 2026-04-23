# =============================================================
# TASK 3.3 — Persona Switcher
# =============================================================
# Goal:
#   User can switch the bot's persona mid-conversation:
#     "be pirate"  → responds as a pirate 🏴‍☠️
#     "be formal"  → responds formally
#     "be casual"  → responds casually
#   Persona is stored in state and applied via SystemMessage.
#
# State: {messages, persona}
# =============================================================

from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ── STEP 1: State ─────────────────────────────────────────────

class PersonaState(TypedDict):
    messages: Annotated[list, add_messages]
    persona: str   # "casual" | "formal" | "pirate"


# ── STEP 2: Persona Prompts ───────────────────────────────────

PERSONA_PROMPTS = {
    "casual": "You are a friendly, casual assistant. Use informal language, contractions, and keep it relaxed.",
    "formal": "You are a formal, professional assistant. Use complete sentences, avoid slang, and be precise.",
    "pirate": (
        "You are a swashbuckling pirate. Speak only in pirate dialect: use 'Arrr', 'matey', "
        "'ye', 'shiver me timbers'. Every response must start with 'Arrr!'."
    ),
}


# ── STEP 3: Persona Detection ─────────────────────────────────
# TODO: Check if the latest HumanMessage contains a persona switch command.
#   "be pirate"  → return "pirate"
#   "be formal"  → return "formal"
#   "be casual"  → return "casual"
#   otherwise    → return current persona unchanged

def detect_persona_switch(message: str, current_persona: str) -> str:
    pass


# ── STEP 4: Chatbot Node ──────────────────────────────────────
# TODO:
#   1. Get latest human message
#   2. Check for persona switch → update state["persona"] if changed
#   3. Build messages with SystemMessage(PERSONA_PROMPTS[persona]) prepended
#   4. Call LLM, return {"messages": [response], "persona": persona}

def persona_node(state: PersonaState) -> dict:
    pass


# ── STEP 5: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(PersonaState)

# TODO: add node, edges, compile

graph = graph_builder.compile()


# ── STEP 6: Chat Helper ───────────────────────────────────────

def chat(user_input: str, history: list, persona: str):
    history.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages": history, "persona": persona})
    return result["messages"][-1].content, result["messages"], result["persona"]


# ── STEP 7: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Persona Switcher — try: 'be pirate', 'be formal', 'be casual'")
    print("=" * 60)

    history = []
    persona = "casual"

    turns = [
        "Hello! What is recursion?",
        "be pirate",
        "Now explain recursion again",
        "be formal",
        "One more time please",
    ]

    for msg in turns:
        print(f"\n[{persona.upper()}] You: {msg}")
        reply, history, persona = chat(msg, history, persona)
        print(f"Bot ({persona}): {reply[:300]}")
