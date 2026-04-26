# =============================================================
# TASK 5.1 — Customer Support Multi-Agent System
# =============================================================
# Goal:
#   Supervisor routes tickets to 4 specialists:
#     billing   — payment, invoice, charge questions
#     technical — bugs, errors, integrations
#     returns   — refunds, exchanges, order issues
#     general   — everything else
#   Test all 4 routing paths.
#
# Pattern: Supervisor → specialist → Supervisor → FINISH → END
# =============================================================

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_ollama_model


# ── STEP 1: State ─────────────────────────────────────────────

class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str  # supervisor's routing decision
    ticket: str  # original ticket text


# ── STEP 2: LLM ───────────────────────────────────────────────

llm = ChatOllama(model=get_ollama_model(), temperature=0)

supervisor_llm = ChatOllama(model=get_ollama_model(), temperature=0)

# ── STEP 3: Supervisor Node ───────────────────────────────────
# TODO:
#   System prompt: route to billing/technical/returns/general.
#   Respond ONLY with one word: billing, technical, returns, general, or FINISH.
#   FINISH when the specialist has answered.

SUPERVISOR_PROMPT = """You are a customer support supervisor.
Route the ticket to the correct specialist:
- billing: payment issues, invoices, charges, subscriptions
- technical: bugs, errors, crashes, API issues, integrations
- returns: refunds, returns, exchanges, damaged goods, order issues
- general: anything else

If a specialist has already responded, respond with: FINISH

Respond with ONLY one word: billing, technical, returns, general, or FINISH"""


def supervisor_node(state: SupportState) -> dict:
    print("[supervisor] Deciding route...")
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    try:
        response = llm.invoke(messages)
        decision = response.content.strip().lower()
        print(f"[supervisor] Routing to: {decision}")
        return {"next": decision}
    except Exception as e:
        print(f"[supervisor] Error routing: {e}")
        return {"next": "FINISH"}


# ── STEP 4: Specialist Nodes ──────────────────────────────────

def billing_agent(state: SupportState) -> dict:
    prompt = SystemMessage(
        content="You are a billing specialist. Resolve payment, invoice, and subscription issues professionally.")
    response = llm.invoke([prompt] + state["messages"])
    print(f"[billing] Responding...")
    return {"messages": [response]}


def technical_agent(state: SupportState) -> dict:
    prompt = SystemMessage(
        content="You are a technical support specialist. Resolve bugs, errors, and integration issues with clear steps.")
    response = llm.invoke([prompt] + state["messages"])
    print(f"[technical] Responding...")
    return {"messages": [response]}


def returns_agent(state: SupportState) -> dict:
    prompt = SystemMessage(
        content="You are a returns and refunds specialist. Handle return requests, refunds, and exchanges empathetically.")
    response = llm.invoke([prompt] + state["messages"])
    print(f"[returns] Responding...")
    return {"messages": [response]}


def general_agent(state: SupportState) -> dict:
    prompt = SystemMessage(
        content="You are a general customer support agent. Be helpful, friendly, and resolve queries efficiently.")
    response = llm.invoke([prompt] + state["messages"])
    print(f"[general] Responding...")
    return {"messages": [response]}


# ── STEP 5: Routing Function ──────────────────────────────────

def route_supervisor(state: SupportState) -> Literal["billing", "technical", "returns", "general", "__end__"]:
    # TODO: map state["next"] → node name or END
    decision = state.get("next", "general").lower().strip()
    if decision == "finish":
        return "__end__"
    if decision in ["billing", "technical", "returns", "general"]:
        return decision
    return "general"  # fallback


# ── STEP 6: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(SupportState)

# TODO: add supervisor + all 4 specialist nodes
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("billing", billing_agent)
graph_builder.add_node("technical", technical_agent)
graph_builder.add_node("returns", returns_agent)
graph_builder.add_node("general", general_agent)

# TODO: START → supervisor
graph_builder.add_edge(START, "supervisor")

# TODO: conditional edges from supervisor (route_supervisor)
graph_builder.add_conditional_edges("supervisor", route_supervisor)

# TODO: each specialist → supervisor (report back)
graph_builder.add_edge("billing", "supervisor")
graph_builder.add_edge("technical", "supervisor")
graph_builder.add_edge("returns", "supervisor")
graph_builder.add_edge("general", "supervisor")

graph = graph_builder.compile()

# ── STEP 7: Test All 4 Paths ──────────────────────────────────

if __name__ == "__main__":
    tickets = [
        "I was charged twice for my subscription last month!",
        "The API keeps returning 500 errors on /users endpoint.",
        "I want to return the laptop I ordered — it arrived damaged.",
        "What are your business hours?",
    ]

    for ticket in tickets:
        print("\n" + "=" * 65)
        print(f"Ticket: {ticket}")
        print("=" * 65)
        result = graph.invoke({
            "messages": [HumanMessage(content=ticket)],
            "next": "",
            "ticket": ticket,
        })
        print(f"\nFinal Response:\n{result['messages'][-1].content[:300]}")
