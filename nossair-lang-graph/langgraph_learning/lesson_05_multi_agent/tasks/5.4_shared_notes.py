# =============================================================
# TASK 5.4 — Shared Notes Multi-Agent
# =============================================================
# Goal:
#   Add shared_notes: list (with append reducer) to state.
#   Each specialist appends their findings to shared_notes.
#   Supervisor reads shared_notes when deciding next routing.
#
# Agents: market_analyst, tech_analyst, risk_analyst
# Task: analyze a business proposal end-to-end
#
# Key concept: shared_notes as a cross-agent communication channel
# =============================================================

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_ollama_model


# ── STEP 1: State ─────────────────────────────────────────────

def append_notes(existing: list, new: list) -> list:
    """Reducer: append new notes to shared_notes list."""
    return (existing or []) + (new or [])


PIPELINE = ["market_analyst", "tech_analyst", "risk_analyst", "summarizer", "finish"]


class AnalysisState(TypedDict):
    messages: Annotated[list, add_messages]
    step: int                                    # current pipeline index (guardrail)
    next: str                                    # supervisor routing
    proposal: str                                # business proposal text
    shared_notes: Annotated[list, append_notes]  # cross-agent findings
    final_report: str                            # supervisor's final summary


# ── STEP 2: LLM ───────────────────────────────────────────────

llm = ChatOllama(model=get_ollama_model(), temperature=0.3)


# ── STEP 3: Supervisor Node (LLM with guardrails) ─────────────
# Supervisor suggests next step, but step index enforces pipeline order

def supervisor_node(state: AnalysisState) -> dict:
    current_step = state.get("step", 0)
    notes_summary = "\n".join([f"- {n}" for n in state["shared_notes"]]) or "None yet"

    prompt = [
        SystemMessage(content=f"""You are an analysis supervisor.
Pipeline: market_analyst → tech_analyst → risk_analyst → summarizer → FINISH
Current step index: {current_step}

Shared notes so far:
{notes_summary}

Suggest the next step. Respond with ONLY one word:
market_analyst, tech_analyst, risk_analyst, summarizer, or FINISH"""),
        HumanMessage(content=f"Proposal: {state['proposal']}"),
    ]
    response = llm.invoke(prompt)
    suggested = response.content.strip().lower()

    # Guardrail: enforce pipeline order using step index
    expected = PIPELINE[min(current_step, len(PIPELINE) - 1)]
    valid = {"market_analyst", "tech_analyst", "risk_analyst", "summarizer", "finish"}

    if suggested not in valid:
        next_step = expected
        print(f"[supervisor] invalid '{suggested}' → enforced {expected}")
    elif PIPELINE.index(suggested) < current_step:
        # LLM tried to go backwards - enforce forward progress
        next_step = expected
        print(f"[supervisor] backwards '{suggested}' → enforced {expected}")
    else:
        next_step = suggested
        print(f"[supervisor] step {current_step} → {next_step}")

    return {"next": next_step, "step": current_step + 1}


# ── STEP 4: Specialist Nodes ──────────────────────────────────

def market_analyst(state: AnalysisState) -> dict:
    prompt = [
        SystemMessage(content="You are a market analyst. Analyze market opportunity in 2 sentences."),
        HumanMessage(content=f"Proposal: {state['proposal']}"),
    ]
    response = llm.invoke(prompt)
    note = f"[MARKET] {response.content[:150]}"
    print(f"[market_analyst] Note added")
    return {"shared_notes": [note], "messages": [response]}


def tech_analyst(state: AnalysisState) -> dict:
    prompt = [
        SystemMessage(content="You are a tech analyst. Assess technical feasibility in 2 sentences."),
        HumanMessage(content=f"Proposal: {state['proposal']}\n\nPrevious notes:\n" + "\n".join(state["shared_notes"])),
    ]
    response = llm.invoke(prompt)
    note = f"[TECH] {response.content[:150]}"
    print(f"[tech_analyst] Note added")
    return {"shared_notes": [note], "messages": [response]}


def risk_analyst(state: AnalysisState) -> dict:
    prompt = [
        SystemMessage(content="You are a risk analyst. Identify the top 2 risks in 2 sentences."),
        HumanMessage(content=f"Proposal: {state['proposal']}\n\nPrevious notes:\n" + "\n".join(state["shared_notes"])),
    ]
    response = llm.invoke(prompt)
    note = f"[RISK] {response.content[:150]}"
    print(f"[risk_analyst] Note added")
    return {"shared_notes": [note], "messages": [response]}


def summarizer(state: AnalysisState) -> dict:
    all_notes = "\n".join(state["shared_notes"])
    prompt = [
        SystemMessage(content="You are a senior business analyst. Write a concise final report combining all findings below."),
        HumanMessage(content=f"Notes:\n{all_notes}"),
    ]
    response = llm.invoke(prompt)
    print(f"[summarizer] Final report done")
    return {"final_report": response.content, "messages": [response]}


# ── STEP 5: Routing Function ──────────────────────────────────

def route_supervisor(state: AnalysisState) -> Literal["market_analyst", "tech_analyst", "risk_analyst", "summarizer", "__end__"]:
    mapping = {
        "market_analyst": "market_analyst",
        "tech_analyst": "tech_analyst",
        "risk_analyst": "risk_analyst",
        "summarizer": "summarizer",
        "finish": "__end__",
    }
    return mapping.get(state["next"], "__end__")


# ── STEP 6: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(AnalysisState)

graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("market_analyst", market_analyst)
graph_builder.add_node("tech_analyst", tech_analyst)
graph_builder.add_node("risk_analyst", risk_analyst)
graph_builder.add_node("summarizer", summarizer)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges("supervisor", route_supervisor, {
    "market_analyst": "market_analyst",
    "tech_analyst": "tech_analyst",
    "risk_analyst": "risk_analyst",
    "summarizer": "summarizer",
    "__end__": END,
})

graph_builder.add_edge("market_analyst", "supervisor")
graph_builder.add_edge("tech_analyst", "supervisor")
graph_builder.add_edge("risk_analyst", "supervisor")
graph_builder.add_edge("summarizer", "supervisor")

graph = graph_builder.compile()


# ── STEP 7: Test ──────────────────────────────────────────────

PROPOSAL = """
We want to build an AI-powered recruitment platform that uses LLMs to screen
CVs, conduct initial interviews via chatbot, and rank candidates. Target market
is mid-size tech companies (50-500 employees). Revenue model: SaaS at $500/month.
We need $200k seed funding. Tech stack: Python, LangGraph, AWS Bedrock.
"""

if __name__ == "__main__":
    print("=" * 65)
    print("Business Proposal Analysis — Shared Notes Multi-Agent")
    print("=" * 65)

    result = graph.invoke({
        "messages": [HumanMessage(content="Analyze this business proposal")],
        "step": 0,
        "next": "",
        "proposal": PROPOSAL,
        "shared_notes": [],
        "final_report": "",
    })

    print(f"\n{'='*65}")
    print("SHARED NOTES COLLECTED:")
    for note in result["shared_notes"]:
        print(f"  {note[:100]}")

    print(f"\n{'='*65}")
    print("FINAL REPORT:")
    print(result.get("final_report", "(not completed — implement summarizer node)"))
