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


# ── STEP 1: State ─────────────────────────────────────────────

def append_notes(existing: list, new: list) -> list:
    """Reducer: append new notes to shared_notes list."""
    return existing + new


class AnalysisState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str                                    # supervisor routing
    proposal: str                                # business proposal text
    shared_notes: Annotated[list, append_notes]  # cross-agent findings
    final_report: str                            # supervisor's final summary


# ── STEP 2: LLM ───────────────────────────────────────────────

llm = ChatOllama(model="llama3", temperature=0.3)


# ── STEP 3: Supervisor Node ───────────────────────────────────
# TODO:
#   Supervisor reads shared_notes to know what's been done.
#   Pipeline: market_analyst → tech_analyst → risk_analyst → summarizer → FINISH
#   Respond with ONLY: market_analyst, tech_analyst, risk_analyst, summarizer, FINISH

def supervisor_node(state: AnalysisState) -> dict:
    notes_summary = "\n".join([f"- {n}" for n in state["shared_notes"]]) or "None yet"
    prompt = [
        SystemMessage(content=f"""You are an analysis supervisor.
Pipeline: market_analyst → tech_analyst → risk_analyst → summarizer → FINISH

Shared notes so far:
{notes_summary}

Decide the next step. Respond with ONLY one word:
market_analyst, tech_analyst, risk_analyst, summarizer, or FINISH"""),
        HumanMessage(content=f"Proposal: {state['proposal'][:200]}"),
    ]
    response = llm.invoke(prompt)
    next_step = response.content.strip().lower()
    valid = {"market_analyst", "tech_analyst", "risk_analyst", "summarizer", "finish"}
    if next_step not in valid:
        next_step = "market_analyst"
    print(f"[supervisor] → {next_step}")
    return {"next": next_step}


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


# TODO: implement risk_analyst node
# def risk_analyst(state: AnalysisState) -> dict:
#     # Read shared_notes for context, identify top 2 risks, append a "[RISK]" note
#     pass


# TODO: implement summarizer node
# def summarizer(state: AnalysisState) -> dict:
#     # Read all shared_notes, write final_report combining all findings
#     pass


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
# TODO: add risk_analyst and summarizer nodes

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges("supervisor", route_supervisor, {
    "market_analyst": "market_analyst",
    "tech_analyst": "tech_analyst",
    # TODO: add risk_analyst and summarizer
    "__end__": END,
})

graph_builder.add_edge("market_analyst", "supervisor")
graph_builder.add_edge("tech_analyst", "supervisor")
# TODO: add edges for risk_analyst and summarizer → supervisor

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
