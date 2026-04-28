# =============================================================
# TASK 7.1 — Budget Approval Workflow
# =============================================================
# Goal:
#   Purchasing agent proposes item + cost.
#   If cost > $500: interrupt for human approval.
#   If cost <= $500: auto-approve.
#   Track all purchases in purchase_log state field.
#
# Key concepts: interrupt(), Command(resume=), MemorySaver
# =============================================================

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ── STEP 1: State ─────────────────────────────────────────────

def append_log(existing: list, new: list) -> list:
    return existing + new


class PurchaseState(TypedDict):
    messages: Annotated[list, add_messages]
    item: str
    cost: float
    approved: bool
    purchase_log: Annotated[list, append_log]


# ── STEP 2: Propose Node ──────────────────────────────────────
# Simulates the purchasing agent proposing an item

def propose_purchase(state: PurchaseState) -> dict:
    print(f"[propose] Item: '{state['item']}' | Cost: ${state['cost']:.2f}")
    return {}


# ── STEP 3: Approval Gate Node ────────────────────────────────
# TODO:
#   If cost <= 500: auto-approve, log it, return
#   If cost > 500: call interrupt() showing item + cost to human
#     - If human returns "approve": approve + log
#     - If human returns "reject": reject + log

def approval_gate(state: PurchaseState) -> dict:
    if state["cost"] <= 500:
        log_entry = f"AUTO-APPROVED: {state['item']} (${state['cost']:.2f})"
        print(f"[gate] {log_entry}")
        return {"approved": True, "purchase_log": [log_entry]}

    # TODO: interrupt for human review when cost > $500
    # human_decision = interrupt({...})
    # handle "approve" / "reject" response
    pass


# ── STEP 4: Build Graph ───────────────────────────────────────

checkpointer = MemorySaver()

graph_builder = StateGraph(PurchaseState)
graph_builder.add_node("propose", propose_purchase)
graph_builder.add_node("gate", approval_gate)
graph_builder.add_edge(START, "propose")
graph_builder.add_edge("propose", "gate")
graph_builder.add_edge("gate", END)

graph = graph_builder.compile(checkpointer=checkpointer)


# ── STEP 5: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    purchases = [
        {"item": "Notebook pack", "cost": 25.00},
        {"item": "Standing desk", "cost": 799.99},
        {"item": "USB hub", "cost": 45.00},
        {"item": "Ergonomic chair", "cost": 650.00},
    ]

    for i, p in enumerate(purchases):
        config = {"configurable": {"thread_id": f"purchase-{i}"}}
        print(f"\n{'='*60}")
        print(f"Purchasing: {p['item']} (${p['cost']:.2f})")

        state = graph.invoke(
            {"item": p["item"], "cost": p["cost"], "approved": False, "purchase_log": [], "messages": []},
            config=config,
        )

        # Check if paused at interrupt
        current = graph.get_state(config)
        if current.next:
            print(f"  ⚠ APPROVAL REQUIRED for ${p['cost']:.2f}")
            print(f"  Interrupt data: {current.tasks[0].interrupts[0].value}")
            # Simulate human decision
            human_decision = "approve" if p["cost"] < 1000 else "reject"
            print(f"  Human decision: {human_decision}")
            state = graph.invoke(Command(resume=human_decision), config=config)

        print(f"  Log: {state.get('purchase_log', [])}")
