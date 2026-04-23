# =============================================================
# TASK 7.3 — Multi-Step Approval Chain (Loan Application)
# =============================================================
# Goal:
#   Loan application goes through a 2-interrupt approval chain:
#     initial_review → [interrupt: loan officer] →
#     credit_check   → [interrupt: senior officer] →
#     decision
#   Each interrupt shows the appropriate data for that stage.
#
# Key concept: multiple sequential interrupts in one graph run
# =============================================================

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ── STEP 1: State ─────────────────────────────────────────────

class LoanState(TypedDict):
    messages: Annotated[list, add_messages]
    applicant_name: str
    loan_amount: float
    annual_income: float
    credit_score: int        # filled by credit_check
    loan_officer_decision: str   # "proceed" or "reject"
    senior_decision: str         # "approve" or "reject"
    final_status: str


# ── STEP 2: Initial Review Node ───────────────────────────────
# Compute debt-to-income ratio, flag risk, then interrupt for loan officer

def initial_review(state: LoanState) -> dict:
    ratio = (state["loan_amount"] / state["annual_income"]) * 100
    risk = "HIGH" if ratio > 40 else "MEDIUM" if ratio > 25 else "LOW"
    print(f"[initial_review] DTI ratio: {ratio:.1f}% | Risk: {risk}")

    # TODO: interrupt showing applicant name, amount, income, ratio, risk
    # loan_officer_decision = interrupt({...})
    # return {"loan_officer_decision": loan_officer_decision}
    pass


# ── STEP 3: Credit Check Node ─────────────────────────────────
# Simulate credit score lookup, then interrupt for senior officer

def credit_check(state: LoanState) -> dict:
    if state["loan_officer_decision"] != "proceed":
        return {"credit_score": 0, "final_status": "REJECTED by loan officer"}

    # Simulate credit score based on income (mock)
    score = min(850, int(state["annual_income"] / 100) + 500)
    print(f"[credit_check] Credit score: {score}")

    # TODO: interrupt showing full application + credit score for senior officer
    # senior_decision = interrupt({...})
    # return {"credit_score": score, "senior_decision": senior_decision}
    pass


# ── STEP 4: Decision Node ─────────────────────────────────────

def make_decision(state: LoanState) -> dict:
    if state.get("final_status"):
        return {}
    if state["senior_decision"] == "approve":
        status = f"APPROVED — ${state['loan_amount']:,.0f} loan for {state['applicant_name']}"
    else:
        status = f"REJECTED by senior officer — {state['applicant_name']}"
    print(f"[decision] {status}")
    return {"final_status": status}


# ── STEP 5: Routing ───────────────────────────────────────────

def route_after_initial(state: LoanState) -> str:
    if state.get("final_status"):
        return "done"
    return "credit_check"


# ── STEP 6: Build Graph ───────────────────────────────────────

checkpointer = MemorySaver()

graph_builder = StateGraph(LoanState)
graph_builder.add_node("initial_review", initial_review)
graph_builder.add_node("credit_check", credit_check)
graph_builder.add_node("decision", make_decision)

graph_builder.add_edge(START, "initial_review")
graph_builder.add_conditional_edges("initial_review", route_after_initial, {
    "credit_check": "credit_check",
    "done": END,
})
graph_builder.add_edge("credit_check", "decision")
graph_builder.add_edge("decision", END)

graph = graph_builder.compile(checkpointer=checkpointer)


# ── STEP 7: Test ──────────────────────────────────────────────

def process_loan(applicant: dict):
    config = {"configurable": {"thread_id": f"loan-{applicant['name'].replace(' ', '-')}"}}
    print(f"\n{'='*65}")
    print(f"Application: {applicant['name']} | ${applicant['amount']:,} | Income: ${applicant['income']:,}")

    initial_state = {
        "messages": [],
        "applicant_name": applicant["name"],
        "loan_amount": float(applicant["amount"]),
        "annual_income": float(applicant["income"]),
        "credit_score": 0,
        "loan_officer_decision": "",
        "senior_decision": "",
        "final_status": "",
    }

    graph.invoke(initial_state, config=config)

    # Handle interrupt 1: loan officer
    state = graph.get_state(config)
    if state.next:
        data = state.tasks[0].interrupts[0].value
        print(f"  [Loan Officer] Review: {data}")
        lo_decision = "proceed" if applicant["amount"] < 200000 else "reject"
        print(f"  Loan Officer Decision: {lo_decision}")
        graph.invoke(Command(resume=lo_decision), config=config)

    # Handle interrupt 2: senior officer
    state = graph.get_state(config)
    if state.next:
        data = state.tasks[0].interrupts[0].value
        print(f"  [Senior Officer] Review: {data}")
        senior_decision = "approve" if applicant.get("income", 0) > 60000 else "reject"
        print(f"  Senior Decision: {senior_decision}")
        graph.invoke(Command(resume=senior_decision), config=config)

    final = graph.get_state(config).values
    print(f"  FINAL STATUS: {final.get('final_status', 'unknown')}")


if __name__ == "__main__":
    applications = [
        {"name": "Alice Johnson", "amount": 50000, "income": 80000},
        {"name": "Bob Smith", "amount": 300000, "income": 45000},
        {"name": "Carol White", "amount": 120000, "income": 95000},
    ]
    for app in applications:
        process_loan(app)
