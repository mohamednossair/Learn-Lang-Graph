# =============================================================
# TASK 2.1 — Support Ticket Router
# Flow: START → classify_priority → route → handler → END
# =============================================================
# Goal:
#   Classify a support ticket as high/medium/low priority.
#   Route to the appropriate handler node:
#     high   → senior_engineer
#     medium → engineer
#     low    → intern
#
# State: {ticket: str, priority: str, assigned_to: str}
# =============================================================

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# ── STEP 1: State ─────────────────────────────────────────────

class TicketState(TypedDict):
    ticket: str  # raw ticket text
    priority: str  # filled by classify_node: high/medium/low
    assigned_to: str  # filled by handler node


# ── STEP 2: Classify Node ─────────────────────────────────────
# TODO: Detect priority by keywords (or your own logic).
# Hint — high keywords: "down", "crash", "urgent", "broken"
#        medium keywords: "slow", "error", "issue", "wrong"
#        low keywords: everything else → "low"

def classify_priority(state: TicketState) -> dict:
    # TODO: implement keyword-based classification
    text = state["ticket"].lower()
    if any(w in text for w in ["down", "crash", "urgent", "broken"]):
        return {"priority": "high"}
    elif any(w in text for w in ["slow", "error", "issue", "wrong"]):
        return {"priority": "medium"}
    else:
        return {"priority": "low"}


# ── STEP 3: Handler Nodes ─────────────────────────────────────

def senior_engineer(state: TicketState) -> dict:
    # TODO: return assigned_to = "Senior Engineer: [ticket summary]"
    print(f"Senior Engineer: {state['ticket']}")

    return {"assigned_to": "Senior Engineer: " + state['ticket']}


def engineer(state: TicketState) -> dict:
    # TODO: return assigned_to = "Engineer: [ticket summary]"
    print(f"Engineer:{state['ticket']}")
    return {"assigned_to": "Engineer: " + state['ticket']}


def intern(state: TicketState) -> dict:
    # TODO: return assigned_to = "Intern: [ticket summary]"
    print(f"Intern:{state['ticket']}")
    return {"assigned_to": "Intern: " + state['ticket']}


# ── STEP 4: Routing Function ──────────────────────────────────

def route_by_priority(state: TicketState) -> Literal["senior_engineer", "engineer", "intern"]:
    # TODO: return the correct node name based on state["priority"]
    priority = state["priority"]
    if priority == "high":
        return "senior_engineer"
    elif priority == "medium":
        return "engineer"
    else:
        return "intern"


# ── STEP 5: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(TicketState)

# TODO: add nodes
graph_builder.add_node(classify_priority)
graph_builder.add_node(senior_engineer)
graph_builder.add_node(engineer)
graph_builder.add_node(intern)
# TODO: add START → classify_priority edge
graph_builder.add_edge(START, "classify_priority")
# TODO: add conditional edges after classify_priority
graph_builder.add_conditional_edges("classify_priority", route_by_priority, {
    "senior_engineer": "senior_engineer",
    "engineer": "engineer",
    "intern": "intern",
})
# TODO: add edges from each handler → END
graph_builder.add_edge("senior_engineer", END)
graph_builder.add_edge("engineer", END)
graph_builder.add_edge("intern", END)

graph = graph_builder.compile()

# ── STEP 6: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    test_tickets = [
        "The production server is DOWN and users cannot login — urgent!",
        "The dashboard loads slowly when filtering by date range.",
        "Can you update the color of the logo on the about page?",
    ]

    for ticket in test_tickets:
        print("\n" + "=" * 55)
        print(f"Ticket: {ticket[:60]}...")
        result = graph.invoke({"ticket": ticket, "priority": "", "assigned_to": ""})
        print(f"Priority : {result['priority']}")
        print(f"Assigned : {result['assigned_to']}")
