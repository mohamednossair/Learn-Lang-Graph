# =============================================================
# TASK 2.4 — Two-Level Router
# =============================================================
# Goal:
#   Level 1: route by department (engineering / sales / hr)
#   Level 2: within engineering, route by task type
#            (bug / feature / review)
#   Every final node returns a specific message.
#
# State: {request: str, department: str, task_type: str, response: str}
#
# Graph shape:
#   START → classify_department
#             → sales_handler → END
#             → hr_handler    → END
#             → engineering_classifier
#                 → bug_handler     → END
#                 → feature_handler → END
#                 → review_handler  → END
# =============================================================

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# ── STEP 1: State ─────────────────────────────────────────────

class RouterState(TypedDict):
    request: str  # raw request text
    department: str  # engineering / sales / hr
    task_type: str  # bug / feature / review (only for engineering)
    response: str  # final response message


# ── STEP 2: Level-1 Classifier ───────────────────────────────
# TODO: Detect department by keywords.
# Hints:
#   engineering: "bug", "feature", "code", "deploy", "review", "crash", "build"
#   sales:       "quote", "deal", "contract", "pricing", "customer", "revenue"
#   hr:          "leave", "salary", "hire", "onboard", "payroll", "vacation"
#   default:     "engineering"

def classify_department(state: RouterState) -> dict:
    request = state["request"]
    if any(keyword in request for keyword in ["bug", "feature", "code", "deploy", "review", "crash", "build"]):
        department = "engineering"
    elif any(keyword in request for keyword in ["quote", "deal", "contract", "pricing", "customer", "revenue"]):
        department = "sales"
    elif any(keyword in request for keyword in ["leave", "salary", "hire", "onboard", "payroll", "vacation"]):
        department = "hr"
    else:
        department = "engineering"

    return {"department": department}


# ── STEP 3: Level-2 Classifier (engineering only) ─────────────
# TODO: Detect task type within engineering.
# Hints:
#   bug:     "bug", "crash", "error", "broken", "fix"
#   feature: "feature", "add", "new", "implement", "build"
#   review:  "review", "pr", "pull request", "check", "approve"
#   default: "feature"

def classify_task_type(state: RouterState) -> dict:
    request = state["request"]
    if any(keyword in request for keyword in ["bug", "crash", "error", "broken", "fix"]):
        task_type = "bug"
    elif any(keyword in request for keyword in ["feature", "add", "new", "implement", "build"]):
        task_type = "feature"
    elif any(keyword in request for keyword in ["review", "pr", "pull request", "check", "approve"]):
        task_type = "review"
    else:
        task_type = "feature"

    return {"task_type": task_type}


# ── STEP 4: Final Handler Nodes ───────────────────────────────

def sales_handler(state: RouterState) -> dict:
    return {"response": f"[SALES] Request forwarded to sales team: '{state['request'][:50]}'"}


def hr_handler(state: RouterState) -> dict:
    return {"response": f"[HR] Request forwarded to HR team: '{state['request'][:50]}'"}


def bug_handler(state: RouterState) -> dict:
    return {"response": f"[ENG/BUG] Bug ticket created for: '{state['request'][:50]}'"}


def feature_handler(state: RouterState) -> dict:
    return {"response": f"[ENG/FEATURE] Feature request logged: '{state['request'][:50]}'"}


def review_handler(state: RouterState) -> dict:
    return {"response": f"[ENG/REVIEW] Code review scheduled for: '{state['request'][:50]}'"}


# ── STEP 5: Routing Functions ─────────────────────────────────

def route_department(state: RouterState) -> Literal["sales_handler", "hr_handler", "classify_task_type"]:
    # TODO: map state["department"] → node name
    # engineering → "classify_task_type" (second level)
    if state["department"] == "sales":
        return "sales_handler"
    elif state["department"] == "hr":
        return "hr_handler"
    else:
        return "classify_task_type"



def route_task_type(state: RouterState) -> Literal["bug_handler", "feature_handler", "review_handler"]:
    # TODO: map state["task_type"] → node name
    if state["task_type"] == "bug":
        return "bug_handler"
    elif state["task_type"] == "review":
        return "review_handler"
    else:
        return "feature_handler"


# ── STEP 6: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(RouterState)

# TODO: add all nodes
graph_builder.add_node("classify_department",classify_department)
graph_builder.add_node("classify_task_type",classify_task_type)
graph_builder.add_node("sales_handler", sales_handler)
graph_builder.add_node("hr_handler", hr_handler)
graph_builder.add_node("bug_handler", bug_handler)
graph_builder.add_node("feature_handler", feature_handler)
graph_builder.add_node("review_handler", review_handler)
# TODO: START → classify_department
graph_builder.add_edge(START,"classify_department")
# TODO: conditional edges for level 1 (route_department)
graph_builder.add_conditional_edges("classify_department", route_department)
# TODO: conditional edges for level 2 (route_task_type)
graph_builder.add_conditional_edges("classify_task_type", route_task_type)
# TODO: all final nodes → END
graph_builder.add_edge("sales_handler", END)
graph_builder.add_edge("hr_handler", END)
graph_builder.add_edge("bug_handler", END)
graph_builder.add_edge("feature_handler", END)
graph_builder.add_edge("review_handler", END)

graph = graph_builder.compile()

# ── STEP 7: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "There is a crash bug in the payment module",
        "Please add a dark mode feature to the dashboard",
        "Can you review my pull request for the auth service?",
        "I need a pricing quote for 50 licenses",
        "I want to request 5 days of annual leave",
    ]

    for request in tests:
        print("\n" + "=" * 60)
        print(f"Request: {request}")
        result = graph.invoke({
            "request": request, "department": "", "task_type": "", "response": ""
        })
        print(f"Dept   : {result['department']}")
        print(f"Type   : {result['task_type']}")
        print(f"Result : {result['response']}")
