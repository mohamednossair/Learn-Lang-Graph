# =============================================================
# TASK 2.3 — Validation Gate
# Flow: START → validate → route → process → format → END
#                               → error_node → END
# =============================================================
# Goal:
#   Accept a number input (expected 0–100).
#   If valid  → process_node → format_node → END
#   If invalid → error_node → END
#
# State: {raw_input: str, number: float, valid: bool,
#         processed: float, output: str}
# =============================================================

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# ── STEP 1: State ─────────────────────────────────────────────

class ValidationState(TypedDict):
    raw_input: str  # string the user typed (may not be a number)
    number: float  # parsed number (0.0 if invalid)
    valid: bool  # True if number is 0–100
    processed: float  # number after processing (e.g. scaled to 0–1)
    output: str  # final formatted string


# ── STEP 2: Validate Node ─────────────────────────────────────
# TODO:
#   - Try to parse raw_input as float
#   - Check 0 <= number <= 100
#   - Return {"number": ..., "valid": True/False}

def validate_node(state: ValidationState) -> dict:
    try:
        number = float(state["raw_input"])
        valid = 0 <= number <= 100
        return {"number": number, "valid": valid}

    except (ValueError, TypeError):
        return {"number": 0, "valid": False}


# ── STEP 3: Process Node ──────────────────────────────────────
# TODO: Scale the number to 0.0–1.0 range (divide by 100)
# Return {"processed": ...}

def process_node(state: ValidationState) -> dict:
    processed = state["number"] / 100
    return {"processed": processed}


# ── STEP 4: Format Node ───────────────────────────────────────
# TODO: Return {"output": f"Score: {state['number']}/100 → ratio: {state['processed']:.2f}"}

def format_node(state: ValidationState) -> dict:
    output = f"Score: {state['number']}/100 → ratio: {state['processed']:.2f}"
    return {"output": output}


# ── STEP 5: Error Node ────────────────────────────────────────
# TODO: Return {"output": f"ERROR: '{state['raw_input']}' is not a valid number between 0 and 100."}

def error_node(state: ValidationState) -> dict:
    output = f"ERROR: '{state['raw_input']}' is not a valid number between 0 and 100."
    return {"output": output}


# ── STEP 6: Routing Function ──────────────────────────────────

def route_valid(state: ValidationState) -> Literal["process_node", "error_node"]:
    # TODO: route based on state["valid"]
    if state["valid"]:
        return "process_node"
    else:
        return "error_node"


# ── STEP 7: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(ValidationState)

# TODO: add all nodes
graph_builder.add_node("validate_node", validate_node)
graph_builder.add_node("process_node", process_node)
graph_builder.add_node("format_node", format_node)
graph_builder.add_node("error_node", error_node)

# TODO: START → validate_node
graph_builder.add_edge(START, "validate_node")
# TODO: conditional edges after validate_node
graph_builder.add_conditional_edges("validate_node", route_valid)
# TODO: process_node → format_node → END
graph_builder.add_edge("process_node", "format_node")
graph_builder.add_edge("format_node", END)
# TODO: error_node → END
graph_builder.add_edge("error_node", END)

graph = graph_builder.compile()

# ── STEP 8: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = ["85", "0", "100", "-5", "200", "abc", "42.5"]

    for raw in tests:
        print("\n" + "=" * 50)
        print(f"Input: '{raw}'")
        result = graph.invoke({
            "raw_input": raw, "number": 0.0,
            "valid": False, "processed": 0.0, "output": ""
        })
        print(f"Output: {result['output']}")
