# =============================================================
# TASK 1.1 — Text Pipeline
# Flow: input → clean → count_words → format_output
# =============================================================

# ── STEP 1: Imports ──────────────────────────────────────────
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ── STEP 2: Define the State ─────────────────────────────────
# Four fields shared across all nodes.
# Each node will fill in its own field.

class TextState(TypedDict):
    raw: str          # original text (set at the start)
    clean: str        # lowercased + stripped text
    word_count: int   # number of words in clean text
    output: str       # final formatted string


# ── STEP 3: Define the Nodes ─────────────────────────────────

# Node 1 — input_node
# Receives raw text and just prints it.
# (In a real pipeline the raw value is already in state,
#  but we add a node to make the step explicit.)
def input_node(state: TextState) -> dict:
    print(f"[input_node]  raw = '{state['raw']}'")
    return {}   # nothing to update — raw is already in state


# Node 2 — clean_node
# Lowercase the raw text and strip leading/trailing whitespace.
def clean_node(state: TextState) -> dict:
    cleaned = state["raw"].lower().strip()
    print(f"[clean_node]  clean = '{cleaned}'")
    return {"clean": cleaned}


# Node 3 — count_words_node
# Split the clean text on whitespace and count the tokens.
def count_words_node(state: TextState) -> dict:
    count = len(state["clean"].split())
    print(f"[count_words_node]  word_count = {count}")
    return {"word_count": count}


# Node 4 — format_output_node
# Build a human-readable summary string.
def format_output_node(state: TextState) -> dict:
    output = (
        f"Text   : '{state['clean']}'\n"
        f"Words  : {state['word_count']}"
    )
    print(f"[format_output_node]\n{output}")
    return {"output": output}


# ── STEP 4: Build the Graph ───────────────────────────────────

graph_builder = StateGraph(TextState)

# Register every node
graph_builder.add_node("input_node",       input_node)
graph_builder.add_node("clean_node",       clean_node)
graph_builder.add_node("count_words_node", count_words_node)
graph_builder.add_node("format_output_node", format_output_node)

# Wire them in sequence
graph_builder.add_edge(START,              "input_node")
graph_builder.add_edge("input_node",       "clean_node")
graph_builder.add_edge("clean_node",       "count_words_node")
graph_builder.add_edge("count_words_node", "format_output_node")
graph_builder.add_edge("format_output_node", END)


# ── STEP 5: Compile ───────────────────────────────────────────

graph = graph_builder.compile()


# ── STEP 6: Run ───────────────────────────────────────────────

if __name__ == "__main__":
    initial_state = {
        "raw": "  Hello LangGraph World!  ",
        "clean": "",
        "word_count": 0,
        "output": "",
    }

    print("=" * 50)
    result = graph.invoke(initial_state)
    print("=" * 50)
    print("Final state:")
    for key, value in result.items():
        print(f"  {key}: {value!r}")
    print("=" * 50)
