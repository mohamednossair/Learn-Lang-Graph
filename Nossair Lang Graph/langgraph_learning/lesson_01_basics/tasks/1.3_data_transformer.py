# =============================================================
# TASK 1.3 — Data Transformer
# Flow: sort_node → filter_node (keep >3) → stats_node (min/max/avg)
# =============================================================

# ── STEP 1: Imports ──────────────────────────────────────────
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ── STEP 2: Define the State ─────────────────────────────────
class DataState(TypedDict):
    numbers: list[int]      # original input numbers
    sorted_numbers: list[int]   # numbers after sorting
    filtered_numbers: list[int] # numbers after filtering (>3)
    min: float              # minimum of filtered numbers
    max: float              # maximum of filtered numbers
    avg: float              # average of filtered numbers


# ── STEP 3: Define the Nodes ─────────────────────────────────

# Node 1 — sort_node
# Sorts the input numbers in ascending order.
def sort_node(state: DataState) -> dict:
    sorted_nums = sorted(state["numbers"])
    print(f"[sort_node]    sorted_numbers = {sorted_nums}")
    return {"sorted_numbers": sorted_nums}


# Node 2 — filter_node
# Keeps only numbers greater than 3.
def filter_node(state: DataState) -> dict:
    filtered = [n for n in state["sorted_numbers"] if n > 3]
    print(f"[filter_node]  filtered_numbers (>3) = {filtered}")
    return {"filtered_numbers": filtered}


# Node 3 — stats_node
# Computes min, max, and average of the filtered numbers.
def stats_node(state: DataState) -> dict:
    nums = state["filtered_numbers"]
    if not nums:
        raise ValueError("No numbers remain after filtering — cannot compute stats.")
    minimum = min(nums)
    maximum = max(nums)
    average = sum(nums) / len(nums)
    print(f"[stats_node]   min={minimum}, max={maximum}, avg={average:.2f}")
    return {"min": minimum, "max": maximum, "avg": average}


# ── STEP 4: Build the Graph ───────────────────────────────────

graph_builder = StateGraph(DataState)

graph_builder.add_node("sort_node",   sort_node)
graph_builder.add_node("filter_node", filter_node)
graph_builder.add_node("stats_node",  stats_node)

graph_builder.add_edge(START,         "sort_node")
graph_builder.add_edge("sort_node",   "filter_node")
graph_builder.add_edge("filter_node", "stats_node")
graph_builder.add_edge("stats_node",  END)


# ── STEP 5: Compile ───────────────────────────────────────────

graph = graph_builder.compile()


# ── STEP 6: Run ───────────────────────────────────────────────

if __name__ == "__main__":
    initial_state = {
        "numbers": [3, 1, 9, 2, 7],
        "sorted_numbers": [],
        "filtered_numbers": [],
        "min": 0.0,
        "max": 0.0,
        "avg": 0.0,
    }

    print("=" * 50)
    result = graph.invoke(initial_state)
    print("=" * 50)
    print("Final state:")
    for key, value in result.items():
        print(f"  {key}: {value!r}")
    print("=" * 50)
