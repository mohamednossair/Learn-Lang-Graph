# =============================================================
# TASK 9.3 — Parallel vs Sequential Benchmark
# =============================================================
# Goal:
#   Process 10 items two ways:
#     Sequential: loop processes items one-by-one in a single node
#     Parallel:   Send() fans out to 10 worker nodes simultaneously
#   Measure and print the time difference.
#
# Key concept: Send() for true parallel node execution
# =============================================================

import time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send


# ── Simulated work (CPU sleep to simulate processing time) ────

WORK_SECONDS = 0.3   # each item takes 0.3s


def process_item(item: str) -> str:
    """Simulate processing one item."""
    time.sleep(WORK_SECONDS)
    return f"processed:{item.upper()}"


# ════════════════════════════════════════════════════════════════
# APPROACH 1 — SEQUENTIAL
# ════════════════════════════════════════════════════════════════

class SeqState(TypedDict):
    items: list
    results: list


def sequential_processor(state: SeqState) -> dict:
    """Process all items one-by-one in a single node."""
    results = []
    for item in state["items"]:
        results.append(process_item(item))
    return {"results": results}


seq_builder = StateGraph(SeqState)
seq_builder.add_node("process", sequential_processor)
seq_builder.add_edge(START, "process")
seq_builder.add_edge("process", END)
seq_graph = seq_builder.compile()


# ════════════════════════════════════════════════════════════════
# APPROACH 2 — PARALLEL with Send()
# ════════════════════════════════════════════════════════════════

def merge_results(existing: list, new: list) -> list:
    return existing + new


class ParState(TypedDict):
    items: list
    results: Annotated[list, merge_results]


class WorkerState(TypedDict):
    item: str
    results: Annotated[list, merge_results]


def worker_node(state: WorkerState) -> dict:
    """Process a single item — runs in parallel."""
    result = process_item(state["item"])
    return {"results": [result]}


# TODO: implement fan_out function
# def fan_out(state: ParState) -> list:
#     """Return a Send() for each item to run workers in parallel."""
#     return [Send("worker", {"item": item, "results": []}) for item in state["items"]]

def fan_out(state: ParState) -> list:
    # TODO: return list of Send() calls
    pass


par_builder = StateGraph(ParState)
par_builder.add_node("worker", worker_node)
# TODO: add conditional edges from START using fan_out
# par_builder.add_conditional_edges(START, fan_out, ["worker"])
par_builder.add_edge("worker", END)
par_graph = par_builder.compile()


# ════════════════════════════════════════════════════════════════
# BENCHMARK
# ════════════════════════════════════════════════════════════════

def benchmark(items: list):
    print("=" * 60)
    print(f"Benchmarking {len(items)} items ({WORK_SECONDS}s each)")
    print(f"Sequential expected: ~{len(items) * WORK_SECONDS:.1f}s")
    print(f"Parallel expected  : ~{WORK_SECONDS:.1f}s (all at once)")
    print("=" * 60)

    # Sequential
    t0 = time.perf_counter()
    seq_result = seq_graph.invoke({"items": items, "results": []})
    seq_time = time.perf_counter() - t0
    print(f"\nSequential: {seq_time:.2f}s | Results: {len(seq_result['results'])} items")

    # Parallel
    t0 = time.perf_counter()
    par_result = par_graph.invoke({"items": items, "results": []})
    par_time = time.perf_counter() - t0
    print(f"Parallel  : {par_time:.2f}s | Results: {len(par_result['results'])} items")

    if par_time > 0:
        speedup = seq_time / par_time
        print(f"\nSpeedup: {speedup:.1f}x faster with Send()")
    else:
        print("\nTODO: implement fan_out() to see parallel speedup")


if __name__ == "__main__":
    items = [f"item_{i}" for i in range(10)]
    benchmark(items)
