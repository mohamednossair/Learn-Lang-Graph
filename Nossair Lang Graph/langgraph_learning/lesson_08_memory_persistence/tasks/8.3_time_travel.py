# =============================================================
# TASK 8.3 — Time-Travel Debugging
# =============================================================
# Goal:
#   Build a 5-step calculation graph.
#   After it runs, use get_state_history() to:
#     1. Print all 5 snapshots (state at each step)
#     2. Re-run from checkpoint 3 (fork execution)
#     3. Compare the two paths
#
# Key concepts: get_state_history(), checkpoint_id fork
# =============================================================

import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# ── STEP 1: State ─────────────────────────────────────────────

class CalcState(TypedDict):
    value: float
    step: int
    history: list


# ── STEP 2: Calculation Nodes ─────────────────────────────────

def step1_double(state: CalcState) -> dict:
    new_val = state["value"] * 2
    print(f"[step1] {state['value']} * 2 = {new_val}")
    return {"value": new_val, "step": 1, "history": state["history"] + [f"step1: *2 = {new_val}"]}


def step2_add_ten(state: CalcState) -> dict:
    new_val = state["value"] + 10
    print(f"[step2] {state['value']} + 10 = {new_val}")
    return {"value": new_val, "step": 2, "history": state["history"] + [f"step2: +10 = {new_val}"]}


def step3_square(state: CalcState) -> dict:
    new_val = state["value"] ** 2
    print(f"[step3] {state['value']} ^ 2 = {new_val}")
    return {"value": new_val, "step": 3, "history": state["history"] + [f"step3: ^2 = {new_val}"]}


def step4_divide(state: CalcState) -> dict:
    new_val = state["value"] / 4
    print(f"[step4] {state['value']} / 4 = {new_val}")
    return {"value": new_val, "step": 4, "history": state["history"] + [f"step4: /4 = {new_val}"]}


def step5_subtract_five(state: CalcState) -> dict:
    new_val = state["value"] - 5
    print(f"[step5] {state['value']} - 5 = {new_val}")
    return {"value": new_val, "step": 5, "history": state["history"] + [f"step5: -5 = {new_val}"]}


# ── STEP 3: Build Graph ───────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "time_travel.db")

with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    graph_builder = StateGraph(CalcState)
    for name, fn in [("step1", step1_double), ("step2", step2_add_ten),
                     ("step3", step3_square), ("step4", step4_divide),
                     ("step5", step5_subtract_five)]:
        graph_builder.add_node(name, fn)
    graph_builder.add_edge(START, "step1")
    graph_builder.add_edge("step1", "step2")
    graph_builder.add_edge("step2", "step3")
    graph_builder.add_edge("step3", "step4")
    graph_builder.add_edge("step4", "step5")
    graph_builder.add_edge("step5", END)
    graph = graph_builder.compile(checkpointer=checkpointer)

    if __name__ == "__main__":
        config = {"configurable": {"thread_id": "calc-001"}}

        # ── Run original path ─────────────────────────────────
        print("=" * 60)
        print("ORIGINAL RUN (starting value: 3)")
        print("=" * 60)
        result = graph.invoke({"value": 3.0, "step": 0, "history": []}, config=config)
        print(f"\nFinal value: {result['value']}")
        print(f"Path: {result['history']}")

        # ── Print all snapshots ───────────────────────────────
        print(f"\n{'='*60}")
        print("ALL CHECKPOINTS (get_state_history):")
        print("=" * 60)
        snapshots = list(graph.get_state_history(config))
        for i, snap in enumerate(reversed(snapshots)):
            print(f"  [{i}] step={snap.values.get('step')} | value={snap.values.get('value')} | next={snap.next}")

        # ── Fork from checkpoint 3 ────────────────────────────
        # TODO:
        #   1. Get snapshot at step 3 (after step3 ran)
        #   2. Create a fork config with that checkpoint_id
        #   3. Run from that checkpoint with a modified value
        #   4. Compare both final values

        print(f"\n{'='*60}")
        print("FORK FROM CHECKPOINT 3 (after step3):")
        print("=" * 60)

        # Find the checkpoint after step3
        fork_snapshot = None
        for snap in snapshots:
            if snap.values.get("step") == 3:
                fork_snapshot = snap
                break

        if fork_snapshot:
            # TODO: build fork_config with checkpoint_id
            # fork_config = {
            #     "configurable": {
            #         "thread_id": "calc-001-fork",
            #         "checkpoint_id": fork_snapshot.config["configurable"]["checkpoint_id"],
            #     }
            # }
            # TODO: invoke from fork with modified value (e.g. halve it)
            # fork_result = graph.invoke({"value": fork_snapshot.values["value"] / 2, ...}, config=fork_config)
            print(f"  Fork point value: {fork_snapshot.values.get('value')}")
            print(f"  TODO: implement fork invocation and compare paths")
        else:
            print("  Checkpoint 3 not found — run the graph first")
