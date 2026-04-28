# =============================================================
# LESSON 1 — LangGraph Basics: StateGraph, Nodes, Edges
# =============================================================
#
# WHAT YOU WILL LEARN:
#   - What a StateGraph is
#   - How to define shared state with TypedDict
#   - How to write node functions
#   - How to connect nodes with edges
#   - How to compile and run the graph
#
# CORE MENTAL MODEL:
#   State → Node A → Node B → END
#   Each node receives the full state and returns a dict
#   with only the keys it wants to update.
# =============================================================

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# -------------------------------------------------------------
# STEP 1 — Define the State
# The state is shared memory between all nodes.
# Every node can read from it and write to it.
# -------------------------------------------------------------

class MyState(TypedDict):
    message: str       # input message
    processed: str     # output from node_a
    final: str         # output from node_b


# -------------------------------------------------------------
# STEP 2 — Define Nodes
# A node is just a Python function.
# It receives the current state and returns a dict with updates.
# You ONLY return the keys you want to change.
# -------------------------------------------------------------

def node_a(state: MyState) -> dict:
    print(f"[node_a] Received: {state['message']}")
    processed = state["message"].upper()
    return {"processed": processed}


def node_b(state: MyState) -> dict:
    print(f"[node_b] Processed text: {state['processed']}")
    final = f"Result: {state['processed']}!!!"
    return {"final": final}


# -------------------------------------------------------------
# STEP 3 — Build the Graph
# StateGraph takes the state type as its schema.
# -------------------------------------------------------------

graph_builder = StateGraph(MyState)

# Add nodes — give each node a name and the function
graph_builder.add_node("node_a", node_a)
graph_builder.add_node("node_b", node_b)

# Add edges — define the flow
# START  → node_a  (entry point)
# node_a → node_b
# node_b → END     (exit point)
graph_builder.add_edge(START, "node_a")
graph_builder.add_edge("node_a", "node_b")
graph_builder.add_edge("node_b", END)


# -------------------------------------------------------------
# STEP 4 — Compile the Graph
# compile() validates and prepares the graph for execution.
# -------------------------------------------------------------

graph = graph_builder.compile()


# -------------------------------------------------------------
# STEP 5 — Run the Graph
# invoke() takes the initial state and runs the full graph.
# Returns the final state after all nodes have executed.
# -------------------------------------------------------------

if __name__ == "__main__":
    initial_state = {"message": "hello langgraph", "processed": "", "final": ""}

    print("=" * 50)
    print("Running the graph...")
    print("=" * 50)

    result = graph.invoke(initial_state)

    print("=" * 50)
    print("Final state:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("=" * 50)


# =============================================================
# EXERCISE:
#   1. Add a third node called "node_c" that reverses the
#      string in state["final"]
#   2. Connect node_b → node_c → END
#   3. Run and verify the result
# =============================================================
