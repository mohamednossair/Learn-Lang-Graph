# imports
from typing import TypedDict
from langgraph.graph import StateGraph, START, END



# define the state
class CalculateState(TypedDict):
    a: float
    b: float
    result: float
    operation: str

# define the nodes
def subtract(state: CalculateState) -> dict:
    """Subtract b from a and store the result in the state."""
    return {"result": state["a"] - state["b"]}
def multiply(state: CalculateState) -> dict:
    """Multiply a by b and store the result in the state."""
    return {"result": state["a"] * state["b"]}

def divide(state: CalculateState) -> dict:
    """Divide a by b and store the result in the state."""
    if state["b"] == 0:
        raise ValueError("Cannot divide by zero")
    return {"result": state["a"] / state["b"]}

def add(state: CalculateState) -> dict:
    """Add a and b and store the result in the state."""
    return {"result": state["a"] + state["b"]}

# define the graph
graph_builder=StateGraph(CalculateState)

# register every node
graph_builder.add_node("subtract", subtract)
graph_builder.add_node("multiply", multiply)
graph_builder.add_node("divide", divide)
graph_builder.add_node("add", add)

# route to the correct operation based on the 'operation' field
def route_operation(state: CalculateState) -> str:
    return state["operation"]

graph_builder.add_conditional_edges(START, route_operation, {
    "subtract": "subtract",
    "multiply": "multiply",
    "divide": "divide",
    "add": "add",
})
graph_builder.add_edge("subtract", END)
graph_builder.add_edge("multiply", END)
graph_builder.add_edge("divide", END)
graph_builder.add_edge("add", END)

# compile the graph
graph = graph_builder.compile()

# run the graph
if __name__=="__main__":
    initial_state= {"a": 5, "b": 3, "result": 0.0, "operation": "multiply"}
    result = graph.invoke(initial_state)
    print(result)

