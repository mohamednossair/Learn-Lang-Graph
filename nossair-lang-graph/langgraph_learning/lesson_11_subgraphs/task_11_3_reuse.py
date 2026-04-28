"""Task 11.3 — Reuse Across Projects (import validation_subgraph into Lesson 4)."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import validation_subgraph from lesson_11_subgraphs
from lesson_11_subgraphs.lesson_11_subgraphs import validation_subgraph

llm = ChatOllama(model=get_ollama_model(), temperature=0)

# ============== REUSED VALIDATION IN LESSON 4 STYLE AGENT ==============
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    content: str
    is_valid: bool
    validation_errors: list
    response: str

def validate_input(state: AgentState) -> dict:
    """Use imported validation_subgraph to validate user input."""
    last_msg = state["messages"][-1].content if state["messages"] else ""
    result = validation_subgraph.invoke({
        "content": last_msg,
        "is_valid": True,
        "validation_errors": []
    })
    return {
        "content": last_msg,
        "is_valid": result["is_valid"],
        "validation_errors": result["validation_errors"]
    }

def route_validation(state: AgentState) -> str:
    return "agent" if state.get("is_valid", False) else "reject"

def agent_node(state: AgentState) -> dict:
    """Main agent that processes validated input."""
    resp = llm.invoke([
        HumanMessage(content=f"Respond to: {state['content']}")
    ])
    return {"response": resp.content, "messages": [resp]}

def reject_node(state: AgentState) -> dict:
    errors = state.get("validation_errors", ["Unknown error"])
    return {"response": f"Input rejected: {'; '.join(errors)}", "messages": []}

# Build graph with validation step
builder = StateGraph(AgentState)
builder.add_node("validate", validate_input)
builder.add_node("agent", agent_node)
builder.add_node("reject", reject_node)
builder.add_edge(START, "validate")
builder.add_conditional_edges("validate", route_validation, {"agent": "agent", "reject": "reject"})
builder.add_edge("agent", END)
builder.add_edge("reject", END)
validation_agent = builder.compile()

if __name__ == "__main__":
    print("=" * 60)
    print("REUSING VALIDATION_SUBGRAPH IN LESSON 4 STYLE AGENT")
    print("=" * 60)
    
    test_inputs = [
        "Tell me about Python programming and its benefits for developers.",
        "Hi",
        "This is spam content with misleading information.",
    ]
    
    for inp in test_inputs:
        print(f"\nInput: {inp}")
        result = validation_agent.invoke({
            "messages": [HumanMessage(content=inp)],
            "content": "", "is_valid": False,
            "validation_errors": [], "response": ""
        })
        print(f"Response: {result['response'][:100]}...")
        print(f"Valid: {result['is_valid']}")
    
    print("\n" + "=" * 60)
    print("Subgraph successfully reused across projects!")
    print("=" * 60)
