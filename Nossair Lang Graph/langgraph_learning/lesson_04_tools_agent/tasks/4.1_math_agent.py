# =============================================================
# TASK 4.1 — Math Agent
# =============================================================
# Goal:
#   Build a ReAct agent with 6 math tools:
#     add, subtract, multiply, divide, power, square_root
#   The agent must handle multi-step calculations.
#
# Test: "What is sqrt((144 + 25) * 2)?"
# Expected reasoning: 144+25=169 → 169*2=338 → sqrt(338)≈18.39
# =============================================================

import math
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_ollama_model


# ── STEP 1: Define Math Tools ─────────────────────────────────

@tool
def add(a: float, b: float) -> float:
    """Add two numbers. Use for addition."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a. Use for subtraction."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers. Use for multiplication."""
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b. Returns error string if b is zero."""
    if b == 0:
        return "ERROR: Cannot divide by zero"
    return str(a / b)


# TODO: implement power tool
# @tool
# def power(base: float, exponent: float) -> float:
#     """Raise base to the power of exponent."""
#     pass

# TODO: implement square_root tool
# @tool
# def square_root(n: float) -> str:
#     """Compute square root of n. Returns error string if n < 0."""
#     pass


# TODO: add all 6 tools to this list
tools = [add, subtract, multiply, divide]


# ── STEP 2: State ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── STEP 3: LLM + Agent Node ──────────────────────────────────

llm = ChatOllama(model=get_ollama_model(), temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    print(f"\n[agent] thinking... ({len(state['messages'])} messages)")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ── STEP 4: Routing ───────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ── STEP 5: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


# ── STEP 6: Test ──────────────────────────────────────────────

def run(question: str):
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    print(f"\nAnswer: {result['messages'][-1].content}")


if __name__ == "__main__":
    run("What is sqrt((144 + 25) * 2)?")
    run("What is 2 to the power of 10?")
    run("Divide 100 by 0")
    run("What is (15 * 4) - (sqrt(81) + 10)?")
