# =============================================================
# LESSON 4 — ReAct Agent with Tools
# =============================================================
#
# WHAT YOU WILL LEARN:
#   - How to define custom tools with @tool decorator
#   - How to bind tools to an LLM
#   - ToolNode — the built-in node that executes tool calls
#   - The ReAct loop: Reason → Act → Observe → Reason...
#   - should_continue — routing between LLM and tools
#
# THE REACT PATTERN:
#   START
#     ↓
#   [agent]  ←──────────────────────┐
#     ↓                             │
#   (has tool calls?)               │
#     ├── YES → [tools] ────────────┘  (loop back)
#     └── NO  → END
#
# This loop continues until the LLM decides it has the answer
# and stops calling tools.
# =============================================================

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model

from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# -------------------------------------------------------------
# STEP 1 — Define Tools
# @tool turns a Python function into a LangChain tool.
# The docstring becomes the tool's description — the LLM reads it
# to decide when to use the tool.
# -------------------------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together. Use this for addition operations."""
    result = a + b
    print(f"  [tool:add] {a} + {b} = {result}")
    return result


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers. Use this for multiplication operations."""
    result = a * b
    print(f"  [tool:multiply] {a} * {b} = {result}")
    return result


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # Simulated weather data (replace with a real API in production)
    weather_db = {
        "london": "Cloudy, 15°C, light rain expected",
        "paris": "Sunny, 22°C, clear skies",
        "new york": "Partly cloudy, 18°C, mild breeze",
        "cairo": "Hot and sunny, 35°C",
        "tokyo": "Humid, 28°C, chance of thunderstorms",
    }
    city_lower = city.lower()
    result = weather_db.get(city_lower, f"No weather data available for {city}")
    print(f"  [tool:get_weather] {city} → {result}")
    return result


@tool
def search_wikipedia(query: str) -> str:
    """Search for factual information about a topic. Use for general knowledge questions."""
    # Simulated search (replace with real Wikipedia API in production)
    facts = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "langchain": "LangChain is a framework for building applications powered by language models.",
        "ai": "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    }
    for key, fact in facts.items():
        if key in query.lower():
            print(f"  [tool:search_wikipedia] Found result for '{query}'")
            return fact
    return f"No specific information found for: {query}"


@tool
def square_root(n: float) -> float:
    """Calculate the square root of a number."""
    return n ** 0.5


@tool
def convert_currency(amount: float) -> float:
    """Convert an amount from one currency to another."""
    return amount * 0.92


# Collect all tools in a list
tools = [add, multiply, get_weather, search_wikipedia, square_root, convert_currency]


# -------------------------------------------------------------
# STEP 2 — State
# -------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# -------------------------------------------------------------
# STEP 3 — LLM with tools bound
# bind_tools() tells the LLM which tools it can call.
# The LLM will return a ToolCall instead of text when it wants
# to use a tool.
# -------------------------------------------------------------

llm = ChatOllama(model=get_ollama_model(), temperature=0)
llm_with_tools = llm.bind_tools(tools)


# -------------------------------------------------------------
# STEP 4 — Agent Node
# Calls the LLM. The LLM decides: answer directly OR call a tool.
# -------------------------------------------------------------

def agent_node(state: AgentState) -> dict:
    print(f"\n[agent] Thinking... (messages: {len(state['messages'])})")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# -------------------------------------------------------------
# STEP 5 — Routing Function
# Check if the last message contains tool calls.
# If yes → go to "tools" node to execute them.
# If no  → the LLM has a final answer → go to END.
# -------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"[router] Tool calls detected: {[tc['name'] for tc in last_message.tool_calls]}")
        return "tools"
    print("[router] No tool calls → going to END")
    return "end"


# -------------------------------------------------------------
# STEP 6 — Build the ReAct Graph
# ToolNode is a built-in node that automatically executes
# any tool calls from the last message.
# -------------------------------------------------------------

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))  # built-in tool executor

graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",  # has tool calls → execute them
        "end": END,  # has final answer → done
    }
)

# After tools run, go BACK to agent (this creates the loop!)
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


# -------------------------------------------------------------
# STEP 7 — Run the Agent
# -------------------------------------------------------------

def run_agent(question: str):
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    result = graph.invoke({
        "messages": [HumanMessage(content=question)]
    })
    print(f"\nFinal Answer: {result['messages'][-1].content}")
    return result


if __name__ == "__main__":
    # Test 1: Math
    run_agent("What is 15 multiplied by 7, then add 23 to the result?")

    # Test 2: Weather
    run_agent("What's the weather like in Paris?")

    # Test 3: Knowledge
    run_agent("Tell me about LangGraph.")

    # Test 4: Multi-tool
    run_agent("What is the weather in London and what is 100 + 250?")

    # Test 5: square_root
    run_agent("What is the square_root of 144?")

  # Test 6: convert_currency
    run_agent("What is the currency conversion of 100 USD to EUR?")

# =============================================================
# EXERCISE:
#   1. Create a new tool called "square_root" that computes sqrt(n)
#   2. Add it to the tools list
#   3. Ask: "What is the square root of 144?"
#   4. BONUS: Create a "convert_currency" tool that converts
#      USD to EUR (use a fixed rate of 0.92)
# =============================================================
