# =============================================================
# LESSON 5 — Multi-Agent System (Supervisor Pattern)
# =============================================================
#
# WHAT YOU WILL LEARN:
#   - How to build multiple specialized agents
#   - The Supervisor pattern — one agent that delegates to others
#   - How agents hand off tasks to each other
#   - Combining everything from Lessons 1-4
#
# ARCHITECTURE:
#
#   User Question
#        ↓
#   [SUPERVISOR] ← decides who should handle this
#        ├── "math_agent"    → handles calculations
#        ├── "weather_agent" → handles weather queries
#        ├── "qa_agent"      → handles general knowledge
#        └── "FINISH"        → done, return to user
#
# The supervisor reads the conversation and routes to the
# right specialist. Each specialist does its job and reports
# back to the supervisor, who then decides what to do next.
# =============================================================

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import json


# -------------------------------------------------------------
# STEP 1 — Tools for each specialist
# -------------------------------------------------------------

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. E.g. '15 * 7 + 23'"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        print(f"  [calc] {expression} = {result}")
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    weather_db = {
        "london": "Cloudy, 15°C", "paris": "Sunny, 22°C",
        "new york": "Partly cloudy, 18°C", "cairo": "Hot, 35°C",
        "tokyo": "Humid, 28°C",
    }
    result = weather_db.get(city.lower(), f"No data for {city}")
    print(f"  [weather] {city} → {result}")
    return result


@tool
def answer_question(question: str) -> str:
    """Answer general knowledge questions."""
    knowledge = {
        "python": "Python is a high-level language created by Guido van Rossum in 1991.",
        "langgraph": "LangGraph builds stateful multi-actor LLM applications as graphs.",
        "langchain": "LangChain is a framework for LLM-powered applications.",
        "ai": "AI is the simulation of human intelligence in machines.",
    }
    for key, fact in knowledge.items():
        if key in question.lower():
            return fact
    return f"I don't have specific info about: {question}"


# -------------------------------------------------------------
# STEP 2 — State
# -------------------------------------------------------------

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str    # which agent should act next


# -------------------------------------------------------------
# STEP 3 — LLMs
# -------------------------------------------------------------

llm = ChatOllama(model="llama3.2", temperature=0)

# Supervisor LLM — makes routing decisions
supervisor_llm = ChatOllama(model="llama3.2", temperature=0)

# Specialist LLMs with their tools bound
math_llm = llm.bind_tools([calculate])
weather_llm = llm.bind_tools([get_weather])
qa_llm = llm.bind_tools([answer_question])


# -------------------------------------------------------------
# STEP 4 — Supervisor Node
# The supervisor reads the conversation and decides:
# which agent should handle the next step, or FINISH.
# -------------------------------------------------------------

SUPERVISOR_SYSTEM = """You are a supervisor that routes tasks to specialist agents.

Available agents:
- math_agent: handles mathematical calculations and arithmetic
- weather_agent: handles weather and climate queries
- qa_agent: handles general knowledge questions and facts

Given the conversation, respond with ONLY a JSON object:
{"next": "math_agent"} or {"next": "weather_agent"} or {"next": "qa_agent"} or {"next": "FINISH"}

Use FINISH when the user's question has been fully answered.
Never add any explanation, ONLY the JSON."""


def supervisor_node(state: MultiAgentState) -> dict:
    print("\n[supervisor] Deciding who handles this...")

    messages = [SystemMessage(content=SUPERVISOR_SYSTEM)] + state["messages"]
    response = supervisor_llm.invoke(messages)

    try:
        raw = response.content.strip()
        # Extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        next_agent = parsed.get("next", "FINISH")
    except Exception:
        next_agent = "FINISH"

    print(f"[supervisor] → routing to: {next_agent}")
    return {"next_agent": next_agent}


# -------------------------------------------------------------
# STEP 5 — Specialist Agent Nodes
# Each specialist uses its own LLM+tools and runs a mini-loop
# until it has a result, then reports back.
# -------------------------------------------------------------

def run_specialist(llm_with_tools, tools_list, role_name: str, state: MultiAgentState) -> dict:
    """Generic runner for a specialist agent."""
    print(f"\n[{role_name}] Starting work...")
    specialist_tools = ToolNode(tools_list)

    # Give the specialist a system prompt
    system = SystemMessage(content=f"You are the {role_name}. Use your tools to answer the user's question.")
    messages = [system] + state["messages"]

    # Run up to 3 iterations (safety limit)
    for iteration in range(3):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If no tool calls, we have the answer
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            break

        # Execute tool calls
        tool_results = specialist_tools.invoke({"messages": messages})
        messages.extend(tool_results["messages"])

    print(f"[{role_name}] Done. Returning to supervisor.")
    # Return only the new messages (exclude the system message)
    new_messages = messages[len(state["messages"]) + 1:]
    return {"messages": new_messages}


def math_agent_node(state: MultiAgentState) -> dict:
    return run_specialist(math_llm, [calculate], "math_agent", state)


def weather_agent_node(state: MultiAgentState) -> dict:
    return run_specialist(weather_llm, [get_weather], "weather_agent", state)


def qa_agent_node(state: MultiAgentState) -> dict:
    return run_specialist(qa_llm, [answer_question], "qa_agent", state)


# -------------------------------------------------------------
# STEP 6 — Routing after Supervisor
# -------------------------------------------------------------

def route_after_supervisor(state: MultiAgentState) -> Literal["math_agent", "weather_agent", "qa_agent", "__end__"]:
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "math_agent":
        return "math_agent"
    elif next_agent == "weather_agent":
        return "weather_agent"
    elif next_agent == "qa_agent":
        return "qa_agent"
    else:
        return "__end__"


# -------------------------------------------------------------
# STEP 7 — Build the Multi-Agent Graph
# -------------------------------------------------------------

graph_builder = StateGraph(MultiAgentState)

graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("math_agent", math_agent_node)
graph_builder.add_node("weather_agent", weather_agent_node)
graph_builder.add_node("qa_agent", qa_agent_node)

graph_builder.add_edge(START, "supervisor")

graph_builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "math_agent":    "math_agent",
        "weather_agent": "weather_agent",
        "qa_agent":      "qa_agent",
        "__end__":       END,
    }
)

# All specialists report back to supervisor
graph_builder.add_edge("math_agent",    "supervisor")
graph_builder.add_edge("weather_agent", "supervisor")
graph_builder.add_edge("qa_agent",      "supervisor")

graph = graph_builder.compile()


# -------------------------------------------------------------
# STEP 8 — Run the Multi-Agent System
# -------------------------------------------------------------

def ask(question: str):
    print("\n" + "=" * 60)
    print(f"User: {question}")
    print("=" * 60)
    result = graph.invoke({
        "messages": [HumanMessage(content=question)],
        "next_agent": ""
    })
    final_answer = result["messages"][-1].content
    print(f"\nFinal Answer: {final_answer}")
    return final_answer


if __name__ == "__main__":
    ask("What is 144 divided by 12?")
    ask("What is the weather in Cairo?")
    ask("Tell me about LangChain.")


# =============================================================
# EXERCISE:
#   1. Add a new "code_agent" specialist that explains Python code
#   2. Create a @tool called "explain_code" that takes a code
#      snippet and returns a simple explanation
#   3. Add it to the supervisor's routing options
#   4. Test with: "Explain this code: [x**2 for x in range(10)]"
# =============================================================
