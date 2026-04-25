# =============================================================
# TASK 4.3 — Date/Time Agent
# =============================================================
# Goal:
#   Build a ReAct agent with date/time tools:
#     get_current_date, day_of_week, days_between, add_days
#
# Test: "What day of the week is it 45 days from today?"
# =============================================================

from datetime import date, timedelta
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ── STEP 1: Date/Time Tools ───────────────────────────────────

@tool
def get_current_date() -> str:
    """Get today's date in YYYY-MM-DD format."""
    today = date.today().isoformat()
    print(f"  [tool:get_current_date] → {today}")
    return today


@tool
def day_of_week(date_str: str) -> str:
    """Get the day of the week for a date in YYYY-MM-DD format. Returns the weekday name."""
    try:
        d = date.fromisoformat(date_str)
        day = d.strftime("%A")
        print(f"  [tool:day_of_week] {date_str} → {day}")
        return day
    except ValueError:
        return f"ERROR: Invalid date format '{date_str}'. Use YYYY-MM-DD."


# TODO: implement days_between tool
# @tool
# def days_between(date1: str, date2: str) -> str:
#     """Calculate the number of days between two dates (YYYY-MM-DD format).
#     Returns the absolute number of days as a string."""
#     pass


# TODO: implement add_days tool
# @tool
# def add_days(date_str: str, n: int) -> str:
#     """Add n days to a date (YYYY-MM-DD format). Returns the resulting date as YYYY-MM-DD."""
#     pass


# TODO: add all 4 tools
tools = [get_current_date, day_of_week]


# ── STEP 2: State + LLM + Agent ───────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    print(f"\n[agent] thinking... ({len(state['messages'])} messages)")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ── STEP 3: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


# ── STEP 4: Test ──────────────────────────────────────────────

def run(question: str):
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    print(f"\nAnswer: {result['messages'][-1].content}")


if __name__ == "__main__":
    run("What day of the week is it 45 days from today?")
    run("How many days are there between 2024-01-01 and 2024-12-31?")
    run("What is the date 100 days from today?")
    run("What day of the week was 2000-01-01?")
