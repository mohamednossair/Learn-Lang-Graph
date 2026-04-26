# =============================================================
# TASK 4.4 — Retry on Tool Failure
# =============================================================
# Goal:
#   One tool fails randomly 50% of the time.
#   The agent must detect the failure (tool returns an error
#   string starting with "ERROR:") and retry up to 3 times.
#   Track retry count in state.
#
# New state field: retries: int
# =============================================================

import random
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_ollama_model


# ── STEP 1: Flaky Tool ────────────────────────────────────────

@tool
def fetch_stock_price(ticker: str) -> str:
    """Fetch the current stock price for a ticker symbol (e.g. AAPL, GOOG).
    Returns the price as a string, or an ERROR string if the service is unavailable."""
    if random.random() < 0.5:
        print(f"  [tool:fetch_stock_price] FAIL — simulated service error")
        return f"ERROR: Stock service temporarily unavailable. Please retry."
    price = round(random.uniform(100, 500), 2)
    print(f"  [tool:fetch_stock_price] SUCCESS — {ticker}: ${price}")
    return f"{ticker}: ${price}"


tools = [fetch_stock_price]


# ── STEP 2: State ─────────────────────────────────────────────

class RetryState(TypedDict):
    messages: Annotated[list, add_messages]
    retries: int  # tracks how many retries have occurred


# ── STEP 3: LLM + Agent Node ──────────────────────────────────

llm = ChatOllama(model=get_ollama_model(), temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(content=(
    "You are a stock price assistant. "
    "If a tool returns an ERROR, retry it. "
    "Stop and report failure after 3 errors for the same request."
))


def agent_node(state: RetryState) -> dict:
    messages = [SYSTEM] + state["messages"]
    print(f"\n[agent] thinking... (retries so far: {state['retries']})")
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ── STEP 4: Retry Counter Node ────────────────────────────────
# TODO:
#   After ToolNode runs, check the last ToolMessage.
#   If its content starts with "ERROR:" → increment retries.
#   Return {"retries": new_count}

def check_retry(state: RetryState) -> dict:
    last = state["messages"][-1]
    if (last.content.startswith("ERROR:")):
        return {"retries": state["retries"] + 1}
    return {"retries": state["retries"]}


# ── STEP 5: Routing ───────────────────────────────────────────

def should_continue(state: RetryState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    # If last message was an error and we haven't reached max retries, force retry
    if hasattr(last, "content") and last.content.startswith("ERROR:") and state["retries"] < 3:
        return "tools"
    return "end"


# ── STEP 6: Build Graph ───────────────────────────────────────
# Flow: agent → tools → check_retry → agent (loop)
#                                   or end if retries >= 3

graph_builder = StateGraph(RetryState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))

# TODO: add check_retry node
graph_builder.add_node("check_retry", check_retry)
# TODO: wire: START → agent
graph_builder.add_edge(START, "agent")
# TODO: conditional edges from agent (should_continue)
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
# TODO: tools → check_retry → agent  (or handle max retries)
graph_builder.add_edge("tools", "check_retry")

def should_retry(state: RetryState) -> str:
    if state["retries"] >= 3:
        return "end"
    return "agent"

graph_builder.add_conditional_edges("check_retry", should_retry, {"agent": "agent", "end": END})
graph = graph_builder.compile()

# ── STEP 7: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Retry Agent — will retry flaky tool up to 3 times")
    print("=" * 60)

    for ticker in ["AAPL", "GOOG", "MSFT"]:
        print(f"\n{'=' * 60}")
        result = graph.invoke({
            "messages": [HumanMessage(content=f"What is the current price of {ticker}?")],
            "retries": 0,
        })
        print(f"Retries used: {result['retries']}")
        print(f"Answer: {result['messages'][-1].content}")
