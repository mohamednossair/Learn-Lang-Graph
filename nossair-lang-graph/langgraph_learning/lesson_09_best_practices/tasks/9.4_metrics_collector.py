# =============================================================
# TASK 9.4 — Metrics Collector
# =============================================================
# Goal:
#   Track metrics in state across every invoke():
#     llm_call_count   : int
#     avg_response_time: float (ms)
#     tool_call_count  : int
#     error_count      : int
#   After each invoke(), print a summary report.
# =============================================================

import time
import logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_ollama_model


logging.basicConfig(level=logging.WARNING)


# ── STEP 1: State ─────────────────────────────────────────────

class MetricsState(TypedDict):
    messages: Annotated[list, add_messages]
    llm_call_count: int
    total_response_time_ms: float   # sum of all LLM call times
    tool_call_count: int
    error_count: int
    error: str


# ── STEP 2: Tools ─────────────────────────────────────────────

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


tools = [add, multiply]


# ── STEP 3: Agent Node with Metrics ───────────────────────────
# TODO:
#   1. Record start time
#   2. Call LLM
#   3. Record end time → compute elapsed_ms
#   4. Count tool_calls in response
#   5. Return updated metrics

llm = ChatOllama(model=get_ollama_model(), temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: MetricsState) -> dict:
    t0 = time.perf_counter()
    try:
        response = llm_with_tools.invoke(state["messages"])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # TODO: count tool calls in response
        tool_calls = len(getattr(response, "tool_calls", [])) or 0

        return {
            "messages": [response],
            "llm_call_count": state["llm_call_count"] + 1,
            "total_response_time_ms": state["total_response_time_ms"] + elapsed_ms,
            "tool_call_count": state["tool_call_count"] + tool_calls,
            "error": "",
        }
    except Exception as e:
        return {
            "error": str(e),
            "error_count": state["error_count"] + 1,
            "llm_call_count": state["llm_call_count"] + 1,
            "total_response_time_ms": state["total_response_time_ms"],
        }


# ── STEP 4: Error Handler Node ────────────────────────────────

def error_handler(state: MetricsState) -> dict:
    return {"messages": [AIMessage(content=f"Error: {state['error']}")]}


# ── STEP 5: Routing ───────────────────────────────────────────

def should_continue(state: MetricsState) -> str:
    if state.get("error"):
        return "error_handler"
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ── STEP 6: Build Graph ───────────────────────────────────────

checkpointer = MemorySaver()
graph_builder = StateGraph(MetricsState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("error_handler", error_handler)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "error_handler": "error_handler",
    "end": END,
})
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("error_handler", END)
graph = graph_builder.compile(checkpointer=checkpointer)


# ── STEP 7: Metrics Report ────────────────────────────────────

def print_metrics(state: dict, label: str = ""):
    llm_calls = state.get("llm_call_count", 0)
    total_ms = state.get("total_response_time_ms", 0.0)
    avg_ms = (total_ms / llm_calls) if llm_calls > 0 else 0.0
    print(f"\n{'─'*50}")
    print(f"  METRICS REPORT{f' — {label}' if label else ''}")
    print(f"{'─'*50}")
    print(f"  LLM calls         : {llm_calls}")
    print(f"  Avg response time : {avg_ms:.0f} ms")
    print(f"  Tool calls        : {state.get('tool_call_count', 0)}")
    print(f"  Errors            : {state.get('error_count', 0)}")
    print(f"{'─'*50}")


# ── STEP 8: Test ──────────────────────────────────────────────

def run(question: str, config: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "llm_call_count": 0,
            "total_response_time_ms": 0.0,
            "tool_call_count": 0,
            "error_count": 0,
            "error": "",
        },
        config=config,
    )
    print(f"A: {result['messages'][-1].content[:150]}")
    print_metrics(result, question[:40])
    return result


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "metrics-demo"}}

    run("What is 25 * 4?", config)
    run("What is (100 + 50) * 3 and then divide by 2?", config)
    run("What is 15 + 8?", config)
