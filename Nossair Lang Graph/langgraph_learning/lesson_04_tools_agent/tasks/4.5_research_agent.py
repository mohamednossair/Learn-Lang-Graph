# =============================================================
# TASK 4.5 — Research Agent (Mocked Tools)
# =============================================================
# Goal:
#   Build a research agent with 3 mocked tools:
#     search_web(query)       — returns relevant URLs + snippets
#     summarize_page(url)     — returns page summary
#     extract_facts(text)     — returns bullet-point facts
#   The agent chains these tools to research a topic end-to-end.
# =============================================================

from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ── STEP 1: Mocked Research Tools ────────────────────────────

MOCK_SEARCH_DB = {
    "langgraph": [
        {"url": "https://langchain-ai.github.io/langgraph/", "snippet": "LangGraph is a library for building stateful, multi-actor LLM apps."},
        {"url": "https://blog.langchain.dev/langgraph/", "snippet": "LangGraph enables cyclic graphs for agentic workflows."},
    ],
    "python": [
        {"url": "https://python.org", "snippet": "Python is a high-level, general-purpose programming language."},
        {"url": "https://docs.python.org/3/", "snippet": "Official Python 3 documentation."},
    ],
    "default": [
        {"url": "https://en.wikipedia.org/wiki/Artificial_intelligence", "snippet": "AI is the simulation of human intelligence in machines."},
    ],
}

MOCK_PAGE_CONTENT = {
    "https://langchain-ai.github.io/langgraph/": "LangGraph lets you model agent workflows as directed graphs. Nodes are Python functions. Edges define execution order. Supports cycles for ReAct loops.",
    "https://python.org": "Python was created by Guido van Rossum. First released in 1991. Known for readability and simplicity. Supports OOP, functional, and procedural styles.",
    "default": "This page contains general information about the topic.",
}


@tool
def search_web(query: str) -> str:
    """Search the web for a query. Returns a list of relevant URLs and snippets."""
    query_lower = query.lower()
    results = MOCK_SEARCH_DB.get("default", [])
    for key in MOCK_SEARCH_DB:
        if key in query_lower:
            results = MOCK_SEARCH_DB[key]
            break
    formatted = "\n".join([f"URL: {r['url']}\nSnippet: {r['snippet']}" for r in results])
    print(f"  [tool:search_web] query='{query}' → {len(results)} results")
    return formatted


@tool
def summarize_page(url: str) -> str:
    """Fetch and summarize the content of a web page given its URL."""
    content = MOCK_PAGE_CONTENT.get(url, MOCK_PAGE_CONTENT["default"])
    print(f"  [tool:summarize_page] url='{url}'")
    return f"Summary of {url}:\n{content}"


# TODO: implement extract_facts tool
# @tool
# def extract_facts(text: str) -> str:
#     """Extract key facts from a block of text. Returns bullet-point facts.
#     Use this after summarize_page to get structured facts."""
#     pass


# TODO: add all 3 tools
tools = [search_web, summarize_page]


# ── STEP 2: State + LLM + Agent ───────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="llama3", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(content=(
    "You are a research assistant. To answer a question: "
    "1. Use search_web to find relevant URLs. "
    "2. Use summarize_page on the most relevant URL. "
    "3. Use extract_facts to get structured facts from the summary. "
    "4. Provide a final answer based on the facts."
))


def agent_node(state: AgentState) -> dict:
    messages = [SYSTEM] + state["messages"]
    print(f"\n[agent] thinking... ({len(state['messages'])} messages)")
    response = llm_with_tools.invoke(messages)
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
    print(f"Research Question: {question}")
    print("=" * 60)
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    print(f"\nFinal Answer:\n{result['messages'][-1].content}")


if __name__ == "__main__":
    run("What is LangGraph and what are its key features?")
    run("Tell me about the Python programming language.")
