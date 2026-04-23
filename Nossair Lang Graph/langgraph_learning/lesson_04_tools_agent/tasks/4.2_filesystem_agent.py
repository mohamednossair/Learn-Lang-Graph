# =============================================================
# TASK 4.2 — File System Agent
# =============================================================
# Goal:
#   Build a ReAct agent that can inspect the local file system.
#   Tools: read_file, list_files, count_lines
#
# Test: "How many lines does requirements.txt have?"
# =============================================================

import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ── STEP 1: File System Tools ─────────────────────────────────

@tool
def read_file(path: str) -> str:
    """Read the full contents of a text file. Returns the file content or an error string."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"  [tool:read_file] Read {len(content)} chars from '{path}'")
        return content
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def list_files(directory: str) -> str:
    """List all files (not subdirectories) in a directory. Returns filenames as a newline-separated string."""
    try:
        items = os.listdir(directory)
        files = [f for f in items if os.path.isfile(os.path.join(directory, f))]
        result = "\n".join(files) if files else "(no files found)"
        print(f"  [tool:list_files] {len(files)} files in '{directory}'")
        return result
    except FileNotFoundError:
        return f"ERROR: Directory not found: {directory}"
    except Exception as e:
        return f"ERROR: {str(e)}"


# TODO: implement count_lines tool
# @tool
# def count_lines(path: str) -> str:
#     """Count the number of lines in a text file. Returns the count as a string, or an error."""
#     pass


# TODO: add all 3 tools to this list
tools = [read_file, list_files]


# ── STEP 2: State + LLM + Agent ───────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="llama3", temperature=0)
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
    # Adjust the base path to your project root
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    req_path = os.path.join(project_root, "requirements.txt")

    run(f"How many lines does the file '{req_path}' have?")
    run(f"List all files in the directory: {project_root}")
    run(f"Read the first 5 lines of '{req_path}' and tell me what packages are listed.")
