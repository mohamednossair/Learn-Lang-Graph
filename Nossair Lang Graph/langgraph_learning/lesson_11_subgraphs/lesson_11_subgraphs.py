# =============================================================
# LESSON 11 — Subgraphs: Composing Graphs Inside Graphs
# =============================================================
#
# WHAT YOU WILL LEARN:
#   1. How to build a subgraph and use it as a node
#   2. How state is shared between parent and subgraph
#   3. Why subgraphs = reusable, testable, team-owned modules
#   4. Parallel subgraphs with Send()
#
# ANALOGY:
#   Subgraph = a function that is itself a full workflow.
#   Parent graph calls it exactly like any other node.
#
# ARCHITECTURE:
#   Parent: generate → validate → clean → publish/reject
#            (subgraph)  (subgraph)
# =============================================================

from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send

llm = ChatOllama(model="llama3", temperature=0)


# =============================================================
# SUBGRAPH 1 — Validation Module
# =============================================================
# This subgraph is reusable — any parent graph can include it.
# It checks length + forbidden words, sets is_valid + errors.

class ValidationState(TypedDict):
    content:           str
    is_valid:          bool
    validation_errors: list


def check_length(state: ValidationState) -> dict:
    errors = list(state.get("validation_errors", []))
    if len(state["content"].strip()) < 10:
        errors.append("Content too short (minimum 10 characters)")
        return {"is_valid": False, "validation_errors": errors}
    return {"is_valid": True, "validation_errors": errors}


def check_forbidden_words(state: ValidationState) -> dict:
    FORBIDDEN = ["spam", "scam", "fake", "misleading"]
    errors = list(state.get("validation_errors", []))
    found = [w for w in FORBIDDEN if w in state["content"].lower()]
    if found:
        errors.append(f"Contains forbidden words: {found}")
        return {"is_valid": False, "validation_errors": errors}
    return {"validation_errors": errors}


def check_format(state: ValidationState) -> dict:
    errors = list(state.get("validation_errors", []))
    if not state["content"][0].isupper():
        errors.append("Content must start with a capital letter")
        return {"is_valid": False, "validation_errors": errors}
    return {"validation_errors": errors}


v_builder = StateGraph(ValidationState)
v_builder.add_node("check_length",   check_length)
v_builder.add_node("check_forbidden", check_forbidden_words)
v_builder.add_node("check_format",   check_format)
v_builder.add_edge(START, "check_length")
v_builder.add_edge("check_length",   "check_forbidden")
v_builder.add_edge("check_forbidden", "check_format")
v_builder.add_edge("check_format",   END)
validation_subgraph = v_builder.compile()


# =============================================================
# SUBGRAPH 2 — Clean & Format Module
# =============================================================

class CleanState(TypedDict):
    content:       str
    clean_content: str


def clean_whitespace(state: CleanState) -> dict:
    clean = " ".join(state["content"].split())
    return {"clean_content": clean}


def fix_capitalization(state: CleanState) -> dict:
    return {"clean_content": state["clean_content"].strip().capitalize()}


def add_punctuation(state: CleanState) -> dict:
    text = state["clean_content"].rstrip()
    if not text.endswith((".", "!", "?")):
        text += "."
    return {"clean_content": text}


c_builder = StateGraph(CleanState)
c_builder.add_node("clean",     clean_whitespace)
c_builder.add_node("capitalize", fix_capitalization)
c_builder.add_node("punctuate", add_punctuation)
c_builder.add_edge(START,       "clean")
c_builder.add_edge("clean",     "capitalize")
c_builder.add_edge("capitalize", "punctuate")
c_builder.add_edge("punctuate", END)
clean_subgraph = c_builder.compile()


# =============================================================
# PARENT GRAPH
# State uses both subgraph state keys + parent-only keys
# Matching key names are automatically shared with subgraphs.
# =============================================================

class ReviewState(TypedDict):
    messages:          Annotated[list, add_messages]
    content:           str       # shared with BOTH subgraphs
    is_valid:          bool      # shared with validation subgraph
    validation_errors: list      # shared with validation subgraph
    clean_content:     str       # shared with clean subgraph
    final_output:      str       # parent only


def generate_node(state: ReviewState) -> dict:
    """Generate content using LLM."""
    system = SystemMessage(content="Write a short 2-sentence product description. Be professional.")
    resp = llm.invoke([system] + state["messages"])
    return {"content": resp.content, "messages": [resp]}


def route_after_validation(state: ReviewState) -> str:
    return "clean" if state.get("is_valid", False) else "reject"


def publish_node(state: ReviewState) -> dict:
    return {"final_output": f"✅ PUBLISHED: {state['clean_content']}"}


def reject_node(state: ReviewState) -> dict:
    errors = state.get("validation_errors", [])
    return {"final_output": f"❌ REJECTED: {'; '.join(errors)}"}


builder = StateGraph(ReviewState)
builder.add_node("generate", generate_node)
builder.add_node("validate", validation_subgraph)   # ← subgraph AS a node
builder.add_node("clean",    clean_subgraph)        # ← subgraph AS a node
builder.add_node("publish",  publish_node)
builder.add_node("reject",   reject_node)

builder.add_edge(START,       "generate")
builder.add_edge("generate",  "validate")
builder.add_conditional_edges("validate", route_after_validation, {"clean": "clean", "reject": "reject"})
builder.add_edge("clean",     "publish")
builder.add_edge("publish",   END)
builder.add_edge("reject",    END)

graph = builder.compile()


# =============================================================
# DEMO: Parallel Subgraphs with Send()
# =============================================================

class ParallelState(TypedDict):
    topics:    list[str]
    summaries: Annotated[list, lambda x, y: x + y]


def summarize_subgraph_node(state: dict) -> dict:
    """Each topic gets its own isolated subgraph execution."""
    topic = state["topic"]
    resp = llm.invoke([HumanMessage(content=f"In one sentence, define: {topic}")])
    return {"summaries": [f"{topic}: {resp.content}"]}


def parallel_fan_out(state: ParallelState):
    """Launch one summarize task per topic in parallel."""
    return [Send("summarize", {"topic": t}) for t in state["topics"]]


p_builder = StateGraph(ParallelState)
p_builder.add_node("summarize", summarize_subgraph_node)
p_builder.add_conditional_edges(START, parallel_fan_out, ["summarize"])
p_builder.add_edge("summarize", END)
parallel_graph = p_builder.compile()


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    # Test 1: Full pipeline with validation
    print("=" * 60)
    print("TEST 1: Content generation pipeline with subgraphs")
    print("=" * 60)
    result = graph.invoke({
        "messages": [HumanMessage(content="Write about our Python developer toolkit.")],
        "content": "", "is_valid": False, "validation_errors": [],
        "clean_content": "", "final_output": ""
    })
    print(f"Result: {result['final_output']}")

    # Test 2: Direct validation subgraph
    print("\n" + "=" * 60)
    print("TEST 2: Validation subgraph in isolation")
    print("=" * 60)
    tests = [
        {"content": "Hi", "is_valid": True, "validation_errors": []},
        {"content": "This product is spam.", "is_valid": True, "validation_errors": []},
        {"content": "Great Python IDE for developers.", "is_valid": True, "validation_errors": []},
    ]
    for t in tests:
        r = validation_subgraph.invoke(t)
        print(f"  '{t['content']}' → valid={r['is_valid']} | errors={r['validation_errors']}")

    # Test 3: Parallel subgraphs
    print("\n" + "=" * 60)
    print("TEST 3: Parallel topic summaries with Send()")
    print("=" * 60)
    result = parallel_graph.invoke({
        "topics": ["LangGraph", "SQLite", "FastAPI", "Pydantic"],
        "summaries": []
    })
    for s in result["summaries"]:
        print(f"  • {s[:100]}")


# =============================================================
# KEY TAKEAWAYS:
#   ✅ A subgraph is compiled ONCE, reused in many parent graphs
#   ✅ Matching state key names = automatic sharing
#   ✅ Test subgraphs independently before integrating
#   ✅ Send() runs multiple subgraph executions in parallel
#   ✅ Each subgraph has its own internal node/edge structure
# =============================================================
