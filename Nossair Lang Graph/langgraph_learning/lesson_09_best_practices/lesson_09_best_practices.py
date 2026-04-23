# =============================================================
# LESSON 9 — Production Best Practices
# =============================================================
#
# WHAT YOU WILL LEARN:
#   1. Streaming — see tokens as they arrive (not wait for full response)
#   2. Error handling & retry logic in nodes
#   3. Structured output — force LLM to return valid JSON/objects
#   4. Logging — trace every step of graph execution
#   5. Input validation — guard against bad inputs
#   6. Recursion limit — prevent infinite loops
#   7. Parallel node execution with Send()
#
# RULE OF THUMB:
#   Lessons 1-5 = "it works on my machine"
#   Lesson 9     = "it works in production"
# =============================================================

import logging
import time
import json
from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, ValidationError
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send


# =============================================================
# PRACTICE 1 — Structured Logging
# =============================================================
# Always set up logging BEFORE building graphs.
# Use it inside every node so you can trace what happened.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("langgraph_app")


def log_node(node_name: str, state: dict):
    """Call this at the start of every node."""
    n_msgs = len(state.get("messages", []))
    logger.info(f"NODE={node_name} | messages={n_msgs} | keys={list(state.keys())}")


# =============================================================
# PRACTICE 2 — Input Validation with Pydantic
# =============================================================

class UserQuery(BaseModel):
    question: str
    max_tokens: Optional[int] = 512
    language: Optional[str] = "english"

    def validate_question(self):
        if len(self.question.strip()) < 3:
            raise ValueError("Question is too short (minimum 3 characters)")
        if len(self.question) > 2000:
            raise ValueError("Question is too long (maximum 2000 characters)")
        return self


def validate_input(raw_input: dict) -> UserQuery:
    """Validate input before passing to the graph."""
    try:
        query = UserQuery(**raw_input)
        query.validate_question()
        logger.info(f"Input validated: '{query.question[:50]}'")
        return query
    except ValidationError as e:
        logger.error(f"Input validation failed: {e}")
        raise


# =============================================================
# PRACTICE 3 — Structured Output (Force LLM to return JSON)
# =============================================================

class SentimentResult(BaseModel):
    sentiment: str         # positive / negative / neutral
    confidence: float      # 0.0 to 1.0
    key_phrases: list[str] # top phrases that led to this
    summary: str           # one sentence summary


def analyze_with_structured_output(text: str) -> SentimentResult:
    """Use with_structured_output to guarantee valid JSON from the LLM."""
    llm = ChatOllama(model="llama3", temperature=0)
    structured_llm = llm.with_structured_output(SentimentResult)

    system = SystemMessage(content="""Analyze the sentiment of the text.
    Return a structured response with sentiment, confidence (0-1), key_phrases, and summary.""")

    result = structured_llm.invoke([
        system,
        HumanMessage(content=f"Analyze: {text}")
    ])
    logger.info(f"Structured output: sentiment={result.sentiment}, confidence={result.confidence}")
    return result


# =============================================================
# PRACTICE 4 — Error Handling & Retry in Nodes
# =============================================================

class ResilientState(TypedDict):
    messages:     Annotated[list, add_messages]
    retry_count:  int
    error:        str
    result:       str


llm = ChatOllama(model="llama3", temperature=0)


def resilient_node(state: ResilientState) -> dict:
    """A node with retry logic and graceful error handling."""
    log_node("resilient_node", state)

    MAX_RETRIES = 3
    retry_count = state.get("retry_count", 0)

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES}")
            response = llm.invoke(state["messages"])
            logger.info("LLM call succeeded")
            return {
                "messages": [response],
                "result": response.content,
                "error": "",
                "retry_count": retry_count
            }
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed: {e}")
                return {
                    "error": str(e),
                    "result": "Sorry, I encountered an error. Please try again.",
                    "retry_count": retry_count + 1
                }


def error_handler_node(state: ResilientState) -> dict:
    """Catch errors and respond gracefully instead of crashing."""
    log_node("error_handler", state)
    if state.get("error"):
        logger.error(f"Error caught by handler: {state['error']}")
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(content=f"I encountered a problem: {state['error']}. Please rephrase your question.")],
            "error": ""
        }
    return {}


def route_after_agent(state: ResilientState) -> str:
    if state.get("error"):
        return "error_handler"
    return "end"


# =============================================================
# PRACTICE 5 — Streaming Responses
# =============================================================

def demo_streaming():
    """Stream tokens as they arrive instead of waiting for full response."""
    print("\n" + "="*60)
    print("PRACTICE 5 — Streaming")
    print("="*60)

    builder = StateGraph({"messages": Annotated[list, add_messages]})

    def stream_node(state):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder.add_node("chatbot", stream_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    graph = builder.compile()

    print("Streaming response (token by token):")
    print("-" * 40)

    # stream_mode="messages" yields (message_chunk, metadata) tuples
    for chunk, metadata in graph.stream(
        {"messages": [HumanMessage(content="Write a haiku about Python programming.")]},
        stream_mode="messages"
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n" + "-" * 40)


# =============================================================
# PRACTICE 6 — Recursion Limit (Prevent Infinite Loops)
# =============================================================

def demo_recursion_limit():
    """Show how to set a recursion limit to prevent runaway graphs."""
    print("\n" + "="*60)
    print("PRACTICE 6 — Recursion Limit")
    print("="*60)

    class LoopState(TypedDict):
        messages: Annotated[list, add_messages]
        count: int

    def loop_node(state: LoopState) -> dict:
        print(f"  Node executed (count={state['count']})")
        return {"count": state["count"] + 1}

    def should_loop(state: LoopState) -> str:
        if state["count"] >= 3:
            return "end"
        return "loop"

    builder = StateGraph(LoopState)
    builder.add_node("loop", loop_node)
    builder.add_edge(START, "loop")
    builder.add_conditional_edges("loop", should_loop, {"loop": "loop", "end": END})
    graph = builder.compile()

    # The recursion_limit in config is a safety net
    config = {"recursion_limit": 10}
    result = graph.invoke({"messages": [], "count": 0}, config=config)
    print(f"Loop completed after {result['count']} iterations")


# =============================================================
# PRACTICE 7 — Parallel Execution with Send()
# =============================================================
# Send() lets you fan out to multiple nodes in parallel.
# Great for: parallel data fetching, batch processing, map-reduce.

class ParallelState(TypedDict):
    topics:   list[str]
    summaries: Annotated[list, lambda x, y: x + y]  # merge by appending


def summarize_topic(state: dict) -> dict:
    """Summarize a single topic — runs in parallel for each topic."""
    topic = state["topic"]
    logger.info(f"Summarizing topic: {topic}")
    response = llm.invoke([HumanMessage(content=f"In one sentence, what is {topic}?")])
    return {"summaries": [f"{topic}: {response.content}"]}


def fan_out(state: ParallelState):
    """Use Send() to launch one task per topic in parallel."""
    return [Send("summarize", {"topic": t}) for t in state["topics"]]


def demo_parallel():
    print("\n" + "="*60)
    print("PRACTICE 7 — Parallel Execution with Send()")
    print("="*60)

    builder = StateGraph(ParallelState)
    builder.add_node("summarize", summarize_topic)
    builder.add_conditional_edges(START, fan_out, ["summarize"])
    builder.add_edge("summarize", END)
    graph = builder.compile()

    result = graph.invoke({
        "topics": ["LangGraph", "SQLite", "Python decorators"],
        "summaries": []
    })

    print("Parallel summaries:")
    for summary in result["summaries"]:
        print(f"  • {summary[:100]}")


# =============================================================
# PRACTICE 8 — Full Resilient Graph with all best practices
# =============================================================

def build_production_graph():
    """A graph that combines: logging + validation + retry + error handling."""
    builder = StateGraph(ResilientState)
    builder.add_node("agent",         resilient_node)
    builder.add_node("error_handler", error_handler_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"error_handler": "error_handler", "end": END})
    builder.add_edge("error_handler", END)
    return builder.compile()


def demo_production_graph():
    print("\n" + "="*60)
    print("PRACTICE 8 — Full Production Graph")
    print("="*60)

    graph = build_production_graph()

    # Validate input first
    raw = {"question": "Explain the benefits of using LangGraph for production AI agents."}
    try:
        validated = validate_input(raw)
    except ValueError as e:
        print(f"Bad input: {e}")
        return

    result = graph.invoke({
        "messages": [HumanMessage(content=validated.question)],
        "retry_count": 0,
        "error": "",
        "result": ""
    })
    print(f"\nAnswer: {result['messages'][-1].content[:300]}")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print("Running all best practice demos...\n")

    # Structured output
    print("="*60)
    print("PRACTICE 3 — Structured Output")
    print("="*60)
    result = analyze_with_structured_output("This product is absolutely amazing, best purchase I've made!")
    print(f"Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
    print(f"Key phrases: {result.key_phrases}")
    print(f"Summary: {result.summary}")

    demo_streaming()
    demo_recursion_limit()
    demo_parallel()
    demo_production_graph()


# =============================================================
# BEST PRACTICES CHECKLIST:
#   ✅ Always log at start of every node
#   ✅ Validate inputs with Pydantic before invoking
#   ✅ Use with_structured_output for predictable LLM responses
#   ✅ Add try/except + retry with exponential backoff in nodes
#   ✅ Add an error_handler node as a safety net
#   ✅ Set recursion_limit in config to prevent runaway loops
#   ✅ Use streaming for better UX in chat applications
#   ✅ Use Send() for parallel workloads (much faster)
#   ✅ Use thread_id + checkpointer for multi-user isolation
# =============================================================
