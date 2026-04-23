# =============================================================
# TASK 9.2 — Token Counter Node
# =============================================================
# Goal:
#   Add a token counter node that runs BEFORE every LLM call.
#   Uses tiktoken to count tokens.
#   If > 3000 tokens: trim oldest messages.
#   Log token count before and after trimming.
#   Track total_tokens_used: int in state.
#
# State: {messages, total_tokens_used, tokens_this_turn}
# =============================================================

import logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("WARNING: tiktoken not installed. Run: pip install tiktoken")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("token_counter")

TOKEN_LIMIT = 3000


# ── STEP 1: State ─────────────────────────────────────────────

class TokenState(TypedDict):
    messages: Annotated[list, add_messages]
    total_tokens_used: int
    tokens_this_turn: int


# ── STEP 2: Token Counter Utility ─────────────────────────────

def count_tokens(messages: list) -> int:
    """Count tokens in message list using tiktoken, with fallback."""
    if not TIKTOKEN_AVAILABLE:
        # Rough estimate: 4 chars ≈ 1 token
        total_chars = sum(len(getattr(m, "content", "")) for m in messages)
        return total_chars // 4

    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "")
        total += len(enc.encode(str(content)))
    return total


# ── STEP 3: Token Counter Node ────────────────────────────────
# TODO:
#   1. Count tokens in current messages
#   2. Log: f"Tokens before: {before}"
#   3. If count > TOKEN_LIMIT: trim — keep last 6 messages
#      Log: f"Trimmed from {before} to {after} tokens"
#   4. Return {"tokens_this_turn": count_after, "total_tokens_used": updated_total}
#
# NOTE: To replace messages (not append), you need to handle this
# at the chat() level since add_messages reducer only appends.
# Return the token counts only; trimming happens in chat().

def token_counter_node(state: TokenState) -> dict:
    messages = state["messages"]
    before = count_tokens(messages)
    logger.info(f"Tokens before LLM call: {before}")
    new_total = state["total_tokens_used"] + before
    return {"tokens_this_turn": before, "total_tokens_used": new_total}


# ── STEP 4: LLM Node ──────────────────────────────────────────

llm = ChatOllama(model="llama3", temperature=0.5)


def llm_node(state: TokenState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ── STEP 5: Build Graph ───────────────────────────────────────
# Flow: START → token_counter → llm_node → END

graph_builder = StateGraph(TokenState)
graph_builder.add_node("token_counter", token_counter_node)
graph_builder.add_node("llm", llm_node)
graph_builder.add_edge(START, "token_counter")
graph_builder.add_edge("token_counter", "llm")
graph_builder.add_edge("llm", END)

graph = graph_builder.compile()


# ── STEP 6: Chat Helper with External Trim ────────────────────

def chat(question: str, history: list, total_tokens: int):
    history.append(HumanMessage(content=question))

    # Trim if over limit
    current_tokens = count_tokens(history)
    if current_tokens > TOKEN_LIMIT:
        before_len = len(history)
        history = history[-6:]  # keep last 6 messages
        after_tokens = count_tokens(history)
        logger.warning(f"Trimmed messages: {before_len} → {len(history)} ({current_tokens} → {after_tokens} tokens)")

    result = graph.invoke({
        "messages": history,
        "total_tokens_used": total_tokens,
        "tokens_this_turn": 0,
    })

    new_msg = result["messages"][-1]
    history.append(new_msg)
    return new_msg.content, history, result["total_tokens_used"]


# ── STEP 7: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Token Counter Node — tracking token usage")
    print("=" * 60)

    history = []
    total_tokens = 0
    questions = [
        "What is Python?",
        "What are the key data structures in Python?",
        "Explain list comprehensions with examples",
        "What is the difference between a list and a tuple?",
        "What is a generator in Python?",
        "How does async/await work in Python?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        reply, history, total_tokens = chat(q, history, total_tokens)
        print(f"Messages in history: {len(history)}")
        print(f"Total tokens used so far: {total_tokens}")
        print(f"A: {reply[:150]}")
