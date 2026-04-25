# =============================================================
# TASK 5.3 — Parallel Code Review System
# =============================================================
# Goal:
#   Supervisor fans out to 3 parallel specialists using Send():
#     syntax_checker   — checks for syntax/logic errors
#     style_checker    — checks code style and naming
#     security_checker — checks for security vulnerabilities
#   Supervisor then combines all 3 reports into a final review.
#
# Key concept: Send() runs all 3 checkers in PARALLEL.
# State uses an Annotated list reducer to collect parallel results.
# =============================================================

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send


# ── STEP 1: State ─────────────────────────────────────────────

def merge_reports(existing: list, new: list) -> list:
    """Reducer: append new review reports to the list."""
    return existing + new


class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    code: str                              # code to review
    review_reports: Annotated[list, merge_reports]  # collected from parallel checkers
    final_review: str                      # combined summary from supervisor


# ── STEP 2: LLM ───────────────────────────────────────────────

llm = ChatOllama(model="llama3.2", temperature=0)


# ── STEP 3: Parallel Checker Nodes ───────────────────────────

def syntax_checker(state: ReviewState) -> dict:
    prompt = [
        SystemMessage(content="You are a syntax and logic reviewer. Find bugs, logic errors, and incorrect code patterns. Be concise."),
        HumanMessage(content=f"Review this code for syntax and logic issues:\n```python\n{state['code']}\n```"),
    ]
    response = llm.invoke(prompt)
    print("[syntax_checker] Done")
    return {"review_reports": [f"[SYNTAX/LOGIC]\n{response.content}"]}


def style_checker(state: ReviewState) -> dict:
    prompt = [
        SystemMessage(content="You are a code style reviewer. Check PEP8 compliance, naming conventions, and readability. Be concise."),
        HumanMessage(content=f"Review this code for style issues:\n```python\n{state['code']}\n```"),
    ]
    response = llm.invoke(prompt)
    print("[style_checker] Done")
    return {"review_reports": [f"[STYLE]\n{response.content}"]}


# TODO: implement security_checker node
# def security_checker(state: ReviewState) -> dict:
#     # Check for: SQL injection, hardcoded secrets, unsafe eval(), unvalidated input
#     pass


# ── STEP 4: Fan-Out Function ──────────────────────────────────
# TODO: return Send() calls to run all 3 checkers in parallel

def fan_out(state: ReviewState) -> list:
    # return [
    #     Send("syntax_checker", state),
    #     Send("style_checker", state),
    #     Send("security_checker", state),
    # ]
    pass


# ── STEP 5: Combiner Node ─────────────────────────────────────
# TODO: supervisor combines all review_reports into a final_review summary

def combine_reviews(state: ReviewState) -> dict:
    all_reports = "\n\n".join(state["review_reports"])
    prompt = [
        SystemMessage(content="You are a senior code reviewer. Summarize the following review reports into one final code review with a priority list of issues to fix."),
        HumanMessage(content=f"Reports:\n{all_reports}"),
    ]
    response = llm.invoke(prompt)
    print("[combiner] Final review done")
    return {"final_review": response.content, "messages": [response]}


# ── STEP 6: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(ReviewState)

# TODO: add all nodes
# TODO: START → fan_out (using add_conditional_edges with fan_out returning Send list)
# TODO: all checkers → combine_reviews
# TODO: combine_reviews → END

graph = graph_builder.compile()


# ── STEP 7: Test ──────────────────────────────────────────────

SAMPLE_CODE = '''
import sqlite3

def get_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    cursor.execute(query)
    return cursor.fetchone()

API_KEY = "sk-1234567890abcdef"

def processData(data):
    result=eval(data)
    return result
'''

if __name__ == "__main__":
    print("=" * 65)
    print("Parallel Code Review System")
    print("=" * 65)

    result = graph.invoke({
        "messages": [HumanMessage(content="Please review this code")],
        "code": SAMPLE_CODE,
        "review_reports": [],
        "final_review": "",
    })

    print(f"\n{'='*65}")
    print("FINAL REVIEW:")
    print(result["final_review"])
