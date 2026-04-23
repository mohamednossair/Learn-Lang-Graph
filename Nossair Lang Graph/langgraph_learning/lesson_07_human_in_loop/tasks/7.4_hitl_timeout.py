# =============================================================
# TASK 7.4 — HITL Timeout
# =============================================================
# Goal:
#   If human doesn't respond within 30 seconds, auto-reject.
#   Implement using threading.Timer that calls
#   graph.invoke(Command(resume="timeout"), config) after 30s.
#
# Key concept: background timer thread driving graph resume
# =============================================================

import threading
import time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ── STEP 1: State ─────────────────────────────────────────────

class ApprovalState(TypedDict):
    messages: Annotated[list, add_messages]
    action: str
    decision: str
    timed_out: bool


# ── STEP 2: Request Node ──────────────────────────────────────

def request_approval(state: ApprovalState) -> dict:
    print(f"[request] Waiting for approval: '{state['action']}'")
    decision = interrupt({
        "message": f"Approve this action? '{state['action']}'",
        "options": ["approve", "reject"],
        "timeout_seconds": 30,
    })
    timed_out = decision == "timeout"
    if timed_out:
        print(f"[request] TIMED OUT — auto-rejecting")
    else:
        print(f"[request] Human decision: {decision}")
    return {"decision": decision, "timed_out": timed_out}


# ── STEP 3: Execute Node ──────────────────────────────────────

def execute_action(state: ApprovalState) -> dict:
    if state["decision"] == "approve":
        print(f"[execute] ✓ Action approved and executed: {state['action']}")
    elif state["decision"] == "timeout":
        print(f"[execute] ✗ Action auto-rejected (timeout): {state['action']}")
    else:
        print(f"[execute] ✗ Action rejected by human: {state['action']}")
    return {}


# ── STEP 4: Build Graph ───────────────────────────────────────

checkpointer = MemorySaver()

graph_builder = StateGraph(ApprovalState)
graph_builder.add_node("request", request_approval)
graph_builder.add_node("execute", execute_action)
graph_builder.add_edge(START, "request")
graph_builder.add_edge("request", "execute")
graph_builder.add_edge("execute", END)

graph = graph_builder.compile(checkpointer=checkpointer)


# ── STEP 5: Timeout Helper ────────────────────────────────────

def schedule_timeout(config: dict, timeout_seconds: int = 30):
    """Schedule an auto-reject if human doesn't respond in time."""
    def fire_timeout():
        print(f"\n[TIMEOUT] {timeout_seconds}s elapsed — auto-rejecting")
        try:
            graph.invoke(Command(resume="timeout"), config=config)
        except Exception as e:
            print(f"[TIMEOUT] Graph already completed: {e}")

    timer = threading.Timer(timeout_seconds, fire_timeout)
    timer.daemon = True
    timer.start()
    return timer


# ── STEP 6: Test ──────────────────────────────────────────────

def run_with_timeout(action: str, human_response: str | None, response_delay: float = 0):
    """
    Simulate HITL with optional human response.
    human_response=None simulates no response (timeout fires).
    response_delay=seconds before human responds.
    """
    config = {"configurable": {"thread_id": f"action-{action[:10].replace(' ', '-')}"}}
    print(f"\n{'='*60}")
    print(f"Action: {action}")

    # Start graph — will pause at interrupt
    graph.invoke(
        {"messages": [], "action": action, "decision": "", "timed_out": False},
        config=config,
    )

    current = graph.get_state(config)
    if not current.next:
        return  # completed without interrupt

    # Schedule timeout
    timer = schedule_timeout(config, timeout_seconds=5)  # 5s for demo

    if human_response is not None:
        # Simulate human responding after a delay
        time.sleep(response_delay)
        timer.cancel()
        print(f"[human] Responding: {human_response}")
        graph.invoke(Command(resume=human_response), config=config)
    else:
        # No human response — let timeout fire
        print("[human] No response — waiting for timeout...")
        time.sleep(7)  # wait longer than the 5s timeout

    final = graph.get_state(config).values
    print(f"Decision: {final.get('decision')} | TimedOut: {final.get('timed_out')}")


if __name__ == "__main__":
    # Test 1: Human approves quickly
    run_with_timeout("Deploy to production", human_response="approve", response_delay=1)

    # Test 2: Human rejects
    run_with_timeout("Delete all logs", human_response="reject", response_delay=1)

    # Test 3: No response — timeout fires
    run_with_timeout("Send bulk email", human_response=None)
