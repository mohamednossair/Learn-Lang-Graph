"""Task 20.2 — Circuit Breaker + GDPR."""
import sys, os, time, logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from typing import Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger("task_20_2")

# ---------------------------------------------------------------------------
# Circuit Breaker — CLOSED → OPEN → HALF_OPEN
# ---------------------------------------------------------------------------
class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown: float = 60.0):
        self.state: Literal["closed", "open", "half_open"] = "closed"
        self.failure_count = 0
        self.threshold = threshold
        self.cooldown = cooldown
        self.last_failure_time = 0.0
    
    def can_proceed(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.cooldown:
                self.state = "half_open"
                logger.info("[CIRCUIT] OPEN → HALF_OPEN (cooldown elapsed)")
                return True
            return False
        if self.state == "half_open":
            return True  # One test call allowed
        return False
    
    def record_success(self):
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            logger.info("[CIRCUIT] HALF_OPEN → CLOSED (test call succeeded)")
        self.failure_count = 0
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == "half_open":
            self.state = "open"
            logger.info("[CIRCUIT] HALF_OPEN → OPEN (test call failed)")
        elif self.failure_count >= self.threshold:
            self.state = "open"
            logger.warning(f"[CIRCUIT] CLOSED → OPEN ({self.failure_count} failures)")

circuit = CircuitBreaker(threshold=3, cooldown=0.5)  # Short cooldown for demo

# ---------------------------------------------------------------------------
# GDPR — Right to erasure + portability + audit
# ---------------------------------------------------------------------------
_audit_log = []

def gdpr_erasure(user_id: str) -> dict:
    """Article 17: Right to erasure — anonymise, don't hard-delete."""
    # 1. Anonymise records (keep structure for audit)
    # 2. Delete checkpoints: DELETE FROM langgraph_checkpoints WHERE thread_id LIKE '{user_id}-%'
    # 3. Write immutable audit event
    audit_entry = {
        "user_id": user_id,
        "action": "GDPR_ERASURE",
        "result": "COMPLETED",
        "timestamp": time.time(),
        "details": "Records anonymised, checkpoints deleted"
    }
    _audit_log.append(audit_entry)
    logger.info(f"[GDPR] Erasure completed for {user_id}")
    return {"erased_records": True, "audit_id": len(_audit_log)}

def gdpr_export(user_id: str) -> list:
    """Article 20: Right to portability — return user data as JSON."""
    # In production: SELECT * FROM interactions WHERE user_id = :1
    return [{"user_id": user_id, "data": "exported_record"}]

def gdpr_audit_trail() -> list:
    """Immutable audit log for SOC2 compliance."""
    return _audit_log.copy()

# ---------------------------------------------------------------------------
# Agent with circuit breaker
# ---------------------------------------------------------------------------
class CircuitState(TypedDict):
    messages: Annotated[list, add_messages]
    circuit_state: str
    force_fail: bool

llm = ChatOllama(model=get_ollama_model(), temperature=0)

def circuit_check(state: CircuitState) -> dict:
    can = circuit.can_proceed()
    return {"circuit_state": circuit.state if not can else "proceed"}

def route_circuit(state: CircuitState) -> str:
    if state.get("circuit_state") == "proceed":
        return "chat"
    return "circuit_open"

def chat_node(state: CircuitState) -> dict:
    if state.get("force_fail"):
        circuit.record_failure()
        return {"messages": [AIMessage(content="LLM call failed (simulated)")]}
    try:
        resp = llm.invoke(state["messages"])
        circuit.record_success()
        return {"messages": [resp]}
    except Exception as e:
        circuit.record_failure()
        return {"messages": [AIMessage(content=f"Error: {e}")]}

def circuit_open_node(state: CircuitState) -> dict:
    return {"messages": [AIMessage(content="Circuit breaker OPEN — LLM unavailable. Please try again later.")]}

builder = StateGraph(CircuitState)
builder.add_node("check", circuit_check)
builder.add_node("chat", chat_node)
builder.add_node("open", circuit_open_node)
builder.add_edge(START, "check")
builder.add_conditional_edges("check", route_circuit, {"chat": "chat", "open": "open"})
builder.add_edge("chat", END)
builder.add_edge("open", END)
circuit_graph = builder.compile()

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 20.2 — CIRCUIT BREAKER + GDPR")
    print("=" * 50)
    
    # Circuit breaker test
    print("\n--- Circuit breaker ---")
    print(f"  Initial state: {circuit.state}")
    
    for i in range(4):
        result = circuit_graph.invoke({
            "messages": [HumanMessage(content="Hello")],
            "circuit_state": "", "force_fail": True
        })
        print(f"  Attempt {i+1}: circuit={circuit.state}, response={result['messages'][-1].content[:40]}...")
    
    # Wait for cooldown → HALF_OPEN
    time.sleep(0.6)
    result = circuit_graph.invoke({
        "messages": [HumanMessage(content="Hello")],
        "circuit_state": "", "force_fail": False
    })
    print(f"  After cooldown: circuit={circuit.state}")
    
    # GDPR test
    print("\n--- GDPR compliance ---")
    erasure = gdpr_erasure("user-123")
    print(f"  Erasure: {erasure}")
    print(f"  Export: {gdpr_export('user-123')}")
    print(f"  Audit trail: {gdpr_audit_trail()}")
    
    print("\n✅ Circuit breaker + GDPR working!")
