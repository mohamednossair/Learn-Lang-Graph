# =============================================================
# TASK 2.5 — Comprehensive Document Processing Workflow
# =============================================================
# This task combines ALL Lesson 02 concepts:
#   1. N-way classification (document type)
#   2. Validation gate (document structure)
#   3. Retry with limit (flaky processing)
#   4. ReAct loop (tool-calling simulation)
#   5. Two-level routing (type → subtype)
#   6. Proper routing best practices (Literal, defaults, .get())
#   7. Unit testing routing functions
#   8. Error handling with explicit defaults
#
# Scenario: An intelligent document processor that:
#   - Classifies documents (invoice, contract, report, email)
#   - Validates structure
#   - Processes with retry on failure
#   - Uses tool-calling pattern for complex extraction
#   - Routes to appropriate final handler
#
# State: {
#   document: str,
#   doc_type: str,
#   subtype: str,
#   valid: bool,
#   validation_errors: list,
#   processed: bool,
#   attempt: int,
#   max_attempts: int,
#   tool_calls_made: int,
#   extracted_data: dict,
#   final_output: str,
#   error: str
# }
# =============================================================

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
import random


# ── STEP 1: State Definition ─────────────────────────────────────

class DocumentState(TypedDict):
    document: str              # raw document text
    doc_type: str              # invoice/contract/report/email
    subtype: str               # subtype within type (e.g., invoice: po/so)
    valid: bool                # validation result
    validation_errors: list    # list of error messages
    processed: bool            # processing success flag
    attempt: int               # current retry attempt
    max_attempts: int          # maximum retry attempts
    tool_calls_made: int       # number of tool calls in ReAct loop
    extracted_data: dict       # data extracted by tools
    final_output: str          # final result message
    error: str                 # error message if any


# ── STEP 2: Document Type Classifier (N-way classification) ────

def classify_document_type(state: DocumentState) -> dict:
    """
    Classify document based on keywords.
    Demonstrates N-way classification pattern.
    """
    doc = state["document"].lower()
    
    # Invoice keywords
    if any(kw in doc for kw in ["invoice", "bill", "payment", "due", "amount"]):
        return {"doc_type": "invoice"}
    
    # Contract keywords
    elif any(kw in doc for kw in ["contract", "agreement", "terms", "clause", "party"]):
        return {"doc_type": "contract"}
    
    # Report keywords
    elif any(kw in doc for kw in ["report", "analysis", "summary", "findings", "metrics"]):
        return {"doc_type": "report"}
    
    # Email keywords (default)
    else:
        return {"doc_type": "email"}


# ── STEP 3: Subtype Classifier (Two-level routing) ─────────────

def classify_subtype(state: DocumentState) -> dict:
    """
    Classify subtype within document type.
    Demonstrates two-level hierarchical routing.
    """
    doc = state["document"].lower()
    doc_type = state["doc_type"]
    
    if doc_type == "invoice":
        if "purchase order" in doc or "po" in doc:
            return {"subtype": "purchase_order"}
        elif "sales order" in doc or "so" in doc:
            return {"subtype": "sales_order"}
        else:
            return {"subtype": "standard"}
    
    elif doc_type == "contract":
        if "nda" in doc or "non-disclosure" in doc:
            return {"subtype": "nda"}
        elif "employment" in doc:
            return {"subtype": "employment"}
        else:
            return {"subtype": "service"}
    
    elif doc_type == "report":
        if "financial" in doc:
            return {"subtype": "financial"}
        elif "performance" in doc:
            return {"subtype": "performance"}
        else:
            return {"subtype": "general"}
    
    else:  # email
        if "urgent" in doc:
            return {"subtype": "urgent"}
        else:
            return {"subtype": "standard"}


# ── STEP 4: Validation Gate (Validation pattern) ─────────────

def validate_document(state: DocumentState) -> dict:
    """
    Validate document structure.
    Demonstrates validation gate pattern.
    """
    doc = state["document"]
    errors = []
    
    # Check minimum length
    if len(doc) < 10:
        errors.append("Document too short (minimum 10 characters)")
    
    # Check for required content based on type
    doc_type = state["doc_type"]
    if doc_type == "invoice" and "amount" not in doc.lower():
        errors.append("Invoice missing amount information")
    elif doc_type == "contract" and "terms" not in doc.lower():
        errors.append("Contract missing terms section")
    elif doc_type == "report" and "summary" not in doc.lower():
        errors.append("Report missing summary section")
    
    valid = len(errors) == 0
    return {"valid": valid, "validation_errors": errors}


# ── STEP 5: Processing Node with Retry Logic ───────────────────

def process_document(state: DocumentState) -> dict:
    """
    Process document with simulated flaky behavior.
    Demonstrates retry with limit pattern.
    """
    # Simulate 70% success rate (flaky processing)
    success = random.random() > 0.3
    
    if success:
        return {"processed": True, "error": ""}
    else:
        return {"processed": False, "error": "Processing failed (simulated transient error)"}


# ── STEP 6: ReAct Agent Node (Tool-calling pattern) ───────────

def react_agent(state: DocumentState) -> dict:
    """
    Simulate the ReAct loop with tool calls.
    Demonstrates ReAct loop pattern.
    """
    tool_calls_made = state["tool_calls_made"]
    extracted = state["extracted_data"].copy() if state["extracted_data"] else {}
    
    # Simulate tool calls based on document type
    if tool_calls_made == 0:
        # First tool call: extract basic info
        extracted["basic_info"] = f"Extracted from {state['doc_type']}"
        return {
            "tool_calls_made": tool_calls_made + 1,
            "extracted_data": extracted
        }
    elif tool_calls_made == 1:
        # Second tool call: extract entities
        extracted["entities"] = ["Company A", "Company B", "Date: 2024-01-15"]
        return {
            "tool_calls_made": tool_calls_made + 1,
            "extracted_data": extracted
        }
    else:
        # Done with tool calls
        return {"tool_calls_made": tool_calls_made}


# ── STEP 7: Final Handler Nodes ─────────────────────────────────

def invoice_handler(state: DocumentState) -> dict:
    subtype = state["subtype"]
    output = f"[INVOICE/{subtype.upper()}] Processed successfully. Extracted: {state['extracted_data']}"
    return {"final_output": output}


def contract_handler(state: DocumentState) -> dict:
    subtype = state["subtype"]
    output = f"[CONTRACT/{subtype.upper()}] Processed successfully. Extracted: {state['extracted_data']}"
    return {"final_output": output}


def report_handler(state: DocumentState) -> dict:
    subtype = state["subtype"]
    output = f"[REPORT/{subtype.upper()}] Processed successfully. Extracted: {state['extracted_data']}"
    return {"final_output": output}


def email_handler(state: DocumentState) -> dict:
    subtype = state["subtype"]
    output = f"[EMAIL/{subtype.upper()}] Processed successfully. Extracted: {state['extracted_data']}"
    return {"final_output": output}


def validation_error_handler(state: DocumentState) -> dict:
    errors = "; ".join(state["validation_errors"])
    output = f"[VALIDATION ERROR] {errors}"
    return {"final_output": output, "error": errors}


def retry_exhausted_handler(state: DocumentState) -> dict:
    output = f"[RETRY EXHAUSTED] Failed after {state['attempt']} attempts. Last error: {state['error']}"
    return {"final_output": output, "error": "Max retries exceeded"}


# ── STEP 8: Routing Functions (Best practices) ─────────────────

def route_after_classification(state: DocumentState) -> Literal["classify_subtype", "validation_error_handler"]:
    """
    Route after initial classification.
    Demonstrates: error state check first, explicit default.
    """
    # Always check error state first (best practice)
    if state.get("error"):
        return "validation_error_handler"
    
    return "classify_subtype"


def route_after_validation(state: DocumentState) -> Literal["process_document", "validation_error_handler"]:
    """
    Route after validation.
    Demonstrates: .get() with default, explicit default return.
    """
    if not state.get("valid", False):
        return "validation_error_handler"
    
    return "process_document"


def route_after_processing(state: DocumentState) -> Literal["react_agent", "retry_document", "success_handler"]:
    """
    Route after processing attempt.
    Demonstrates: retry with limit pattern.
    """
    # Success path
    if state.get("processed", False):
        return "react_agent"
    
    # Retry path
    if state.get("attempt", 0) < state.get("max_attempts", 3):
        return "retry_document"
    
    # Give up path
    return "success_handler"  # Will route to retry_exhausted


def route_react_loop(state: DocumentState) -> Literal["react_agent", "route_to_final"]:
    """
    Route in ReAct loop.
    Demonstrates: ReAct loop pattern.
    """
    # Continue loop if less than 2 tool calls made
    if state.get("tool_calls_made", 0) < 2:
        return "react_agent"
    
    # Exit loop
    return "route_to_final"


def route_to_final(state: DocumentState) -> Literal["invoice_handler", "contract_handler", "report_handler", "email_handler"]:
    """
    Route to final handler based on document type.
    Demonstrates: semantic routing with mapping dict.
    """
    doc_type = state.get("doc_type", "email")
    
    if doc_type == "invoice":
        return "invoice_handler"
    elif doc_type == "contract":
        return "contract_handler"
    elif doc_type == "report":
        return "report_handler"
    else:
        return "email_handler"  # Explicit default


def route_retry_or_giveup(state: DocumentState) -> Literal["process_document", "retry_exhausted_handler"]:
    """
    Route for retry logic.
    Demonstrates: retry with limit.
    """
    # Increment attempt counter
    new_attempt = state.get("attempt", 0) + 1
    
    if new_attempt <= state.get("max_attempts", 3):
        # Retry - update attempt counter
        return "process_document"
    else:
        # Give up
        return "retry_exhausted_handler"


# ── STEP 9: Increment Attempt Node ─────────────────────────────

def increment_attempt(state: DocumentState) -> dict:
    """Increment retry attempt counter."""
    return {"attempt": state.get("attempt", 0) + 1}


# ── STEP 10: Build Graph ─────────────────────────────────────────

graph_builder = StateGraph(DocumentState)

# Add all nodes
graph_builder.add_node("classify_document_type", classify_document_type)
graph_builder.add_node("classify_subtype", classify_subtype)
graph_builder.add_node("validate_document", validate_document)
graph_builder.add_node("process_document", process_document)
graph_builder.add_node("react_agent", react_agent)
graph_builder.add_node("increment_attempt", increment_attempt)
graph_builder.add_node("invoice_handler", invoice_handler)
graph_builder.add_node("contract_handler", contract_handler)
graph_builder.add_node("report_handler", report_handler)
graph_builder.add_node("email_handler", email_handler)
graph_builder.add_node("validation_error_handler", validation_error_handler)
graph_builder.add_node("retry_exhausted_handler", retry_exhausted_handler)

# Add edges
graph_builder.add_edge(START, "classify_document_type")
graph_builder.add_edge("classify_document_type", "classify_subtype")
graph_builder.add_edge("classify_subtype", "validate_document")

# Conditional edge: validation gate
graph_builder.add_conditional_edges(
    "validate_document",
    route_after_validation,
    {
        "process_document": "process_document",
        "validation_error_handler": "validation_error_handler"
    }
)

# Conditional edge: after processing (retry or continue)
graph_builder.add_conditional_edges(
    "process_document",
    route_after_processing,
    {
        "react_agent": "react_agent",
        "retry_document": "increment_attempt",
        "success_handler": "retry_exhausted_handler"
    }
)

# Retry loop
graph_builder.add_edge("increment_attempt", "process_document")

# ReAct loop
graph_builder.add_conditional_edges(
    "react_agent",
    route_react_loop,
    {
        "react_agent": "react_agent",
        "route_to_final": "route_to_final"
    }
)

# Route to final handler (using a pass-through node for routing)
graph_builder.add_node("route_to_final", lambda s: {})
graph_builder.add_conditional_edges(
    "route_to_final",
    route_to_final,
    {
        "invoice_handler": "invoice_handler",
        "contract_handler": "contract_handler",
        "report_handler": "report_handler",
        "email_handler": "email_handler"
    }
)

# All final handlers to END
graph_builder.add_edge("invoice_handler", END)
graph_builder.add_edge("contract_handler", END)
graph_builder.add_edge("report_handler", END)
graph_builder.add_edge("email_handler", END)
graph_builder.add_edge("validation_error_handler", END)
graph_builder.add_edge("retry_exhausted_handler", END)

graph = graph_builder.compile()


# ── STEP 11: Unit Tests for Routing Functions ──────────────────

def test_routing_functions():
    """
    Unit tests for all routing functions.
    Demonstrates: testing routing functions independently.
    """
    print("\n" + "=" * 60)
    print("UNIT TESTS FOR ROUTING FUNCTIONS")
    print("=" * 60)
    
    # Test route_after_validation
    print("\n--- Testing route_after_validation ---")
    assert route_after_validation({"valid": True}) == "process_document"
    assert route_after_validation({"valid": False}) == "validation_error_handler"
    assert route_after_validation({}) == "validation_error_handler"  # Missing key test
    print("✓ route_after_validation tests passed")
    
    # Test route_to_final
    print("\n--- Testing route_to_final ---")
    assert route_to_final({"doc_type": "invoice"}) == "invoice_handler"
    assert route_to_final({"doc_type": "contract"}) == "contract_handler"
    assert route_to_final({"doc_type": "report"}) == "report_handler"
    assert route_to_final({"doc_type": "email"}) == "email_handler"
    assert route_to_final({}) == "email_handler"  # Default test
    assert route_to_final({"doc_type": "unknown"}) == "email_handler"  # Default test
    print("✓ route_to_final tests passed")
    
    # Test route_react_loop
    print("\n--- Testing route_react_loop ---")
    assert route_react_loop({"tool_calls_made": 0}) == "react_agent"
    assert route_react_loop({"tool_calls_made": 1}) == "react_agent"
    assert route_react_loop({"tool_calls_made": 2}) == "route_to_final"
    assert route_react_loop({}) == "react_agent"  # Missing key defaults to 0, continues loop
    print("✓ route_react_loop tests passed")
    
    print("\n" + "=" * 60)
    print("ALL ROUTING FUNCTION TESTS PASSED ✓")
    print("=" * 60)


# ── STEP 12: Integration Tests ───────────────────────────────────

if __name__ == "__main__":
    # Run unit tests first
    test_routing_functions()
    
    # Integration tests
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS - FULL WORKFLOW")
    print("=" * 60)
    
    test_documents = [
        ("Invoice #1234 for $5000 - Purchase Order PO-5678", "invoice"),
        ("Contract agreement with terms and conditions - NDA required", "contract"),
        ("Financial report summary showing Q4 metrics and analysis", "report"),
        ("Urgent: Please review the attached document ASAP", "email"),
        ("Short", "invalid"),  # Too short - should fail validation
    ]
    
    for doc, expected_type in test_documents:
        print("\n" + "-" * 60)
        print(f"Document: {doc[:50]}...")
        print(f"Expected type: {expected_type}")
        
        # Set random seed for reproducible testing
        random.seed(42)
        
        initial_state = {
            "document": doc,
            "doc_type": "",
            "subtype": "",
            "valid": False,
            "validation_errors": [],
            "processed": False,
            "attempt": 0,
            "max_attempts": 3,
            "tool_calls_made": 0,
            "extracted_data": {},
            "final_output": "",
            "error": ""
        }
        
        try:
            result = graph.invoke(initial_state, config={"recursion_limit": 20})
            print(f"Type: {result['doc_type']}")
            print(f"Subtype: {result['subtype']}")
            print(f"Valid: {result['valid']}")
            print(f"Processed: {result['processed']}")
            print(f"Tool calls: {result['tool_calls_made']}")
            print(f"Output: {result['final_output'][:100]}...")
            
            if result.get("error"):
                print(f"Error: {result['error']}")
        except Exception as e:
            print(f"Execution failed: {e}")
    
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS COMPLETED")
    print("=" * 60)
    
    # Print graph structure
    print("\n" + "=" * 60)
    print("GRAPH STRUCTURE (Mermaid)")
    print("=" * 60)
    try:
        print(graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"Could not generate mermaid: {e}")
