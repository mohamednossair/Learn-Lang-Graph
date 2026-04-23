# Lesson 2 — Conditional Edges & Branching: Complete Deep Dive

> **Prerequisite:** Open `lesson_02_conditional/lesson_02_conditional.ipynb` and run all cells first.

---

## Real-World Analogy

Think of **airport immigration control**:
- Every passenger goes through a **classifier** (officer checks passport type)
- Based on classification: citizen → fast lane, resident → standard, visitor → full inspection
- The officer's decision logic = the routing function
- Each lane = a different handler node
- There is ALWAYS a lane assignment — no passenger is left unrouted (no missing default)

---

## How Conditional Edges Work — Under the Hood

```python
builder.add_conditional_edges(
    "classify",           # (1) After this node runs...
    route_by_sentiment,   # (2) LangGraph calls this function with current state
    {                     # (3) Maps return value → next node name
        "positive": "handle_positive",
        "negative": "handle_negative",
        "neutral":  "handle_neutral",
    }
)
```

Internal execution order:
```
1. "classify" node runs → updates state["sentiment"] = "positive"
2. LangGraph calls route_by_sentiment(current_state)
3. route_by_sentiment reads state["sentiment"] → returns "positive"
4. LangGraph looks up "positive" in mapping → finds "handle_positive"
5. LangGraph executes "handle_positive" node
6. Continues from there
```

**The mapping dict is optional.** If your routing function already returns exact node names, omit it:
```python
# These are identical if route() returns "handle_positive", "handle_negative", "handle_neutral"
builder.add_conditional_edges("classify", route)
builder.add_conditional_edges("classify", route, {"handle_positive": "handle_positive", ...})
```

Use the mapping dict when you want **semantic routing names** that differ from node names:
```python
def route(state) -> Literal["high", "medium", "low"]:   # semantic names
    ...

builder.add_conditional_edges("triage", route, {
    "high":   "emergency_room",     # semantic → actual node
    "medium": "urgent_care",
    "low":    "general_practice",
})
```

---

## The Routing Function — Complete Guide

### Anatomy of a perfect routing function

```python
from typing import Literal

def route_request(state: RequestState) -> Literal["db_agent", "analyst", "human", "finish"]:
    """
    Route request to appropriate handler.
    Called by LangGraph after the classify node runs.
    """
    # 1. Always check error state first
    if state.get("error"):
        return "human"

    # 2. Main routing logic — use .get() with defaults to avoid KeyError
    req_type = state.get("request_type", "").lower().strip()

    if req_type == "data_query":  return "db_agent"
    if req_type == "analysis":    return "analyst"
    if req_type == "approval":    return "human"

    # 3. ALWAYS have an explicit default — never return None
    return "finish"
```

### Rules for routing functions

| Rule | Why |
|------|-----|
| Use `Literal[...]` return type | LangGraph validates all paths at compile time |
| Always have a default `return` | Missing default → `InvalidUpdateError` at runtime |
| Use `.get("key", default)` | Prevents `KeyError` if key not set |
| Pure function — read state only | No side effects, no external calls |
| Extract to named function | Not inline lambda — testable independently |
| Never call LLM inside routing | Slow; do LLM classification in a node, routing reads result |

### What happens with a missing default — the most common bug

```python
# ❌ DANGEROUS
def route(state) -> Literal["a", "b"]:
    if state["x"] == "a": return "a"
    if state["x"] == "b": return "b"
    # If state["x"] is anything else → returns None → InvalidUpdateError

# ✅ SAFE
def route(state) -> Literal["a", "b", "default"]:
    if state["x"] == "a": return "a"
    if state["x"] == "b": return "b"
    return "default"   # explicit catch-all
```

---

## 6 Essential Branching Patterns

### Pattern 1: N-way Classification (most common)

```
classify → route() → handler_A → END
                   → handler_B → END
                   → handler_C → END
```

```python
builder.add_node("classify", classify_fn)
builder.add_node("handler_a", handler_a_fn)
builder.add_node("handler_b", handler_b_fn)
builder.add_node("handler_c", handler_c_fn)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route, {"a": "handler_a", "b": "handler_b", "c": "handler_c"})
builder.add_edge("handler_a", END)
builder.add_edge("handler_b", END)
builder.add_edge("handler_c", END)
```

### Pattern 2: Validation Gate (pass/fail)

```
validate → route() → process → END   (valid)
                   → error   → END   (invalid)
```

```python
def route_validation(state) -> Literal["process", "error"]:
    return "error" if state.get("validation_error") else "process"
```

### Pattern 3: ReAct Loop (run until done)

```
agent → route() → tools → agent  (loop — has tool calls)
               → END             (done — no tool calls)
```

```python
def should_continue(state) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")   # ← this edge creates the loop
```

### Pattern 4: Retry with Limit

```
process → route() → retry_node → process  (retry — attempt < max)
                  → success    → END      (done — has result)
                  → give_up    → END      (max attempts reached)
```

```python
def route_retry(state) -> Literal["retry", "success", "give_up"]:
    if state["result"]:                          return "success"
    if state["attempt"] >= state["max_attempt"]: return "give_up"
    return "retry"
```

### Pattern 5: Fan-out (parallel branches) — advanced

```
START → fan_out → worker_1 ↘
                → worker_2 → merge → END
                → worker_3 ↗
```

```python
from langgraph.types import Send

def fan_out(state):
    return [Send("worker", {"item": item}) for item in state["items"]]

builder.add_conditional_edges(START, fan_out, ["worker"])
builder.add_edge("worker", "merge")
```

### Pattern 6: Two-Level Routing (hierarchical)

```
classify_dept → route_dept() → engineering → classify_task → route_task() → bug_handler
                             → sales       →                              → feature_handler
                             → hr          → hr_handler
```

---

## Before vs After — Routing Examples

### Before: Logic inside node (wrong approach)

```python
# ❌ Node doing both work AND routing
def classify_and_route(state: MyState) -> dict:
    sentiment = detect_sentiment(state["text"])
    if sentiment == "positive":
        response = "Thank you! We're glad you're happy."
    elif sentiment == "negative":
        response = "We're sorry to hear that."
    else:
        response = "Thank you for your feedback."
    return {"sentiment": sentiment, "response": response}

# Problems:
# 1. Node has multiple responsibilities (classify + respond)
# 2. Routing logic hidden inside node — invisible in graph diagram
# 3. Hard to test each response independently
# 4. Can't add new handlers without modifying the node
```

### After: Separation of concerns (correct approach)

```python
# ✅ Node does ONE thing: classify
def classify_node(state: MyState) -> dict:
    return {"sentiment": detect_sentiment(state["text"])}

# ✅ Routing function reads result — pure
def route_by_sentiment(state: MyState) -> Literal["positive", "negative", "neutral"]:
    return state["sentiment"]

# ✅ Each handler does ONE thing: respond
def handle_positive(state: MyState) -> dict:
    return {"response": "Thank you! We're glad you're happy."}

def handle_negative(state: MyState) -> dict:
    return {"response": "We're sorry to hear that."}

def handle_neutral(state: MyState) -> dict:
    return {"response": "Thank you for your feedback."}

# Benefits:
# 1. Each function has one responsibility
# 2. Routing visible in graph diagram
# 3. Each handler testable independently
# 4. Add new handler = add one node + update routing
```

---

## Anti-Patterns — What NOT to Do

| Anti-pattern | Code | Why it's wrong | Fix |
|-------------|------|----------------|-----|
| **No default return** | Routing ends without returning | `InvalidUpdateError` at runtime | Add `return "default"` as last line |
| **Lambda routing** | `add_conditional_edges("n", lambda s: "a" if s["x"] else "b")` | Untestable, unreadable | Extract to named function |
| **KeyError in routing** | `state["type"]` (may not exist) | Crash during routing | `state.get("type", "default")` |
| **LLM call in routing** | Slow, unpredictable | Routing should be instant | Do LLM work in a node, routing reads result |
| **Return None** | `if x: return "a"` (no else) | `InvalidUpdateError` | Always cover all cases |
| **Missing Literal hint** | Return type not declared | No compile-time validation | Use `Literal["a", "b", "c"]` |

---

## Debugging Guide — Lesson 2

### Problem: `InvalidUpdateError` during graph execution

**Cause:** Routing function returned a string not in the mapping dict, or returned `None`.
```python
# Debug: print what routing returns
def route(state):
    result = my_routing_logic(state)
    print(f"[ROUTING] returning: '{result}'")   # add this
    return result
```

### Problem: Graph always takes the same path

**Cause:** Classification node not writing to state correctly, OR routing reads wrong key.
```python
# Check classification output
for step in graph.stream(state, stream_mode="updates"):
    print(step)   # see what "classify" actually wrote
```

### Problem: `ValueError: Node 'X' not found`

**Cause:** String in mapping dict doesn't match a registered node name (typo).
```python
# Check: registered names vs mapping names
builder.add_node("handle_positive", ...)      # registered name
{"positive": "handle_positiv"}                # ← typo! missing 'e'
```

### Problem: Routing function crashes with KeyError

**Cause:** Reading a state key that's not set yet.
```python
# Fix: always use .get() with default
state["sentiment"]          # ❌ crashes if "sentiment" not set
state.get("sentiment", "")  # ✅ returns "" if not set
```

---

## Best Practices Checklist — Lesson 2

```
ROUTING FUNCTION:
  □ Return type uses Literal["a", "b", "c"] — all options listed
  □ Always has an explicit default return (last else clause)
  □ Uses state.get("key", default) — never state["key"]
  □ Pure function — reads state only, no external calls, no side effects
  □ Extracted to named function — not an inline lambda
  □ Can be unit tested independently without running the full graph

GRAPH STRUCTURE:
  □ All values in mapping dict match registered node names exactly
  □ Every conditional destination node has an exit path (→ END or → other node)
  □ Classification happens in a NODE, not in the routing function
  □ Each handler node does ONE thing (single responsibility)

TESTING:
  □ Test routing function directly: assert route({"sentiment": "positive"}) == "positive"
  □ Test each path end-to-end with invoke()
  □ Test the default/fallback path explicitly
  □ Test with unexpected input values to verify default is hit
```

---

## Tasks

### Task 2.1 — Support Ticket Router
State: `{ticket: str, priority: str, assigned_to: str, sla_hours: int}`
Keyword lists:
- High: "urgent", "critical", "down", "broken", "emergency"
- Medium: "slow", "issue", "problem", "error", "bug"  
- Low: everything else

Routes: high → `senior_engineer` (SLA 4h), medium → `engineer` (8h), low → `intern` (24h).
Each handler sets both `assigned_to` and `sla_hours`.
Test: verify all 3 paths. What happens if ticket is empty string?

### Task 2.2 — Multi-Condition Validator
State: `{value: int, name: str, email: str, errors: list}`
Build a chain of 3 validators, each routing to either the next validator (pass) or an error collector:
- `validate_range`: value must be 1-100
- `validate_name`: name length 2-50, letters and spaces only
- `validate_email`: must contain "@" and ".", length < 100
Error collector appends to `errors` list. After all validators, a final `report_node` produces a summary.

### Task 2.3 — Retry Loop
State: `{query: str, result: str, attempt: int, max_attempts: int}`
Simulate a flaky API: `import random; success = random.random() > 0.6`
Graph: `process → route() → retry → process (loop) | success → format → END | give_up → END`
Track all attempts. Print which attempt succeeded.

### Task 2.4 — Hierarchical Router (Two Levels)
State: `{department: str, task_type: str, assigned_to: str, sla: str}`
Level 1: engineering / sales / hr
Level 2 (engineering only): bug / feature / review / deploy
Each of the 6 terminal nodes sets `assigned_to` and `sla` differently.
Test all paths. Visualize with `draw_mermaid()`.

### Task 2.5 — Build and Test Routing Functions
For each routing function below, write 5 unit tests (use `assert`):
```python
def route_priority(state) -> Literal["high", "medium", "low"]:
    score = state.get("urgency_score", 0)
    if score >= 8:   return "high"
    elif score >= 4: return "medium"
    return "low"
```
Test: score=10, score=8, score=4, score=0, score=-1, score=None (missing key case).
Which tests fail? Fix the routing function.

---

## Interview Q&A — Lesson 2

**Q1: What is the difference between `add_conditional_edges()` and putting if/else logic inside a node?**

Conditional edges make branching a first-class graph feature. Benefits: (1) **Visible structure** — `draw_mermaid_png()` shows all possible paths, making the graph self-documenting. (2) **Single responsibility** — nodes do work, routing functions make decisions. (3) **Independent testability** — routing functions are plain Python functions, testable with simple `assert` statements without running the graph. (4) **Compile-time validation** — LangGraph can detect dead paths and unreachable nodes. Putting if/else inside a node works but hides routing logic, mixes concerns, and makes the graph opaque.

**Q2: Can a routing function return `END` directly to immediately terminate?**

Yes. Map `END` as a destination: `{"abort": END, "continue": "next_node"}`. The routing function returns the string `"abort"` which maps to `END`. Alternatively, without a mapping dict, the routing function can return `END` directly (the sentinel object, not the string). Use this for "early exit" patterns: a validation failure that should immediately stop the graph without going through other nodes.

**Q3: How do you handle a routing function that needs to make a complex decision requiring multiple state fields?**

Routing functions can read any combination of state fields:
```python
def route_complex(state) -> Literal["fast_path", "slow_path", "error_path"]:
    if state.get("error"):                           return "error_path"
    if state["score"] > 0.9 and not state["needs_review"]: return "fast_path"
    return "slow_path"
```
The key is that the routing function only READS — all the computation that produces those fields should happen in classification/processing nodes that ran before this routing point.

**Q4: What happens when routing returns a value not in the mapping dict?**

LangGraph raises `InvalidUpdateError` (or similar) at runtime, crashing the graph execution. The error message shows the invalid return value and the expected options. Prevention: (1) Always have a default `return` statement, (2) Use `Literal[...]` type hints — some IDEs and type checkers will warn if your routing function can return a value not in the Literal options.

**Q5: How do you implement a routing function that routes based on LLM output?**

Two-step approach — never call LLM inside the routing function itself:
```python
# Step 1: Node calls LLM to classify
def classify_node(state: MyState) -> dict:
    resp = llm.with_structured_output(Classification).invoke(state["messages"])
    return {"category": resp.category}   # writes result to state

# Step 2: Routing function reads the classification (instant, pure)
def route(state: MyState) -> Literal["billing", "technical", "general"]:
    return state.get("category", "general")   # reads what classify_node wrote
```
This keeps routing pure and fast, while the LLM work stays in a node.

**Q6: How do you make a routing function that can route back to a PREVIOUS node (create a cycle)?**

```python
# Graph structure:
builder.add_node("generate", generate_fn)
builder.add_node("validate", validate_fn)
builder.add_conditional_edges("validate", route, {
    "valid":   "publish",   # forward
    "invalid": "generate",  # ← BACK to generate (cycle)
    "abort":   END          # exit
})
builder.add_edge("generate", "validate")   # validate always follows generate

# CRITICAL: always set recursion_limit to prevent infinite cycles
config = {"recursion_limit": 10}
graph.invoke(state, config=config)
```

**Q7: How do you visualize and verify that all routing paths are correctly wired?**

```python
# Method 1: Mermaid diagram
print(graph.get_graph().draw_mermaid())
# Paste at https://mermaid.live — visually verify all nodes and edges

# Method 2: Stream execution and observe which path was taken
for step in graph.stream(test_state, stream_mode="updates"):
    node_name = list(step.keys())[0]
    print(f"Executed: {node_name}")

# Method 3: Check state after invoke
result = graph.invoke(test_state)
print(f"Routing took path through: {result['path_taken']}")
# (requires path_taken field in state, updated by each handler)

# Method 4: Unit test each path
for test_input, expected_path in test_cases:
    result = graph.invoke(test_input)
    assert result["handled_by"] == expected_path
```
