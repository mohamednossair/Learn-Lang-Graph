# Lessons 9–10 Deep Dive: Production Best Practices & Capstone

---

# Lesson 9 — Production Best Practices: Complete Deep Dive

> **Prerequisite:** Open `lesson_09_best_practices/lesson_09_best_practices.ipynb` and run all cells first.

---

## The Gap Between Demo and Production

A demo agent:
- No logging → impossible to debug failures
- No validation → crashes on bad input
- No retry → fails on first network blip
- No limits → hangs forever in infinite loops
- No error handling → crashes and loses state

A production agent is not more complex — it's the same agent with **10 extra layers of protection**. This lesson adds those layers systematically.

---

## Pillar 1: Structured Logging

### Why structured logging matters

```python
# ❌ Unstructured — useless in production
print("processing started")
print("done")

# ✅ Structured — searchable, parseable, auditable
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("langgraph.agent")
```

### Logging pattern for every node

```python
def my_node(state: MyState) -> dict:
    # ← LOG AT THE START — capture inputs
    logger.info(f"[my_node] START | thread={get_thread_id()} | "
                f"msgs={len(state.get('messages', []))} | "
                f"user={state.get('user_id', 'unknown')}")
    try:
        result = do_work(state)
        # ← LOG SUCCESS — capture outputs
        logger.info(f"[my_node] DONE | result_len={len(str(result))}")
        return result
    except Exception as e:
        # ← LOG ERRORS — always with exception info
        logger.error(f"[my_node] FAILED | error={e}", exc_info=True)
        return {"error": str(e)}
```

### Log levels guide

| Level | When | Example |
|-------|------|---------|
| `DEBUG` | Detailed internals, dev only | SQL query details, state dump |
| `INFO` | Normal operation milestones | Node start/end, routing decisions |
| `WARNING` | Recoverable issues | Retry attempt, slow response |
| `ERROR` | Failures requiring attention | Tool failure, LLM error |
| `CRITICAL` | System-level failures | Database down, auth failure |

### Structured log output for observability

```python
import json
import time

def log_node_execution(node_name: str, state: dict, result: dict, elapsed: float):
    """Emit a structured JSON log entry — compatible with log aggregators."""
    entry = {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level":       "INFO",
        "node":        node_name,
        "thread_id":   state.get("thread_id", "unknown"),
        "user_id":     state.get("user_id", "unknown"),
        "elapsed_ms":  round(elapsed * 1000, 2),
        "keys_changed": list(result.keys()),
    }
    logger.info(json.dumps(entry))
```

---

## Pillar 2: Input Validation with Pydantic

### Why validate at the boundary

```python
# ❌ No validation — crashes deep inside the graph
result = graph.invoke({"question": "", "user_id": None})
# → KeyError or NoneType error 5 nodes deep — hard to trace

# ✅ Validate before the graph — fail fast at the boundary
class AgentInput(BaseModel):
    question: str
    user_id:  str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) < 3:
            raise ValueError(f"Question too short: {len(v)} chars (min 3)")
        if len(v) > 2000:
            raise ValueError(f"Question too long: {len(v)} chars (max 2000)")
        return v

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("user_id is required")
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("user_id must be alphanumeric")
        if len(v) > 50:
            raise ValueError("user_id too long")
        return v.strip()

# Usage
try:
    validated = AgentInput(question="  ", user_id="ahmed")  # ← raises ValueError
except ValueError as e:
    return {"error": str(e), "status": 400}   # reject before entering graph
```

---

## Pillar 3: Structured Output

### Never parse LLM text manually

```python
# ❌ Fragile — LLM format varies, regex breaks
resp = llm.invoke(msgs).content
json_str = re.search(r'\{.*\}', resp, re.DOTALL).group()
data = json.loads(json_str)   # fails if LLM adds ```json or whitespace

# ✅ Reliable — Pydantic schema enforced
from pydantic import BaseModel

class RoutingDecision(BaseModel):
    next_agent:  str
    confidence:  float
    reasoning:   str

class SentimentResult(BaseModel):
    sentiment:   Literal["positive", "negative", "neutral"]
    score:       float   # 0.0 to 1.0
    key_phrases: list[str]

# Usage — guaranteed to return valid RoutingDecision or raise
decision = llm.with_structured_output(RoutingDecision).invoke(messages)
print(decision.next_agent)   # always a string, never None
print(decision.confidence)   # always a float, never a string
```

---

## Pillar 4: Retry with Exponential Backoff

### When to retry and when not to

| Error | Retry? | Reason |
|-------|--------|--------|
| Network timeout | Yes | Transient |
| Rate limit (429) | Yes, with backoff | Will resolve |
| Server error (500) | Yes, limited | May be transient |
| Bad request (400) | No | Your fault, retrying won't help |
| Auth failure (401) | No | Fix credentials |
| Tool logic error | No | Code bug, not transient |

### Complete retry implementation

```python
import time
import random
import logging

logger = logging.getLogger("retry")

def llm_with_retry(llm, messages: list, max_retries: int = 3):
    """Call LLM with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except ConnectionError as e:
            if attempt == max_retries - 1:
                raise   # last attempt — re-raise
            wait = (2 ** attempt) + random.uniform(0, 1)   # exponential + jitter
            logger.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}). "
                          f"Retrying in {wait:.1f}s. Error: {e}")
            time.sleep(wait)
        except Exception as e:
            # Non-retryable error — don't retry
            logger.error(f"LLM call failed with non-retryable error: {e}")
            raise

# In a node:
def agent_node(state: MyState) -> dict:
    try:
        response = llm_with_retry(llm, state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"error": str(e)}   # node never crashes the graph
```

### Using tenacity (production library)

```python
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),   # 1s → 2s → 4s → ... → 60s max
    stop=stop_after_attempt(5),                            # max 5 attempts
    retry=retry_if_exception_type(ConnectionError),        # only retry connection errors
    reraise=True                                           # raise if all attempts fail
)
def call_llm_reliable(messages: list):
    return llm.invoke(messages)
```

---

## Pillar 5: Error Handler Node

### Every production graph needs one

```python
def error_handler_node(state: MyState) -> dict:
    """Safety net — handles any error that reached this node."""
    error = state.get("error", "Unknown error")
    user_id = state.get("user_id", "unknown")

    # 1. Log for debugging
    logger.error(f"[error_handler] Error for user={user_id}: {error}")

    # 2. Alert ops team (production)
    # alert_ops_team(error, user_id, state)

    # 3. Return graceful response
    return {
        "final_answer": "I encountered an issue and couldn't complete your request. "
                       "Please try again or contact support.",
        "error_handled": True
    }

# Wire it up — route to error_handler if error in state
def route_with_error_check(state: MyState) -> str:
    if state.get("error"):
        return "error_handler"
    return "normal_path"

builder.add_conditional_edges("agent", route_with_error_check, {
    "error_handler": "error_handler",
    "normal_path":   "tools"
})
builder.add_edge("error_handler", END)
```

---

## Pillar 6: Recursion Limit

```python
# Always set this — no exceptions
config = {
    "recursion_limit": 25,   # default is 25, set explicitly
    "configurable": {"thread_id": "..."}
}

# For multi-agent systems with many specialists, increase:
config = {"recursion_limit": 50, "configurable": {"thread_id": "..."}}

# On hit: GraphRecursionError is raised — catch it
from langgraph.errors import GraphRecursionError

try:
    result = graph.invoke(state, config=config)
except GraphRecursionError as e:
    return {"error": "Agent exceeded maximum steps. Please simplify your request."}
```

Choosing the right limit:
- Simple ReAct: 15-25 (enough for 3-5 tool calls)
- Database agent: 25-35 (schema inspection + query + retry)
- Multi-agent: 40-60 (supervisor + specialist turns)
- Never: unlimited (production graphs must have a limit)

---

## Pillar 7: Streaming

### Three streaming modes

```python
# Mode 1: "values" — full state after each node (monitoring)
for snapshot in graph.stream(state, config=config, stream_mode="values"):
    msgs = snapshot.get("messages", [])
    print(f"After some node: {len(msgs)} messages")

# Mode 2: "updates" — only changed keys (efficient monitoring)
for node_name, updates in graph.stream(state, config=config, stream_mode="updates"):
    print(f"{node_name} changed: {list(updates.keys())}")

# Mode 3: "messages" — LLM tokens as they arrive (chat UI)
for chunk, metadata in graph.stream(state, config=config, stream_mode="messages"):
    if hasattr(chunk, "content") and chunk.content:
        print(chunk.content, end="", flush=True)   # token-by-token

# Combine modes
for event in graph.stream(state, config=config, stream_mode=["updates", "messages"]):
    ...
```

### Building a streaming chat UI

```python
def stream_response(question: str, user_id: str, thread_id: str) -> Generator:
    """Yield tokens as they arrive from the LLM."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    state  = {"messages": [HumanMessage(content=question)], "user_id": user_id}

    for chunk, metadata in graph.stream(state, config=config, stream_mode="messages"):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content   # each yield = one or more tokens

# In FastAPI (Server-Sent Events):
@app.get("/stream")
async def stream_endpoint(question: str, user_id: str, thread_id: str):
    def event_generator():
        for token in stream_response(question, user_id, thread_id):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## Pillar 8: Parallel Execution with Send()

### When to use parallel execution

Use `Send()` when you have independent tasks that don't depend on each other's results:

```python
# Sequential (slow):  3 tasks × 5 seconds each = 15 seconds total
# Parallel  (fast):   3 tasks × 5 seconds each = 5 seconds total (3x speedup)

from langgraph.types import Send

def fan_out(state: MyState):
    """Launch one worker per item, all running in parallel."""
    return [Send("worker", {"item": item, "context": state["context"]})
            for item in state["items"]]

def worker_node(state: dict) -> dict:
    """Process one item. Runs in parallel with other workers."""
    result = process_item(state["item"])
    return {"results": [result]}   # reducer appends to shared results list

class MyState(TypedDict):
    items:   list
    results: Annotated[list, lambda x, y: x + y]   # merge parallel results
    context: str

builder.add_conditional_edges(START, fan_out, ["worker"])
builder.add_edge("worker", "combine")   # all workers complete before combine runs
```

---

## Pillar 9: Token Cost Management

```python
import tiktoken

def count_tokens_cl100k(messages: list) -> int:
    """Count tokens for models using cl100k_base encoding (GPT-4, Claude)."""
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        total += len(enc.encode(content)) + 4   # 4 tokens per message overhead
    total += 2   # conversation overhead
    return total

def token_guard_node(state: ChatState) -> dict:
    """Trim messages if approaching token limit."""
    messages = state["messages"]
    token_count = count_tokens_cl100k(messages)

    if token_count > 6000:   # 75% of 8K limit
        logger.warning(f"Token count {token_count} exceeds threshold. Trimming.")
        # Keep system messages + last 4 messages
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        recent_msgs  = [m for m in messages if not isinstance(m, SystemMessage)][-4:]
        messages = system_msgs + recent_msgs
        new_count = count_tokens_cl100k(messages)
        logger.info(f"Trimmed from {token_count} to {new_count} tokens")

    return {"messages": messages, "token_count": token_count}
```

---

## Anti-Patterns — Lesson 9

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| **No logging in nodes** | Can't debug production failures | Log at start of every node |
| **No input validation** | Graph crashes deep inside on bad input | Pydantic validation before invoke() |
| **Manual JSON parsing** | Fragile, breaks with LLM variation | `with_structured_output()` |
| **No retry on LLM calls** | Single network blip = failure | Exponential backoff retry |
| **No error handler node** | Exceptions crash the graph | Error handler with graceful response |
| **No recursion_limit** | Agent loops forever | Always set in config |
| **Blocking on invoke()** for streaming | User sees nothing until done | Use `stream_mode="messages"` |
| **Unlimited token history** | Context window exceeded | Token counter + trim |
| **Same model for all tasks** | Expensive and slow for routing | Small model (routing) + large (generation) |

---

## Best Practices Checklist — Lesson 9

```
LOGGING:
  □ Structured logger at module level (not print statements)
  □ Log at start of every node: node name, user_id, message count
  □ Log errors with exc_info=True for full traceback
  □ Log routing decisions in supervisor

VALIDATION:
  □ Pydantic model for all external inputs
  □ Validate before invoke() — reject at boundary
  □ All validators wrapped in try/except with meaningful error messages

LLM CALLS:
  □ with_structured_output() for all JSON responses
  □ Retry wrapper on all LLM calls (3-5 attempts, exponential backoff)
  □ Temperature appropriate for task (0.0 for agents)

GRAPH STRUCTURE:
  □ Error handler node wired to all critical nodes
  □ recursion_limit in every config dict
  □ Token counter node before LLM in long-running conversations

STREAMING:
  □ stream_mode="messages" for chat UIs
  □ stream_mode="updates" for monitoring dashboards
  □ Never use blocking invoke() for interactive applications
```

---

## Tasks — Lesson 9

**Task 9.1 — Production Upgrade**
Take your Lesson 4 ReAct agent and add ALL 9 pillars systematically. Work through them one at a time. After each pillar: run the agent and verify it still works correctly with the new layer added.

**Task 9.2 — Token Budget Enforcer**
Build a node that runs before every LLM call. It counts tokens, logs the count, and if > 3000: trims the oldest non-system messages. Track `total_tokens_used: int` in state (accumulate reducer). After a 10-turn conversation, print the token usage report.

**Task 9.3 — Benchmark: Sequential vs Parallel**
Process 8 "analysis tasks" (each sleeps 1 second to simulate work):
- Sequential: one node loops through all 8 → measure time
- Parallel: `Send()` to 8 worker nodes → measure time
Print: "Sequential: Xs, Parallel: Ys, Speedup: Zx"

**Task 9.4 — Chaos Engineering**
Build an agent where 30% of tool calls fail randomly. Add ALL resilience layers:
- Retry up to 3 times per tool call
- Error handler node
- After 3 failures: route to human review (interrupt)
- Metrics: track `success_count`, `failure_count`, `retry_count` in state
Run 20 questions and print the metrics summary.

---

## Interview Q&A — Lesson 9

**Q1: How do you handle rate limiting (429 errors) from LLM APIs in production?**
Exponential backoff with jitter: `wait = (2 ** attempt) + random.uniform(0, 1)`. The jitter prevents "thundering herd" — multiple retries hitting at the same time. For high-volume systems: (1) request queue with rate limiter, (2) multiple API keys rotated round-robin, (3) fallback to a local model (Ollama) when cloud API is rate-limited, (4) cache frequent queries so repeat questions don't hit the LLM.

**Q2: What streaming mode should you use for each use case?**
`"values"`: yields full state snapshot after every node — use for: monitoring systems, debugging, cases where you need complete state at each step. `"updates"`: yields only changed keys (diff) — use for: efficient state monitoring, logging what each node did. `"messages"`: yields LLM token chunks as they arrive — use for: chat UIs where users see responses typed character by character. Combine: `stream_mode=["updates", "messages"]` for both monitoring and live output simultaneously.

**Q3: How do you implement a production-grade error handling strategy?**
Four layers: (1) **Tool level** — every tool catches exceptions and returns error strings. (2) **Node level** — every node wraps LLM calls in try/except, writes error to state. (3) **Graph level** — `error_handler` node is wired from every critical node; it logs, alerts, and returns graceful message. (4) **Caller level** — catch `GraphRecursionError` and any uncaught exceptions from `invoke()`. Never let exceptions propagate to the user without handling.

**Q4: When would you use `Send()` for parallel execution and when would it be wrong to use it?**
Use `Send()` when: tasks are independent (don't need each other's results), tasks are roughly equal in duration (uneven tasks waste parallel efficiency), the overhead of parallelism is worth it (each task takes >1 second). Do NOT use when: tasks depend on each other's results (use sequential edges), tasks are very fast (<100ms — parallelism overhead exceeds benefit), shared state writes conflict without reducers.

---

# Lesson 10 — Capstone: Integration and System Thinking

> **Prerequisite:** Run `python lesson_10_capstone/lesson_10_capstone.py test` and verify it works.

---

## What the Capstone Teaches

The capstone doesn't introduce new concepts — it teaches you to **see how patterns compose**. The hard skill in senior engineering is not knowing each pattern individually, but knowing how to combine them correctly.

### Every component traces back to a lesson

```
lesson_10_capstone.py
├── StateGraph setup (Lesson 1)
├── Conditional routing (Lesson 2)
├── MessagesState + Ollama (Lesson 3)
├── @tool + ToolNode + ReAct loop (Lesson 4)
├── Supervisor + specialists (Lesson 5)
├── list_tables + describe_table + run_sql (Lesson 6)
├── interrupt() + Command(resume) (Lesson 7)
├── MemorySaver + thread_id (Lesson 8)
└── Logging + retry + recursion_limit (Lesson 9)
```

---

## Architecture Deep Dive

```
User Message
     ↓
[SUPERVISOR] ← Reads conversation context, decides routing
     │
     ├── → [DB_AGENT]        # Has: list_tables, describe_table, run_sql tools
     │        ↓                # Runs: ReAct loop (agent ↔ tools)
     │   Reports back to supervisor
     │
     ├── → [ANALYST]          # Has: calculate_stats, compare_data, format_insight tools
     │        ↓                # Runs: ReAct loop
     │   Reports back to supervisor
     │
     ├── → [HUMAN_REVIEW]    # Has: interrupt() for sensitive actions
     │        ↓                # Pauses: waits for human Command(resume=...)
     │   Reports back to supervisor
     │
     └── → FINISH → END      # Supervisor decides task is complete
```

### The state schema — why every field is there

```python
class CapstoneState(TypedDict):
    messages:       Annotated[list, add_messages]   # full conversation (L3)
    user_id:        str                              # session isolation (L8)
    next_agent:     str                              # supervisor routing (L5)
    needs_approval: bool                             # HITL gate flag (L7)
    error:          str                              # error handler path (L9)
    visited:        Annotated[list, lambda x,y:x+y] # loop prevention (L5)
```

---

## The Supervisor Prompt — Making It Robust

```python
SUPERVISOR_PROMPT = """You are a task router for a company analytics system.

AVAILABLE SPECIALISTS:
- db_agent:      Query database for data (employee counts, salaries, departments, sales)
- analyst:       Analyze and interpret data (trends, comparisons, statistics)
- human_review:  Any action that modifies data, sends messages, or requires approval

ROUTING RULES:
1. For questions ABOUT data → db_agent
2. For questions REQUIRING ANALYSIS of data → analyst (may need db_agent first)
3. For any WRITE operations or sensitive actions → human_review
4. When task is fully answered → FINISH

CONTEXT:
- Specialists already visited: {visited}
- Do NOT route to a specialist already in visited unless their previous output was an error
- If you see an answer in the conversation → route to FINISH

Respond ONLY with valid JSON: {{"next": "agent_name"}}"""
```

### Making supervisor output reliable

```python
def supervisor_node(state: CapstoneState) -> dict:
    visited = state.get("visited", [])
    prompt  = SUPERVISOR_PROMPT.format(visited=visited)
    msgs    = [SystemMessage(content=prompt)] + state["messages"]

    # Use structured output for reliable JSON — never free text
    class Routing(BaseModel):
        next: str

    try:
        decision = llm.with_structured_output(Routing).invoke(msgs)
        agent = decision.next
    except Exception:
        agent = "FINISH"   # safe fallback

    # Validate against allowlist
    allowed = {"db_agent", "analyst", "human_review", "FINISH"}
    if agent not in allowed:
        logger.warning(f"Supervisor returned unknown agent: {agent}")
        agent = "FINISH"

    logger.info(f"[supervisor] Routing to: {agent} | visited: {visited}")
    return {"next_agent": agent}
```

---

## Production Deployment Checklist

Use this before deploying any LangGraph agent to production:

```
STATE:
  □ TypedDict with all fields typed
  □ Annotated reducers on all list fields (messages, visited, results)
  □ user_id field for multi-user tracking
  □ error field for graceful error routing

TOOLS:
  □ Every tool has clear, specific docstring
  □ Every tool catches all exceptions, returns error string
  □ Read-only tools clearly documented as read-only
  □ Sensitive tools (write, send, delete) require interrupt() approval
  □ Database tools: SELECT-only guard + LIMIT enforced

GRAPH:
  □ Error handler node wired from all critical nodes
  □ recursion_limit set in every invoke() config
  □ Checkpointer attached (SqliteSaver minimum for production)
  □ thread_id in every invoke() config

NODES:
  □ Logging at start of every node (node name, user_id, message count)
  □ try/except in every node calling external services
  □ Retry logic for all LLM calls

INPUT/OUTPUT:
  □ Pydantic validation before invoke()
  □ with_structured_output() for all JSON responses
  □ streaming for interactive applications

TESTING:
  □ Unit test for every tool function (all edge cases)
  □ Integration test for every supervisor routing path
  □ HITL interrupt → resume flow tested end-to-end
  □ Persistence tested: state survives process restart
  □ Recursion limit tested: graph stops cleanly at limit
  □ Error handler tested: graph recovers from node failures

SECURITY:
  □ No credentials in code (use environment variables)
  □ SQL injection prevention (SELECT-only + parameterized queries)
  □ User data isolation (thread_id per user)
  □ GDPR delete endpoint implemented
  □ API authentication (if exposed via API)
```

---

## Anti-Patterns — Lesson 10

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| **Supervisor does domain work** | Supervisor should only route | Supervisor returns routing only — never answers directly |
| **No visited tracking** | Supervisor loops between two agents | `visited: Annotated[list, append]` + checked in prompt |
| **MemorySaver in production** | State lost on restart | SqliteSaver for production |
| **No recursion_limit** | Multi-agent can loop 100+ times | Set limit: 40-60 for multi-agent |
| **Agents don't report back** | Supervisor loses track | Every specialist → supervisor edge |
| **HITL for every action** | Kills automation value | Only interrupt for genuinely risky actions |
| **No error handler** | One node failure = crash | Error handler wired from every agent |

---

## Tasks — Lesson 10

**Task 10.1 — Add HR Specialist**
`hr_agent` handles: hire dates, department transfers, org chart queries, headcount by team.
Tools: `get_hire_date(employee_id)`, `get_org_chart()`, `list_employees_by_dept(dept)`.
Update supervisor prompt and routing. Test 3 HR-specific questions.

**Task 10.2 — Upgrade to SqliteSaver**
Replace MemorySaver in capstone with SqliteSaver.
1. Verify: 3 turns → restart Python → run again → history persists
2. Add `--list-sessions` CLI flag: `python lesson_10_capstone.py --list-sessions`
3. Add `--clear` CLI flag: deletes the checkpoints DB

**Task 10.3 — Production Monitoring**
Add a `metrics: dict` field to state with `add` reducer.
Track: llm_calls, tool_calls, routing_decisions, errors, total_tokens.
After every `invoke()`, print:
```
=== Session Metrics ===
LLM calls:        8
Tool calls:       5
Routing decisions: 3
Errors:           0
Tokens used:      ~2400
```

**Task 10.4 — System Load Test**
Run 10 different questions through the capstone in sequence.
Measure: total time, average time per question, success rate.
Categorize which agents handled each question.
Identify which question type is slowest (schema discovery overhead?).

---

## Interview Q&A — Lesson 10

**Q1: Walk me through how you would debug a production capstone agent that's giving wrong answers.**
Step-by-step: (1) Check logs — which agent did the supervisor route to? (2) Check tool calls in message history — what SQL was generated? (3) Run the SQL manually and verify the result is correct. (4) If SQL is correct but answer is wrong, the LLM's interpretation failed — improve the agent's system prompt. (5) If SQL is wrong, check if `describe_table` was called first — if not, the schema discovery phase is broken. (6) Use `get_state_history(config)` to replay the exact execution and see state at each step.

**Q2: How would you scale the capstone for 500 concurrent users?**
(1) Replace MemorySaver with PostgresSaver — shared state across server instances. (2) Deploy 3-5 API server instances (FastAPI + uvicorn) behind a load balancer (nginx/AWS ALB). (3) Use connection pooling for the company database (SQLAlchemy pool). (4) Redis cache for frequent SQL queries (same question = same result, no LLM needed). (5) Use `Send()` to parallelize supervisor routing for multi-part questions. (6) Separate LLM server (Ollama) from API server — scale each independently.

**Q3: How do you maintain the capstone agent as requirements change?**
(1) New specialist: add node + update supervisor prompt + add routing edge — no other changes needed. (2) New database table: no agent changes — `list_tables` + `describe_table` discover it automatically. (3) Changed schema: no agent changes — schema inspection happens at query time. (4) New approval requirement: add `interrupt()` to relevant tool — no graph structure change. (5) Track all changes: keep a CHANGELOG.md, version the supervisor prompt, store version in state metadata.
