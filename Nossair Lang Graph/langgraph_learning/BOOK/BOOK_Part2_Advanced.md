# LangGraph Senior Engineer Guide — Part 2: Advanced (Lessons 6–10)

---

## LESSON 6 — Database Agent (Natural Language → SQL)

### Theory

#### Why a database agent?
Business data lives in databases. A database agent lets anyone ask "What was revenue last quarter?" and get accurate SQL-backed answers — no SQL knowledge required.

#### Two-Phase Strategy (Never Skip Phase 1)

```
Phase 1 — Schema Discovery:
  list_tables()          → what tables exist?
  describe_table(name)   → exact column names + types + sample rows
  get_relationships()    → how tables JOIN to each other

Phase 2 — Query + Answer:
  run_sql(query)         → execute the SELECT
  interpret results      → natural language answer
```

**Why Phase 1?** LLMs hallucinate column names with high frequency. Schema discovery eliminates this completely.

#### Read-Only Safety Guard
```python
@tool
def run_sql(query: str) -> str:
    """Execute a SELECT query. READ-ONLY only."""
    if not query.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT allowed. Use request_data_change for modifications."
    # execute...
```

This one check prevents any accidental `DROP TABLE`, `DELETE`, or `UPDATE`.

#### SQL rules for agents
| Rule | Why |
|------|-----|
| Always LIMIT results | Prevent millions of rows |
| Use explicit column names | `SELECT name, salary` not `SELECT *` |
| Describe table before querying | No hallucinated columns |
| Catch SQL errors, return string | Never crash |
| Use JOINs explicitly | Agent must know relationships |

#### Connecting to real databases
The pattern is identical for all databases — just swap the connection:
```python
# PostgreSQL:  psycopg2.connect(host=..., database=..., user=..., password=...)
# MySQL:       mysql.connector.connect(...)
# SQLite:      sqlite3.connect("file.db")
# All use the same cursor.execute(query) interface
```

---

### Tasks — Lesson 6

**Task 6.1 — Extend Schema**
Add `projects(id, name, department_id, budget, status, start_date)` and `project_assignments(employee_id, project_id, role, hours_per_week)`. Insert 5 projects and 10 assignments.
Ask: "Which employees work on more than 2 projects?" and "What is the total active project budget?"

**Task 6.2 — Complex Aggregations**
Ask the agent:
- "Top 3 departments by average salary with headcount"
- "Employee with highest revenue-to-salary ratio"
- "Monthly sales trend for 2024"

**Task 6.3 — Error Recovery**
Ask: "Show me employee performance scores" (column doesn't exist).
Agent should: detect SQL error → re-inspect schema → honestly respond the data doesn't exist.

**Task 6.4 — Multi-Database Agent**
Build an agent that queries two separate SQLite files: `company.db` and `inventory.db`.
Agent must detect which DB the question is about before querying.

**Task 6.5 — Report Generator**
Agent queries top metrics, composes a markdown report, saves to `reports/weekly_YYYY-MM-DD.md` using a `save_report` tool.

---

### Interview Q&A — Lesson 6

**Q: How do you prevent SQL injection from LLM-generated queries?**
Two layers: (1) operation whitelist — only `SELECT` is allowed, all others rejected immediately. (2) For WHERE clause user values, use parameterized queries: `cursor.execute("SELECT * FROM t WHERE name=?", (value,))` — never string format.

**Q: How do you handle a database with 200 tables?**
Two strategies: (1) semantic table routing — embed the question, find top-5 most similar table names using cosine similarity, only inspect those. (2) category-based routing — group tables by domain (sales/HR/inventory) and route to the right category first using a classifier tool.

**Q: How do you cache frequently asked questions?**
Add `query_cache: dict` to state mapping question hash → answer. Before the LLM call, check if question exists in cache. For semantic similarity (same meaning, different wording), use a vector store. Update cache after each new answer.

**Q: What happens when the LLM generates invalid SQL?**
The `run_sql` tool catches `sqlite3.Error` (or equivalent) and returns the error message as a string. The LLM reads this, understands what went wrong (e.g., "no such column"), re-inspects the schema, and generates a corrected query. This is the self-healing loop.

---

## LESSON 7 — Human-in-the-Loop (HITL)

### Theory

#### What is HITL?
Pausing the automated workflow at critical points and giving a human control. Essential for:
- **Safety** — prevent irreversible actions (delete data, send emails, spend money)
- **Quality** — review AI output before it reaches customers
- **Compliance** — some decisions legally require human sign-off
- **Feedback** — correct the AI in real time

#### interrupt() mechanism
```python
from langgraph.types import interrupt

def my_node(state):
    # Graph PAUSES HERE — state saved to checkpointer
    human_answer = interrupt({
        "message": "Please review this action",
        "data": state["pending_action"]
    })
    # Resumes here after Command(resume=answer)
    if human_answer == "approve":
        return {"result": "executed"}
    return {"result": "rejected"}
```

#### Requirements
`interrupt()` **requires** a checkpointer. Without one: `RuntimeError`.
```python
graph = builder.compile(checkpointer=MemorySaver())  # required
```

#### The full resume flow
```python
# Step 1: Run — pauses at interrupt
graph.invoke(initial_state, config={"configurable": {"thread_id": "t1"}})

# Step 2: Inspect interrupt
state = graph.get_state(config)
if state.next:  # non-empty = paused
    data = state.tasks[0].interrupts[0].value  # show to human

# Step 3: Resume with human's decision
from langgraph.types import Command
graph.invoke(Command(resume="approve"), config=config)
```

#### 3 Production HITL Patterns

**Pattern A — Approve/Reject before action**
```
agent wants to call sensitive tool → interrupt → human says yes/no → execute or skip
```
Best for: send email, delete data, charge payment, run code.

**Pattern B — Collect missing information**
```
agent detects missing data → interrupt asking for it → human provides → agent continues with data
```
Best for: personalization, user preferences, missing context.

**Pattern C — Review and edit output**
```
agent generates draft → interrupt showing draft → human approves or gives feedback → revise → repeat
```
Best for: content generation, decisions, customer-facing responses.

#### When NOT to use HITL
| Situation | HITL? |
|-----------|-------|
| High-volume automation (>100/hour) | No — kills throughput |
| Read-only operations | No — no risk |
| Legally required decisions | Yes |
| Irreversible expensive actions | Yes |
| Content shown to customers | Optional |

---

### Tasks — Lesson 7

**Task 7.1 — Budget Approval Workflow**
Purchasing agent proposes item + cost. If cost > $500: interrupt for approval (show item, cost). If ≤ $500: auto-approve. Track all purchases in `purchase_log` state field.

**Task 7.2 — Content Moderation**
`generator_node` writes social media post. `moderation_node` checks forbidden words. If clean: publish (mock). If flagged: interrupt showing content. Human edits, resumes with edited version.

**Task 7.3 — Multi-Step Approval Chain**
Loan application: initial_review → [interrupt: loan officer] → credit_check → [interrupt: senior officer] → decision. Each interrupt shows appropriate data for that stage.

**Task 7.4 — HITL Timeout**
If human doesn't respond within 30 seconds, auto-reject. Implement using `threading.Timer` that calls `graph.invoke(Command(resume="timeout"), config)` after 30 seconds.

---

### Interview Q&A — Lesson 7

**Q: interrupt() vs input() inside a node — what's the difference?**
`input()` blocks the Python process entirely — impossible for web UIs or async systems. `interrupt()` suspends the graph cleanly, saves state to checkpointer, frees the process. The graph can be resumed hours later from a different process or server. This is the fundamental difference between toy demos and production systems.

**Q: Can you have multiple interrupts in one graph run?**
Yes. Each `interrupt()` is one pause point. When resumed, execution continues to the next `interrupt()` if there is one. Only one interrupt is active at a time per thread. Multiple sequential interrupts are each handled one at a time.

**Q: How do you pass the interrupt return value back to the graph?**
`interrupt(value)` returns whatever was passed to `Command(resume=return_value)`. Example:
```python
choice = interrupt({"question": "Approve?"})  # pauses
# Later: graph.invoke(Command(resume="yes"), config)
# choice is now "yes"
```

**Q: How does HITL work in a web application?**
(1) User submits request → graph runs until interrupt → interrupt data saved in checkpointer. (2) API returns "pending approval" to frontend with `thread_id`. (3) Human reviews in the UI. (4) Human clicks approve → API calls `graph.invoke(Command(resume="approve"), config)`. (5) Graph resumes, completes, API returns final result.

---

## LESSON 8 — Persistent Memory & Checkpointers

### Theory

#### Memory hierarchy
```
None (default)  → Stateless — every invoke() starts fresh
MemorySaver     → In-RAM — lost on restart — dev/testing only
SqliteSaver     → Disk-based — survives restarts — single server
PostgresSaver   → DB-backed — multi-server — production scale
Custom saver    → Implement BaseCheckpointSaver for any backend
```

#### thread_id — Session isolation
```python
# User A — isolated conversation
config_a = {"configurable": {"thread_id": "user-a-001"}}

# User B — completely separate — never sees A's data
config_b = {"configurable": {"thread_id": "user-b-001"}}
```
Every thread_id = one isolated session. This is how multi-user systems are built.

#### What the checkpointer saves
After every node execution, it saves:
- Full state snapshot (all fields and values)
- Which node just ran
- Which node runs next
- Timestamps and run metadata
This enables **time-travel debugging** — replay any past state.

#### State history
```python
for snapshot in graph.get_state_history(config):
    print(snapshot.values)    # state at that point
    print(snapshot.next)      # what runs next
    print(snapshot.metadata)  # when it happened
```

#### Short-term vs Long-term memory
| | Short-term (messages) | Long-term (profile) |
|---|----------------------|---------------------|
| What | Conversation history | Facts about the user |
| Storage | `messages` list | Separate `user_profile` dict |
| Grows | Every turn | Only when new facts learned |
| Passed to LLM | Full history | As SystemMessage context |
| Lives | Current thread | Across all threads |

---

### Tasks — Lesson 8

**Task 8.1 — Persistent Notebook**
State: `{messages, notes: list, tags: dict}` with SqliteSaver.
- "add note: X" → appends to notes
- "show notes" → lists from state (no LLM needed)
- "search: X" → finds notes by keyword
Restart Python and verify notes persist.

**Task 8.2 — Multi-User Profile System**
Each user has thread_id. Chatbot stores name, language, preferences. Implement `list_all_sessions()` reading all thread IDs from checkpointer.

**Task 8.3 — Time-Travel Debugging**
Build a 5-step calculation graph. After it runs, use `get_state_history()` to: print all 5 snapshots, then re-run from checkpoint 3 (fork execution), compare the two paths.

**Task 8.4 — Session Cleanup**
Implement `cleanup_old_sessions()` that reads all threads from SqliteSaver, deletes threads older than 30 days, prints count of cleaned sessions.

---

### Interview Q&A — Lesson 8

**Q: MemorySaver vs SqliteSaver in production?**
`MemorySaver` stores state in RAM — lost on restart, can't be shared between processes, grows until OOM. Use only for development. `SqliteSaver` writes to disk — persists across restarts, survives deployments, but single-file so only one writer at a time. For multi-server production, use `PostgresSaver` or a cloud-backed custom saver.

**Q: How does LangGraph retrieve the right state for 10,000 users efficiently?**
The checkpointer uses `thread_id` as the primary key with a B-tree index. Retrieval is O(log n). Every `invoke()` call passes `config = {"configurable": {"thread_id": "..."}}` and the checkpointer loads only that thread's state.

**Q: How do you implement GDPR "right to be forgotten"?**
Add `delete_user_data(thread_id)` that calls the checkpointer's delete methods to remove all checkpoints for that thread. For SqliteSaver: `DELETE FROM checkpoints WHERE thread_id = ?`. Also delete any vector store entries (semantic memory) linked to that user.

**Q: How do you fork execution from a past checkpoint?**
Get a past snapshot: `snapshot = list(graph.get_state_history(config))[N]`. Then invoke with `config = {**config, "configurable": {**config["configurable"], "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"]}}`. LangGraph runs from that checkpoint forward with a new execution path.

---

## LESSON 9 — Production Best Practices

### Theory

#### The 9 Production Pillars

**1. Structured Logging**
```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent")

def log_node(name, state):
    logger.info(f"NODE={name} | thread={config.get('thread_id','?')} | msgs={len(state.get('messages',[]))}")
```
Call `log_node` at start of **every** node. Creates an audit trail.

**2. Input Validation with Pydantic**
```python
class UserQuery(BaseModel):
    question: str
    model_post_init(self, _):
        if len(self.question.strip()) < 3: raise ValueError("Too short")
        if len(self.question) > 2000: raise ValueError("Too long")
```
Validate before entering graph. Reject early, fail loudly.

**3. Structured Output**
```python
class SentimentResult(BaseModel):
    sentiment: str        # positive/negative/neutral
    confidence: float     # 0.0 to 1.0
    summary: str

result = llm.with_structured_output(SentimentResult).invoke(messages)
# result is GUARANTEED to be a valid SentimentResult
```
Never parse LLM text with regex or `json.loads()` — use `with_structured_output()`.

**4. Retry with Exponential Backoff**
```python
for attempt in range(MAX_RETRIES):
    try:
        return llm.invoke(messages)
    except Exception as e:
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
        else:
            return {"error": str(e)}
```

**5. Error Handler Node**
```python
builder.add_node("error_handler", error_handler_node)
builder.add_conditional_edges("agent",
    lambda s: "error_handler" if s.get("error") else "end",
    {"error_handler": "error_handler", "end": END})
builder.add_edge("error_handler", END)
```
Every agent graph should have an error handler as a safety net.

**6. Recursion Limit**
```python
config = {"recursion_limit": 25}  # default is 25
```
Circuit breaker against infinite loops. Raises `GraphRecursionError` when hit.

**7. Streaming modes**
| Mode | What you get | Use for |
|------|-------------|---------|
| `"values"` | Full state after each node | Monitoring |
| `"updates"` | Only changed keys | Efficient monitoring |
| `"messages"` | LLM tokens as they arrive | Chat UIs |

**8. Parallel Execution with Send()**
```python
from langgraph.types import Send
def fan_out(state):
    return [Send("worker", {"item": item}) for item in state["items"]]
# All workers run in parallel — results merged by reducers
```

**9. Token Cost Management**
- Trim history to last N messages
- Count tokens with `tiktoken` before LLM call
- Cache repeated queries
- Use small models (llama3:8b) for routing, large for generation

---

### Tasks — Lesson 9

**Task 9.1 — Production Upgrade**
Take your Lesson 4 ReAct agent. Add ALL 9 pillars: logging, Pydantic validation, retry with backoff, error handler node, recursion limit 20, streaming output.

**Task 9.2 — Token Counter Node**
Node runs before every LLM call. Counts tokens with `tiktoken`. If > 3000: trim oldest messages. Log token count before and after. Track `total_tokens_used: int` in state.

**Task 9.3 — Parallel vs Sequential Benchmark**
Process 10 items: compare sequential (loop in one node) vs parallel (Send() to 10 worker nodes). Measure and print the time difference.

**Task 9.4 — Metrics Collector**
Add metrics tracking to state: `llm_call_count`, `avg_response_time`, `tool_call_count`, `error_count`. After each `invoke()`, print a summary report.

---

### Interview Q&A — Lesson 9

**Q: How do you handle 429 rate limiting from LLM providers?**
Exponential backoff with jitter: `wait = (2 ** attempt) + random.uniform(0, 1); time.sleep(wait)`. Use the `tenacity` library for production: `@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))`. For Ollama, handle `ConnectionError` — the model may still be loading.

**Q: What is stream_mode="updates" vs "values" vs "messages"?**
`"values"`: full state after every node — useful for monitoring but sends a lot of data. `"updates"`: only the changed keys (diff) — efficient for large states. `"messages"`: individual LLM token chunks — for streaming chat UIs. All three can be combined: `stream_mode=["values", "messages"]`.

**Q: How do you implement circuit breakers for external API tools?**
Three states: Closed (normal), Open (failing — reject all calls immediately), Half-Open (one test call). Track consecutive failures in state or Redis. After N failures: open circuit, return cached fallback. After timeout: allow one test call. If succeeds: close. If fails: reopen.

---

## LESSON 10 — Capstone: Full Production Agent

### Theory

The capstone is not a new concept — it's the **integration test** of everything. The real learning here is recognizing how the pieces compose together.

#### Architecture
```
User → SUPERVISOR → db_agent (SQL ReAct loop) → back to SUPERVISOR
                  → analyst (stats + insights) → back to SUPERVISOR
                  → human_review (interrupt)   → back to SUPERVISOR
                  → FINISH → END
```

#### What each lesson contributes
| Lesson | Feature in capstone |
|--------|---------------------|
| 1–2 | StateGraph structure, conditional routing |
| 3 | MessagesState, Ollama |
| 4 | @tool, ToolNode, ReAct loop inside db_agent |
| 5 | Supervisor routing pattern |
| 6 | list_tables + describe_table + run_sql |
| 7 | interrupt() for data change approval |
| 8 | MemorySaver + thread_id per user |
| 9 | Logging, structured output, recursion_limit |

#### Production deployment checklist
```
STATE:
  □ All fields typed with TypedDict
  □ Reducers on all list fields
  □ user_id field for multi-user tracking
  □ error field for graceful handling

TOOLS:
  □ Docstrings clear and specific
  □ Read-only guard on DB tools
  □ Sensitive tools behind interrupt()
  □ All exceptions caught — return strings

GRAPH:
  □ Error handler node
  □ recursion_limit in config
  □ Checkpointer attached
  □ thread_id on every invoke()

NODES:
  □ Logging at start of every node
  □ try/except for external calls
  □ Retry for LLM calls

TESTING:
  □ Unit test each tool
  □ Integration test each agent path
  □ Test HITL interrupt + resume
  □ Test persistence (restart + verify)
```

---

### Tasks — Lesson 10

**Task 10.1 — HR Specialist**
Add `hr_agent` to the capstone for: hire dates, department transfers, org chart queries. Update supervisor routing.

**Task 10.2 — Weekly Report Agent**
Add `report_agent`: queries metrics, generates markdown report, saves to `reports/weekly_YYYY-MM-DD.md`.

**Task 10.3 — Upgrade to SqliteSaver**
Replace MemorySaver. Verify persistence across restarts. Add `--clear` CLI flag to delete DB. Add `--list-sessions` to show all sessions.

**Task 10.4 — Input Validation Layer**
Add Pydantic validation before every `invoke()`. Rules: question 3-500 chars, user_id alphanumeric max 20 chars. Invalid input never enters the graph — return error immediately.

---

### Interview Q&A — Lesson 10

**Q: Walk me through how you would debug a capstone-style agent that returns wrong answers.**
1. Enable full logging — check which agent the supervisor routed to. 2. Check which tools were called (in message history). 3. Check tool results — did the SQL query return correct data? 4. Check if the LLM interpreted the data correctly. 5. Use `get_state_history()` to replay the exact execution. Each step is isolated, so you know exactly where the error occurred.

**Q: How would you scale this agent for 1,000 concurrent users?**
(1) Replace `MemorySaver` with `PostgresSaver` — shared state across servers. (2) Deploy multiple graph server instances behind a load balancer. (3) Use `thread_id` for routing — all requests for the same user go to the same session. (4) Use connection pooling for the database. (5) Cache frequent SQL queries in Redis. (6) Use `Send()` to parallelize within each request.

**Q: How do you handle a scenario where the supervisor routes incorrectly?**
(1) Improve the supervisor prompt with clearer agent descriptions and examples. (2) Add a "confidence threshold" — if supervisor uncertainty is high, route to `human_review`. (3) Add supervisor output validation — if the JSON is malformed or agent name unknown, default to `db_agent`. (4) Log all routing decisions and review them weekly to identify patterns.
