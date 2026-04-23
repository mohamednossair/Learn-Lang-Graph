# Lessons 3–5 Deep Dive: Chatbot, ReAct Agents, Multi-Agent

---

# Lesson 3 — Chatbot with Memory: Complete Deep Dive

> **Prerequisite:** Open `lesson_03_chatbot/lesson_03_chatbot.ipynb` and run all cells first.

---

## Real-World Analogy

A **whiteboard in a team meeting** — without a reducer, every person who walks up **erases the board and writes fresh notes**. With `add_messages`, every person **adds their notes below the previous ones**. The whiteboard grows over time. This is the difference between `messages: list` (erases) and `messages: Annotated[list, add_messages]` (appends).

---

## The Reducer System — Deep Dive

### Why reducers exist

The default behavior in LangGraph is **replace**: when a node returns `{"field": new_value}`, the old value is replaced entirely. This works perfectly for simple fields like `score: float` or `status: str`. But for a conversation history, "replace" means history is lost.

Reducers let you override this behavior **per field**.

### How add_messages works

```python
from langgraph.graph.message import add_messages
from typing import Annotated

# Syntax: field: Annotated[type, reducer_function]
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

Internally, when a node returns `{"messages": [new_message]}`:
```python
# LangGraph calls the reducer:
new_state["messages"] = add_messages(
    old_state["messages"],   # existing messages
    [new_message]            # what the node returned
)
# Result: existing + new — history preserved
```

### Special behavior of add_messages

1. **Appends new messages**: `[msg1, msg2] + [msg3]` → `[msg1, msg2, msg3]`
2. **Deduplicates by ID**: if new message has same ID as existing → **replaces** (not appends)
3. **Handles RemoveMessage**: `RemoveMessage(id=x)` deletes message with id=x
4. **Works with all message types**: `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`

### Custom reducers for other use cases

```python
from typing import Annotated

# Append items to a list (like add_messages but for plain strings)
def append_strings(existing: list, new: list) -> list:
    return existing + (new if isinstance(new, list) else [new])

# Merge two dicts (newer keys override older)
def merge_dicts(existing: dict, new: dict) -> dict:
    return {**existing, **new}

# Keep running total
def accumulate(existing: int, new: int) -> int:
    return existing + new

class AgentState(TypedDict):
    messages:     Annotated[list, add_messages]   # chat history
    notes:        Annotated[list, append_strings] # collected notes
    metadata:     Annotated[dict, merge_dicts]    # merged config
    token_total:  Annotated[int,  accumulate]     # running total
```

---

## Message Types — Complete Reference

| Class | Import | Role | Who creates it | Key attributes |
|-------|--------|------|---------------|---------------|
| `HumanMessage` | `langchain_core.messages` | User's turn | You | `.content` |
| `AIMessage` | `langchain_core.messages` | LLM's response | LLM returns it | `.content`, `.tool_calls` |
| `SystemMessage` | `langchain_core.messages` | Instructions/persona | You | `.content` |
| `ToolMessage` | `langchain_core.messages` | Tool result | `ToolNode` creates it | `.content`, `.tool_call_id` |
| `RemoveMessage` | `langgraph.graph.message` | Delete a past message | You | `.id` |

### The power of SystemMessage

```python
# ❌ Generic, unhelpful
llm.invoke([HumanMessage("Explain APIs")])

# ✅ Focused expert — same model, dramatically better
llm.invoke([
    SystemMessage(content="""You are a senior backend engineer with 10 years of API design experience.
When explaining concepts:
- Always give a concrete code example first
- Then explain what the code does line by line
- Point out 1-2 common mistakes beginners make
- Mention one production best practice"""),
    HumanMessage(content="Explain APIs")
])
```

The SystemMessage is **always first** in the messages list you pass to the LLM. It sets the entire context for the conversation.

---

## The Multi-Turn Chatbot Pattern

### Without checkpointer — manual history management

```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}   # add_messages appends this

graph = builder.compile()  # no checkpointer

# You must maintain history yourself
history = []
while True:
    user_input = input("You: ")
    history.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages": history})
    history = result["messages"]   # update history with AI response
    print(f"AI: {result['messages'][-1].content}")
```

### With checkpointer — automatic history management

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "my-session"}}

# LangGraph loads and saves history automatically via thread_id
while True:
    user_input = input("You: ")
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},  # only send NEW message
        config=config
    )
    print(f"AI: {result['messages'][-1].content}")
# add_messages reducer + checkpointer = full automatic history
```

---

## Message Trimming — Production Pattern

### Why you need it

Every LLM has a context window limit. For llama3: ~8K tokens. GPT-4: 128K. At ~3-4 tokens per word, a long conversation with detailed responses fills the window quickly. Without trimming → `ContextLengthExceededError`.

### Strategy 1: Hard trim (simple, loses context)

```python
def trim_node(state: ChatState) -> dict:
    messages = state["messages"]
    if len(messages) <= 10:
        return {}
    return {"messages": messages[-10:]}  # keep only last 10
```

### Strategy 2: Summarize + trim (production standard)

```python
def summarize_and_trim(state: ChatState) -> dict:
    messages = state["messages"]
    if len(messages) <= 12:
        return {}   # nothing to do

    to_summarize = messages[:-4]   # everything except last 4
    keep_recent  = messages[-4:]   # always keep last 4 verbatim

    # Ask LLM to summarize old messages
    summary_prompt = "Summarize this conversation in 3-5 concise bullet points:\n\n"
    summary_prompt += "\n".join(f"{type(m).__name__}: {m.content}" for m in to_summarize)
    summary = llm.invoke([HumanMessage(content=summary_prompt)]).content

    return {"messages": [SystemMessage(content=f"[Summary of earlier conversation]:\n{summary}")] + keep_recent}
```

### Token-based trimming (most precise)

```python
import tiktoken  # pip install tiktoken

def count_tokens(messages: list) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        total += len(enc.encode(msg.content)) + 4  # 4 tokens per message overhead
    return total

def token_trim_node(state: ChatState) -> dict:
    messages = state["messages"]
    while count_tokens(messages) > 4000 and len(messages) > 4:
        messages = messages[1:]   # remove oldest message
    return {"messages": messages}
```

---

## Temperature Guide — When to Use What

| Temperature | Behavior | Use for | Avoid for |
|-------------|---------|---------|-----------|
| 0.0 | Fully deterministic | SQL generation, tool routing, agents, JSON output | Creative tasks |
| 0.1 | Nearly deterministic | Factual Q&A, code generation, summaries | n/a |
| 0.3 | Slight variation | Professional chat, explanations | Agent tool calls |
| 0.5 | Balanced | General conversation | Precise tasks |
| 0.7 | Creative variation | Casual chat, brainstorming | Agents, tools |
| 0.9 | High creativity | Creative writing, poetry | Any precise task |
| 1.0+ | Unpredictable | Experimental only | Everything else |

**Production rule:** Start at `0.0` for every agent. Only increase when you need creative variation and can accept inconsistency.

---

## Anti-Patterns — Lesson 3

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| `messages: list` without reducer | History erased each turn | `Annotated[list, add_messages]` |
| SystemMessage not first | LLM ignores persona | Always prepend: `[SystemMessage(...)] + state["messages"]` |
| No message trimming | Context limit error in production | Implement trim after 10-15 messages |
| Temperature 0.7 for agents | Inconsistent tool use, hallucinations | `temperature=0.0` for all agents |
| Passing raw strings instead of message objects | LLM can't distinguish roles | Always use `HumanMessage()`, `AIMessage()` etc. |
| LLM initialized inside node | Recreated on every call — slow | Initialize LLM once at module level |
| Duplicate SystemMessages | Confuses LLM, wastes tokens | One SystemMessage, maintain it across turns |

---

## Best Practices Checklist — Lesson 3

```
□ messages: Annotated[list, add_messages] — never plain list
□ SystemMessage is always the first message passed to LLM
□ LLM initialized once at module level, not inside node function
□ temperature=0.0 for agents, higher only for creative tasks
□ Message trimming implemented for conversations that may exceed 10 turns
□ All messages use proper classes (never raw strings)
□ Conversation history tested by checking messages[-1].content for AI response
□ With checkpointer: only new messages passed to invoke(), history is automatic
□ Without checkpointer: full history passed to invoke(), maintained manually
```

---

## Tasks — Lesson 3

**Task 3.1 — Python Tutor Bot**
Build a chatbot with system prompt: "You are a senior Python engineer. Rate every question 1-5 (specificity score). Always respond with: Rating: X/5, then a code example, then explanation."
Have a 5-turn conversation. Verify the rating appears every time.

**Task 3.2 — Custom Reducer**
Create a state with: `messages`, `facts: list`, `turn_count: int`.
Write a custom reducer for `facts` that appends but prevents duplicates.
Write a custom reducer for `turn_count` that increments (never decreases).
Test by running 5 turns and printing both fields.

**Task 3.3 — Message Archaeology**
After a 6-turn conversation, use the message history to:
- Count how many times each message type appeared
- Find the longest AI response
- Print messages 2-4 (slice the list)
- Find a message by its ID: `messages[i].id`

**Task 3.4 — Summarize and Trim**
Implement the summarize-and-trim pattern from the theory section.
Test: have a 15-turn conversation. At turn 12, trimming kicks in.
After trimming: ask "What did we discuss at the start?" — verify the summary is used.

**Task 3.5 — Persona Store**
State: `{messages, active_persona: str, personas: dict}`.
Personas dict: `{"pirate": "...", "teacher": "...", "engineer": "..."}`.
User command "switch to [persona]" → updates active_persona → SystemMessage changes.
Verify: persona switches mid-conversation without losing history.

---

## Interview Q&A — Lesson 3

**Q1: What is a reducer in LangGraph and why do messages specifically need one?**
A reducer is a function `(old_value, new_value) → merged_value` that defines how state fields are updated. Default behavior is replace. For a `messages: list` field, replace means every node call destroys history — each turn starts fresh. `add_messages` is a built-in reducer that **appends** new messages to existing history, which is how multi-turn conversations work. Without this reducer, your chatbot can't remember anything beyond the current turn.

**Q2: How does message deduplication work in add_messages?**
`add_messages` uses message IDs for deduplication. Every `HumanMessage`, `AIMessage`, etc. has a unique `.id` (UUID). When `add_messages` receives a new message with the same ID as an existing one, it **replaces** the old one instead of appending. This enables editing past messages: create a new message with the same ID as the old one, return it from a node, and `add_messages` replaces the old version. This is how "edit my last message" UIs work.

**Q3: How do you implement message history trimming without losing important context?**
Three-step approach: (1) Keep the original SystemMessage (persona/instructions) — never trim it. (2) When history exceeds N messages, call the LLM to summarize the oldest chunk into bullet points. (3) Replace the summarized messages with one `SystemMessage(content="[Summary]: ...")` and keep the last 4-6 exchanges verbatim. This way the LLM always has the full persona + summary of what happened + recent context — without hitting token limits.

**Q4: What is the difference between using a checkpointer and manually passing history?**
Without checkpointer: you must maintain the history list in your code and pass the full list to every `invoke()` call. History dies when your process dies. With checkpointer: LangGraph saves state after every node execution. On the next `invoke()` with the same `thread_id`, LangGraph automatically loads the saved state and merges the new message. History persists across process restarts. In production: always use a checkpointer (`SqliteSaver` minimum). Use manual history only in simple scripts.

**Q5: How do you build a chatbot that maintains different conversation contexts for different users?**
Use `thread_id` in config — each user gets their own isolated session: `config = {"configurable": {"thread_id": f"user-{user_id}"}}`. LangGraph's checkpointer stores state separately per `thread_id`. User A's messages never mix with User B's. For multi-user web apps: generate a unique `thread_id` when a user starts a new conversation, store it in your web session, pass it with every subsequent request to the agent.

---

# Lesson 4 — ReAct Agent with Tools: Complete Deep Dive

> **Prerequisite:** Open `lesson_04_tools_agent/lesson_04_tools_agent.ipynb` and run all cells first.

---

## Real-World Analogy

Think of a **detective with a team of specialists**:
- The detective (LLM) **reasons** about the case — what evidence is needed
- They **dispatch** to forensics (a tool) to get data they don't have
- Forensics **reports back** with results (ToolMessage)
- The detective **reasons again** with new evidence — maybe dispatch again
- Eventually: enough evidence → **final conclusion** (answer without tool calls)

The detective never guesses — they verify with evidence. The agent never makes up data — it always uses tools.

---

## The @tool Decorator — Everything You Need

### Basic usage

```python
from langchain_core.tools import tool

@tool
def get_employee_count(department: str) -> str:
    """
    Get the number of employees in a specific department.
    Use this when asked about headcount, team size, or how many people work in a department.
    Department names are case-insensitive: "Engineering", "engineering", "ENGINEERING" all work.
    Returns the count as a string, e.g., "Engineering has 45 employees."
    """
    department = department.strip().lower()
    counts = {"engineering": 45, "sales": 32, "marketing": 18}
    if department not in counts:
        return f"Unknown department: {department}. Known departments: {list(counts.keys())}"
    return f"{department.title()} has {counts[department]} employees."
```

### The docstring is the most important part

The LLM reads the docstring to decide:
1. **When** to call this tool (what situations call for it)
2. **How** to call it (parameter format, examples)
3. **What** to expect from it (return format)

```python
# ❌ Bad docstring — LLM won't know when or how to use this
@tool
def calculate(a: int, b: int) -> int:
    """Calculate."""
    return a + b

# ✅ Good docstring — LLM knows exactly when and how
@tool
def add_numbers(a: int, b: int) -> str:
    """
    Add two integers together. Use this for addition operations.
    Both parameters must be whole numbers (integers), not decimals.
    Example: add_numbers(5, 3) returns "5 + 3 = 8"
    Returns the equation and result as a formatted string.
    """
    return f"{a} + {b} = {a + b}"
```

### Tool parameter types the LLM understands best

| Type | LLM reliability | Notes |
|------|----------------|-------|
| `str` | Excellent | Most flexible |
| `int` | Good | LLM may pass floats — validate inside |
| `float` | Good | Be explicit in docstring |
| `bool` | Good | Use "true"/"false" in docstring |
| `list[str]` | Moderate | Specify separator in docstring |
| `dict` | Poor | Avoid — LLM struggles with nested dicts |

### Tool error handling — never raise

```python
@tool
def divide_numbers(a: float, b: float) -> str:
    """Divide a by b. Returns the quotient. Cannot divide by zero."""
    try:
        if b == 0:
            return "ERROR: Cannot divide by zero. Please provide a non-zero divisor."
        return f"{a} ÷ {b} = {a / b:.4f}"
    except Exception as e:
        return f"ERROR: {str(e)}"   # always return a string, never raise
```

When a tool raises an exception, `ToolNode` catches it and wraps it as an error `ToolMessage`. But the LLM may not handle this well. Better: return a descriptive error string — the LLM reads it and can retry intelligently.

---

## How the ReAct Loop Works — Step by Step

```python
# Setup
tools = [tool1, tool2, tool3]
llm_with_tools = llm.bind_tools(tools)   # attaches tool schemas to every LLM call

def agent_node(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    # AIMessage with tool_calls → agent wants to call a tool
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"   # AIMessage without tool_calls → final answer

tool_node = ToolNode(tools)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")   # ← THE LOOP: after tools, always back to agent
```

### What ToolNode does internally

```
1. Read state["messages"][-1] — should be AIMessage with tool_calls
2. For each tool_call in tool_calls:
   a. Find the tool function by name from tools list
   b. Call function(**tool_call.args)
   c. Wrap result in ToolMessage(content=result, tool_call_id=tool_call.id)
3. Return {"messages": [ToolMessage1, ToolMessage2, ...]}
```

### The message flow in a ReAct loop

```
Turn 1:
  HumanMessage: "What is 15% of 2,450?"
  → agent_node: LLM decides to use multiply tool
  → AIMessage(tool_calls=[{name:"multiply", args:{a:2450, b:0.15}}])
  → should_continue → "tools"

Turn 2:
  → tool_node: calls multiply(2450, 0.15) = 367.5
  → ToolMessage(content="2450 × 0.15 = 367.5")
  → back to agent_node

Turn 3:
  → agent_node: LLM sees the tool result
  → AIMessage(content="15% of 2,450 is 367.5")  ← no tool_calls this time
  → should_continue → "end" → END
```

---

## Tool Design Patterns

### Pattern 1: Read-only data tool

```python
@tool
def search_products(query: str, max_results: int = 5) -> str:
    """
    Search the product catalog by name or description.
    Use for questions about available products, prices, or categories.
    Returns up to max_results products matching the query.
    Example: search_products("laptop", 3) returns top 3 laptop products.
    """
    # Always read-only — never write to database from a tool without approval
    results = db.search(query, limit=max_results)
    if not results:
        return f"No products found matching '{query}'"
    return "\n".join(f"- {p.name}: ${p.price}" for p in results)
```

### Pattern 2: Write tool with explicit warning

```python
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email to the specified recipient.
    IMPORTANT: This action sends a real email and cannot be undone.
    Only use this when the user explicitly asks to send an email and confirms the recipient.
    Returns confirmation with the message ID if successful.
    """
    # In production, this should be behind interrupt() approval
    result = email_service.send(to=to, subject=subject, body=body)
    return f"Email sent successfully. Message ID: {result.id}"
```

### Pattern 3: Stateful tool (reads from shared context)

```python
# Pass context via closure — tool accesses data from outer scope
def make_db_tools(db_connection):
    @tool
    def run_query(sql: str) -> str:
        """Execute a SELECT query. Read-only only."""
        if not sql.strip().upper().startswith("SELECT"):
            return "ERROR: Only SELECT queries are allowed."
        cursor = db_connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return str(rows[:20])   # limit output
    return [run_query]

# Usage:
tools = make_db_tools(my_connection)
tool_node = ToolNode(tools)
```

---

## Preventing Infinite Loops — 3 Defenses

### Defense 1: recursion_limit (always use)

```python
config = {"recursion_limit": 15}   # max 15 node executions
# Raises GraphRecursionError if exceeded — does NOT silently loop forever
result = graph.invoke(state, config=config)
```

### Defense 2: Iteration counter in state

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iterations: int

def agent_node(state: AgentState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])],
            "iterations": state["iterations"] + 1}

def should_continue(state: AgentState) -> str:
    if state["iterations"] >= 10:
        return "end"   # force stop regardless of tool_calls
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"
```

### Defense 3: Detect repeated tool calls

```python
def should_continue(state: AgentState) -> str:
    msgs = state["messages"]
    last = msgs[-1]
    if not (hasattr(last, "tool_calls") and last.tool_calls):
        return "end"

    # Check if this exact tool call was made in the last 3 iterations
    recent_calls = [m.tool_calls for m in msgs[-6:] if hasattr(m, "tool_calls") and m.tool_calls]
    if len(recent_calls) >= 2:
        if recent_calls[-1] == recent_calls[-2]:
            return "end"   # stuck in a loop — exit

    return "tools"
```

---

## Anti-Patterns — Lesson 4

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| **Vague docstring** | LLM doesn't know when/how to call tool | Detailed: when, how, what returns |
| **Tool raises exception** | `ToolNode` may crash or LLM gets confused | Catch all exceptions, return error string |
| **No recursion_limit** | Agent loops forever if LLM is confused | Always set `recursion_limit` in config |
| **Too many tools** | LLM gets confused choosing (>10 tools) | Group related tools, use sub-agents |
| **Tool does multiple things** | Docstring unclear, LLM misuses it | One tool = one responsibility |
| **Tool modifies data without approval** | Irreversible actions without oversight | Sensitive tools should use `interrupt()` |
| **complex dict parameters** | LLM passes wrong format | Use `str` parameters, parse inside tool |

---

## Best Practices Checklist — Lesson 4

```
TOOL DESIGN:
  □ Docstring explains: WHEN to use, HOW to call, WHAT it returns
  □ Each tool does one thing (single responsibility)
  □ All exceptions caught — return error string, never raise
  □ Return type is always str (LLM reads it as text)
  □ Input validation inside tool body (check types, ranges)
  □ Read-only tools clearly marked as read-only in docstring
  □ Sensitive tools (write, send, delete) flagged clearly

REACT LOOP:
  □ recursion_limit always set in config (recommend 15-25)
  □ should_continue handles both tool_calls present and absent
  □ add_edge("tools", "agent") — the loop edge is present
  □ ToolNode initialized with same tools list as bind_tools()

TESTING:
  □ Each tool unit-tested independently (no LLM needed)
  □ Test tool error handling (bad input, division by zero etc.)
  □ Test should_continue with mock messages (tool_calls present and absent)
  □ Integration test: end-to-end question requiring 2+ tool calls
```

---

## Tasks — Lesson 4

**Task 4.1 — Math Agent**
Tools: `add(a,b)`, `subtract(a,b)`, `multiply(a,b)`, `divide(a,b)`, `power(base,exp)`, `sqrt(n)`.
Test: "What is sqrt((144 + 25) × 2)?" — requires 3 tool calls.
Unit test each tool independently before running the agent.

**Task 4.2 — File Inspector Agent**
Tools: `read_file(path)`, `list_directory(path)`, `count_lines(path)`, `find_in_file(path, keyword)`.
All tools must handle: file not found, permission error, directory vs file confusion.
Test: "How many lines contain the word 'import' in lesson_01_basics.py?"

**Task 4.3 — Date/Time Agent**
Tools: `get_today()`, `day_of_week(date_str)`, `days_between(d1, d2)`, `add_days(date_str, n)`, `is_weekend(date_str)`.
All dates as "YYYY-MM-DD" strings.
Test: "Is the date 100 days from today a weekend?" — requires 2 tool calls.

**Task 4.4 — Flaky Tool Retry**
Build a tool that fails 60% of the time (random). Track failures in state.
Agent must detect failure (tool returned "ERROR: ...") and retry same tool up to 3 times.
Add `retry_counts: dict` to state. After 3 failures, agent gives up gracefully.

**Task 4.5 — Tool Authorization**
Add `user_role: str` to state ("admin" or "user").
Build a "tool guard" node before ToolNode that:
- Blocks "delete_record" tool for non-admin users → replaces with error ToolMessage
- Allows all other tools for everyone
Test both roles with the same question.

---

## Interview Q&A — Lesson 4

**Q1: How does the LLM decide which tool to call and when?**
When you call `bind_tools(tools)`, LangGraph attaches the schema of each tool (name, description from docstring, parameter types) to every LLM request as part of the system context. The LLM reads this list and decides based on: the docstring content, parameter names, and the current conversation. When it decides to call a tool, it returns an `AIMessage` with `tool_calls` populated instead of `content`. This is why docstrings are the most critical part of tool design.

**Q2: What is the difference between `bind_tools()` and `with_structured_output()`?**
`bind_tools()` makes tools optional and gives the LLM agency — it decides when to call a tool and can also return plain text. `with_structured_output(Schema)` forces the LLM to always return a specific JSON structure — no tool calling involved. Use `bind_tools()` when you want the LLM to autonomously decide what actions to take. Use `with_structured_output()` when you need guaranteed structured data from the LLM (routing decisions, sentiment scores, extracted entities).

**Q3: How does ToolNode know which function to call?**
`ToolNode` receives the message history and reads `state["messages"][-1].tool_calls`. Each tool call contains `{"name": "tool_name", "args": {...}, "id": "call_id"}`. `ToolNode` looks up the function by name in the list you passed to it (`ToolNode([tool1, tool2])`), calls it with the `args` dict unpacked as keyword arguments, and wraps the return value in `ToolMessage(content=str(result), tool_call_id=call_id)`.

**Q4: What happens if the LLM calls a tool that doesn't exist?**
`ToolNode` raises `KeyError: "nonexistent_tool"` — it can't find the function in its tool list. Prevention: never rename tools after the LLM has learned to call them (breaks existing prompts), keep tool names snake_case and descriptive. If using dynamic tool lists: validate that the LLM can only call tools in your registered list.

**Q5: How do you handle a tool that takes 30+ seconds to execute (slow external API)?**
Option 1: Async tool with timeout:
```python
import asyncio
@tool
async def slow_api_call(query: str) -> str:
    try:
        result = await asyncio.wait_for(make_api_request(query), timeout=30)
        return str(result)
    except asyncio.TimeoutError:
        return "ERROR: API request timed out after 30 seconds. Try a simpler query."
```
Option 2: Job queue pattern — tool submits job, returns job_id. A `check_status(job_id)` tool polls until done. Option 3: For very long tasks, use `interrupt()` — tool submits job, graph pauses, resumes when human (or webhook) signals completion.

---

# Lesson 5 — Multi-Agent Systems: Complete Deep Dive

> **Prerequisite:** Open `lesson_05_multi_agent/lesson_05_multi_agent.ipynb` and run all cells first.

---

## Real-World Analogy

Think of a **hospital system**:
- The **triage nurse** (supervisor) assesses each patient and routes to the right department
- **Cardiology** (specialist) handles heart cases
- **Neurology** (specialist) handles brain cases
- **General Practice** (specialist) handles everything else
- Each department reports back to triage for discharge (routing to FINISH)
- No specialist talks directly to another — everything goes through the triage system

This is exactly the supervisor pattern. The supervisor never does the actual medical work — it only decides who should.

---

## Why Multi-Agent? The Core Problem

A single agent with 20 tools has a 500-token tool list. The LLM must process all 20 tool schemas on every call — including irrelevant ones. This degrades reasoning quality.

```
SINGLE AGENT (20 tools):
  "Calculate employee salary increase AND check product inventory AND send email"
  → LLM reads all 20 tool schemas
  → Gets confused about which tools to use when
  → Quality: ❌ poor

MULTI-AGENT (3 specialists × 6-7 tools each):
  Supervisor → "Calculate salary increase" → hr_agent (7 HR tools)
  Supervisor → "Check inventory" → inventory_agent (6 inventory tools)
  Supervisor → "Send email" → comms_agent (5 communication tools)
  → Each agent reads only its relevant 6-7 tool schemas
  → Quality: ✅ much better
```

---

## The Supervisor Pattern — Complete Implementation

### Supervisor state design

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

SPECIALISTS = ["db_agent", "analyst", "writer", "human_review"]
ROUTES = Literal["db_agent", "analyst", "writer", "human_review", "FINISH"]

class SupervisorState(TypedDict):
    messages:     Annotated[list, add_messages]
    user_id:      str
    next_agent:   str                           # set by supervisor
    task_summary: str                           # optional: what was accomplished
    visited:      Annotated[list, lambda x,y: x+y]  # track routing history
```

### Supervisor node — the router

```python
SUPERVISOR_PROMPT = """You are a task router. Route to the right specialist:

- db_agent:     Questions about data, databases, SQL queries, employee counts, salaries
- analyst:      Data analysis, trends, comparisons, statistics, insights
- writer:       Writing, summarizing, formatting reports, drafting content
- human_review: Sensitive actions, data modifications, anything requiring approval
- FINISH:       When the task is fully completed and user question is answered

Previous steps: {visited}
Current conversation context: {context}

Respond with ONLY valid JSON: {{"next": "agent_name"}}
Never route to the same agent twice in a row unless the first attempt failed."""

def supervisor_node(state: SupervisorState) -> dict:
    visited = state.get("visited", [])
    context = state["messages"][-1].content if state["messages"] else ""

    prompt = SUPERVISOR_PROMPT.format(
        visited=visited,
        context=context[:500]   # truncate for efficiency
    )
    msgs = [SystemMessage(content=prompt)] + state["messages"]
    resp = llm.invoke(msgs)

    try:
        import json, re
        match = re.search(r'\{[^}]+\}', resp.content)
        if match:
            data = json.loads(match.group())
            agent = data.get("next", "FINISH")
        else:
            agent = "FINISH"
    except Exception:
        agent = "FINISH"   # safe default

    # Validate the agent name
    valid = SPECIALISTS + ["FINISH"]
    if agent not in valid:
        agent = "FINISH"

    return {"next_agent": agent}
```

### Routing function

```python
def route_supervisor(state: SupervisorState) -> str:
    agent = state.get("next_agent", "FINISH")
    if agent == "FINISH":
        return "__end__"
    if agent in SPECIALISTS:
        return agent
    return "__end__"   # safe default for any unexpected value
```

### Specialist agents — each is its own mini ReAct loop

```python
def make_specialist(name: str, tools: list, system_prompt: str):
    """Factory function to create a specialist agent."""
    specialist_llm = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: SupervisorState) -> dict:
        msgs = [SystemMessage(content=system_prompt)] + state["messages"]
        resp = specialist_llm.invoke(msgs)
        return {"messages": [resp], "visited": [name]}

    def should_continue(state) -> str:
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "done"

    sub = StateGraph(SupervisorState)
    sub.add_node("agent", agent_node)
    sub.add_node("tools", tool_node)
    sub.add_edge(START, "agent")
    sub.add_conditional_edges("agent", should_continue, {"tools": "tools", "done": END})
    sub.add_edge("tools", "agent")
    return sub.compile()
```

---

## Supervisor vs Network (Swarm) Patterns

### Supervisor Pattern

```
All routing through one central agent:
  User → SUPERVISOR → specialist_A → SUPERVISOR → specialist_B → SUPERVISOR → FINISH → END
```

**When to use:**
- Clear hierarchical workflows
- Need audit trail of all routing decisions
- Debuggability is critical
- Predictable task types

**Pros:** Centralized control, easy to debug, one place to fix routing bugs
**Cons:** Single point of failure, supervisor becomes bottleneck with many agents

### Network / Swarm Pattern

```
Agents can hand off directly to each other:
  researcher → hands off to → writer → hands off to → publisher → END
```

```python
from langgraph.types import Command

def researcher_node(state):
    # Do research work...
    result = do_research(state)
    # Decide to hand off to writer directly
    return Command(
        update={"messages": [AIMessage(content=result)], "visited": ["researcher"]},
        goto="writer"   # ← direct handoff, no supervisor needed
    )
```

**When to use:**
- Agents have natural handoff sequences
- Each agent knows who should do the next step
- Low overhead needed

**Pros:** Fast (no supervisor overhead), natural agent-to-agent collaboration
**Cons:** Harder to debug, routing logic spread across multiple nodes

### Comparison table

| Dimension | Supervisor | Network/Swarm |
|-----------|-----------|--------------|
| Routing authority | Central supervisor | Each agent decides |
| Debugging | Easy — one place | Hard — distributed |
| Adding new agents | Update supervisor prompt | Update all agents |
| Visibility | Full routing trace | Partial |
| Best for | Business workflows | Research/collaboration |
| State complexity | Lower | Higher |

---

## Anti-Patterns — Lesson 5

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| **Supervisor does actual work** | Supervisor should only route | Keep supervisor output to routing decisions only |
| **No visited/step tracking** | Supervisor loops infinitely | Add `visited: list` or `step_count: int` |
| **No malformed JSON handling** | Supervisor LLM output isn't always valid JSON | Always wrap JSON parsing in try/except with safe default |
| **Unknown agent name not handled** | Routes to non-existent node | Validate agent name against allowlist |
| **Specialists share all state** | State pollution between agents | Use subgraphs with their own state for isolation |
| **Too many specialists** | Supervisor prompt too long, routing quality degrades | Max ~6 specialists; group into departments otherwise |
| **Specialists report final answers to supervisor** | Final answer lost after routing | Use shared `messages` field with `add_messages` reducer |

---

## Best Practices Checklist — Lesson 5

```
SUPERVISOR DESIGN:
  □ Supervisor only routes — never does domain work
  □ Supervisor prompt lists all agents with clear, distinct descriptions
  □ JSON parsing wrapped in try/except with safe default
  □ Agent name validated against allowlist before routing
  □ visited/step tracking to prevent infinite loops
  □ recursion_limit set in config (recommend 40 for multi-agent)

SPECIALIST DESIGN:
  □ Each specialist has focused, non-overlapping responsibility
  □ Specialist tools are domain-specific (no overlap with other specialists)
  □ Specialists always report back to supervisor after completing work
  □ Specialist final message is a clear summary for the supervisor to read

STATE DESIGN:
  □ messages: Annotated[list, add_messages] — shared across all agents
  □ next_agent: str — set by supervisor
  □ visited: Annotated[list, append] — routing history for loop prevention
  □ All specialist outputs written to messages (not separate fields)
```

---

## Tasks — Lesson 5

**Task 5.1 — Customer Support System**
3 specialists: `billing_agent` (invoices, payments), `technical_agent` (bugs, setup), `general_agent` (other).
Test 6 different support tickets — verify correct routing each time.
Add `visited` tracking. Verify supervisor never routes to same agent twice in one session.

**Task 5.2 — Content Creation Pipeline**
Sequential pipeline: `researcher → writer → editor → publisher`.
Supervisor routes through them in sequence — each reports back before next is called.
Final `publisher` node saves content to a file.

**Task 5.3 — Parallel Specialists with Send()**
Build a "document analysis" system:
- `summary_agent`: produces a 3-sentence summary
- `sentiment_agent`: produces sentiment score
- `keywords_agent`: extracts 5 keywords
Use `Send()` to run all 3 in parallel.
`combine_node` merges all results into a structured report.

**Task 5.4 — Add Confidence Scoring**
Modify your supervisor to output a confidence score along with the routing decision:
```json
{"next": "db_agent", "confidence": 0.9, "reason": "Question about employee data"}
```
If confidence < 0.6, route to `human_review` instead of the chosen specialist.
Track confidence scores in state.

---

## Interview Q&A — Lesson 5

**Q1: How do you prevent the supervisor from routing to the same agent in an infinite loop?**
Three complementary approaches: (1) `visited: Annotated[list, lambda x,y: x+y]` in state — append agent name each time supervisor routes. Include in supervisor prompt: "Do not route to agents already in visited list unless previous attempt failed." (2) `step_count: int` with a maximum check in the routing function. (3) `recursion_limit` in config as final safety net. Use all three for production systems.

**Q2: What is the difference between supervisor pattern and swarm pattern in LangGraph?**
Supervisor: one central LLM makes all routing decisions. Every agent reports back to the supervisor after completing work. Advantages: easy to debug (one place for routing logic), clear audit trail, easy to add new agents. Swarm: agents use `Command(goto="agent_name")` to hand off directly to other agents — no central supervisor. Advantages: lower latency (skip supervisor roundtrip), more natural for collaborative tasks. Disadvantages: routing logic distributed, harder to debug, agents must know about each other.

**Q3: How do you add a new specialist to an existing supervisor system?**
(1) Build the new specialist node/subgraph with its tools and system prompt. (2) Register it: `builder.add_node("new_specialist", new_specialist_fn)`. (3) Add return edge: `builder.add_edge("new_specialist", "supervisor")`. (4) Update supervisor prompt to include new specialist name and description. (5) Update routing function's `Literal[...]` type hint to include new name. (6) Test with queries that should route to new specialist. No other code changes needed — this is the key benefit of the supervisor pattern.

**Q4: How do you share information between specialists without routing through the supervisor?**
Use the shared `messages` field with `add_messages` reducer. Every specialist's output is appended to `messages`. The supervisor can read this history to understand what was already done. Alternatively, add dedicated shared fields: `shared_context: str`, `accumulated_results: Annotated[list, append]`. All specialists can write to shared fields; supervisor reads them to make better routing decisions.

**Q5: How do you handle a specialist that fails (throws an exception)?**
Add an error handler to each specialist subgraph. The specialist catches exceptions, writes `{"error": str(e)}` to state, and still reports back to the supervisor. The supervisor sees the error in state and either: routes to a different specialist that can handle it, routes to `human_review`, or routes to FINISH with an error explanation. Never let specialist exceptions propagate to the supervisor node — it breaks the whole graph.
