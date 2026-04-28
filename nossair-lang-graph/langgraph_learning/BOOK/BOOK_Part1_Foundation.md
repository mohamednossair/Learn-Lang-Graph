# LangGraph Senior Engineer Guide — Part 1: Foundation (Lessons 1–5)

---

## LESSON 1 — StateGraph, Nodes, and Edges

### Theory

LangGraph models your AI workflow as a **directed graph**:
- **State** — a shared `TypedDict` dict flowing between nodes
- **Nodes** — plain Python functions that read/update state
- **Edges** — connections defining execution order

#### The State
```python
class MyState(TypedDict):
    message: str    # input
    result: str     # output written by a node
```
**Rule:** Nodes only return the keys they changed. LangGraph merges the dict automatically.

#### Node signature
```python
def my_node(state: MyState) -> dict:
    return {"result": state["message"].upper()}  # only return changed keys
```

#### Edge types
| Type | Code | Meaning |
|------|------|---------|
| Entry | `add_edge(START, "node_a")` | First node to run |
| Fixed | `add_edge("node_a", "node_b")` | Always go A → B |
| Exit | `add_edge("node_b", END)` | Last node |

#### Execution flow
```
invoke(initial_state) → runs nodes in edge order → each node merges updates → returns final state
```

#### Why TypedDict not plain dict?
- IDE autocomplete and type checking
- LangGraph validates structure at compile time
- Self-documenting code

#### Common mistakes
| Mistake | Fix |
|---------|-----|
| Return full state from node | Only return changed keys |
| Call `invoke()` before `compile()` | Always compile first |
| Forget START/END | Every graph needs entry and exit |

---

### Tasks — Lesson 1

**Task 1.1 — Text Pipeline**
Build 4 nodes in sequence: `input → clean (lowercase+strip) → count_words → format_output`.
State: `{raw: str, clean: str, word_count: int, output: str}`

**Task 1.2 — Calculator Pipeline**
State: `{a: float, b: float, result: float}`.
Build separate graphs for add, subtract, multiply, divide.
Test each with different numbers.

**Task 1.3 — Data Transformer**
Input: `{"numbers": [3,1,9,2,7]}`.
Nodes: `sort_node → filter_node (keep >3) → stats_node (min/max/avg)`.

**Task 1.4 — Bug Hunt**
Find and fix 3 bugs:
```python
class State(TypedDict):
    x: str

def node(state):
    return state          # BUG 1

builder = StateGraph(dict)  # BUG 2
builder.add_node("n", node)
builder.add_edge("n", END)  # BUG 3
graph = builder.compile()
```

---

### Interview Q&A — Lesson 1

**Q: What is the difference between `invoke()` and `stream()`?**
`invoke()` runs the full graph and returns the final state as one dict. `stream()` yields snapshots after each node — essential for streaming tokens to chat UIs and for real-time debugging.

**Q: Why do nodes return only the keys they changed?**
This is the "partial update" pattern. It prevents nodes accidentally overwriting keys they don't own. LangGraph merges using reducers (default: replace). Keeps nodes single-responsibility and composable.

**Q: What happens if you forget `compile()`?**
`invoke()` will raise `AttributeError` — you called it on a `StateGraph` builder object, not a compiled graph. Always call `graph = builder.compile()` first.

**Q: TypedDict vs Pydantic for state?**
`TypedDict` is preferred: no runtime validation overhead, works naturally with type hints. Use Pydantic for tool inputs/outputs where validation matters, `TypedDict` for graph state.

---

## LESSON 2 — Conditional Edges & Branching

### Theory

Real workflows branch. After classifying a document, you route to different handlers. This is `add_conditional_edges()`.

#### The routing function
```python
def route(state: MyState) -> Literal["node_a", "node_b", "node_c"]:
    if state["type"] == "a": return "node_a"
    if state["type"] == "b": return "node_b"
    return "node_c"   # always have a default!
```

#### Wiring it up
```python
builder.add_conditional_edges(
    "source_node",   # triggers routing after this node
    route,           # routing function
    {"node_a": "node_a", "node_b": "node_b", "node_c": "node_c"}
)
```

#### Common patterns
```
# Branch
classify → route() → handler_A → END
                   → handler_B → END

# ReAct Loop  
agent → route() → tools → agent  (loop)
               → END

# Guard / Gate
validate → route() → process → END
                   → error   → END
```

#### The missing-branch bug
If routing returns a value not in the mapping dict → `InvalidUpdateError`. Always cover every case.

---

### Tasks — Lesson 2

**Task 2.1 — Support Ticket Router**
State: `{ticket: str, priority: str, assigned_to: str}`.
Classify as high/medium/low. Route to senior_engineer / engineer / intern.

**Task 2.2 — Language Detector**
Detect if text is English/Arabic/French (by keywords). Route to appropriate translator node.

**Task 2.3 — Validation Gate**
Number input (0-100). Route valid → process → format. Invalid → error node.

**Task 2.4 — Two-Level Router**
Level 1: department (engineering/sales/hr).
Level 2: within engineering, route by task type (bug/feature/review).
Each final node returns a specific message.

---

### Interview Q&A — Lesson 2

**Q: Conditional edges vs if/else inside a node — what's the difference?**
Conditional edges make branching **visible** in the graph structure (shows in `draw_mermaid_png()`). Nodes stay single-responsibility. LangGraph can also optimize parallel execution only if branching is declared via edges. Always prefer conditional edges.

**Q: Can a routing function return END directly?**
Yes. `END` can be a value in the mapping dict: `{"done": END, "continue": "next_node"}`.

**Q: What error occurs if routing returns an unmapped value?**
`InvalidUpdateError`. Always have a default `else` branch in your routing function.

---

## LESSON 3 — Chatbot with Memory (Reducers)

### Theory

#### The Reducer Problem
Without a reducer, every node return **replaces** the state field. For messages this destroys history.

```python
messages: list                        # replace — history lost ❌
messages: Annotated[list, add_messages]  # append — history kept ✅
```

#### Message types
| Class | Role |
|-------|------|
| `HumanMessage` | User input |
| `AIMessage` | LLM output |
| `SystemMessage` | AI persona/rules — goes first |
| `ToolMessage` | Tool execution result |

#### Temperature guide
| Task | Temp |
|------|------|
| Agents/tools | 0.0 |
| Q&A/facts | 0.1–0.3 |
| Chat | 0.5–0.7 |
| Creative | 0.8–1.0 |

#### Message trimming (production pattern)
When history exceeds N messages, either:
1. Hard trim — keep last N
2. Summarize — LLM summarizes old messages into one `SystemMessage`, then trim

---

### Tasks — Lesson 3

**Task 3.1 — Python Tutor Bot**
System prompt: "You are a senior Python tutor. Always give code examples. Rate user questions 1-5 on specificity."

**Task 3.2 — Context Tracker**
Add to state: `topic: str`, `turn_count: int`. At turn 5, add a system note to summarize.

**Task 3.3 — Persona Switcher**
User types "be pirate" / "be formal" / "be casual". Switch persona stored in state.

**Task 3.4 — Message Trimmer**
When messages > 10: summarize first 8 into one SystemMessage. Keep last 2. Implement and test this production pattern.

---

### Interview Q&A — Lesson 3

**Q: What is a reducer and why do messages need one?**
A reducer defines how a state field is updated when a node returns a new value. Default is replace. `add_messages` is a reducer that appends new messages instead, preserving chat history across multiple `invoke()` calls.

**Q: How do you implement message trimming in production?**
In a pre-processing node, count tokens with `tiktoken`. If over budget: call LLM to summarize old messages into a `SystemMessage`, replace old messages with summary, keep last 2 turns verbatim. Return updated messages list.

**Q: MessagesState vs custom TypedDict?**
`MessagesState` (from `langgraph.graph`) is a convenience class with `messages: Annotated[list, add_messages]` already defined. Use it for simple chatbots. Use custom `TypedDict` when you need extra state fields alongside messages.

---

## LESSON 4 — ReAct Agent with Tools

### Theory

#### ReAct = Reason + Act
1. LLM reasons about the question
2. Decides to call a tool
3. Tool runs, result returned
4. LLM reasons with result
5. Repeat until final answer

#### @tool decorator
```python
@tool
def my_tool(city: str) -> str:
    """Get weather for a city. Returns temperature and conditions."""
    # Docstring IS the LLM's instruction for when/how to use this tool
    return f"Weather in {city}: Sunny, 22°C"
```

**The docstring is the most important part.** The LLM reads it to decide when and how to call the tool.

#### The loop
```python
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")  # THIS creates the loop
```

#### should_continue logic
```python
def should_continue(state) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"
```

#### Tool design rules
- One responsibility per tool
- Clear, specific docstring
- Always validate input inside tool
- Return strings (LLM reads tool results as text)
- Handle all exceptions — return error string, never raise

---

### Tasks — Lesson 4

**Task 4.1 — Math Agent**
Tools: add, subtract, multiply, divide, power, square_root. Test: "What is sqrt((144+25)*2)?"

**Task 4.2 — File System Agent**
Tools: read_file(path), list_files(dir), count_lines(path). Ask: "How many lines does requirements.txt have?"

**Task 4.3 — Date/Time Agent**
Tools: get_current_date(), day_of_week(date), days_between(d1,d2), add_days(date,n). Test: "What day is it 45 days from today?"

**Task 4.4 — Retry on Tool Failure**
One tool fails randomly 50% of the time. Agent must detect failure (tool returns error string) and retry up to 3 times. Track retries in state.

**Task 4.5 — Research Agent**
Tools: search_web(query), summarize_page(url), extract_facts(text) — all mocked. Test end-to-end research flow.

---

### Interview Q&A — Lesson 4

**Q: bind_tools() vs with_structured_output()?**
`bind_tools()` attaches optional tools — LLM decides when to use them and can still return plain text. `with_structured_output(PydanticModel)` forces the LLM to always return a specific schema — no tool calling involved. Use tools for actions, structured output for reliable JSON responses.

**Q: How does ToolNode know which function to call?**
The LLM returns an `AIMessage` with `tool_calls = [{"name": "tool_name", "args": {...}, "id": "..."}]`. `ToolNode` looks up the function by name from the list you passed to it, calls it with `args`, wraps result in `ToolMessage` with matching `tool_call_id`.

**Q: How do you prevent infinite ReAct loops?**
Three defenses: (1) `recursion_limit` in config raises `GraphRecursionError`, (2) iteration counter in state — route to END when exceeded, (3) detect repeated identical tool calls and break the loop.

**Q: How do you add user authorization to tools?**
Add a "tool guard" node before `ToolNode`. It reads `tool_calls` from the last message, checks each tool name against a user-role allowlist stored in state, and either passes through or replaces with an error `ToolMessage`.

---

## LESSON 5 — Multi-Agent Systems (Supervisor Pattern)

### Theory

#### Why multi-agent?
A single LLM trying to be expert at everything fails on complex tasks. Specialists outperform generalists because:
- Smaller focused context = better LLM reasoning
- Each agent has fewer tools = less confusion
- Parallel execution of independent sub-tasks
- Easy to add new specialists without breaking existing ones

#### Supervisor pattern
```
User question → SUPERVISOR (routes) → specialist_A → reports back → SUPERVISOR
                                     → specialist_B → reports back → SUPERVISOR → FINISH → END
```

#### Supervisor prompt design
```python
PROMPT = """Route to: researcher (facts), writer (content), coder (code).
Respond ONLY: {"next": "researcher"} or {"next": "FINISH"}
Use FINISH when task is fully completed."""
```

#### Specialist → Supervisor reporting
```python
builder.add_edge("specialist_a", "supervisor")  # always report back
builder.add_edge("specialist_b", "supervisor")
```

#### Supervisor vs Network (peer-to-peer)
| | Supervisor | Network |
|---|-----------|---------|
| Routing | Central | Each agent decides |
| Debugging | Easy | Hard |
| Control | High | Low |
| Use when | Clear hierarchy | Agents collaborate freely |

---

### Tasks — Lesson 5

**Task 5.1 — Customer Support System**
4 specialists: billing, technical, returns, general. Route by ticket content. Test all 4 paths.

**Task 5.2 — Content Pipeline**
researcher → writer → editor → seo_agent. Supervisor routes sequentially.

**Task 5.3 — Code Review System**
3 parallel specialists: syntax_checker, style_checker, security_checker. Supervisor combines all reports into final review.

**Task 5.4 — Shared Notes**
Add `shared_notes: list` to state (with append reducer). Each specialist appends findings. Supervisor reads notes when deciding next routing.

---

### Interview Q&A — Lesson 5

**Q: How do you prevent supervisor from looping to the same agent infinitely?**
Add `visited_agents: list` to state. Append agent name each time supervisor routes to it. Include in supervisor prompt: "Do not re-route to agents in visited_agents." Or use a `max_steps: int` counter.

**Q: Supervisor pattern vs swarm pattern?**
Supervisor: one central agent makes all routing decisions — agents don't talk directly. Swarm: any agent can hand off to any other using `Command(goto="agent_name")`. Supervisor is easier to debug and control. Swarm is more flexible for dynamic collaboration.

**Q: How do you run multiple specialists in parallel?**
Use `Send()`: the routing function returns `[Send("specialist_a", state), Send("specialist_b", state)]`. LangGraph runs them concurrently and merges results using reducers.
