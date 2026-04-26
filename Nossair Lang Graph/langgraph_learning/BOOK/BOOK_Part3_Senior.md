# LangGraph Senior Engineer Guide — Part 3: Senior Level (Lessons 11–15)

These lessons cover the topics that separate **senior** AI engineers from intermediate ones.

---

## LESSON 11 — Subgraphs: Composing Graphs Inside Graphs

### Theory

As systems grow, a single graph becomes a 500-line monolith. Subgraphs solve this — they are compiled graphs used as nodes inside a parent graph.

#### What is a subgraph?
```python
# Build subgraph
sub = StateGraph(SubState)
sub.add_node("step1", step1_fn)
sub.add_edge(START, "step1")
sub.add_edge("step1", END)
subgraph = sub.compile()

# Use as a node in parent graph
parent = StateGraph(ParentState)
parent.add_node("validation_module", subgraph)  # ← subgraph IS the node
```

#### State sharing between parent and subgraph
Keys with **matching names** are automatically shared. Private keys stay local.
```python
class ParentState(TypedDict):
    messages: Annotated[list, add_messages]  # shared — matching name
    user_id: str                             # shared
    final_result: str                        # populated by subgraph

class SubState(TypedDict):
    messages: Annotated[list, add_messages]  # shared with parent
    internal_flag: bool                      # private to subgraph
    final_result: str                        # written back to parent
```

#### Why subgraphs?
- **Reusable modules** — build a "validation subgraph" used by 10 different agents
- **Team ownership** — each team owns their subgraph, tests it independently
- **Version swap** — replace one subgraph's implementation without touching parent
- **Testability** — unit test each subgraph before integrating
- **Readability** — parent graph is clean, each subgraph is focused

#### Parallel subgraphs with Send()
```python
def fan_out_to_subgraphs(state):
    return [
        Send("research_module",  {"topic": state["topic"]}),
        Send("analysis_module",  {"data":  state["data"]}),
        Send("writing_module",   {"brief": state["brief"]}),
    ]
# All 3 subgraphs run in parallel
```

### File: `lesson_11_subgraphs/lesson_11_subgraphs.py`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send

llm = ChatOllama(model="llama3.2", temperature=0)

# --- Shared state fields ---
class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    content: str
    is_valid: bool
    validation_errors: list
    clean_content: str
    final_output: str

# ============================================================
# SUBGRAPH 1: Validation Module
# ============================================================
class ValidationState(TypedDict):
    content: str
    is_valid: bool
    validation_errors: list

def check_length(state: ValidationState) -> dict:
    errors = list(state.get("validation_errors", []))
    if len(state["content"].strip()) < 10:
        errors.append("Content too short (min 10 chars)")
        return {"is_valid": False, "validation_errors": errors}
    return {"is_valid": True, "validation_errors": errors}

def check_forbidden_words(state: ValidationState) -> dict:
    FORBIDDEN = ["spam", "scam", "fake"]
    errors = list(state.get("validation_errors", []))
    found = [w for w in FORBIDDEN if w in state["content"].lower()]
    if found:
        errors.append(f"Forbidden words: {found}")
        return {"is_valid": False, "validation_errors": errors}
    return {"validation_errors": errors}

v = StateGraph(ValidationState)
v.add_node("check_length", check_length)
v.add_node("check_forbidden", check_forbidden_words)
v.add_edge(START, "check_length")
v.add_edge("check_length", "check_forbidden")
v.add_edge("check_forbidden", END)
validation_subgraph = v.compile()

# ============================================================
# SUBGRAPH 2: Clean & Format Module
# ============================================================
class CleanState(TypedDict):
    content: str
    clean_content: str

def clean_text(state: CleanState) -> dict:
    clean = state["content"].strip().lower()
    clean = " ".join(clean.split())  # normalize whitespace
    return {"clean_content": clean}

def format_text(state: CleanState) -> dict:
    return {"clean_content": state["clean_content"].capitalize()}

c = StateGraph(CleanState)
c.add_node("clean", clean_text)
c.add_node("format", format_text)
c.add_edge(START, "clean")
c.add_edge("clean", "format")
c.add_edge("format", END)
clean_subgraph = c.compile()

# ============================================================
# PARENT GRAPH
# ============================================================
def generate_node(state: ReviewState) -> dict:
    resp = llm.invoke([SystemMessage(content="Write a short product description.")] + state["messages"])
    return {"content": resp.content, "messages": [resp]}

def route_after_validation(state: ReviewState) -> str:
    return "clean" if state.get("is_valid", False) else "end"

def publish_node(state: ReviewState) -> dict:
    return {"final_output": f"PUBLISHED: {state['clean_content']}"}

def reject_node(state: ReviewState) -> dict:
    errors = state.get("validation_errors", [])
    return {"final_output": f"REJECTED: {'; '.join(errors)}"}

builder = StateGraph(ReviewState)
builder.add_node("generate",   generate_node)
builder.add_node("validate",   validation_subgraph)   # ← subgraph as node
builder.add_node("clean",      clean_subgraph)        # ← subgraph as node
builder.add_node("publish",    publish_node)
builder.add_node("reject",     reject_node)
builder.add_edge(START, "generate")
builder.add_edge("generate", "validate")
builder.add_conditional_edges("validate", route_after_validation, {"clean": "clean", "end": "reject"})
builder.add_edge("clean", "publish")
builder.add_edge("publish", END)
builder.add_edge("reject", END)
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({
        "messages": [HumanMessage(content="Write about our new laptop product.")],
        "content": "", "is_valid": False, "validation_errors": [],
        "clean_content": "", "final_output": ""
    })
    print(result["final_output"])
```

---

### Tasks — Lesson 11

**Task 11.1 — Reusable Validation Subgraph**
Build a `validation_subgraph` used by 3 different parent graphs:
- Lesson 6's database agent (validate the SQL query before executing)
- Lesson 5's supervisor (validate user question before routing)
- A new document processing pipeline

**Task 11.2 — Nested 3-Level Hierarchy**
Top supervisor → department supervisors (engineering/sales) → specialist agents.
Each level is a separate compiled subgraph.

**Task 11.3 — Parallel Content Pipeline**
Use `Send()` to run 3 subgraphs in parallel:
- `research_subgraph` — finds facts (mock)
- `analysis_subgraph` — analyzes sentiment
- `writing_subgraph` — writes a draft
`combine_node` merges all 3 results into a final document.

---

### Interview Q&A — Lesson 11

**Q: How does state flow between parent and subgraph?**
LangGraph uses key-based state sharing: subgraph keys matching parent keys are automatically passed in and written back. Keys only in the subgraph are private. You can also use `input_schema` and `output_schema` on the subgraph for explicit control.

**Q: Can a subgraph have its own checkpointer?**
Yes. By default, subgraphs inherit the parent's checkpointer and thread_id. You can attach a separate checkpointer for isolation — useful for reusable modules that should maintain independent execution history.

**Q: When would you use subgraphs vs just calling a function from a node?**
Use subgraphs when: (1) the module has its own internal branching/loops, (2) you want independent testability, (3) multiple teams own different modules, (4) you need the module to be independently versioned and swappable.

---

## LESSON 12 — RAG Agent (Retrieval-Augmented Generation)

### Theory

#### The Problem RAG Solves
LLMs have a training cutoff — they don't know about your company's documents, your latest data, or recent events. RAG grounds the LLM's answers in real, current documents.

#### RAG pipeline
```
User question
     ↓
1. EMBED question → vector representation [0.1, 0.8, -0.3, ...]
     ↓
2. SEARCH vector store → find top-K most similar document chunks
     ↓
3. BUILD context → combine retrieved chunks with question
     ↓
4. GENERATE → LLM reads question + chunks → produces grounded answer
```

#### Component guide
| Component | Purpose | Open source option |
|-----------|---------|-------------------|
| Document loader | Load PDFs, text, web pages | LangChain loaders |
| Text splitter | Chunk large docs | `RecursiveCharacterTextSplitter` |
| Embedding model | Text → vector | `OllamaEmbeddings` (free, local) |
| Vector store | Store + similarity search | `Chroma` (local), `FAISS` (local) |
| Retriever | Query the vector store | `.as_retriever()` |

#### Agentic RAG vs Basic RAG
| | Basic RAG | Agentic RAG |
|---|-----------|------------|
| Decision making | Always retrieves | Agent decides if retrieval is needed |
| Query strategy | Single query | Can reformulate, multi-query |
| Result validation | No | Checks relevance of retrieved docs |
| Fallback | None | Falls back to LLM knowledge |

#### Chunking strategy matters
| Document type | Chunk size | Overlap |
|--------------|------------|---------|
| Short articles | 300–500 | 50 |
| Books / long docs | 500–1000 | 100 |
| Code files | By function/class | 0 |
| Tables | By row or section | 0 |

### File: `lesson_12_rag_agent/lesson_12_rag_agent.py`

```python
import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- Setup vector store ---
DOCUMENTS = [
    "LangGraph is a library for building stateful multi-actor LLM applications.",
    "LangGraph uses a StateGraph to define the workflow as a directed graph.",
    "Nodes are Python functions that read and update the shared state.",
    "Edges define execution order. Conditional edges allow branching.",
    "Human-in-the-loop uses interrupt() to pause the graph for human input.",
    "MemorySaver saves state in RAM. SqliteSaver saves to disk.",
    "ReAct agents loop between agent and tools until the task is complete.",
    "The supervisor pattern routes tasks to specialist agents.",
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
docs = [Document(page_content=d) for d in DOCUMENTS]
chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="llama3.2", temperature=0)

# --- Tool: retrieval ---
@tool
def retrieve_documents(query: str) -> str:
    """
    Search the knowledge base for relevant information about the query.
    Use this for any question about LangGraph concepts, features, or patterns.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(f"[Doc {i+1}]: {d.page_content}" for i, d in enumerate(docs))

# --- State ---
class RAGState(TypedDict):
    messages: Annotated[list, add_messages]

# --- Agent ---
rag_llm = llm.bind_tools([retrieve_documents])
SYSTEM = """You are a LangGraph expert assistant.
ALWAYS use retrieve_documents() first to find relevant information before answering.
Base your answer on the retrieved documents. If documents don't contain the answer, say so."""

def rag_agent(state: RAGState) -> dict:
    msgs = [SystemMessage(content=SYSTEM)] + state["messages"]
    return {"messages": [rag_llm.invoke(msgs)]}

def should_retrieve(state: RAGState) -> str:
    last = state["messages"][-1]
    return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "end"

builder = StateGraph(RAGState)
builder.add_node("agent", rag_agent)
builder.add_node("tools", ToolNode([retrieve_documents]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_retrieve, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")
graph = builder.compile()

if __name__ == "__main__":
    questions = [
        "What is a StateGraph?",
        "How does human-in-the-loop work?",
        "What is the difference between MemorySaver and SqliteSaver?",
    ]
    for q in questions:
        print(f"\n❓ {q}")
        result = graph.invoke({"messages": [HumanMessage(content=q)]})
        print(f"💬 {result['messages'][-1].content[:300]}")
```

---

### Tasks — Lesson 12

**Task 12.1 — Document Q&A**
Load a PDF or text file of your choice into Chroma. Build a RAG agent that retrieves top-3 relevant chunks and **cites** which chunk each fact came from.

**Task 12.2 — Multi-Source RAG**
Build a RAG agent with 3 separate Chroma collections: `technical_docs`, `company_policies`, `product_catalog`. Agent selects which collection to search based on question type.

**Task 12.3 — RAG with Self-Correction**
After retrieving, add a `relevance_check` node:
- LLM evaluates if retrieved docs actually answer the question (score 1-5)
- If score < 3: reformulate the query and retrieve again (max 2 retries)
- If still not relevant: answer from LLM knowledge and flag as "not from docs"

**Task 12.4 — Incremental Ingestion**
Build a tool `add_document(text: str, source: str)` that the agent can call to add new documents to the vector store during a conversation. Test: tell the agent a new fact, then ask it about that fact.

---

### Interview Q&A — Lesson 12

**Q: What is the difference between dense retrieval and sparse retrieval?**
Dense retrieval uses embedding vectors (semantic similarity) — finds conceptually similar content even with different words. Sparse retrieval (BM25, TF-IDF) uses keyword matching — finds exact or near-exact word matches. Production systems often use **hybrid retrieval** — combine both for better recall.

**Q: How do you handle chunking for PDFs with tables and images?**
Tables need special parsing — use `pdfplumber` to extract table data as structured text, then chunk by table row or section. Images need a vision model to extract text (OCR + LLM). For critical document types, always test retrieval quality manually before deploying.

**Q: How do you evaluate RAG quality?**
Use the RAGAS framework: (1) **Faithfulness** — is the answer grounded in retrieved docs? (2) **Answer Relevancy** — does the answer address the question? (3) **Context Recall** — did we retrieve the right docs? (4) **Context Precision** — are retrieved docs all relevant?

---

## LESSON 13 — Long-Term Memory with Vector Stores

### Theory

#### Why vector-based long-term memory?
Checkpointers (Lessons 8) save **structured state** per thread. But what about:
- Facts a user mentioned 3 weeks ago in a different thread?
- Preferences stated once but relevant everywhere?
- Knowledge accumulated across thousands of conversations?

Vector stores enable **semantic memory** — retrieve relevant memories by meaning, not by exact key.

#### Memory types in AI agents
```
In-context memory:    Full conversation history in the LLM prompt (limited by context window)
External DB memory:   Structured facts in a database (exact lookup)
Vector store memory:  Semantic memories retrievable by meaning similarity (what we build here)
```

#### The Memory Pattern
```
1. STORE:    After each conversation, extract key facts → embed → store in vector DB
2. RETRIEVE: At start of new conversation, search for relevant past memories
3. INJECT:   Add retrieved memories as SystemMessage context before the LLM call
```

### File: `lesson_13_vector_memory/lesson_13_vector_memory.py`

```python
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="llama3.2")
memory_store = Chroma(persist_directory="./memory_db", embedding_function=embeddings)

class MemoryState(TypedDict):
    messages:       Annotated[list, add_messages]
    user_id:        str
    loaded_memories: list

def load_memories(state: MemoryState) -> dict:
    """Retrieve relevant past memories before responding."""
    if not state["messages"]:
        return {"loaded_memories": []}
    last_msg = state["messages"][-1].content
    results = memory_store.similarity_search(last_msg, k=3, filter={"user_id": state["user_id"]})
    memories = [doc.page_content for doc in results]
    return {"loaded_memories": memories}

def chat_node(state: MemoryState) -> dict:
    """Chat with memory context."""
    memory_ctx = ""
    if state.get("loaded_memories"):
        memory_ctx = "Relevant memories from past conversations:\n" + "\n".join(f"- {m}" for m in state["loaded_memories"])
    system = SystemMessage(content=f"You are a helpful assistant with memory.\n{memory_ctx}")
    resp = llm.invoke([system] + state["messages"])
    return {"messages": [resp]}

def save_memories(state: MemoryState) -> dict:
    """Extract and store key facts from this conversation turn."""
    if len(state["messages"]) < 2:
        return {}
    last_human = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    last_ai    = state["messages"][-1].content if isinstance(state["messages"][-1], AIMessage) else ""
    extract_prompt = f"""Extract 1-3 key facts to remember about the user from this exchange.
User said: {last_human}
AI responded: {last_ai}
Return ONLY a JSON list: ["fact1", "fact2"] or [] if nothing worth remembering."""
    raw = llm.invoke([HumanMessage(content=extract_prompt)]).content
    try:
        start = raw.find("[")
        facts = json.loads(raw[start:raw.rfind("]")+1])
        for fact in facts:
            memory_store.add_documents([Document(
                page_content=fact,
                metadata={"user_id": state["user_id"]}
            )])
    except Exception:
        pass
    return {}

builder = StateGraph(MemoryState)
builder.add_node("load_memories", load_memories)
builder.add_node("chat",          chat_node)
builder.add_node("save_memories", save_memories)
builder.add_edge(START,          "load_memories")
builder.add_edge("load_memories", "chat")
builder.add_edge("chat",          "save_memories")
builder.add_edge("save_memories", END)
graph = builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    user_id = "user-001"
    config = {"configurable": {"thread_id": f"mem-{user_id}"}}
    conversations = [
        "My name is Ahmed and I work as a data engineer.",
        "I really love Python and LangGraph.",
        "I have 5 years of experience with machine learning.",
        "What do you know about me?",  # Should recall previous facts
    ]
    for msg in conversations:
        print(f"\n👤 {msg}")
        result = graph.invoke({"messages": [HumanMessage(content=msg)], "user_id": user_id, "loaded_memories": []}, config=config)
        print(f"🤖 {result['messages'][-1].content[:200]}")
```

---

### Tasks — Lesson 13

**Task 13.1 — Memory Categories**
Organize memories into categories: `preferences`, `facts`, `goals`, `dislikes`. Store category as metadata. When retrieving, search by relevant category based on question type.

**Task 13.2 — Memory Decay**
Implement memory importance scoring. Each memory has a score (1-10) and a last_accessed timestamp. When memory store reaches 100 items, delete the lowest-scored oldest memories. Increment score each time a memory is retrieved (reinforcement).

**Task 13.3 — Cross-Thread Memory**
Build a system where user profile memories are global (accessible from any thread_id) while conversation memories are thread-specific. Implement with two separate Chroma collections.

**Task 13.4 — Memory Contradiction Handling**
When adding a new fact that contradicts an existing memory (e.g., "I hate Python" when memory says "loves Python"), detect the contradiction and update/replace the old memory instead of adding a duplicate.

---

### Interview Q&A — Lesson 13

**Q: What is the difference between episodic and semantic memory in AI agents?**
Episodic memory = specific past events ("User mentioned X in conversation on Jan 5"). Semantic memory = general knowledge ("User prefers Python over Java"). Production systems combine both: episodic for precise recall, semantic for general personalization.

**Q: How do you handle memory privacy and user data separation?**
Use metadata filtering: every memory document includes `{"user_id": "..."}`. All similarity searches pass `filter={"user_id": user_id}`. Chroma, Pinecone, and Weaviate all support metadata filtering. For deletion: use `collection.delete(where={"user_id": user_id})`.

**Q: When does vector memory outperform a simple database for agent memory?**
When you need to retrieve by **meaning** rather than exact key. Example: user asks "What did I say about food?" — a vector search finds all food-related memories even if they don't contain the word "food". A SQL `WHERE content LIKE '%food%'` would miss "I love pizza" but not "I love food".

---

## LESSON 14 — Testing and Evaluating Agents

### Theory

#### Why agents are hard to test
- Non-deterministic outputs (LLMs are probabilistic)
- Long execution paths with many conditional branches
- External dependencies (database, tools, LLM)
- State-dependent behavior

#### 4 testing levels

**1. Unit tests — test each tool independently**
```python
def test_run_sql_rejects_delete():
    result = run_sql.invoke({"query": "DELETE FROM employees"})
    assert "ERROR" in result

def test_run_sql_returns_results():
    result = run_sql.invoke({"query": "SELECT * FROM employees LIMIT 1"})
    assert "Columns:" in result
```

**2. Node tests — test each node function**
```python
def test_supervisor_routes_to_db_agent():
    state = {"messages": [HumanMessage("How many employees?")], "user_id": "test", "next_agent": ""}
    result = supervisor(state)
    assert result["next_agent"] in ["db_agent", "analyst"]
```

**3. Integration tests — test full graph paths**
```python
def test_full_db_query_path():
    result = graph.invoke({
        "messages": [HumanMessage("How many employees are in Engineering?")],
        "user_id": "test", "next_agent": ""
    }, config={"configurable": {"thread_id": "test-1"}, "recursion_limit": 25})
    # Check the answer contains a number
    last_msg = result["messages"][-1].content
    assert any(char.isdigit() for char in last_msg)
```

**4. Evaluation — assess quality of LLM outputs**
Use reference answers (ground truth) to score agent responses:
```python
EVAL_CASES = [
    {"question": "How many employees?", "expected_contains": ["6", "six"]},
    {"question": "Top paid employee?",  "expected_contains": ["Alice", "95000"]},
]
```

### File: `lesson_14_testing/test_agents.py`

```python
import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

# --- Tool Unit Tests ---
def test_list_tables_returns_tables():
    """list_tables should return all table names."""
    from lesson_06_database_agent.lesson_06_database_agent import list_tables
    result = list_tables.invoke({})
    assert "employees" in result.lower() or "Tables" in result

def test_run_sql_blocks_delete():
    """run_sql must reject DELETE queries."""
    from lesson_10_capstone.lesson_10_capstone import run_sql
    result = run_sql.invoke({"query": "DELETE FROM employees WHERE id=1"})
    assert "ERROR" in result or "Only SELECT" in result

def test_run_sql_accepts_select():
    """run_sql must execute SELECT queries."""
    from lesson_10_capstone.lesson_10_capstone import run_sql, setup_database
    setup_database()
    result = run_sql.invoke({"query": "SELECT COUNT(*) FROM employees"})
    assert "Cols:" in result or "1" in result

# --- Graph Unit Tests ---
def test_graph_compiles():
    """Graph compilation should not raise errors."""
    from lesson_10_capstone.lesson_10_capstone import build_capstone_graph, setup_database
    setup_database()
    graph = build_capstone_graph()
    assert graph is not None

# --- Mock LLM Tests ---
def test_supervisor_routing_with_mock_llm():
    """Supervisor should route correctly — test without real LLM."""
    import json
    from unittest.mock import patch
    from langchain_core.messages import AIMessage

    mock_response = AIMessage(content='{"next": "db_agent"}')
    with patch("lesson_10_capstone.lesson_10_capstone.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        from lesson_10_capstone.lesson_10_capstone import supervisor_node
        state = {"messages": [HumanMessage("How many employees?")], "user_id": "test", "next_agent": ""}
        result = supervisor_node(state)
        assert result["next_agent"] == "db_agent"

# --- Evaluation Tests ---
EVAL_CASES = [
    {
        "question": "How many employees are there?",
        "expected_keywords": ["6", "six", "employees"],
        "description": "Should answer with count"
    },
    {
        "question": "What department has the highest budget?",
        "expected_keywords": ["engineering", "500000", "500,000"],
        "description": "Should mention Engineering department"
    }
]

@pytest.mark.integration
@pytest.mark.parametrize("case", EVAL_CASES)
def test_agent_evaluation(case):
    """Integration test: agent answers questions correctly."""
    from lesson_10_capstone.lesson_10_capstone import build_capstone_graph, setup_database
    from langgraph.checkpoint.memory import MemorySaver
    setup_database()
    graph = build_capstone_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": f"eval-{case['question'][:20]}"}, "recursion_limit": 25}
    result = graph.invoke({
        "messages": [HumanMessage(content=case["question"])],
        "user_id": "eval-user", "next_agent": ""
    }, config=config)
    last_msg = next((m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)), "")
    matched = any(kw.lower() in last_msg.lower() for kw in case["expected_keywords"])
    assert matched, f"FAIL: '{case['description']}'\nAnswer: {last_msg[:200]}\nExpected one of: {case['expected_keywords']}"
```

---

### Tasks — Lesson 14

**Task 14.1 — Tool Test Suite**
Write unit tests for every tool in Lesson 6 and Lesson 10. Cover: normal case, error case, boundary case (empty input, very long input, SQL injection attempt).

**Task 14.2 — Mock LLM Tests**
Write tests for your supervisor routing using `unittest.mock.patch`. Test all routing paths (db_agent, analyst, human_review, FINISH) by controlling what the mock LLM returns.

**Task 14.3 — Evaluation Dataset**
Create a 10-question evaluation dataset with expected answers for your capstone agent. Write a script that runs all 10 questions and reports: pass rate, which questions failed, average response time.

**Task 14.4 — CI/CD Pipeline**
Write a `Makefile` or `pytest.ini` that runs: (1) fast unit tests (no LLM, <5 seconds), (2) integration tests (with LLM, marked `@pytest.mark.integration`), (3) evaluation tests (with scoring). Add instructions for running only fast tests during development.

---

### Interview Q&A — Lesson 14

**Q: How do you test non-deterministic LLM behavior reliably?**
Three approaches: (1) Mock the LLM for unit tests — test logic, not the LLM. (2) Use `temperature=0` for eval tests — maximizes reproducibility. (3) Run evaluation N times and use statistical pass/fail thresholds (e.g., answer must be correct 8/10 runs). For production monitoring, track answer quality metrics over time.

**Q: What is LLM-as-a-judge evaluation?**
Use a second, more powerful LLM to evaluate the first LLM's answers. Prompt it: "Given question X and answer Y, is this answer correct, complete, and well-formatted? Score 1-5." More scalable than human evaluation for large test sets. Automate with structured output.

**Q: How do you do regression testing when you change the agent?**
Maintain a "golden dataset" — a set of question/expected-answer pairs that cover all critical paths. Run this dataset before and after any change. If pass rate drops by more than 5%, block the change. Store golden dataset results in version control.

---

## LESSON 15 — Deploying Agents (LangGraph Platform & FastAPI)

### Theory

#### Deployment options

| Option | Best for | Complexity |
|--------|---------|------------|
| FastAPI wrapper | Custom control | Medium |
| LangGraph Platform | Managed service | Low |
| Docker container | Any cloud | Medium |
| Cloud Functions | Serverless | Medium |

#### FastAPI deployment pattern

```python
# api.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from lesson_10_capstone.lesson_10_capstone import build_capstone_graph, setup_database
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

app = FastAPI(title="LangGraph Agent API")
setup_database()
checkpointer = SqliteSaver.from_conn_string("production.db").__enter__()
graph = build_capstone_graph(checkpointer=checkpointer)

class QuestionRequest(BaseModel):
    question: str
    user_id: str
    thread_id: str = None

class ApprovalRequest(BaseModel):
    thread_id: str
    user_id: str
    decision: str  # "approve" or "reject"

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    thread_id = req.thread_id or f"{req.user_id}-{int(__import__('time').time())}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    result = graph.invoke({
        "messages": [HumanMessage(content=req.question)],
        "user_id": req.user_id, "next_agent": ""
    }, config=config)
    state = graph.get_state(config)
    if state.next:
        interrupt_data = state.tasks[0].interrupts[0].value if state.tasks else {}
        return {"status": "awaiting_approval", "thread_id": thread_id, "interrupt": interrupt_data}
    last_ai = next((m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)), "")
    return {"status": "complete", "thread_id": thread_id, "answer": last_ai}

@app.post("/approve")
async def approve_action(req: ApprovalRequest):
    config = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 25}
    result = graph.invoke(Command(resume=req.decision), config=config)
    last_ai = next((m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)), "")
    return {"status": "complete", "answer": last_ai}

@app.get("/sessions/{user_id}")
async def get_sessions(user_id: str):
    return {"user_id": user_id, "sessions": []}  # implement with checkpointer listing
```

#### Running the API
```bash
pip install fastapi uvicorn
uvicorn lesson_15_deployment.api:app --host 0.0.0.0 --port 8000 --reload
```

#### Key deployment considerations

| Concern | Solution |
|---------|---------|
| Concurrent users | PostgresSaver (thread-safe) |
| API authentication | FastAPI middleware + JWT tokens |
| Rate limiting | FastAPI `slowapi` middleware |
| Secrets management | Environment variables, never hardcode |
| Monitoring | Prometheus + Grafana, LangSmith |
| Scaling | Horizontal with shared DB checkpointer |

---

### Tasks — Lesson 15

**Task 15.1 — FastAPI Wrapper**
Build a complete FastAPI application for your capstone agent with endpoints:
- `POST /ask` — submit a question
- `POST /approve` — approve/reject HITL decisions
- `GET /sessions/{user_id}` — list user's sessions
- `DELETE /sessions/{thread_id}` — delete a session (GDPR)

**Task 15.2 — Dockerize**
Write a `Dockerfile` for your API. Build and run the container. Test all endpoints.
`Dockerfile` hint: use `python:3.11-slim`, install requirements, copy code, expose port 8000.

**Task 15.3 — Authentication Middleware**
Add API key authentication: clients must pass `X-API-Key: your-key` header. Return 401 if missing/invalid. Store valid keys in a `.env` file (never hardcode them).

**Task 15.4 — LangSmith Tracing**
Set up LangSmith (free tier) to trace your agent's execution:
- Set environment variables: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=...`
- Run 5 questions through your agent
- In LangSmith UI, inspect the full trace for each question

---

### Interview Q&A — Lesson 15

**Q: How do you handle concurrent users in production LangGraph?**
(1) Replace `SqliteSaver` with `PostgresSaver` — thread-safe, multi-process. (2) Deploy multiple API instances behind a load balancer. (3) Use connection pooling (`asyncpg` or `psycopg2.pool`). (4) Each user's `thread_id` is isolated — no shared mutable state between users. (5) LLM calls are the bottleneck — scale with multiple Ollama instances or use cloud LLM APIs.

**Q: How do you version your LangGraph agents without breaking existing sessions?**
(1) Store graph version in checkpoint metadata alongside state. (2) When loading a checkpoint, check its version number. (3) Implement migration functions: `migrate_v1_to_v2(state)`. (4) Run migrations lazily (when session is first loaded after deployment). (5) Keep old graph versions importable for a transition period.

**Q: What is LangSmith and why do you need it in production?**
LangSmith is a tracing and evaluation platform by LangChain. It records every LLM call, tool execution, and node transition. In production you need it because: (1) debugging production failures without traces is nearly impossible, (2) you can replay failed traces exactly, (3) it provides latency and cost monitoring per node, (4) you can build evaluation datasets from real production traffic.
