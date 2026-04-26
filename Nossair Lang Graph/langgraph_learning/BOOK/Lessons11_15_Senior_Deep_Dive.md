# Lessons 11–15 Senior Deep Dive: Subgraphs, RAG, Memory, Testing, Deployment

---

# Lesson 11 — Subgraphs: Complete Deep Dive

> **Prerequisite:** Open `lesson_11_subgraphs/lesson_11_subgraphs.py` and run it first.

---

## Real-World Analogy

Think of **microservices in software architecture**:
- A microservice is a self-contained service with its own API, database, logic
- Other services call it without knowing its internals
- You can deploy, scale, test, and update it independently
- Multiple systems reuse the same microservice

A **subgraph** is the LangGraph equivalent: a fully compiled graph used as a single node in a parent graph. Same isolation, same reusability, same testability.

---

## What Is a Subgraph and Why Use It?

### Without subgraphs (everything in one graph)

```python
# ❌ Monolithic graph — all validation, cleaning, generation in one flat graph
builder.add_node("check_length",   check_length_fn)
builder.add_node("check_forbidden", check_forbidden_fn)
builder.add_node("check_format",   check_format_fn)
builder.add_node("clean_whitespace", clean_ws_fn)
builder.add_node("fix_caps",       fix_caps_fn)
builder.add_node("add_punctuation", add_punct_fn)
builder.add_node("generate",       generate_fn)
builder.add_node("publish",        publish_fn)
# 8 nodes, all in one graph, no reuse possible
# Can't test validation independently
# Can't share validation logic with another graph
```

### With subgraphs (modular)

```python
# ✅ Validation compiled once, reusable everywhere
validation_subgraph = build_validation_graph()  # 3 nodes internally
clean_subgraph      = build_clean_graph()        # 3 nodes internally

# Parent graph sees them as single nodes
parent_builder.add_node("validate", validation_subgraph)  # subgraph AS node
parent_builder.add_node("clean",    clean_subgraph)       # subgraph AS node
parent_builder.add_node("generate", generate_fn)
parent_builder.add_node("publish",  publish_fn)
# Only 4 nodes visible in parent — clean architecture
```

---

## How State Sharing Works

### The key rule: shared field names

When a subgraph is called, LangGraph automatically shares state fields whose **names match** between parent and subgraph:

```python
# Parent state
class ParentState(TypedDict):
    content:           str    # ← SHARED (same name in subgraph)
    is_valid:          bool   # ← SHARED
    validation_errors: list   # ← SHARED
    parent_only_field: str    # ← NOT shared (doesn't exist in subgraph)

# Subgraph state
class ValidationState(TypedDict):
    content:           str    # ← matches parent → automatically shared
    is_valid:          bool   # ← matches parent → automatically shared
    validation_errors: list   # ← matches parent → automatically shared
    # No parent_only_field — subgraph doesn't see it
```

### State flow diagram

```
Parent invokes subgraph node:
  1. Parent state: {content: "Hello", is_valid: False, parent_only: "X"}
  2. LangGraph extracts matching keys for subgraph: {content: "Hello", is_valid: False}
  3. Subgraph runs its internal nodes using these values
  4. Subgraph returns: {is_valid: True, validation_errors: []}
  5. LangGraph merges back to parent state — only matching keys updated
  6. Parent state: {content: "Hello", is_valid: True, parent_only: "X"}
```

### Custom state transformation (when field names differ)

If your parent and subgraph use different field names, use a wrapper:

```python
def validation_wrapper(state: ParentState) -> dict:
    """Translate parent state → subgraph input → run subgraph → translate back."""
    subgraph_input = {
        "text":   state["content"],    # parent uses "content", subgraph uses "text"
        "valid":  state["is_valid"],
    }
    result = validation_subgraph.invoke(subgraph_input)
    return {
        "is_valid":          result["valid"],
        "validation_errors": result.get("errors", [])
    }
```

---

## Building and Testing Subgraphs

### Build the subgraph independently

```python
def build_validation_subgraph() -> CompiledStateGraph:
    """Build and compile validation logic as a standalone reusable subgraph."""
    builder = StateGraph(ValidationState)
    builder.add_node("check_length",    check_length_fn)
    builder.add_node("check_forbidden", check_forbidden_fn)
    builder.add_node("check_format",    check_format_fn)
    builder.add_edge(START, "check_length")
    builder.add_edge("check_length",    "check_forbidden")
    builder.add_edge("check_forbidden", "check_format")
    builder.add_edge("check_format",    END)
    return builder.compile()

# Compile ONCE at module level
validation_subgraph = build_validation_subgraph()
```

### Test the subgraph independently (no parent graph needed)

```python
# Unit test the subgraph in complete isolation
def test_validation_subgraph():
    # Test 1: valid input passes all checks
    result = validation_subgraph.invoke({
        "content": "Great Python toolkit for developers.",
        "is_valid": True,
        "validation_errors": []
    })
    assert result["is_valid"] == True
    assert len(result["validation_errors"]) == 0

    # Test 2: short content fails length check
    result = validation_subgraph.invoke({
        "content": "Hi",
        "is_valid": True,
        "validation_errors": []
    })
    assert result["is_valid"] == False
    assert any("short" in e.lower() for e in result["validation_errors"])
```

---

## Parallel Subgraphs with Send()

```python
from langgraph.types import Send

class ParallelAnalysisState(TypedDict):
    documents: list[str]
    analyses:  Annotated[list, lambda x, y: x + y]  # merge parallel results

def fan_out_to_subgraphs(state: ParallelAnalysisState):
    """Launch one analysis subgraph per document, all in parallel."""
    return [Send("analyze_doc", {"document": doc, "result": ""})
            for doc in state["documents"]]

# Result: 5 documents analyzed in parallel instead of sequentially
# Each runs the full analysis subgraph independently
# Results merged by the lambda reducer
```

---

## Anti-Patterns — Lesson 11

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| **Compiling subgraph inside node** | Recompiled on every call — very slow | Compile once at module level |
| **Subgraph with 20+ nodes** | Defeats the purpose — split further | Keep subgraphs focused (3-8 nodes) |
| **Parent and subgraph state fully identical** | No isolation benefit | Subgraph state = only the fields it needs |
| **Not testing subgraph independently** | Bugs buried in parent integration | Always test subgraph.invoke() in isolation first |
| **Passing parent checkpointer to subgraph** | State bleeding between subgraph instances | Subgraphs compile without checkpointer unless explicitly needed |

---

## Best Practices Checklist — Lesson 11

```
□ Each subgraph solves ONE cohesive responsibility
□ Subgraph compiled once at module level (not inside nodes)
□ Subgraph state contains only fields it actually uses
□ Subgraph unit tested in isolation before parent integration
□ State sharing uses matching field names (no translation needed)
□ Parallel subgraphs use Send() with Annotated merge reducers
□ Subgraphs documented: what inputs they need, what they output
□ Subgraph can be imported and used in OTHER projects (true reusability)
```

---

## Tasks — Lesson 11

**Task 11.1 — Email Processing Pipeline**
Build 3 subgraphs:
- `spam_filter_subgraph`: checks sender, keywords, formatting → `{is_spam: bool, reason: str}`
- `categorize_subgraph`: routes to inbox/promotions/updates → `{category: str}`
- `summarize_subgraph`: LLM produces 1-sentence summary → `{summary: str}`
Parent graph runs all 3 in sequence. Test each subgraph independently first.

**Task 11.2 — Document Processing with Parallel Subgraphs**
Given a list of 5 documents, use `Send()` to run an `analyze_subgraph` on all 5 in parallel.
Each subgraph: count words, detect language, extract top 3 keywords.
Parent `combine_node` merges results and produces a comparison table.
Benchmark: compare time for parallel vs sequential processing.

**Task 11.3 — Reuse Across Projects**
Take your `validation_subgraph` from Task 11.1.
Import it into a completely different graph (e.g., your Lesson 4 agent).
Add it as a validation step before the agent processes each input.
This proves true subgraph reusability.

---

## Interview Q&A — Lesson 11

**Q1: What is a subgraph in LangGraph and why would you use one?**
A subgraph is a fully compiled `StateGraph` used as a node inside a parent graph. The parent calls it like any other node. Benefits: (1) **Reusability** — compile once, use in multiple parent graphs. (2) **Isolation** — subgraph state is independent; only shared field names cross the boundary. (3) **Testability** — test the subgraph with `subgraph.invoke()` completely independently. (4) **Team ownership** — different teams own different subgraphs; integrate via shared state schema. (5) **Complexity management** — parent graph stays clean (4-6 nodes) even when total logic is 20+ nodes.

**Q2: How does state sharing work between parent and subgraph?**
LangGraph automatically shares state fields whose names match between parent TypedDict and subgraph TypedDict. Parent-only fields are invisible to the subgraph. Subgraph-only fields don't appear in parent state. When the subgraph completes, only matching keys are merged back. This is intentional isolation — subgraphs can't accidentally corrupt parent state fields they don't know about.

**Q3: When should you use a subgraph vs a regular node?**
Use a subgraph when: (1) the logic requires its own branching or conditional edges, (2) you want to reuse the logic in multiple parent graphs, (3) the logic has 3+ steps that belong together, (4) you want to test the logic independently. Use a regular node when: the logic is a single function call, it has no internal branching, it's specific to this one graph. Rule of thumb: if you could imagine it as a separate microservice, make it a subgraph.

---

# Lesson 12 — RAG Agent: Complete Deep Dive

> **Prerequisite:** Run `pip install langchain-community chromadb` first.

---

## Why RAG Exists — The LLM Knowledge Problem

LLMs are trained on data up to a cutoff date. They have no knowledge of:
- Your company's internal documents
- Your product's specific features
- Events after their training cutoff
- Your proprietary data and processes

RAG (Retrieval-Augmented Generation) solves this by giving the agent access to your documents at query time.

```
WITHOUT RAG: "What is our refund policy?" → LLM guesses or makes up answer
WITH RAG:    "What is our refund policy?" → Search docs → Find policy doc → Answer from real policy
```

---

## RAG Architecture — All 4 Stages

### Stage 1: Indexing (done once, offline)

```python
# 1. Load your documents
from langchain_community.document_loaders import TextLoader, DirectoryLoader
loader = DirectoryLoader("./company_docs/", glob="**/*.txt")
docs = loader.load()

# 2. Split into chunks (each chunk = one vector)
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # characters per chunk
    chunk_overlap=50,     # overlap prevents context loss at boundaries
    separators=["\n\n", "\n", ".", " "]  # split on paragraphs first
)
chunks = splitter.split_documents(docs)

# 3. Embed each chunk (convert text → vector of numbers)
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="llama3.2")

# 4. Store in vector database
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # persists to disk
)
```

### Stage 2: Retrieval (at query time)

```python
# User's question is embedded, compared to all stored vectors
# Returns top-k most similar chunks (by cosine similarity)
results = vectorstore.similarity_search(
    "What is the refund policy?",
    k=3   # return top 3 chunks
)
# results[i].page_content = the text chunk
# results[i].metadata = source, page, etc.
```

### Stage 3: Augmentation (inject into prompt)

```python
context = "\n\n".join(doc.page_content for doc in results)
augmented_prompt = f"""Answer based ONLY on the following documents.
If the answer is not in the documents, say "I don't have information about that."

DOCUMENTS:
{context}

QUESTION: {question}"""
```

### Stage 4: Generation (LLM answers from context)

```python
answer = llm.invoke([
    SystemMessage(content="You are a helpful assistant. Answer only from provided documents."),
    HumanMessage(content=augmented_prompt)
])
```

---

## Chunk Size — The Most Important Hyperparameter

| Chunk Size | Pros | Cons | Best for |
|-----------|------|------|---------|
| 100-200 chars | Very precise retrieval | Too small — loses context | Short facts, Q&A |
| 300-500 chars | Good balance | May split mid-sentence | General use |
| 500-1000 chars | Rich context | Less precise matching | Long-form docs |
| 1000-2000 chars | Full paragraphs | Returns too much | Technical manuals |

**Practical rule:** Start at 500 with 50 overlap. Test retrieval quality. Adjust based on your document structure.

---

## Agentic RAG vs Simple RAG

### Simple RAG (fixed pipeline)

```
Question → Retrieve → Augment → Generate → Answer
```

Problem: retrieves based on literal question text. If question is ambiguous → retrieves wrong docs.

### Agentic RAG (agent decides when/how to retrieve)

```python
# Agent has retrieve_documents as a TOOL
# LLM decides: does this question need retrieval? which query to use?
# Agent can rephrase, retrieve again, or combine multiple searches

@tool
def retrieve_documents(query: str, k: int = 3) -> str:
    """
    Search the knowledge base for relevant information.
    Use this for any question that may be answered by our documentation.
    Rephrase the query to use keywords likely to appear in the documents.
    Returns the most relevant document chunks with source information.
    """
    results = vectorstore.similarity_search(query, k=k)
    if not results:
        return "No relevant documents found."
    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Doc {i} from {source}]:\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)
```

---

## Self-Correcting RAG — Handling Poor Retrieval

```
Initial query → retrieve → check relevance score
                              ↓
                    score < 0.5 → refine query → retrieve again (max 2 retries)
                    score ≥ 0.5 → generate answer
```

```python
def check_relevance(state: RAGState) -> dict:
    """Score how well retrieved docs match the question."""
    check_prompt = f"""On a scale 0.0 to 1.0, how relevant are these documents to the question?
Question: {state['query']}
Documents: {state['retrieved_docs'][:500]}
Reply with ONLY a number like: 0.8"""
    resp = llm.invoke([HumanMessage(content=check_prompt)])
    try:
        score = float(resp.content.strip().split()[0])
    except (ValueError, IndexError):
        score = 0.5   # default if parsing fails
    return {"relevance_score": score}

def route_by_relevance(state: RAGState) -> str:
    if state["relevance_score"] >= 0.5 or state["retry_count"] >= 2:
        return "generate"
    return "refine"   # rephrase query and try again
```

---

## Anti-Patterns — Lesson 12

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| **Very large chunks (2000+ chars)** | Poor retrieval precision | Use 300-500 chars with overlap |
| **No overlap between chunks** | Context lost at chunk boundaries | Use chunk_overlap=50-100 |
| **LLM answers without retrieval** | Hallucinations from training data | Strict system prompt: answer ONLY from docs |
| **k=1 (top result only)** | Single chunk rarely has full answer | k=3 to k=5 for coverage |
| **No source citation** | Can't verify accuracy | Always include source metadata in retrieved text |
| **Rebuilding index on every query** | Extremely slow | Index once offline, load from persist_directory |
| **No relevance check** | Returns unrelated docs, LLM makes up answers | Check similarity score, refine if < threshold |

---

## Tasks — Lesson 12

**Task 12.1 — Your Own Knowledge Base**
Create a knowledge base from 3-5 text files about topics you know well (Python, databases, your company's processes).
Index them, then test 10 questions. For each: print retrieved chunks + final answer.
Which questions get good answers? Which don't? Why?

**Task 12.2 — Source Attribution**
Modify the RAG agent so every answer ends with:
```
Sources: doc1.txt (paragraph 3), doc2.txt (paragraph 1)
```
Implement this by including `metadata["source"]` and `metadata["paragraph"]` in the retrieval tool output.

**Task 12.3 — Multi-Collection RAG**
Build 2 separate Chroma collections: `technical_docs` and `hr_policies`.
The agent first classifies the question (technical vs HR), then searches only the relevant collection.
Test: "How does authentication work?" (→ technical), "What is the vacation policy?" (→ HR).

**Task 12.4 — Evaluate RAG Quality**
Create a test set: 10 questions + expected answers.
Run each through RAG agent. Score: 1 if answer contains expected keywords, 0 if not.
Calculate accuracy. Identify failing questions and diagnose why (wrong chunk size? bad retrieval?).

---

## Interview Q&A — Lesson 12

**Q1: What is RAG and why is it better than fine-tuning for most use cases?**
RAG retrieves relevant documents at query time and injects them into the LLM prompt. Fine-tuning bakes knowledge into model weights. RAG is better when: (1) data changes frequently (fine-tuned model would be stale), (2) you need source attribution ("answer based on doc X"), (3) you have a small dataset (fine-tuning needs thousands of examples), (4) you need to audit what the LLM used to answer. Fine-tuning is better when: you need behavior change (new writing style, new task format) rather than new knowledge.

**Q2: What is chunk size and how do you choose the right value?**
Chunk size is the maximum character/token count per stored vector. Too small (< 200 chars): precise retrieval but each chunk lacks full context. Too large (> 1000 chars): rich context but imprecise retrieval (unrelated parts retrieved). Choose based on: (1) document structure — paragraphs naturally 200-500 chars? use 400, (2) question specificity — very specific questions need smaller chunks, (3) test empirically — try 300 and 600, compare answer quality on 20 test questions.

**Q3: How do you prevent the LLM from hallucinating when relevant documents aren't found?**
Three approaches: (1) Strict system prompt: "Answer ONLY from the provided documents. If the answer is not in the documents, say 'I don't have information about that in my knowledge base.'" (2) Relevance scoring: if similarity score < 0.4, skip LLM call and return "no relevant documents found" directly. (3) Confidence field: use `with_structured_output` to force LLM to return `{answer: str, confidence: float, found_in_docs: bool}` — display low-confidence answers with a warning.

---

# Lesson 13 — Long-Term Vector Memory: Complete Deep Dive

> **Prerequisite:** Open `lesson_13_vector_memory/lesson_13_vector_memory.py` and read it fully.

---

## The 3 Memory Types — Choose the Right One

| Type | Storage | Best for | Limitation |
|------|---------|---------|-----------|
| **In-context** | Message list | Current conversation | Context window limit; lost after session |
| **Structured DB** | SQL table / dict | Exact facts: name, email, preferences | Only finds exact matches; no semantic search |
| **Vector store** | Embedded vectors | Semantic facts: "user is interested in X" | Approximate — may not find exact matches |

### When to use vector memory

Use vector memory when:
- Facts need to be retrieved by **meaning**, not exact key
- "User likes Python" should be retrieved when asked about "programming preferences"
- You have many facts (100+) that can't all fit in every prompt

---

## The Memory Lifecycle — Per Conversation Turn

```
User: "I'm a data engineer building a real-time pipeline with Kafka"

Turn Start:
  1. LOAD: search vector store for memories relevant to this message
     Query: "data engineer real-time pipeline kafka"
     Retrieved: ["User works with Python", "User prefers cloud solutions"]
  2. INJECT: add retrieved memories to system prompt as context

Turn Processing:
  3. CHAT: LLM has user message + retrieved context → better personalized response

Turn End:
  4. EXTRACT: LLM identifies memorable facts from this turn
     → "User works as a data engineer"
     → "User builds real-time pipelines"
     → "User uses Kafka"
  5. STORE: embed new facts → save to vector store with user_id metadata
```

---

## Fact Extraction — Making it Reliable

```python
EXTRACT_PROMPT = """Extract 1-3 facts worth remembering about the user from this message.
Focus ONLY on: name, job title, skills, preferences, goals, projects, location, company.
Ignore: questions they asked, general topics, things you said.

User message: {message}

Return ONLY a JSON list of strings. Examples:
["User's name is Ahmed", "User is a senior data engineer", "User prefers Python over Java"]
Return [] if nothing worth remembering."""

def extract_facts(message: str, llm) -> list[str]:
    resp = llm.invoke([HumanMessage(content=EXTRACT_PROMPT.format(message=message))])
    try:
        start, end = resp.content.find("["), resp.content.rfind("]")
        if start == -1 or end == -1:
            return []
        facts = json.loads(resp.content[start:end+1])
        return [f for f in facts if isinstance(f, str) and len(f.strip()) > 5]
    except Exception:
        return []   # never crash on extraction failure
```

---

## User Isolation — Critical for Multi-User Systems

```python
# ✅ CORRECT: filter by user_id so users never see each other's memories
results = vectorstore.similarity_search(
    query,
    k=5,
    filter={"user_id": user_id}   # ← user isolation enforced at retrieval
)

# Each memory stored with user_id metadata:
vectorstore.add_documents([Document(
    page_content="User prefers Python",
    metadata={"user_id": "ahmed-001", "stored_at": "2024-01-15"}
)])

# GDPR: delete all memories for a user
vectorstore.delete(where={"user_id": "ahmed-001"})
```

---

## Memory Quality Issues and Solutions

| Problem | Symptom | Fix |
|---------|---------|-----|
| **Contradicting memories** | "User likes Python" AND "User hates Python" | Before storing, search for contradictions; replace if found |
| **Too many low-quality facts** | "User said hello" stored as fact | Tighten extraction prompt; only store job/preference/goal facts |
| **Memory not retrieved** | Agent forgets recent facts | Increase k, lower similarity threshold |
| **Wrong memories retrieved** | Facts from different context | Add temporal weight; recent memories score higher |
| **Duplicate memories** | Same fact stored 10 times | Dedup before storing: cosine similarity > 0.95 = duplicate |

---

## Tasks — Lesson 13

**Task 13.1 — Personalized Tutor**
Build a tutoring agent that remembers: student's skill level, topics mastered, topics struggling with, preferred explanation style.
After 5 sessions, the agent should adapt its explanations without being re-told context.

**Task 13.2 — Memory Categories**
Store memories with categories: `{"user_id": "...", "category": "work"}`.
Categories: work, personal, preferences, goals.
Retrieve by category based on question type (work question → search work memories first).

**Task 13.3 — Memory Contradiction Detector**
Before storing "User dislikes Python", search for "User likes Python" (high similarity).
If found: replace the old memory instead of adding a contradiction.
Test: tell agent "I love Python" then "I hate Python". Only the latest should be stored.

**Task 13.4 — Memory Audit Tool**
Build `memory_audit(user_id)` that:
- Lists all stored memories for a user
- Groups by category
- Shows which memories were retrieved most (add a "hit_count" metadata field, increment on each retrieval)
- Identifies low-value memories (never retrieved in last 10 turns) for pruning

---

## Interview Q&A — Lesson 13

**Q1: How is vector memory different from just storing a long message history?**
Message history stores every conversation turn verbatim — it grows every turn, hits context limits, and the LLM must process all of it to find relevant facts. Vector memory stores only distilled facts ("User is a senior data engineer") embedded as vectors. At retrieval time, only the 3-5 most relevant facts are loaded — not the full history. This enables: (1) cross-session memory (facts persist across conversations), (2) scalable memory (10,000 facts don't slow down retrieval), (3) semantic search (find facts by meaning, not keyword).

**Q2: How do you ensure user A's memories never leak to user B?**
Store `user_id` in every document's metadata. At retrieval, always filter: `similarity_search(query, filter={"user_id": user_id})`. At deletion (GDPR): `vectorstore.delete(where={"user_id": user_id})`. The vector store enforces isolation at the database level — even if the query is identical, results are scoped to the requesting user's data.

---

# Lesson 14 — Testing Agents: Complete Deep Dive

> **Prerequisite:** Run `python -m pytest lesson_14_testing/test_agents.py -m "not integration" -v`

---

## Why Testing Agents Is Uniquely Hard

**Traditional function tests:** same input → same output. Deterministic.

**Agent tests:** same input → varies by LLM state, temperature, context, API rate limits. Non-deterministic.

This requires a different testing philosophy:

```
❌ Don't test: "LLM returned exactly 'The answer is 42'"
✅ Do test:    "Answer contains '42' or 'forty-two'"

❌ Don't test: Exact LLM routing decisions (varies by model version)
✅ Do test:    Routing function logic (pure Python, deterministic)

❌ Don't test: LLM in every test (slow, expensive, flaky)
✅ Do test:    Tools directly, routing functions directly, full graph with mocks
```

---

## The 5 Testing Levels

### Level 1: Tool Unit Tests (fastest — milliseconds, no LLM)

```python
def test_run_sql_blocks_delete():
    """SQL safety guard must block DELETE operations."""
    from lesson_10_capstone.lesson_10_capstone import run_sql
    result = run_sql.invoke({"query": "DELETE FROM employees WHERE id=1"})
    assert "ERROR" in result.upper() or "Only SELECT" in result

def test_run_sql_blocks_drop():
    result = run_sql.invoke({"query": "DROP TABLE employees"})
    assert "ERROR" in result.upper()

def test_run_sql_accepts_select():
    result = run_sql.invoke({"query": "SELECT COUNT(*) FROM employees"})
    assert "ERROR" not in result.upper()

def test_run_sql_handles_invalid_query():
    result = run_sql.invoke({"query": "SELECT * FROM nonexistent_table_xyz"})
    assert isinstance(result, str)   # must return string, not raise
```

### Level 2: Routing Function Tests (fast — pure Python)

```python
def test_route_supervisor_to_db():
    from lesson_10_capstone.lesson_10_capstone import route_supervisor
    assert route_supervisor({"next_agent": "db_agent"}) == "db_agent"

def test_route_supervisor_to_end():
    assert route_supervisor({"next_agent": "FINISH"}) == "__end__"

def test_route_supervisor_default():
    assert route_supervisor({"next_agent": "unknown_xyz"}) == "__end__"

def test_route_supervisor_empty():
    assert route_supervisor({"next_agent": ""}) == "__end__"
```

### Level 3: Node Tests (fast — mock the LLM)

```python
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

def test_supervisor_routes_correctly():
    """Test supervisor routing logic with mocked LLM."""
    mock_response = AIMessage(content='{"next": "db_agent"}')

    with patch("lesson_10_capstone.lesson_10_capstone.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        from lesson_10_capstone.lesson_10_capstone import supervisor_node
        state = {
            "messages":   [HumanMessage(content="How many employees?")],
            "user_id":    "test",
            "next_agent": ""
        }
        result = supervisor_node(state)
        assert result["next_agent"] == "db_agent"

def test_supervisor_handles_malformed_json():
    """Supervisor must not crash on bad JSON output."""
    mock_response = AIMessage(content="I cannot decide")
    with patch("lesson_10_capstone.lesson_10_capstone.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        from lesson_10_capstone.lesson_10_capstone import supervisor_node
        state = {"messages": [HumanMessage("unclear")], "user_id": "t", "next_agent": ""}
        result = supervisor_node(state)   # must not raise
        assert "next_agent" in result
```

### Level 4: Compilation Tests (instant)

```python
def test_all_lesson_graphs_compile():
    """Verify every lesson graph compiles without errors."""
    from lesson_01_basics.lesson_01_basics import graph as g1
    from lesson_02_conditional.lesson_02_conditional import graph as g2
    from lesson_11_subgraphs.lesson_11_subgraphs import graph as g11

    assert g1 is not None
    assert g2 is not None
    assert g11 is not None
```

### Level 5: Integration Tests (slow — require real LLM)

```python
@pytest.mark.integration   # run separately: pytest -m integration
def test_lesson01_full_execution():
    from lesson_01_basics.lesson_01_basics import graph
    result = graph.invoke({"message": "hello", "processed": "", "final": ""})
    assert result["processed"] == "HELLO"
    assert "HELLO" in result["final"]

@pytest.mark.integration
def test_db_agent_answers_employee_count():
    """Agent must find the correct employee count."""
    from lesson_06_database_agent.lesson_06_database_agent import graph, build_sample_database
    build_sample_database()
    result = graph.invoke({"messages": [HumanMessage("How many employees are there?")]})
    answer = result["messages"][-1].content.lower()
    # "3" or "three" or "three employees" — any of these is correct
    assert any(kw in answer for kw in ["3", "three"])
```

---

## Evaluation Tests — Measuring Answer Quality

```python
EVAL_DATASET = [
    {
        "id": "E01",
        "question": "How many employees are in Engineering?",
        "expected_keywords": ["2", "two", "alice", "bob"],
        "description": "Count Engineering employees"
    },
    {
        "id": "E02",
        "question": "What is the highest salary?",
        "expected_keywords": ["95000", "95,000"],
        "description": "Max salary query"
    }
]

def run_evaluation(graph, dataset: list) -> dict:
    """Run evaluation dataset and return metrics."""
    results = {"passed": 0, "failed": 0, "errors": 0, "details": []}

    for case in dataset:
        try:
            output = graph.invoke(
                {"messages": [HumanMessage(content=case["question"])], "user_id": "eval"},
                config={"configurable": {"thread_id": f"eval-{case['id']}"}, "recursion_limit": 25}
            )
            last_msg = output["messages"][-1].content.lower()
            passed = any(kw.lower() in last_msg for kw in case["expected_keywords"])
            results["passed" if passed else "failed"] += 1
            results["details"].append({"id": case["id"], "passed": passed})
        except Exception as e:
            results["errors"] += 1
            results["details"].append({"id": case["id"], "error": str(e)})

    total = len(dataset)
    results["score"] = results["passed"] / total if total > 0 else 0
    return results
```

---

## Running Tests — Command Reference

```bash
# Run all fast tests (no LLM, < 3 seconds)
python -m pytest lesson_14_testing/test_agents.py -m "not integration" -v

# Run only integration tests (requires Ollama running)
python -m pytest lesson_14_testing/test_agents.py -m integration -v

# Run specific test class
python -m pytest lesson_14_testing/test_agents.py::TestDatabaseTools -v

# Run specific test
python -m pytest lesson_14_testing/test_agents.py::TestDatabaseTools::test_run_sql_rejects_delete -v

# Run with coverage report
python -m pytest lesson_14_testing/ --cov=lesson_10_capstone --cov-report=term-missing

# Run evaluation report
python lesson_14_testing/test_agents.py
```

---

## Tasks — Lesson 14

**Task 14.1 — Test Your Lesson 4 Agent**
Write unit tests for every tool in your Lesson 4 ReAct agent.
Write node tests (mock LLM) for: the agent node, the should_continue routing.
Write integration test: "What is sqrt(144)?" → answer contains "12".

**Task 14.2 — Mutation Testing**
Deliberately introduce 3 bugs into lesson_06:
1. Remove the SELECT-only guard
2. Remove the LIMIT enforcement
3. Make describe_table return empty string

Write tests that catch each bug. Run tests before and after the bug. All 3 should fail when bug is present.

**Task 14.3 — Build an Evaluation Suite**
Create 15 question-answer pairs for the capstone agent.
5 easy (direct queries), 5 medium (aggregations), 5 hard (multi-table JOINs).
Run the evaluation. Target: ≥ 13/15 passing.
Identify failing cases and explain why they fail.

**Task 14.4 — CI Integration**
Create a `run_tests.sh` script:
```bash
#!/bin/bash
echo "Running fast tests..."
python -m pytest lesson_14_testing/ -m "not integration" -v --tb=short
echo "Fast tests done. Run with -m integration for LLM tests."
```
This is how you'd integrate with GitHub Actions or GitLab CI.

---

## Interview Q&A — Lesson 14

**Q1: Why is testing AI agents harder than testing regular software?**
Three core challenges: (1) **Non-determinism** — same input can produce different outputs on different runs (temperature > 0, model updates). (2) **External dependencies** — tests depend on LLM availability, database state, API quotas. (3) **Subjective correctness** — "Did the agent answer correctly?" is hard to assert programmatically. Solutions: test routing logic (deterministic) without LLM, mock LLMs for node tests, use keyword presence instead of exact string matching for integration tests, separate fast tests from slow integration tests.

**Q2: How do you mock the LLM for unit tests?**
```python
from unittest.mock import patch
from langchain_core.messages import AIMessage

with patch("my_module.llm") as mock_llm:
    mock_llm.invoke.return_value = AIMessage(content='{"next": "db_agent"}')
    result = supervisor_node(test_state)
    assert result["next_agent"] == "db_agent"
```
The mock replaces the real LLM with a deterministic stub that always returns the specified value. This makes tests fast (no LLM call), reliable (no network), and free (no API cost).

**Q3: What is the difference between unit tests, integration tests, and evaluation tests for agents?**
**Unit tests**: test one component in isolation (a tool function, a routing function) — fast, no LLM, deterministic. **Integration tests**: test the full graph end-to-end with real LLM — slow, require Ollama running, may have variance. **Evaluation tests**: measure answer quality against a ground truth dataset — like unit tests but for the LLM's knowledge and reasoning quality. All three are needed; run unit tests in CI, integration and evaluation tests periodically or before releases.

---

# Lesson 15 — Deployment: Complete Deep Dive

> **Prerequisite:** Run `pip install fastapi uvicorn python-dotenv` first.

---

## The Deployment Architecture

```
Internet
    ↓
[Load Balancer] (nginx / AWS ALB)
    ↓
[FastAPI Server 1]  [FastAPI Server 2]  [FastAPI Server 3]
    ↓                      ↓                    ↓
[LangGraph Agent] with PostgresSaver (shared state across all servers)
    ↓
[Database Server] + [Ollama LLM Server] + [Vector Store]
```

Each FastAPI server hosts the same agent. Users are load-balanced across servers. State is shared via PostgresSaver so any server can resume any user's session.

---

## FastAPI Wrapper — Complete Pattern

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger("api")

# Graph initialized ONCE on startup, reused for all requests
_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    from lesson_10_capstone.lesson_10_capstone import build_capstone_graph, setup_database
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3
    setup_database()
    conn = sqlite3.connect("production.db", check_same_thread=False)
    _graph = build_capstone_graph(checkpointer=SqliteSaver(conn))
    logger.info("Agent initialized and ready")
    yield
    logger.info("Shutting down")

app = FastAPI(title="LangGraph Agent API", version="1.0.0", lifespan=lifespan)

class QuestionRequest(BaseModel):
    question: str
    user_id:  str
    thread_id: Optional[str] = None

    @field_validator("question")
    @classmethod
    def validate_q(cls, v):
        v = v.strip()
        if len(v) < 3:  raise ValueError("Too short")
        if len(v) > 500: raise ValueError("Too long")
        return v

@app.post("/ask")
async def ask(req: QuestionRequest, api_key: str = Header(alias="X-Api-Key")):
    if api_key not in VALID_KEYS:
        raise HTTPException(401, "Invalid API key")

    thread_id = req.thread_id or f"{req.user_id}-{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

    try:
        result = _graph.invoke(
            {"messages": [HumanMessage(content=req.question)],
             "user_id": req.user_id, "next_agent": ""},
            config=config
        )
        last_ai = next((m.content for m in reversed(result["messages"])
                       if isinstance(m, AIMessage)), "No response")
        return {"status": "complete", "thread_id": thread_id, "answer": last_ai}
    except GraphRecursionError:
        raise HTTPException(500, "Agent exceeded maximum steps")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(500, "Internal error")
```

---

## LangSmith Tracing — Observability in Production

```python
import os

# Set these in .env file (never hardcode)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "my-agent-production"
os.environ["LANGCHAIN_API_KEY"]    = "ls__your_key_here"

# That's it — every LLM call, tool call, and node execution
# is now automatically traced and visible in the LangSmith dashboard
# URL: https://smith.langchain.com
```

What LangSmith shows per trace:
- Which nodes ran and in what order
- LLM input/output for every call
- Tool calls and their results
- Latency per node
- Token usage
- Errors and their location

---

## Environment Configuration

```bash
# .env file (never commit to git)
DB_HOST=localhost
DB_NAME=company_db
DB_USER=agent_user
DB_PASSWORD=secure_password_here
API_KEYS=key1,key2,key3
OLLAMA_BASE_URL=http://localhost:11434
LANGCHAIN_API_KEY=ls__your_key
LANGCHAIN_PROJECT=my-agent
```

```python
# Load in Python
from dotenv import load_dotenv
load_dotenv()   # reads .env file

db_password = os.getenv("DB_PASSWORD")   # never os.environ["DB_PASSWORD"] directly
if not db_password:
    raise RuntimeError("DB_PASSWORD not set — check .env file")
```

---

## Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Run with multiple workers for production
CMD ["uvicorn", "lesson_15_deployment.api:app",
     "--host", "0.0.0.0",
     "--port", "8000",
     "--workers", "4"]
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=postgres
      - DB_NAME=company_db
      - DB_USER=agent
      - DB_PASSWORD=${DB_PASSWORD}   # from host environment
    depends_on:
      - postgres
      - ollama

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB:       company_db
      POSTGRES_USER:     agent
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"

volumes:
  postgres_data:
  ollama_models:
```

```bash
# Deploy
docker-compose up -d          # start all services
docker-compose logs -f agent-api   # watch logs
docker-compose down           # stop all services
```

---

## API Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: dev-key-123" \
     -d '{"question": "How many employees are there?", "user_id": "ahmed"}'

# Approve a pending HITL action
curl -X POST http://localhost:8000/approve \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: dev-key-123" \
     -d '{"thread_id": "ahmed-1234567890", "decision": "approve"}'

# Get conversation history
curl http://localhost:8000/history/ahmed-1234567890 \
     -H "X-Api-Key: dev-key-123"

# Stream response (Server-Sent Events)
curl -N "http://localhost:8000/stream/my-thread?question=Hello&user_id=ahmed" \
     -H "X-Api-Key: dev-key-123"

# Interactive API docs (browser)
open http://localhost:8000/docs
```

---

## Anti-Patterns — Lesson 15

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| **Graph built per request** | Very slow (compile overhead) | Build once in lifespan, reuse |
| **Hardcoded API keys** | Security breach | Environment variables + .env file |
| **No API authentication** | Anyone can call your agent | API key or JWT auth on all endpoints |
| **No rate limiting** | One user can overwhelm the service | Add rate limiting middleware |
| **Blocking invoke() in async** | FastAPI performance destroyed | Use `asyncio.to_thread(graph.invoke, ...)` |
| **MemorySaver in multi-server** | State not shared across servers | PostgresSaver for multi-server |
| **No health check endpoint** | Load balancer can't detect failures | `/health` endpoint always present |
| **No request logging** | Can't debug production issues | Log every request: method, path, status, latency |

---

## Best Practices Checklist — Lesson 15

```
API DESIGN:
  □ Graph compiled once in lifespan startup, never per request
  □ API key authentication on all non-health endpoints
  □ Pydantic models for all request bodies (validation)
  □ /health endpoint returns 200 if ready, 503 if not
  □ Thread_id returned to client for multi-turn conversations
  □ Structured error responses: {status, error, detail}

CONFIGURATION:
  □ All credentials in .env file, loaded with python-dotenv
  □ .env file in .gitignore (NEVER committed)
  □ Required env vars checked at startup (raise if missing)

DEPLOYMENT:
  □ Dockerfile uses slim base image
  □ Docker-compose for local testing with all services
  □ uvicorn workers = 2 × CPU cores + 1
  □ LangSmith tracing enabled for production observability

OPERATIONS:
  □ Structured request logging (method, path, user_id, latency)
  □ GraphRecursionError caught and returned as 500
  □ Database connections in pool (not per-request)
  □ GDPR delete endpoint implemented
```

---

## Tasks — Lesson 15

**Task 15.1 — Start and Test the API**
```bash
uvicorn lesson_15_deployment.api:app --port 8000 --reload
```
Use curl or http://localhost:8000/docs to:
1. Check `/health`
2. Ask 3 questions via `/ask` — verify different thread_ids
3. Get history via `/history/{thread_id}`
4. Try wrong API key — verify 401 response

**Task 15.2 — Add Rate Limiting**
Install `slowapi`: `pip install slowapi`.
Add: max 10 requests per minute per IP for `/ask`.
```python
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/ask")
@limiter.limit("10/minute")
async def ask(request: Request, ...):
    ...
```
Test: send 11 requests in a minute. Verify 429 response on the 11th.

**Task 15.3 — Add Request Logging Middleware**
```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)")
    return response
```
Run 5 requests, print the log output. This is how production logs look.

**Task 15.4 — Docker Deployment**
Write the Dockerfile for the project. Build it: `docker build -t langgraph-agent .`.
Run it: `docker run -p 8000:8000 -e API_KEYS=test-key langgraph-agent`.
Test the running container with curl.

---

## Interview Q&A — Lesson 15

**Q1: How do you handle multiple concurrent users in a LangGraph API?**
(1) **FastAPI + uvicorn**: run with `--workers 4` (or `2 × CPUs + 1`) — each worker handles requests in its own process. (2) **Thread isolation**: every user has unique `thread_id` in config — LangGraph's checkpointer guarantees no state mixing. (3) **Shared checkpointer**: use PostgresSaver so all worker processes read/write to the same database. (4) **Connection pooling**: use SQLAlchemy pool for PostgreSQL connections — don't open a new connection per request. (5) **Load balancer**: nginx or AWS ALB distributes traffic across multiple server instances.

**Q2: What is the difference between uvicorn and gunicorn and when do you use each?**
uvicorn is an ASGI server — runs one async event loop per worker, handles many concurrent requests via async I/O. gunicorn is a WSGI process manager — manages multiple workers (processes). Production pattern: `gunicorn -k uvicorn.workers.UvicornWorker app:app -w 4` — gunicorn manages 4 uvicorn workers. For Docker: simpler to just use `uvicorn --workers 4`. For raw performance: gunicorn + uvicorn workers on bare metal.

**Q3: How do you implement zero-downtime deployment for a LangGraph agent?**
(1) **Rolling deployment**: deploy one server instance at a time. Load balancer routes traffic to running instances while one is updating. (2) **Health checks**: load balancer checks `/health` before routing traffic to newly started instance. (3) **Graceful shutdown**: uvicorn completes in-progress requests before shutdown. (4) **State in external DB**: since state is in PostgresSaver (not RAM), new server instances immediately have full context. (5) **Blue-green**: run two full environments (blue=current, green=new). Switch load balancer from blue to green atomically.

**Q4: How do you implement LangSmith tracing and what does it give you?**
Set three environment variables: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_PROJECT="my-project"`, `LANGCHAIN_API_KEY="ls__..."`. Every LangGraph execution is then automatically traced — no code changes. LangSmith dashboard shows: full execution tree (nodes, LLM calls, tool calls), input/output for every step, latency breakdown per node, token usage and cost estimate, error traces with full context, ability to replay any trace with different inputs. Critical for production debugging — without tracing, agent failures are nearly impossible to diagnose remotely.
