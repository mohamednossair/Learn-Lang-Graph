# LangGraph Deep Dive — Lessons 25–26: Mem0 & Solr RAG

> **Who this is for:** Engineers who completed Lessons 1–24 and want staff/principal-level
> understanding of production memory systems and enterprise search-backed RAG.
> Each section: internals, failure modes, anti-patterns, and 7 interview Q&A.

---

# Lesson 25 — Mem0 Long-Term Memory Deep Dive

> **File:** `lesson_25_mem0/lesson_25_mem0.py`

---

## How Mem0 Fits Into LangGraph State

Mem0 is an external side-effect called from inside nodes — exactly like S3 (Lesson 22).
LangGraph does not know about Mem0. The state carries retrieved memory strings, not Mem0 objects.

```
START
  ↓
[load_memories_node]  ──► m.search(query, user_id) ──► Qdrant + LLM extraction
  ↓  returns: {"retrieved_memories": [...], "memory_context": "..."}
[chat_node]           ──► LLM with memory injected into system prompt
  ↓  returns: {"messages": [AIMessage(...)]}
[save_memories_node]  ──► m.add(messages, user_id) ──► LLM extracts, Qdrant stores
  ↓  returns: {}   ← no state mutation needed
END
```

**Critical rule:** Never put a Mem0 client object into the state. The client is a module-level singleton. State must stay serializable for the checkpointer.

```python
class Mem0AgentState(TypedDict):
    messages:           Annotated[list, add_messages]
    user_id:            str
    retrieved_memories: list   # ✅ list of plain dicts — serializable
    memory_context:     str    # ✅ plain string — serializable
    # mem0_client: Memory      # ❌ NEVER — not serializable
```

---

## Mem0 Internal Architecture

When you call `m.add(messages, user_id)`, Mem0 runs a **5-step pipeline** internally:

```
Step 1 — Extract
  LLM receives: (user message, assistant message)
  LLM returns:  ["User's name is Sarah", "User prefers Go for performance services"]

Step 2 — Fetch existing
  Mem0 calls m.search() to find existing memories semantically similar to each new fact

Step 3 — Classify (LLM-driven)
  For each (new_fact, existing_facts) pair the LLM classifies:
    ADD:    no existing memory covers this topic
    UPDATE: existing memory covers the same topic — replace it
    DELETE: new fact explicitly invalidates existing one ("I no longer use X")
    NONE:   fact is not worth storing (question, transient statement)

Step 4 — Execute
  ADD    → Qdrant.upsert(new_fact_vector)
  UPDATE → Qdrant.update(existing_id, new_text + new_vector)
  DELETE → Qdrant.delete(existing_id)

Step 5 — Return result
  {"results": [{"id": "...", "memory": "...", "event": "ADD"|"UPDATE"|"DELETE"|"NONE"}]}
```

This is why Mem0 prevents contradictions: it runs a **comparison LLM call** (step 3) before writing.
Chroma has no equivalent — it blindly appends every extracted fact.

---

## Memory Scoping — user_id, agent_id, run_id

Mem0 supports three independent scope dimensions:

| Scope | Purpose | Example |
|---|---|---|
| `user_id` | Per-user long-term memory | `"user-42"` — survives across sessions |
| `agent_id` | Per-agent memory namespace | `"data-agent"` vs `"db-agent"` |
| `run_id` | Per-session (ephemeral) | `thread_id` — deleted at session end |

All three are optional but any combination filters the Qdrant query.

```python
# Only user memory (cross-agent):
m.search(query=q, user_id="user-42")

# Agent-specific memory for this user:
m.search(query=q, user_id="user-42", agent_id="data-agent")

# Session-only memory (not persisted):
m.add(messages=msgs, user_id="user-42", run_id="thread-xyz")
```

Multi-tenant rule: **never call any Mem0 API without `user_id`**. Without it, Mem0 returns all memories across all users — a data isolation violation equivalent to a missing `WHERE tenant_id = ?` in SQL.

---

## Qdrant Under the Hood

Mem0 self-hosted uses Qdrant as its vector store. Understanding Qdrant helps when debugging Mem0.

```
Qdrant collection: "langgraph_memories"
  Each point:
    id:       UUID (Mem0-generated)
    vector:   [float32 × embedding_dims]   ← from Ollama/OpenAI embeddings
    payload:  {
                "data":       "User prefers Go",
                "user_id":    "user-42",
                "agent_id":   "data-agent",   ← optional
                "created_at": "2025-01-15T10:30:00Z",
                "updated_at": "2025-01-20T14:00:00Z",
              }
```

When `m.search(query, user_id)` is called:
1. Embed the query → float32 vector
2. Qdrant kNN search with filter: `payload.user_id == user_id`
3. Return top-K results by cosine similarity

When `m.delete_all(user_id)` is called:
- Qdrant `delete(collection, filter={"user_id": user_id})` — deletes all matching points

---

## Mem0 Cloud vs Self-Hosted — Internals Comparison

| Aspect | Self-hosted (Ollama + Qdrant) | Mem0 Cloud (MemoryClient) |
|---|---|---|
| LLM for extraction | Your Ollama model | OpenAI (GPT-4o-mini) |
| Vector store | Your Qdrant instance | Managed Qdrant |
| Latency of `m.add()` | Ollama latency (local, ~1-3s) | OpenAI API latency (~0.5-1s) |
| Data residency | Your infrastructure | Mem0's US servers |
| Cost | Compute only | $0 (free tier) + usage |
| Contradiction quality | Depends on Ollama model | High (GPT-4o-mini) |
| Graph memory (v1.1) | Available if configured | Available |

**Graph memory** (`"version": "v1.1"`): Mem0 also builds a **knowledge graph** of entity relationships alongside the vector store. `"Sarah works at CompanyX"` + `"CompanyX uses Kubernetes"` creates a graph edge `Sarah → CompanyX → Kubernetes`. This enables reasoning like "What infrastructure does Sarah's company use?" — not possible with pure vector similarity.

---

## Failure Modes and Mitigations

### Mem0 `add()` latency spike
**Cause:** The extraction LLM call runs synchronously inside `m.add()`. Ollama on a slow machine can take 5-15 seconds.
**Mitigation:** Run `save_memories_node` as a background task (fire-and-forget):
```python
import asyncio

async def save_memories_node_async(state):
    asyncio.create_task(_mem0_add_background(state))
    return {}  # return immediately, don't await
```
The user sees the AI response instantly; memory saving happens in the background.

### Qdrant connection failure
**Cause:** Docker container not running, network partition, OOM kill.
**Pattern:** Mem0 raises `ConnectionError`. Wrap all mem0 calls in try/except — the graph should degrade gracefully (no memories loaded/saved) rather than crash.
```python
def load_memories_node(state):
    try:
        memories = mem0_search(query, user_id)
    except Exception:
        memories = []   # ← degrade gracefully
    return {"retrieved_memories": memories, "memory_context": ""}
```

### Memory hallucination
**Cause:** Low-quality Ollama model extracts false facts ("User said they hate Python" when user just asked a Python question).
**Detection:** Log all `m.add()` results. Add a daily review job that scans recent memories with score < 0.3.
**Mitigation:** Use `"version": "v1.1"` which includes contradiction checking, and use a capable model (llama3:70b or Claude via Bedrock for extraction).

### GDPR erasure incompleteness
**Cause:** `m.delete_all(user_id)` deletes Qdrant vectors but the Qdrant WAL (write-ahead log) may retain tombstone records.
**Mitigation:** After `delete_all()`, also call Qdrant's collection-level vacuum if Qdrant version supports it. Document in your GDPR runbook that Qdrant WAL cleanup has a 24-hour SLA. For Mem0 Cloud, erasure is instant (managed).

---

## Anti-Patterns

❌ **Calling `m.search()` without a `user_id`**
Returns all memories from all users. Zero data isolation. Treat this the same as missing `WHERE tenant_id=?` in SQL — a critical security bug.

❌ **Storing the Mem0 client in LangGraph state**
`Memory` objects contain thread locks, HTTP clients, and Qdrant connections — none are serializable. Always use a module-level singleton (`_mem0_client`).

❌ **Not wrapping `m.add()` in try/except**
If Qdrant is down and the graph crashes in `save_memories_node`, the user loses their entire response. Memory saving must never block or crash the main conversational flow.

❌ **Using `run_id` for long-term memory**
`run_id`-scoped memories are designed for ephemeral session context and may be garbage-collected. Use `user_id` for anything that must persist across sessions.

❌ **Setting `recursion_limit` too low when combining Mem0 with tools**
Each `load_memories → chat → save_memories` cycle counts as 3 steps. A 10-step limit means only 3 full cycles before the graph aborts. Set `recursion_limit: 50` minimum for memory agents.

---

## 7 Interview Q&A

**Q1: What is the core architectural difference between storing memories in Chroma (Lesson 13) and Mem0?**
A: Chroma is a vector database — it stores and retrieves vectors but has no memory lifecycle logic. You are responsible for writing the extraction prompt, parsing JSON, deduplicating, and handling contradictions. Mem0 wraps a vector database (Qdrant) with an LLM-driven memory lifecycle: it runs its own extraction call, compares new facts against existing memories, and decides whether to ADD, UPDATE, DELETE, or ignore each fact. The Chroma approach gives you full control; the Mem0 approach gives you correctness and deduplication for free at the cost of one additional LLM inference call per `add()`.

**Q2: Explain how Mem0 handles the contradiction "User prefers Python" followed by "I now use Rust exclusively".**
A: When `m.add()` is called with the Rust message: (1) Mem0 extracts the new fact: "User now uses Rust exclusively". (2) Mem0 searches existing memories for semantic similarity to this fact — finds "User prefers Python". (3) A classification LLM call receives both facts and classifies the relationship as UPDATE (same subject: language preference; different value). (4) Mem0 issues a Qdrant UPDATE on the existing memory's ID, replacing the text and re-embedding. The old fact is gone; only "User now uses Rust exclusively" remains. `m.history(memory_id)` still shows the change log for audit purposes.

**Q3: How does the `agent_id` scope in Mem0 enable multi-agent memory isolation in a LangGraph supervisor?**
A: In a supervisor graph with a `data_agent` and `db_agent` specialist, each specialist calls `m.add(messages, user_id=uid, agent_id="data-agent")` or `agent_id="db-agent"`. This partitions the Qdrant namespace: a data-agent memory about "User's preferred chart type is bar chart" will not appear when `db_agent` calls `m.search(query, user_id=uid, agent_id="db-agent")`. The user's cross-agent preferences (name, company, role) should be stored without `agent_id` so all agents can retrieve them.

**Q4: How would you implement async Mem0 memory saving so it does not add latency to the user response?**
A: Move `save_memories_node` to a background task using `asyncio.create_task()` or a Celery queue (Lesson 19). The graph returns the AI response immediately. The memory extraction/saving runs asynchronously. Tradeoff: if the process crashes between response and memory save, the memory is lost for that turn. Mitigate with a durable queue (Celery + Redis) rather than a raw `asyncio.create_task()`.

**Q5: A user complains the agent "forgot" something they told it 3 sessions ago. How do you debug this?**
A: Systematic debug steps: (1) Call `mem0_list_all(user_id)` — is the memory present at all? (2) If present, call `mem0_search(query, user_id)` with the relevant query manually — does it return the memory? (3) If not returned, check `limit` — is it set too low, or is score-based filtering excluding it? (4) If not stored, check `save_memories_node` logs — did the extraction LLM classify the fact as NONE? (5) Check if a subsequent UPDATE/DELETE incorrectly overwrote the memory. (6) Check `m.history(memory_id)` for the change trail. Most "forgetting" issues are either extraction quality (LLM classified the fact as not worth storing) or contradiction over-triggering (UPDATE incorrectly fired).

**Q6: How would you implement a GDPR right-to-erasure that covers both Mem0 and the rest of the system?**
A: A complete GDPR purge requires hitting every store: (1) `mem0_delete_all(user_id)` — erases all Qdrant vectors for the user. (2) `s3_erase_user_data(tenant_id, thread_id)` — deletes S3 conversation snapshots and documents (Lesson 22). (3) Oracle: `DELETE FROM langgraph_checkpoints WHERE thread_id LIKE user_id || '%'` — removes checkpoint history (Lesson 16). (4) Redis: `DEL rate_limit:{user_id}:*` — removes rate limit counters (Lesson 17). Log the erasure event to CloudWatch with a `gdpr_erase` metric. Return confirmation within 30 days per GDPR Article 17.

**Q7: What are the cost implications of self-hosted Mem0 vs Mem0 Cloud at 100,000 messages/day?**
A: Self-hosted: each `m.add()` triggers one Ollama LLM call (~50-200ms on GPU, compute cost only). At 100K messages/day that is 100K extraction calls. On a single A10G GPU (~$0.75/hr on AWS), you can do ~3,000 extractions/hr → 72,000/day → need 2 GPUs → ~$36/day. Plus Qdrant storage: 100K memories × ~1.5KB = ~150MB/day, negligible. Mem0 Cloud: free tier covers light usage; paid tier is usage-based (check current pricing at mem0.ai). For privacy-sensitive enterprises, self-hosted wins regardless of cost because data never leaves your VPC. For teams without GPU infrastructure, Mem0 Cloud is more cost-effective until ~50K daily active users.

---

## Systems Design: Persistent Cross-Session User Memory at Scale

**Scenario:** Design a memory system for a chatbot serving 50,000 daily active users. Each user should have their preferences, goals, and facts remembered indefinitely across sessions. The system must support GDPR erasure within 24 hours and not add more than 500ms latency to responses.

**Answer:**

```
Request path (latency-critical):
  API → load_memories_node
           ↓
        mem0_search(query, user_id, limit=5)  [~100ms, async]
           ↓
        Qdrant cluster (3 nodes, sharded by user_id hash)
           ↓
        inject top-5 memories into system prompt
           ↓
        LLM inference (Bedrock Claude Haiku, ~800ms)
           ↓
        return response to user  [total: ~900ms ✅]

Save path (background, after response):
  [save_memories_node] → Celery task queue (Redis)
           ↓
        Celery worker: m.add(turn, user_id) [~300ms Ollama extraction]
           ↓
        Qdrant upsert (vector + metadata)
```

Key design decisions:
- **Why Celery for saves?** Moves the 300ms extraction call off the critical path. User sees response in ~900ms, not 1.2s.
- **Why Qdrant cluster (3 nodes)?** Qdrant supports horizontal sharding — shard by `user_id` hash so user A's searches only hit one shard. 50K DAU × ~100 memories each = 5M vectors → easily handled by 3×16GB Qdrant nodes.
- **Why `limit=5` for search?** More than 5 memories in the system prompt exceeds the benefit — context window fills up and the LLM starts ignoring the later memories. 5 is the sweet spot from empirical testing.
- **GDPR 24h SLA:** Celery task `gdpr_erase_user` → `mem0_delete_all()` + S3 purge + Oracle purge. Enqueue with high priority. CloudWatch alarm if task age > 1h.
- **Monitoring:** Track `mem0_add_latency`, `mem0_search_latency`, and `memory_extraction_classification` (ADD/UPDATE/DELETE/NONE ratios). High NONE ratio = poor extraction model; high UPDATE ratio = users frequently changing preferences (normal).

---

# Lesson 26 — Solr RAG Deep Dive

> **File:** `lesson_26_solr_rag/lesson_26_solr_rag.py`

---

## How Solr Fits Into LangGraph

Solr is called from `@tool` functions bound to the LLM — the same pattern as any other tool in Lesson 4. The agent decides when to call Solr (agentic RAG). LangGraph doesn't know about Solr — it sees a tool call result message.

```
[agent node]  ──► LLM with bound tools ──► tool_call: solr_hybrid_search_tool(query)
     ↑                                                           ↓
     └──────── ToolMessage(results) ──────── [ToolNode] ──► pysolr query
```

State is minimal — just messages:
```python
class SolrRAGState(TypedDict):
    messages: Annotated[list, add_messages]
    # No Solr client in state — use module-level singleton
```

---

## Solr Internals: Lucene Index

Solr is built on top of Apache Lucene. Understanding the Lucene index explains BM25 behavior.

```
Document indexing:
  "MemorySaver stores checkpoints in RAM"
       ↓ Analyzer (tokenize, lowercase, stem)
  tokens: ["memorysaver", "store", "checkpoint", "ram"]
       ↓ Inverted index
  Term     │ DocIDs (postings)
  ─────────┼──────────────────
  memorysaver │ [doc_12, doc_34]
  checkpoint  │ [doc_12, doc_7, doc_45]
  ram         │ [doc_12]
```

**BM25 scoring** (the default in Solr since Solr 6):
```
BM25(term, doc) = IDF(term) × TF_normalized(term, doc)

IDF(term) = log(1 + (N - df + 0.5) / (df + 0.5))
  N  = total docs in collection
  df = number of docs containing the term
  → rare terms score higher than common terms

TF_normalized(term, doc) = tf / (tf + k1 × (1 - b + b × dl/avgdl))
  tf    = term frequency in this doc
  k1    = 1.2 (saturation parameter — controls how much extra TF helps)
  b     = 0.75 (length normalization — longer docs penalized)
  dl    = doc length
  avgdl = average doc length in collection
```

**Why this matters for RAG:** "MemorySaver" is a rare term (appears in few docs) so it scores very high. "the" would score near zero. When writing knowledge base documents, include the technical term names explicitly — BM25 rewards them.

---

## Solr kNN — Dense Vector Field Internals

Solr 9+ uses HNSW (Hierarchical Navigable Small World) graphs for approximate kNN.

```
Index time:
  text → OllamaEmbeddings.embed_query() → float32[4096]
       → Solr DenseVectorField → HNSW graph node added

Query time:
  query → embed → float32[4096] query vector
        → {!knn f=vector topK=5}<vec> Solr query
        → HNSW graph traversal: ~O(log N) hops
        → cosine similarity computed for candidate set
        → top-K results returned
```

HNSW parameters in Solr (fieldType definition):
```xml
<fieldType name="knn_vector" class="solr.DenseVectorField"
  vectorDimension="4096"
  similarityFunction="cosine"
  knnAlgorithm="hnsw"
  hnswMaxConnections="16"    <!-- higher = better recall, more memory -->
  hnswBeamWidth="100"        <!-- higher = better index quality, slower indexing -->
/>
```

**Why HNSW is approximate:** HNSW trades recall for speed. With default settings, recall@10 is typically 95-99%. For RAG this is acceptable — the difference between the 1st and 2nd most similar doc is usually negligible.

---

## Hybrid Search — Why Simple Merging Works

The hybrid approach in Lesson 26 is **late fusion** (merge after retrieval). The alternative is **early fusion** (a single Solr query that combines both signals).

**Late fusion (what we implement):**
```python
bm25_results = solr_bm25_search(query, top_k=10)   # run separately
knn_results  = solr_knn_search(query,  top_k=10)   # run separately
# normalize → merge → re-rank
final_score = 0.4 * bm25_norm + 0.6 * knn_norm
```

**Why 0.6 kNN weight?** Semantic recall matters more than keyword precision for natural-language questions. The LLM generates varied phrasing; kNN finds the right document even when the exact terms differ.

**Reciprocal Rank Fusion (RRF) — the alternative:**
```python
# Instead of score normalization, use rank position
rrf_score = 1/(k + rank_bm25) + 1/(k + rank_knn)   # k=60 is standard
```
RRF is robust to score scale differences between BM25 and kNN — no normalization needed. Use RRF when you can't control score ranges (different Solr versions).

---

## Solr Schema Design for RAG

A well-designed Solr schema for RAG includes these fields:

```xml
<!-- BM25 search field — full text, tokenized, stemmed -->
<field name="content"     type="text_general" indexed="true" stored="true"/>

<!-- Metadata fields for filtering -->
<field name="source"      type="string"       indexed="true" stored="true"/>
<field name="doc_type"    type="string"       indexed="true" stored="true"/>   <!-- "faq"|"manual"|"code" -->
<field name="tenant_id"   type="string"       indexed="true" stored="false"/>  <!-- multi-tenant filter -->
<field name="created_at"  type="pdate"        indexed="true" stored="true"/>

<!-- kNN vector field (Solr 9+) -->
<field name="vector"      type="knn_vector"   indexed="true" stored="true"/>
```

**Multi-tenant filtering with Solr:**
```python
# Always add tenant_id to every search query
results = client.search(
    f"content:({query}) AND tenant_id:{tenant_id}",
    rows=top_k, fl="content,source,score"
)
```
Never skip the `tenant_id` filter — it is the Solr equivalent of `WHERE tenant_id=?` in SQL.

---

## Solr Deployment: SolrCloud vs Standalone

| Mode | Use for | Config |
|---|---|---|
| Standalone | Development, single-server | `bin/solr start -p 8983` |
| SolrCloud | Production, HA, scale-out | Requires ZooKeeper for cluster coordination |

**SolrCloud setup** (3-node cluster):
```bash
# Node 1 (also runs embedded ZooKeeper)
bin/solr start -cloud -p 8983 -z localhost:9983

# Nodes 2 and 3
bin/solr start -cloud -p 8984 -z localhost:9983
bin/solr start -cloud -p 8985 -z localhost:9983

# Create collection: 2 shards, replication factor 2
bin/solr create -c langgraph_docs -shards 2 -replicationFactor 2
```

With 2 shards + RF=2: each shard has a leader + 1 replica. Any single node failure → zero downtime. 2 simultaneous node failures → potential unavailability.

---

## Self-Correcting RAG — When to Re-Query

The `check_relevance → route → refine_query` loop is expensive (1 extra LLM call per retry).

**When it helps:**
- Ambiguous queries where the first phrasing misses the target document
- Queries using synonyms not present in the knowledge base ("checkpointing" vs "persistence")
- Multi-hop questions where the first retrieval returns context for step 2, not step 1

**When it hurts:**
- Real-time chatbots where p99 latency must stay under 2s (each retry adds ~500ms)
- Well-indexed knowledge bases where BM25 recall is already >90%

**Production pattern:** Use a score threshold instead of a retry loop for latency-sensitive paths:
```python
results = solr_hybrid_search(query, top_k=5)
top_score = results[0]["score"] if results else 0
if top_score < 0.3:
    # Low confidence — add disclaimer to answer, don't retry
    answer_prefix = "I have limited information on this topic. "
else:
    answer_prefix = ""
```

---

## Failure Modes and Mitigations

### Solr is down / collection not found
**Cause:** Docker container crashed, network partition, collection not yet created.
**Pattern:** `create_solr_client()` catches connection errors and returns `None`. All search functions check `if client is None: return []`. The `SimulatedSolr` fallback ensures the graph always returns something.
**Production:** Add Solr liveness to the `/health/ready` endpoint (Lesson 24 pattern):
```python
def check_solr_connectivity():
    try:
        get_solr().ping()
        return True
    except Exception:
        return False
```

### kNN search returns empty results after indexing
**Cause:** `vector` field was not defined before indexing — documents were indexed without embeddings. The knnDenseVector field requires schema setup **before** the first `add()`.
**Fix:** Always call `setup_solr_schema()` once before indexing. If already indexed without vectors, delete the collection and re-index.

### Hybrid search scores all zeros
**Cause:** BM25 returns results but kNN returns nothing (embedding model not available), so `knn_norm` is all zeros. With `knn_weight=0.6`, the hybrid score is deflated.
**Fix:** Fall back to BM25-only when kNN results are empty:
```python
if not knn_results:
    return bm25_results[:top_k]   # pure BM25 fallback
```

### Solr query injection
**Cause:** User input passed directly to Solr query string. `content:(foo) OR id:*)` breaks out of the field query.
**Fix:** Escape all user input before passing to Solr:
```python
import pysolr
safe_query = pysolr.escape(user_input)
client.search(f"content:({safe_query})")
```

### Memory pressure from large `vector` fields
**Cause:** Each 4096-dim float32 vector = 16KB. 1M documents = 16GB vector data. HNSW graph adds ~4× overhead = 64GB RAM on a single Solr node.
**Fix:** Use quantization: Solr 9.4+ supports `int8` quantization → 4× memory reduction. Or reduce embedding dims using a smaller model (384-dim `all-MiniLM` = 1.5KB per doc).

---

## Anti-Patterns

❌ **No `tenant_id` filter in Solr queries**
In a multi-tenant system, omitting `AND tenant_id:{tenant_id}` returns documents from all tenants to every user. Solr has no built-in row-level security — you must enforce it in every query.

❌ **Running `setup_solr_schema()` on every request**
Schema API calls are slow (100-200ms each) and fail silently if the field already exists in some Solr versions. Call it once at startup (in FastAPI's `lifespan`) and cache the result.

❌ **Indexing without committing**
By default, pysolr auto-commits. But in batch indexing, disable auto-commit and call `client.commit()` once at the end — it is 10-100× faster.
```python
client = pysolr.Solr(url, always_commit=False)
client.add(batch_of_1000_docs)
client.commit()  # one commit for 1000 docs, not 1000 commits
```

❌ **Using `rows=10000` to "get everything"**
Solr deep pagination is expensive. For bulk retrieval use Solr's cursor-based pagination:
```python
results = client.search("*:*", rows=100, sort="id asc", cursorMark="*")
while results.nextCursorMark != cursorMark:
    cursorMark = results.nextCursorMark
    results = client.search("*:*", rows=100, sort="id asc", cursorMark=cursorMark)
```

❌ **Not using `fl` (field list) to limit returned fields**
Returning `vector` (16KB per doc) in search results wastes bandwidth and memory. Always specify `fl="id,content,source,score"` — never return the `vector` field in search responses.

---

## 7 Interview Q&A

**Q1: What is the difference between BM25 and TF-IDF, and why did Solr switch from TF-IDF to BM25 as the default?**
A: Both are term-frequency-based ranking functions. TF-IDF = TF × IDF with no length normalization beyond raw term count. BM25 adds two improvements: (1) TF saturation — BM25 uses a saturating function so a term appearing 100 times in a document doesn't score proportionally 100× higher than appearing 10 times (controlled by parameter k1=1.2). (2) Document length normalization — shorter documents that use the query term are ranked higher than longer documents with the same term frequency (controlled by parameter b=0.75). Solr switched to BM25 in Solr 6 because it produces substantially better ranking quality on most IR benchmarks, especially for mixed-length document collections.

**Q2: Explain the trade-off between `hnswMaxConnections` (M) and recall/memory in Solr's kNN field.**
A: `hnswMaxConnections` (M) controls the maximum number of edges each node in the HNSW graph can have. Higher M: more graph edges → better recall (more candidate nodes examined per query) → higher memory consumption (each node stores M × 8 bytes of edge pointers). Lower M: less memory → faster index construction → slightly lower recall. For most RAG use cases, M=16 gives 95-99% recall with acceptable memory overhead. Only reduce M below 16 if Solr is memory-constrained (e.g., 8GB nodes with millions of documents).

**Q3: How do you implement multi-tenant document isolation in Solr without row-level security?**
A: Solr does not have built-in row-level security (unlike Oracle VPD). You enforce isolation at the query level: (1) Index every document with a `tenant_id` field. (2) Append `AND tenant_id:{tenant_id}` to every search query — make this a non-negotiable rule in the search client wrapper, not in ad-hoc query construction. (3) In SolrCloud, consider using separate collections per tenant for the largest tenants (collection-level isolation is stronger). (4) Monitor Solr access logs for queries missing the `tenant_id` filter — set up an alert. This is analogous to application-level `WHERE tenant_id=?` enforcement in SQL.

**Q4: Why does late-fusion hybrid search (BM25 + kNN separately, then merge) usually outperform using just one mode?**
A: Each mode has a complementary weakness. BM25 misses semantically relevant documents that use different terminology (synonym problem: "persistence" vs "checkpointing"). kNN misses documents where the query embedding and document embedding diverge despite exact keyword overlap (embedding collapse for rare technical terms). Late fusion is robust because: if BM25 scores a document 0.0 (no keyword match) but kNN scores it 0.9 (strong semantic match), the document still surfaces in the merged results. Empirically, hybrid retrieval improves NDCG@10 by 5-15% over either single mode on most enterprise document sets.

**Q5: A relevance score drops from 0.85 to 0.12 for the same query after you added 50,000 new documents. What happened?**
A: BM25 IDF (inverse document frequency) component decreased. When the collection had 1,000 documents and the query term appeared in 10, `IDF = log(1 + (1000-10+0.5)/(10+0.5)) ≈ 4.6`. After adding 50,000 more documents (many containing the same term), if the term now appears in 5,000 docs: `IDF = log(1 + (51000-5000+0.5)/(5000+0.5)) ≈ 2.3` — half the original score. This is expected and correct BM25 behavior (common terms should score lower). If the ranking quality degraded, review whether the new documents correctly expanded the knowledge base or introduced noise.

**Q6: How would you set up hot/warm document tiering in Solr to reduce costs?**
A: SolrCloud supports the Data Management API for tiered storage. Recent documents (< 30 days) live on SSD-backed nodes (hot tier — fast retrieval). Older documents are moved to HDD-backed nodes or S3 via HDFS (warm/cold tier — cheaper storage). Configure: `<policy name="hot"><rule replica="1" sysprop.solr.node.type:HOT"/></policy>`. LangGraph RAG typically only retrieves recent knowledge base updates — cold-tier docs (old policy versions, archived manuals) can safely be demoted. This reduces active SSD costs by 60-80% for large collections.

**Q7: How do you test that your Solr RAG agent returns relevant results before deploying to production?**
A: Three-level testing: (1) **Retrieval evaluation** — compile a test set of 50 (query, expected_doc_ids) pairs. Compute Recall@5 (are the expected docs in top 5?) and NDCG@10. Target >85% Recall@5. (2) **End-to-end RAG evaluation** — for 20 question/answer pairs, run the full agent and score: does the answer mention the correct fact from the source document? Use an LLM-as-judge (Claude) to score each answer 1-5. Target mean score >3.5. (3) **Regression test** — run the 20 Q&A pairs as a pytest suite after every knowledge base update. If any answer score drops below 3, block the deployment. This mirrors the pattern from Lesson 14 (testing agents) applied to RAG specifically.

---

## Systems Design: Enterprise Search-Backed RAG at Scale

**Scenario:** Your company has an existing Solr 9 SolrCloud cluster (5 nodes, 50M documents across 20 collections) used for internal document management. The data team wants to add a LangGraph RAG chatbot that answers questions from this corpus. Design the integration.

**Answer:**

```
Existing Solr cluster (do not modify):
  5 nodes × 10M docs each, 20 collections (legal, hr, engineering, ...)

New RAG layer (additive only):
  1. Add vector field to relevant collections (Schema API, zero downtime)
  2. Backfill embeddings via batch job (index_documents() in chunks of 500)
  3. New read-only Solr user for the LangGraph app (no write access)

LangGraph agent architecture:
  User question
      ↓
  [classify_collection_node]   ← LLM: "legal" | "hr" | "engineering" | "all"
      ↓
  [retrieve_node]              ← solr_hybrid_search(query,
                                    collection=classified_collection,
                                    filter=f"tenant_id:{user_tenant}")
      ↓
  [generate_node]              ← Bedrock Claude Sonnet (legal requires audit)
      ↓
  [citation_node]              ← append Solr doc IDs as citations
      ↓
  Response + citations
```

Key decisions:
- **Why collection classification first?** 50M documents is too many to search at once. Routing the query to the right collection (legal, hr, etc.) reduces the search space to 2-5M docs and improves relevance.
- **Why read-only Solr user?** The RAG app should never write to the document store. Create a Solr user with `schema-read`, `read` roles only. The existing document ingestion pipeline retains write access.
- **Why Bedrock Sonnet for legal?** Legal queries require higher accuracy and the responses may be used in proceedings — Sonnet's higher capability justifies the cost. HR and engineering can use Haiku (cheaper, faster).
- **Backfill strategy:** Run `index_documents()` in a nightly batch job for documents older than the new vector field. New documents get embedded at index time by the existing ingestion pipeline (add an embedding step). The backfill takes 50M docs / 500 per batch = 100,000 batches — at 1 batch/second, ~28 hours on one GPU worker.
- **Schema migration risk:** `add-field` via Schema API is non-destructive — existing documents remain queryable during backfill. kNN queries return no results until a document has its vector populated (graceful degradation: fall back to BM25 only).
