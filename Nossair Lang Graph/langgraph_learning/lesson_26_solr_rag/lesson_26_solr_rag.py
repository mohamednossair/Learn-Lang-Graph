# =============================================================
# LESSON 26 — RAG with Apache Solr
# =============================================================
#
# WHAT YOU WILL LEARN:
#   1. What Apache Solr is and why enterprises use it over Chroma
#   2. How to index documents into Solr from a LangGraph agent
#   3. Two Solr search modes: BM25 (keyword) and kNN (vector/semantic)
#   4. Hybrid search: combining BM25 + kNN scores for best results
#   5. Agentic RAG: agent decides WHEN to retrieve (same pattern as L12)
#   6. Self-correcting Solr RAG: re-query on low relevance
#   7. LangChain SolrVectorStore integration
#   8. Solr vs Chroma comparison (when to use each)
#
# WHY SOLR OVER CHROMA?
#
#   | Feature                     | Chroma (L12)          | Solr (L26)                  |
#   |-----------------------------|-----------------------|-----------------------------|
#   | Scale                       | Single process        | Cluster (SolrCloud)         |
#   | Text search (BM25)          | No                    | Yes (full Lucene engine)    |
#   | Vector search (kNN)         | Yes                   | Yes (Dense Vector Field)    |
#   | Hybrid search               | No                    | Yes (BM25 + kNN)            |
#   | Faceting & filtering        | Basic metadata filter | Full Solr facets + pivots   |
#   | Enterprise auth             | None                  | Kerberos, PKI, JWT          |
#   | Production SLA              | DIY                   | Managed SolrCloud or Fusion |
#   | Existing enterprise data    | Re-ingest required    | Already in Solr?  Use it   |
#
#   Use Chroma for: quick local RAG prototypes, embedding-only search.
#   Use Solr for:  existing enterprise Solr clusters, BM25 keyword search,
#                  hybrid retrieval, filtered faceted search at scale.
#
# MENTAL MODEL:
#   User question
#      ↓
#   [agent] → solr_search(query) → [BM25 + kNN Solr query]
#      ↑             ↓
#      └── reads retrieved docs ──┘
#      ↓
#   [grounded answer based on Solr results]
#
# INSTALL:
#   pip install pysolr langchain-community
#   (For kNN/vector search: Solr 9.0+ required)
#
# LOCAL SOLR SETUP (Docker):
#   docker run -p 8983:8983 -t solr:9 solr-demo
#   OR
#   docker run -d -p 8983:8983 --name solr solr:9 solr-precreate langgraph_docs
#
# ENVIRONMENT VARIABLES:
#   SOLR_URL         = http://localhost:8983/solr   (default)
#   SOLR_COLLECTION  = langgraph_docs               (default)
#   SOLR_USERNAME    = (optional, for auth)
#   SOLR_PASSWORD    = (optional, for auth)
# =============================================================

import json
import logging
import os
from typing import Annotated, Any, Optional
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("lesson_26")

# ===========================================================================
# SECTION 1 — SOLR CLIENT SETUP
# ===========================================================================

SOLR_URL = os.getenv("SOLR_URL", "http://localhost:8983/solr")
SOLR_COLLECTION = os.getenv("SOLR_COLLECTION", "langgraph_docs")
SOLR_USERNAME = os.getenv("SOLR_USERNAME")
SOLR_PASSWORD = os.getenv("SOLR_PASSWORD")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# ---------------------------------------------------------------------------
# Try to import pysolr. Fall back to simulation if not installed.
# ---------------------------------------------------------------------------
try:
    import pysolr  # pip install pysolr
    PYSOLR_AVAILABLE = True
except ImportError:
    PYSOLR_AVAILABLE = False
    logger.warning("pysolr not installed. Run: pip install pysolr")

# ---------------------------------------------------------------------------
# Try to import LangChain Solr vector store (langchain-community >= 0.2.0)
# ---------------------------------------------------------------------------
try:
    from langchain_community.vectorstores import SolrVectorStore
    LANGCHAIN_SOLR_AVAILABLE = True
except ImportError:
    LANGCHAIN_SOLR_AVAILABLE = False
    logger.warning("SolrVectorStore not available in langchain-community. Using pysolr directly.")

# ---------------------------------------------------------------------------
# Try to import Ollama embeddings
# ---------------------------------------------------------------------------
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    LLM_AVAILABLE = True
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
except Exception:
    LLM_AVAILABLE = False
    llm = None
    embeddings = None


# ===========================================================================
# SECTION 2 — KNOWLEDGE BASE
# ===========================================================================
#
# Same LangGraph knowledge base as Lesson 12.
# In production this would be loaded from files, databases, or a data pipeline.

KNOWLEDGE_BASE = [
    "LangGraph is a library for building stateful, multi-step AI agent workflows as directed graphs.",
    "A StateGraph defines the workflow. Nodes are Python functions. Edges define execution order.",
    "The state is a TypedDict shared between all nodes. Reducers control how fields are updated.",
    "add_messages is a reducer that appends messages to a list instead of replacing them.",
    "compile() validates the graph and returns a runnable. invoke() runs the graph synchronously.",
    "stream() runs the graph and yields state snapshots after each node — used for real-time UIs.",
    "The @tool decorator creates tools. The docstring tells the LLM when and how to use the tool.",
    "bind_tools() attaches tool schemas to the LLM so it can choose to call them.",
    "ToolNode executes tool calls from the LLM and returns ToolMessage results.",
    "The ReAct pattern: LLM reasons → calls tool → reads result → reasons again → final answer.",
    "should_continue() routing function: if last message has tool_calls, go to tools; else END.",
    "MemorySaver stores checkpoints in RAM. State is lost on process restart.",
    "SqliteSaver stores checkpoints on disk. State survives process restarts.",
    "thread_id identifies a conversation session. Each user should have their own thread_id.",
    "get_state_history() returns all past snapshots — enables time-travel debugging.",
    "interrupt() pauses the graph and waits for human input. Requires a checkpointer.",
    "Command(resume=value) resumes a paused graph, passing the value to the interrupt() call.",
    "Always set recursion_limit in config to prevent infinite loops.",
    "Use with_structured_output(PydanticModel) to force the LLM to return valid structured data.",
    "Use Send() to run multiple tasks in parallel. Results merged by reducers.",
    "The supervisor pattern: one LLM routes tasks to specialist agents, each with focused tools.",
    "Always call list_tables() and describe_table() before writing SQL queries.",
    "Only allow SELECT queries — reject INSERT, UPDATE, DELETE with a safety guard.",
    "Parameterized queries prevent SQL injection: cursor.execute('SELECT * FROM t WHERE id=?', (x,))",
    "Mem0 provides automatic memory extraction, deduplication, and contradiction resolution.",
    "SqliteSaver and MemorySaver are short-term per-thread checkpointers.",
    "For enterprise memory at scale, use Mem0 with Qdrant or Mem0 Cloud.",
    "Chroma is a simple local vector store — good for prototypes, not for enterprise clusters.",
    "Apache Solr supports BM25 keyword search, kNN vector search, and hybrid combinations.",
    "SolrCloud scales horizontally across multiple nodes with automatic sharding and replication.",
]


# ===========================================================================
# SECTION 3 — SOLR CLIENT FACTORY
# ===========================================================================

def create_solr_client() -> Optional[Any]:
    """
    Create a pysolr client.

    Connection:
      - No auth: pysolr.Solr(url)
      - Basic auth: pysolr.Solr(url, auth=(user, password))
      - Kerberos: configure SOLR_USERNAME/PASSWORD or Kerberos ticket
    """
    if not PYSOLR_AVAILABLE:
        return None
    url = f"{SOLR_URL}/{SOLR_COLLECTION}"
    kwargs = {"always_commit": True, "timeout": 10}
    if SOLR_USERNAME and SOLR_PASSWORD:
        kwargs["auth"] = (SOLR_USERNAME, SOLR_PASSWORD)
    try:
        client = pysolr.Solr(url, **kwargs)
        client.ping()
        logger.info(f"Connected to Solr: {url}")
        return client
    except Exception as e:
        logger.warning(f"Cannot connect to Solr ({url}): {e}")
        return None


# Lazy singleton
_solr_client = None


def get_solr() -> Optional[Any]:
    global _solr_client
    if _solr_client is None:
        _solr_client = create_solr_client()
    return _solr_client


# ===========================================================================
# SECTION 4 — SOLR SCHEMA SETUP
# ===========================================================================
#
# Solr requires field definitions. For BM25+kNN hybrid search we need:
#   - id         (string, unique key)
#   - content    (text_general — full-text BM25 indexed)
#   - source     (string — metadata)
#   - vector     (knnDenseVector, dims=<embedding_size>)
#
# Schema API call (run once to set up the collection):

SOLR_SCHEMA_FIELDS = [
    {
        "name": "content",
        "type": "text_general",
        "stored": True,
        "indexed": True,
        "multiValued": False,
    },
    {
        "name": "source",
        "type": "string",
        "stored": True,
        "indexed": True,
    },
    # Dense vector field for kNN search (Solr 9.0+)
    # dim must match the embedding model output dimension
    # llama3 OllamaEmbeddings → 4096 dims
    {
        "name": "vector",
        "type": "knn_vector",    # defined as knnDenseVector in Solr fieldType
        "stored": True,
        "indexed": True,
    },
]


def setup_solr_schema():
    """
    Add required fields to the Solr collection schema via Schema API.

    Run this ONCE when setting up a new collection.
    Safe to run multiple times — existing fields are not duplicated.

    Note: Requires Solr 9+ for knnDenseVector (kNN search).
    """
    import urllib.request
    import urllib.error

    schema_url = f"{SOLR_URL}/{SOLR_COLLECTION}/schema"
    for field in SOLR_SCHEMA_FIELDS:
        payload = json.dumps({"add-field": field}).encode()
        req = urllib.request.Request(
            schema_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
                logger.info(f"Schema field '{field['name']}': {body[:80]}")
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if "already exists" in body:
                logger.info(f"Field '{field['name']}' already exists (ok)")
            else:
                logger.warning(f"Schema add-field '{field['name']}' failed: {body[:120]}")
        except Exception as e:
            logger.warning(f"Schema setup failed: {e}")


# ===========================================================================
# SECTION 5 — INDEXING
# ===========================================================================

def index_documents(documents: list[str], source_prefix: str = "kb") -> int:
    """
    Index a list of text strings into Solr.

    Each document gets:
      - id:      uuid (unique)
      - content: the text (BM25 indexed)
      - source:  "kb_0", "kb_1", etc. (metadata)
      - vector:  embedding if available (kNN indexed)

    Returns the number of documents successfully indexed.
    """
    client = get_solr()
    if client is None:
        logger.warning("Solr not available — documents not indexed")
        return 0

    docs_to_add = []
    for i, text in enumerate(documents):
        doc = {
            "id":      str(uuid4()),
            "content": text,
            "source":  f"{source_prefix}_{i}",
        }
        # Add vector if embeddings are available
        if embeddings is not None:
            try:
                vec = embeddings.embed_query(text)
                doc["vector"] = vec
            except Exception as e:
                logger.warning(f"Embedding failed for doc {i}: {e}")
        docs_to_add.append(doc)

    try:
        client.add(docs_to_add)
        logger.info(f"Indexed {len(docs_to_add)} documents into Solr collection '{SOLR_COLLECTION}'")
        return len(docs_to_add)
    except Exception as e:
        logger.error(f"Solr indexing error: {e}")
        return 0


# ===========================================================================
# SECTION 6 — SEARCH MODES
# ===========================================================================

def solr_bm25_search(query: str, top_k: int = 3) -> list[dict]:
    """
    BM25 keyword search using Solr's standard query parser.

    Best for: exact keyword matches, structured queries, enterprise full-text search.
    Solr query: q=content:(<words>) OR content:(<phrase>)

    Returns: [{"content": "...", "source": "...", "score": 1.23}]
    """
    client = get_solr()
    if client is None:
        return []
    try:
        # Escape special Solr chars
        safe_query = query.replace(":", "\\:").replace('"', '\\"')
        results = client.search(
            f"content:({safe_query})",
            rows=top_k,
            fl="id,content,source,score",
        )
        return [
            {
                "content": r.get("content", ""),
                "source":  r.get("source",  ""),
                "score":   r.get("score",   0.0),
                "mode":    "bm25",
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return []


def solr_knn_search(query: str, top_k: int = 3) -> list[dict]:
    """
    kNN (dense vector / semantic) search using Solr Dense Vector Field.

    Requires: Solr 9+, 'vector' field indexed with knnDenseVector type.
    Best for: semantic similarity, "meaning over keywords".

    Solr KNN query syntax:
      {{!knn f=vector topK=N}}<comma-separated-vector>

    Returns: [{"content": "...", "source": "...", "score": 0.87}]
    """
    client = get_solr()
    if client is None or embeddings is None:
        return []
    try:
        vec = embeddings.embed_query(query)
        vec_str = ",".join(str(v) for v in vec)
        results = client.search(
            f"{{!knn f=vector topK={top_k}}}{vec_str}",
            rows=top_k,
            fl="id,content,source,score",
        )
        return [
            {
                "content": r.get("content", ""),
                "source":  r.get("source",  ""),
                "score":   r.get("score",   0.0),
                "mode":    "knn",
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"kNN search error: {e}")
        return []


def solr_hybrid_search(query: str, top_k: int = 5, bm25_weight: float = 0.4, knn_weight: float = 0.6) -> list[dict]:
    """
    Hybrid search: combine BM25 + kNN results with weighted re-ranking.

    Strategy:
      1. Run BM25 search → normalize scores to [0, 1]
      2. Run kNN search  → normalize scores to [0, 1]
      3. Merge by doc ID, compute: final_score = bm25_weight*bm25 + knn_weight*knn
      4. Sort by final_score descending

    This is the production-grade approach used by enterprise search teams.
    kNN weight > BM25 weight = prefer semantic matching.
    Tune weights based on your use case and relevance judgments.

    Returns: [{"content": "...", "source": "...", "score": 0.91, "mode": "hybrid"}]
    """
    bm25_results = solr_bm25_search(query, top_k=top_k * 2)
    knn_results  = solr_knn_search(query,  top_k=top_k * 2)

    # Normalize helper
    def normalize(results: list[dict]) -> dict[str, float]:
        if not results:
            return {}
        scores = [r["score"] for r in results]
        max_s, min_s = max(scores), min(scores)
        denom = max_s - min_s if max_s != min_s else 1.0
        return {r["content"]: (r["score"] - min_s) / denom for r in results}

    bm25_norm = normalize(bm25_results)
    knn_norm  = normalize(knn_results)

    # Merge by content (use content as unique key since IDs differ between calls)
    merged: dict[str, dict] = {}
    all_results = bm25_results + knn_results
    for r in all_results:
        key = r["content"]
        if key not in merged:
            merged[key] = {"content": r["content"], "source": r["source"], "mode": "hybrid"}

    for key, item in merged.items():
        b = bm25_norm.get(key, 0.0)
        k = knn_norm.get(key, 0.0)
        item["score"] = round(bm25_weight * b + knn_weight * k, 4)

    sorted_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:top_k]


# ===========================================================================
# SECTION 7 — SIMULATION MODE
# ===========================================================================
#
# When Solr is not reachable, fall back to a simple in-process search
# so the LangGraph agent can still run and demonstrate the pattern.

class SimulatedSolr:
    """
    In-process Solr simulation using keyword overlap.
    Supports BM25-style search only (no embeddings).
    Replace with real Solr for production.
    """

    def __init__(self, documents: list[str]):
        self._docs = [
            {"id": str(i), "content": d, "source": f"kb_{i}"}
            for i, d in enumerate(documents)
        ]

    def bm25_search(self, query: str, top_k: int = 3) -> list[dict]:
        q_words = set(query.lower().split())
        scored = []
        for doc in self._docs:
            d_words = set(doc["content"].lower().split())
            overlap = len(q_words & d_words)
            if overlap > 0:
                scored.append({**doc, "score": overlap / len(q_words), "mode": "bm25_sim"})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def knn_search(self, query: str, top_k: int = 3) -> list[dict]:
        return self.bm25_search(query, top_k)  # Fallback: no vectors in sim

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict]:
        return self.bm25_search(query, top_k)


_simulated_solr: Optional[SimulatedSolr] = None


def get_search_backend():
    """
    Return real Solr client if reachable, else SimulatedSolr.
    Used by the search tool to be transparent about availability.
    """
    if get_solr() is not None:
        return None  # Use real Solr functions
    global _simulated_solr
    if _simulated_solr is None:
        _simulated_solr = SimulatedSolr(KNOWLEDGE_BASE)
        logger.info("Using SimulatedSolr (start Solr + set SOLR_URL for real search)")
    return _simulated_solr


# ===========================================================================
# SECTION 8 — RETRIEVAL TOOLS (used by the LangGraph agent)
# ===========================================================================

@tool
def solr_keyword_search(query: str) -> str:
    """
    Search the LangGraph knowledge base using BM25 keyword matching (Solr).
    Best for: exact terms, structured queries, and keyword-heavy questions.
    Returns the top 3 most relevant documents.
    """
    sim = get_search_backend()
    if sim:
        results = sim.bm25_search(query, top_k=3)
    else:
        results = solr_bm25_search(query, top_k=3)

    if not results:
        return "No results found. Try a different query."

    lines = [f"[BM25 Source {i+1} | score={r['score']:.2f}] {r['content']}"
             for i, r in enumerate(results)]
    return "\n\n".join(lines)


@tool
def solr_semantic_search(query: str) -> str:
    """
    Search the LangGraph knowledge base using semantic kNN vector search (Solr).
    Best for: meaning-based queries, paraphrase matching, fuzzy concept search.
    Returns the top 3 semantically similar documents.
    """
    sim = get_search_backend()
    if sim:
        results = sim.knn_search(query, top_k=3)
    else:
        results = solr_knn_search(query, top_k=3)

    if not results:
        return "No results found. Try a different query."

    lines = [f"[kNN Source {i+1} | score={r['score']:.2f}] {r['content']}"
             for i, r in enumerate(results)]
    return "\n\n".join(lines)


@tool
def solr_hybrid_search_tool(query: str) -> str:
    """
    Search the LangGraph knowledge base using hybrid BM25+kNN search (Solr).
    Best for: most queries — combines keyword precision with semantic recall.
    Use this as your PRIMARY retrieval tool.
    Returns the top 5 documents ranked by combined BM25 + semantic score.
    """
    sim = get_search_backend()
    if sim:
        results = sim.hybrid_search(query, top_k=5)
    else:
        results = solr_hybrid_search(query, top_k=5)

    if not results:
        return "No results found. Try a different query or check Solr connectivity."

    lines = [f"[Hybrid Source {i+1} | score={r['score']:.4f}] {r['content']}"
             for i, r in enumerate(results)]
    return "\n\n".join(lines)


# ===========================================================================
# SECTION 9 — BASIC SOLR RAG AGENT
# ===========================================================================

class SolrRAGState(TypedDict):
    messages: Annotated[list, add_messages]


SOLR_RAG_SYSTEM_PROMPT = """You are a LangGraph expert assistant with access to a Solr knowledge base.

ALWAYS follow this process:
1. Call solr_hybrid_search_tool() with the user's question FIRST
2. Read the retrieved documents carefully
3. Base your answer ONLY on the retrieved documents
4. If documents don't contain the answer, say "I don't have that in my knowledge base."
5. Cite which source number your answer comes from

For keyword-heavy questions (exact terms): also try solr_keyword_search().
For meaning-based questions (concepts): also try solr_semantic_search().
Never answer from general knowledge alone — always retrieve first."""

_rag_tools = [solr_hybrid_search_tool, solr_keyword_search, solr_semantic_search]


def make_solr_rag_agent():
    """Build and return the basic Solr RAG agent graph."""
    if not LLM_AVAILABLE:
        logger.warning("LLM not available — agent will not function")
        return None

    rag_llm = llm.bind_tools(_rag_tools)

    def agent_node(state: SolrRAGState) -> dict:
        msgs = [SystemMessage(content=SOLR_RAG_SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [rag_llm.invoke(msgs)]}

    def should_retrieve(state: SolrRAGState) -> str:
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "end"

    builder = StateGraph(SolrRAGState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(_rag_tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_retrieve, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")
    return builder.compile()


# ===========================================================================
# SECTION 10 — SELF-CORRECTING SOLR RAG AGENT
# ===========================================================================
#
# Same pattern as Lesson 12, but using Solr hybrid search.
# If retrieved docs are not relevant enough → re-query with refined terms.

class SelfCorrectingSolrRAGState(TypedDict):
    messages:        Annotated[list, add_messages]
    query:           str
    retrieved_docs:  str
    relevance_score: float
    retry_count:     int
    final_answer:    str


def solr_retrieve_node(state: SelfCorrectingSolrRAGState) -> dict:
    """Retrieve documents using hybrid search."""
    sim = get_search_backend()
    if sim:
        results = sim.hybrid_search(state["query"], top_k=5)
    else:
        results = solr_hybrid_search(state["query"], top_k=5)
    formatted = "\n".join(f"[{r['mode']} | {r['score']:.3f}] {r['content']}"
                          for r in results)
    return {"retrieved_docs": formatted or "No results found."}


def check_solr_relevance(state: SelfCorrectingSolrRAGState) -> dict:
    """Use LLM to score relevance of Solr results to the query."""
    if not LLM_AVAILABLE:
        return {"relevance_score": 0.5}
    check_prompt = f"""Question: {state['query']}

Retrieved documents:
{state['retrieved_docs']}

Score relevance from 0.0 to 1.0 (how well do the docs answer the question?).
Reply with ONLY a number like: 0.8"""
    resp = llm.invoke([HumanMessage(content=check_prompt)])
    try:
        score = float(resp.content.strip().split()[0])
    except (ValueError, IndexError):
        score = 0.5
    return {"relevance_score": score}


def route_solr_relevance(state: SelfCorrectingSolrRAGState) -> str:
    if state["relevance_score"] >= 0.5 or state["retry_count"] >= 2:
        return "generate"
    return "refine_query"


def refine_solr_query(state: SelfCorrectingSolrRAGState) -> dict:
    """Rephrase the query for better Solr BM25 keyword coverage."""
    if not LLM_AVAILABLE:
        return {"query": state["query"], "retry_count": state["retry_count"] + 1}
    refine_prompt = f"""The query '{state['query']}' retrieved low-relevance documents from Solr.

Rephrase it using more specific keywords that would match documentation better.
Consider: technical terms, function names, pattern names.
Reply with ONLY the new query string."""
    resp = llm.invoke([HumanMessage(content=refine_prompt)])
    return {"query": resp.content.strip(), "retry_count": state["retry_count"] + 1}


def generate_solr_answer(state: SelfCorrectingSolrRAGState) -> dict:
    """Generate final answer from retrieved Solr documents."""
    if not LLM_AVAILABLE:
        answer = f"[LLM unavailable] Retrieved docs:\n{state['retrieved_docs']}"
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
    msgs = [
        SystemMessage(content="Answer based on the retrieved documents. Cite source numbers."),
        HumanMessage(content=f"Question: {state['query']}\n\nSources:\n{state['retrieved_docs']}")
    ]
    resp = llm.invoke(msgs)
    return {"final_answer": resp.content, "messages": [resp]}


def make_self_correcting_solr_rag():
    """Build and return the self-correcting Solr RAG graph."""
    builder = StateGraph(SelfCorrectingSolrRAGState)
    builder.add_node("retrieve",      solr_retrieve_node)
    builder.add_node("check_rel",     check_solr_relevance)
    builder.add_node("refine_query",  refine_solr_query)
    builder.add_node("generate",      generate_solr_answer)
    builder.add_edge(START,           "retrieve")
    builder.add_edge("retrieve",      "check_rel")
    builder.add_conditional_edges(
        "check_rel", route_solr_relevance,
        {"generate": "generate", "refine_query": "refine_query"}
    )
    builder.add_edge("refine_query",  "retrieve")
    builder.add_edge("generate",      END)
    return builder.compile()


# ===========================================================================
# SECTION 11 — LANGCHAIN SolrVectorStore INTEGRATION
# ===========================================================================
#
# LangChain provides a SolrVectorStore that wraps Solr with the standard
# VectorStore interface (add_documents, similarity_search, etc.)
# This lets you swap Chroma ↔ Solr with minimal code changes.

def build_langchain_solr_vectorstore():
    """
    Create a LangChain SolrVectorStore for use with LCEL chains or agents.

    Requires: langchain-community>=0.2.0 with SolrVectorStore support.
    Equivalent to lesson_12_rag_agent.build_vector_store() but backed by Solr.
    """
    if not LANGCHAIN_SOLR_AVAILABLE:
        logger.warning("SolrVectorStore not in langchain-community. Use pysolr directly (see above).")
        return None
    if embeddings is None:
        logger.warning("Embeddings not available. SolrVectorStore needs OllamaEmbeddings.")
        return None

    docs = [
        Document(page_content=text, metadata={"source": f"doc_{i}"})
        for i, text in enumerate(KNOWLEDGE_BASE)
    ]

    try:
        vectorstore = SolrVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            solr_url=f"{SOLR_URL}/{SOLR_COLLECTION}",
        )
        logger.info(f"LangChain SolrVectorStore ready: {len(docs)} docs")
        return vectorstore
    except Exception as e:
        logger.error(f"SolrVectorStore init failed: {e}")
        return None


# ===========================================================================
# MAIN — Demo
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("LESSON 26 — RAG with Apache Solr")
    print("=" * 65)

    if not PYSOLR_AVAILABLE:
        print("\n⚠️  pysolr not installed. Running with SimulatedSolr.")
        print("   Install for real Solr: pip install pysolr")
        print("   Start Solr: docker run -d -p 8983:8983 solr:9 solr-precreate langgraph_docs")
        print()

    sim = get_search_backend()
    if sim is None:
        print(f"Connected to Solr: {SOLR_URL}/{SOLR_COLLECTION}")
        indexed = index_documents(KNOWLEDGE_BASE)
        print(f"Indexed {indexed} documents")
    else:
        print("Using SimulatedSolr (in-process keyword search)\n")

    # -----------------------------------------------------------------------
    # Demo 1: Search mode comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SEARCH MODE COMPARISON")
    print("=" * 65)

    test_queries = [
        "What is the difference between MemorySaver and SqliteSaver?",
        "How do I prevent SQL injection?",
        "How does interrupt() work with HITL patterns?",
    ]

    for q in test_queries:
        print(f"\n❓ Query: {q}")

        if sim:
            bm25 = sim.bm25_search(q, top_k=2)
            hybrid = sim.hybrid_search(q, top_k=2)
        else:
            bm25 = solr_bm25_search(q, top_k=2)
            hybrid = solr_hybrid_search(q, top_k=2)

        print(f"  BM25:   {bm25[0]['content'][:80] if bm25 else 'no results'}")
        print(f"  Hybrid: {hybrid[0]['content'][:80] if hybrid else 'no results'}")

    # -----------------------------------------------------------------------
    # Demo 2: Basic Solr RAG Agent
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("BASIC SOLR RAG AGENT")
    print("=" * 65)

    agent = make_solr_rag_agent()
    if agent is None:
        print("⚠️  LLM not available. Skipping agent demo.")
    else:
        questions = [
            "What is a StateGraph and how does it work?",
            "How does the ReAct pattern work in LangGraph?",
            "What is the difference between Chroma and Solr for RAG?",
        ]
        for q in questions:
            print(f"\n❓ {q}")
            result = agent.invoke({"messages": [HumanMessage(content=q)]})
            last_ai = next(
                (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
                ""
            )
            print(f"💬 {last_ai[:300]}")

    # -----------------------------------------------------------------------
    # Demo 3: Self-correcting Solr RAG
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SELF-CORRECTING SOLR RAG AGENT")
    print("=" * 65)

    sc_graph = make_self_correcting_solr_rag()
    result = sc_graph.invoke({
        "messages":        [],
        "query":           "How do I handle parallel fan-out in LangGraph?",
        "retrieved_docs":  "",
        "relevance_score": 0.0,
        "retry_count":     0,
        "final_answer":    "",
    })
    print(f"Final answer:\n{result['final_answer'][:400]}")
    print(f"Retries used: {result['retry_count']}")
    print(f"Relevance score: {result['relevance_score']}")


# =============================================================
# EXERCISES:
#
#   1. Tune BM25 vs kNN weights in solr_hybrid_search():
#      bm25_weight=0.6, knn_weight=0.4 → more keyword precision
#      bm25_weight=0.2, knn_weight=0.8 → more semantic recall
#      Measure result quality on 10 test queries.
#
#   2. Add Solr faceted search:
#      After retrieving docs, run a facet query to show topic breakdown.
#      client.search("*:*", facet=True, **{"facet.field": "source", "facet.limit": 5})
#
#   3. Compare Solr vs Chroma (Lesson 12):
#      Run the same 5 questions through both.
#      Which returns more relevant top-1 results? Which is faster?
#
#   4. Implement Solr document update:
#      When knowledge base changes, re-index only changed documents.
#      client.add([{"id": existing_id, "content": new_text}])
#
#   5. Add Solr authentication:
#      Set SOLR_USERNAME and SOLR_PASSWORD env vars.
#      Verify that unauthenticated requests are rejected.
# =============================================================
