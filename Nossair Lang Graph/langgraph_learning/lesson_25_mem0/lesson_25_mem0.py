# =============================================================
# LESSON 25 — Long-Term Memory with Mem0
# =============================================================
#
# WHAT YOU WILL LEARN:
#   1. What Mem0 is and how it differs from Chroma/vector stores
#   2. How Mem0 automatically extracts, deduplicates, and updates memories
#   3. How to integrate Mem0 as the memory layer in a LangGraph agent
#   4. Cross-session, cross-thread user memory (user_id scoped)
#   5. Searching, listing, updating, and deleting Mem0 memories (GDPR)
#   6. Mem0 with a local Ollama LLM + local Qdrant vector store (no API key)
#   7. Mem0 Cloud (managed) vs self-hosted comparison
#
# WHY Mem0 OVER PLAIN CHROMA (Lesson 13)?
#
#   | Feature                         | Chroma (L13)      | Mem0 (L25)              |
#   |---------------------------------|-------------------|-------------------------|
#   | Fact extraction                 | You write prompt  | Automatic               |
#   | Deduplication                   | None              | Built-in (LLM-driven)   |
#   | Contradiction handling          | None              | Auto-update/replace     |
#   | Memory categories               | Manual metadata   | Auto-classified         |
#   | Search                          | similarity_search | m.search(query, user_id)|
#   | GDPR delete                     | Manual filter     | m.delete_all(user_id)   |
#   | Production ready                | DIY               | Managed or self-hosted  |
#
# PATTERN (per turn):
#   1. Search Mem0 for relevant memories → inject as system context
#   2. Chat with LLM
#   3. Add conversation turn to Mem0 → auto-extracts + deduplicates facts
#
# INSTALL:
#   pip install mem0ai qdrant-client
#   (Qdrant: self-hosted vector store used by Mem0 locally)
#   (Or use Mem0 cloud: set MEM0_API_KEY env var)
#
# LOCAL SETUP:
#   docker run -p 6333:6333 qdrant/qdrant   ← needed for local mode
#   (or use in-memory Qdrant: no Docker needed, data lost on restart)
# =============================================================

import json
import logging
import os
from typing import Annotated, Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("lesson_25")

# ===========================================================================
# SECTION 1 — MEM0 SETUP
# ===========================================================================
#
# Mem0 config hierarchy:
#   Option A — Mem0 Cloud (managed, needs MEM0_API_KEY)
#   Option B — Self-hosted: local Ollama LLM + Qdrant vector store
#   Option C — Simulation mode (this file, when neither is available)
#
# Mem0 config reference: https://docs.mem0.ai/open-source/llms/ollama

MEM0_API_KEY = os.getenv("MEM0_API_KEY")          # Option A: Mem0 Cloud
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ---------------------------------------------------------------------------
# Try to import Mem0. Fall back to simulation if not installed.
# ---------------------------------------------------------------------------
try:
    from mem0 import Memory, MemoryClient  # pip install mem0ai
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    logger.warning("mem0ai not installed. Run: pip install mem0ai qdrant-client")


def create_mem0_client():
    """
    Build the correct Mem0 client depending on environment.

    Priority:
      1. MEM0_API_KEY set  → Mem0 Cloud (MemoryClient — fully managed)
      2. Qdrant reachable  → Self-hosted Memory with local Ollama + Qdrant
      3. Fallback          → In-process Qdrant (ephemeral, no Docker needed)
    """
    if not MEM0_AVAILABLE:
        return None

    if MEM0_API_KEY:
        logger.info("Using Mem0 Cloud (MemoryClient)")
        return MemoryClient(api_key=MEM0_API_KEY)

    # Self-hosted config: local Ollama LLM + local Qdrant vector store
    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": OLLAMA_MODEL,
                "temperature": 0,
                "ollama_base_url": OLLAMA_BASE_URL,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": OLLAMA_MODEL,
                "ollama_base_url": OLLAMA_BASE_URL,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "langgraph_memories",
                "url": QDRANT_URL,
                # For in-process Qdrant (no Docker): use path instead of url
                # "path": "/tmp/qdrant_mem0",
            },
        },
        "version": "v1.1",  # enables graph memory (relationship tracking)
    }

    try:
        logger.info("Creating self-hosted Mem0 (Ollama + Qdrant)")
        return Memory.from_config(config)
    except Exception as e:
        logger.warning(f"Qdrant not reachable ({e}). Falling back to in-process Qdrant.")
        # Fallback: replace url with path (in-process, no Docker)
        config["vector_store"]["config"] = {
            "collection_name": "langgraph_memories",
            "path": os.path.join(os.path.dirname(__file__), "qdrant_local_db"),
        }
        try:
            return Memory.from_config(config)
        except Exception as e2:
            logger.error(f"Mem0 init failed: {e2}")
            return None


# Lazy singleton — initialized once per process
_mem0_client = None


def get_mem0() -> Optional[Any]:
    global _mem0_client
    if _mem0_client is None:
        _mem0_client = create_mem0_client()
    return _mem0_client


# ===========================================================================
# SECTION 2 — MEM0 CORE OPERATIONS
# ===========================================================================

def mem0_search(query: str, user_id: str, limit: int = 5) -> list[dict]:
    """
    Search Mem0 for memories relevant to the query.

    Returns list of dicts: [{"memory": "...", "score": 0.9, "id": "..."}]
    """
    m = get_mem0()
    if m is None:
        return []
    try:
        results = m.search(query=query, user_id=user_id, limit=limit)
        # Mem0 returns: {"results": [{"memory": ..., "score": ..., "id": ...}]}
        if isinstance(results, dict):
            return results.get("results", [])
        return results if isinstance(results, list) else []
    except Exception as e:
        logger.error(f"mem0_search error: {e}")
        return []


def mem0_add(messages: list[dict], user_id: str) -> dict:
    """
    Add a conversation turn to Mem0.

    Mem0 automatically:
      - Extracts facts worth remembering
      - Deduplicates against existing memories
      - Updates contradictory memories (e.g., "User prefers Java" replaces
        "User prefers Python" if the new message says so)

    messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    m = get_mem0()
    if m is None:
        return {}
    try:
        result = m.add(messages=messages, user_id=user_id)
        logger.info(f"mem0_add → stored {len(result.get('results', []))} memories for {user_id}")
        return result
    except Exception as e:
        logger.error(f"mem0_add error: {e}")
        return {}


def mem0_list_all(user_id: str) -> list[dict]:
    """Return all stored memories for a user."""
    m = get_mem0()
    if m is None:
        return []
    try:
        results = m.get_all(user_id=user_id)
        if isinstance(results, dict):
            return results.get("results", [])
        return results if isinstance(results, list) else []
    except Exception as e:
        logger.error(f"mem0_list_all error: {e}")
        return []


def mem0_delete_memory(memory_id: str) -> bool:
    """Delete a single memory by its ID."""
    m = get_mem0()
    if m is None:
        return False
    try:
        m.delete(memory_id=memory_id)
        logger.info(f"Deleted memory: {memory_id}")
        return True
    except Exception as e:
        logger.error(f"mem0_delete error: {e}")
        return False


def mem0_delete_all(user_id: str) -> bool:
    """
    Delete ALL memories for a user (GDPR right-to-erasure).

    In Mem0 Cloud: m.delete_all(user_id=user_id)
    In self-hosted: iterate m.get_all() → delete each
    """
    m = get_mem0()
    if m is None:
        return False
    try:
        m.delete_all(user_id=user_id)
        logger.info(f"GDPR erasure complete for user: {user_id}")
        return True
    except AttributeError:
        # Older mem0ai versions use reset()
        try:
            memories = mem0_list_all(user_id)
            for mem in memories:
                m.delete(memory_id=mem["id"])
            logger.info(f"GDPR erasure complete for user: {user_id}")
            return True
        except Exception as e2:
            logger.error(f"mem0_delete_all fallback error: {e2}")
            return False
    except Exception as e:
        logger.error(f"mem0_delete_all error: {e}")
        return False


def mem0_update_memory(memory_id: str, new_content: str) -> bool:
    """Update the text content of a specific memory."""
    m = get_mem0()
    if m is None:
        return False
    try:
        m.update(memory_id=memory_id, data=new_content)
        logger.info(f"Updated memory {memory_id}: {new_content}")
        return True
    except Exception as e:
        logger.error(f"mem0_update error: {e}")
        return False


# ===========================================================================
# SECTION 3 — LANGGRAPH STATE
# ===========================================================================

class Mem0AgentState(TypedDict):
    messages:         Annotated[list, add_messages]  # conversation history
    user_id:          str                             # per-user memory namespace
    retrieved_memories: list                          # memories loaded this turn
    memory_context:   str                             # formatted system context


# ===========================================================================
# SECTION 4 — LANGGRAPH NODES
# ===========================================================================

# ---------------------------------------------------------------------------
# NODE 1 — Load Memories
# Runs BEFORE LLM call. Searches Mem0 and builds a system context string.
# ---------------------------------------------------------------------------

def load_memories_node(state: Mem0AgentState) -> dict:
    """Search Mem0 for memories relevant to the current user message."""
    if not state["messages"]:
        return {"retrieved_memories": [], "memory_context": ""}

    last_msg = state["messages"][-1]
    query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    user_id = state["user_id"]

    memories = mem0_search(query=query, user_id=user_id, limit=5)

    if memories:
        lines = [f"• {m['memory']}" for m in memories]
        context = "Relevant memories from past conversations:\n" + "\n".join(lines)
        logger.info(f"Loaded {len(memories)} memories for {user_id}")
    else:
        context = ""
        logger.info(f"No memories found for {user_id} — first interaction or no relevant history")

    return {"retrieved_memories": memories, "memory_context": context}


# ---------------------------------------------------------------------------
# NODE 2 — Chat
# Runs LLM with memory context injected into system prompt.
# ---------------------------------------------------------------------------

try:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False
    llm = None


def chat_node(state: Mem0AgentState) -> dict:
    """Generate a response using LLM, with Mem0 memories as context."""
    context = state.get("memory_context", "")

    system_parts = ["You are a helpful, personalized assistant."]
    if context:
        system_parts.append(f"\n{context}")
        system_parts.append("\nUse this context to personalize your response naturally.")

    system_prompt = SystemMessage(content="\n".join(system_parts))
    msgs = [system_prompt] + list(state["messages"])

    if not LLM_AVAILABLE:
        response = AIMessage(content="[LLM not available — install langchain-ollama and start Ollama]")
    else:
        response = llm.invoke(msgs)

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# NODE 3 — Save Memories
# Runs AFTER LLM call. Sends the (user, assistant) turn to Mem0.
# Mem0 handles extraction, deduplication, contradiction resolution.
# ---------------------------------------------------------------------------

def save_memories_node(state: Mem0AgentState) -> dict:
    """
    Pass the latest conversation turn to Mem0.

    Mem0 automatically:
      1. Runs an extraction LLM call to find facts worth storing
      2. Checks existing memories for duplicates/contradictions
      3. Adds/updates/ignores appropriately
    """
    msgs = state["messages"]
    if len(msgs) < 2:
        return {}

    # Collect the last human and assistant messages
    last_human_content = ""
    last_ai_content = ""
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage) and not last_ai_content:
            last_ai_content = msg.content
        elif isinstance(msg, HumanMessage) and not last_human_content:
            last_human_content = msg.content
        if last_human_content and last_ai_content:
            break

    if not last_human_content:
        return {}

    turn = [
        {"role": "user",      "content": last_human_content},
        {"role": "assistant", "content": last_ai_content or ""},
    ]

    mem0_add(messages=turn, user_id=state["user_id"])
    return {}


# ===========================================================================
# SECTION 5 — BUILD THE GRAPH
# ===========================================================================

def build_mem0_graph(checkpointer=None):
    """
    Graph topology:
      START → load_memories → chat → save_memories → END

    The same pattern as Lesson 13 (Chroma), but using Mem0 instead.
    Mem0 gives us automatic extraction, deduplication, and updates for free.
    """
    builder = StateGraph(Mem0AgentState)

    builder.add_node("load_memories", load_memories_node)
    builder.add_node("chat",          chat_node)
    builder.add_node("save_memories", save_memories_node)

    builder.add_edge(START,            "load_memories")
    builder.add_edge("load_memories",  "chat")
    builder.add_edge("chat",           "save_memories")
    builder.add_edge("save_memories",  END)

    return builder.compile(checkpointer=checkpointer)


# ===========================================================================
# SECTION 6 — SIMULATION MODE (no Mem0 / no Ollama required)
# ===========================================================================
#
# When mem0ai is not installed, we simulate the Mem0 API with a simple
# in-process dict so the graph can still run and demonstrate the pattern.

class SimulatedMem0:
    """
    A minimal in-process Mem0 substitute for demo/testing.
    Stores memories per user_id in a dict. No embedding, no LLM extraction.
    Replace with real Mem0 in production.
    """

    def __init__(self):
        self._store: dict[str, list[dict]] = {}
        self._id_counter = 0

    def search(self, query: str, user_id: str, limit: int = 5) -> dict:
        memories = self._store.get(user_id, [])
        # Simple keyword overlap scoring (no embeddings)
        q_words = set(query.lower().split())
        scored = []
        for mem in memories:
            m_words = set(mem["memory"].lower().split())
            score = len(q_words & m_words) / max(len(q_words), 1)
            scored.append({**mem, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"results": scored[:limit]}

    def add(self, messages: list[dict], user_id: str) -> dict:
        if user_id not in self._store:
            self._store[user_id] = []
        # Extract a naive "memory" from the user message
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        added = []
        for content in user_msgs:
            if len(content) > 20:  # ignore trivial messages
                self._id_counter += 1
                entry = {"id": str(self._id_counter), "memory": content}
                self._store[user_id].append(entry)
                added.append(entry)
        return {"results": added}

    def get_all(self, user_id: str) -> dict:
        return {"results": self._store.get(user_id, [])}

    def delete(self, memory_id: str):
        for memories in self._store.values():
            for m in memories:
                if m["id"] == memory_id:
                    memories.remove(m)
                    return

    def delete_all(self, user_id: str):
        self._store.pop(user_id, None)

    def update(self, memory_id: str, data: str):
        for memories in self._store.values():
            for m in memories:
                if m["id"] == memory_id:
                    m["memory"] = data
                    return


def get_mem0_or_simulation():
    """Return real Mem0 client if available, else SimulatedMem0."""
    real = get_mem0()
    if real is not None:
        return real
    logger.info("Using SimulatedMem0 (install mem0ai + qdrant-client for real Mem0)")
    return SimulatedMem0()


# ===========================================================================
# SECTION 7 — UTILITY FUNCTIONS
# ===========================================================================

def print_user_memories(user_id: str):
    """Print all memories stored for a user (admin/debug utility)."""
    m = get_mem0()
    if m is None:
        print(f"  ⚠️  Mem0 not available")
        return
    memories = mem0_list_all(user_id)
    if not memories:
        print(f"  No memories stored for {user_id}")
        return
    print(f"  All memories for {user_id}:")
    for i, mem in enumerate(memories, 1):
        score_str = f" (score: {mem.get('score', '?')})" if "score" in mem else ""
        print(f"    {i}. {mem.get('memory', mem)}{score_str}")


def gdpr_erase_user(user_id: str) -> bool:
    """Full GDPR erasure: delete all Mem0 memories for a user."""
    success = mem0_delete_all(user_id)
    if success:
        print(f"✅ GDPR erasure complete for user: {user_id}")
    else:
        print(f"❌ GDPR erasure failed for user: {user_id}")
    return success


# ===========================================================================
# SECTION 8 — COMPARISON: Mem0 vs Chroma (Lesson 13) — code comparison
# ===========================================================================
#
# LESSON 13 (Chroma) — you wrote all the plumbing:
#
#   extract_prompt = f"""Extract 1-3 concise facts worth remembering...
#       User said: {last_human}
#       Return ONLY a JSON list..."""
#   raw = llm.invoke([HumanMessage(content=extract_prompt)]).content
#   facts = json.loads(raw[start:end+1])
#   for fact in facts:
#       memory_store.add_documents([Document(page_content=fact, metadata={"user_id": user_id})])
#
# LESSON 25 (Mem0) — Mem0 does all of this automatically:
#
#   mem0_add(messages=[{"role": "user", "content": last_human}], user_id=user_id)
#   # Mem0 internally: extracts facts, deduplicates, resolves contradictions
#
# The payoff: Mem0 also detects "User prefers Python" vs "User now prefers Java"
# and UPDATES the stored memory instead of storing a contradiction.


# ===========================================================================
# MAIN — Demo multi-turn conversation with Mem0
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("LESSON 25 — Long-Term Memory with Mem0")
    print("=" * 65)

    if not MEM0_AVAILABLE:
        print("\n⚠️  mem0ai not installed. Running with SimulatedMem0.")
        print("   Install for real Mem0: pip install mem0ai qdrant-client")
        print("   (For Mem0 Cloud: set MEM0_API_KEY env var)")
        print()

    # Patch get_mem0() to use simulation if real mem0 is unavailable
    if not MEM0_AVAILABLE:
        _sim = SimulatedMem0()
        import lesson_25_mem0.lesson_25_mem0 as _self
        _self._mem0_client = _sim

    graph = build_mem0_graph(checkpointer=MemorySaver())

    user_id = "user-mem0-demo-001"
    config = {"configurable": {"thread_id": f"mem0-{user_id}"}}

    print(f"\nUser ID: {user_id}")
    print("-" * 65)

    conversation = [
        # Session 1: Introduce yourself
        "Hi! I'm Sarah, a backend engineer with 7 years of experience.",
        # Tell preferences
        "I mainly use Python and Go. I really prefer Go for performance-critical services.",
        # Add a project context
        "I'm currently building a distributed task scheduler for my company.",
        # Test recall (Mem0 should have extracted facts from above)
        "What programming languages do I prefer?",
        # Test contradiction handling — Mem0 should UPDATE, not duplicate
        "Actually, I switched teams. Now I mostly write TypeScript for a Node.js microservice.",
        # Final recall — should show updated preference
        "What do you know about my tech stack?",
    ]

    for i, turn_text in enumerate(conversation, 1):
        print(f"\n[Turn {i}]")
        print(f"👤 User: {turn_text}")

        state = {
            "messages":           [HumanMessage(content=turn_text)],
            "user_id":            user_id,
            "retrieved_memories": [],
            "memory_context":     "",
        }

        result = graph.invoke(state, config=config)

        last_ai = next(
            (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            ""
        )
        print(f"🤖 AI: {last_ai[:250]}")

        if result.get("retrieved_memories"):
            mems = result["retrieved_memories"][:3]
            print(f"   📚 Retrieved {len(mems)} memories:")
            for mem in mems:
                print(f"      • {mem.get('memory', mem)}")

    # Show all stored memories
    print("\n" + "=" * 65)
    print(f"ALL STORED MEMORIES — {user_id}")
    print("=" * 65)
    print_user_memories(user_id)

    # GDPR Demo
    print("\n" + "=" * 65)
    print("GDPR ERASURE DEMO")
    print("=" * 65)
    gdpr_erase_user(user_id)
    print("After erasure:")
    print_user_memories(user_id)


# =============================================================
# EXERCISES:
#
#   1. Multi-user isolation:
#      Run two users (user_a, user_b) through the graph.
#      Verify user_a's memories never appear in user_b's search results.
#      (Mem0 scopes memories by user_id automatically)
#
#   2. Contradiction resolution test:
#      Turn 1: "I love Python"
#      Turn 5: "I switched to Rust, I don't use Python anymore"
#      Check mem0_list_all() — should have ONE updated memory, not two.
#
#   3. Mem0 Cloud vs self-hosted:
#      Set MEM0_API_KEY and re-run. Compare memory quality and latency.
#      Cloud uses OpenAI for extraction; self-hosted uses your local Ollama.
#
#   4. Add agent_id scoping:
#      Mem0 also supports agent_id for per-agent memory namespaces.
#      mem0_add(messages=..., user_id=..., agent_id="data-agent")
#      Try building a memory-aware multi-agent system (combine with L5).
#
#   5. Memory history / versioning:
#      m.history(memory_id=...) shows all updates to a memory over time.
#      Implement an audit endpoint that exposes memory change history.
# =============================================================
