# =============================================================
# LESSON 13 — Long-Term Memory with Vector Stores
# =============================================================
#
# WHAT YOU WILL LEARN:
#   1. The 3 memory types in AI agents
#   2. How to extract facts from conversations
#   3. How to store facts in a vector store per user
#   4. How to retrieve relevant memories at start of each turn
#   5. Cross-thread memory (global user profile)
#
# MEMORY HIERARCHY:
#   In-context:    Full message history in the LLM prompt  (limited, temporary)
#   Structured DB: Exact key/value facts in a database    (precise lookup)
#   Vector store:  Semantic memories by meaning           (what we build here)
#
# PATTERN (per conversation turn):
#   1. Load memories: search vector store for relevant past facts
#   2. Chat: include loaded memories as context
#   3. Save memories: extract new facts → embed → store
# =============================================================

import json
import os
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️  Run: pip install langchain-community chromadb")

llm = ChatOllama(model="llama3.2", temperature=0)


# =============================================================
# SETUP: Vector store for long-term memory
# =============================================================

def get_memory_store():
    if not CHROMA_AVAILABLE:
        return None
    embeddings = OllamaEmbeddings(model="llama3.2")
    db_path = os.path.join(os.path.dirname(__file__), "user_memory_db")
    return Chroma(persist_directory=db_path, embedding_function=embeddings)


# =============================================================
# STATE
# =============================================================

class MemoryState(TypedDict):
    messages:        Annotated[list, add_messages]
    user_id:         str
    loaded_memories: list     # memories retrieved this turn
    memory_context:  str      # formatted for the system prompt


# =============================================================
# NODE 1 — Load Memories
# Retrieve relevant past facts BEFORE generating response
# =============================================================

def load_memories_node(state: MemoryState) -> dict:
    """Search vector store for memories relevant to the current question."""
    memory_store = get_memory_store()
    if memory_store is None or not state["messages"]:
        return {"loaded_memories": [], "memory_context": ""}

    last_msg = state["messages"][-1].content
    user_id  = state["user_id"]

    try:
        # Filter by user_id so users never see each other's memories
        results = memory_store.similarity_search(
            last_msg, k=5,
            filter={"user_id": user_id}
        )
        memories = [doc.page_content for doc in results]
    except Exception:
        memories = []

    if memories:
        context = "What I remember about you from past conversations:\n" + "\n".join(f"• {m}" for m in memories)
    else:
        context = ""

    return {"loaded_memories": memories, "memory_context": context}


# =============================================================
# NODE 2 — Chat with Memory Context
# =============================================================

def chat_node(state: MemoryState) -> dict:
    """Generate response using LLM, injecting memory as context."""
    context = state.get("memory_context", "")
    system_content = "You are a helpful, personalized assistant."
    if context:
        system_content += f"\n\nContext from memory:\n{context}"
        system_content += "\n\nUse this context to personalize your response naturally."

    msgs = [SystemMessage(content=system_content)] + state["messages"]
    resp = llm.invoke(msgs)
    return {"messages": [resp]}


# =============================================================
# NODE 3 — Save Memories
# Extract key facts from this turn and store in vector store
# =============================================================

def save_memories_node(state: MemoryState) -> dict:
    """Extract important facts from this conversation turn and store them."""
    memory_store = get_memory_store()
    if memory_store is None or len(state["messages"]) < 2:
        return {}

    # Get the last human and AI message
    last_human = ""
    last_ai    = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not last_ai:
            last_ai = msg.content
        elif isinstance(msg, HumanMessage) and not last_human:
            last_human = msg.content
        if last_human and last_ai:
            break

    if not last_human:
        return {}

    extract_prompt = f"""Extract 1-3 concise facts worth remembering about the user from this conversation.
Focus on: name, job, preferences, goals, skills, dislikes, location, projects.
Ignore: questions, requests, general chat.

User said: {last_human}

Return ONLY a JSON list of strings, or [] if nothing worth remembering.
Example: ["User's name is Ahmed", "User works as a data engineer", "User prefers Python"]"""

    try:
        raw = llm.invoke([HumanMessage(content=extract_prompt)]).content
        start = raw.find("[")
        end   = raw.rfind("]")
        if start == -1 or end == -1:
            return {}
        facts = json.loads(raw[start:end+1])

        user_id = state["user_id"]
        for fact in facts:
            if isinstance(fact, str) and len(fact.strip()) > 5:
                memory_store.add_documents([Document(
                    page_content=fact,
                    metadata={"user_id": user_id}
                )])
    except Exception as e:
        pass  # Don't fail the graph if memory saving fails

    return {}


# =============================================================
# BUILD GRAPH
# =============================================================

def build_memory_graph(checkpointer=None):
    builder = StateGraph(MemoryState)
    builder.add_node("load_memories", load_memories_node)
    builder.add_node("chat",          chat_node)
    builder.add_node("save_memories", save_memories_node)
    builder.add_edge(START,           "load_memories")
    builder.add_edge("load_memories", "chat")
    builder.add_edge("chat",          "save_memories")
    builder.add_edge("save_memories", END)
    return builder.compile(checkpointer=checkpointer)


# =============================================================
# MEMORY UTILITIES
# =============================================================

def list_user_memories(user_id: str, max_memories: int = 20) -> list:
    """List all stored memories for a user."""
    memory_store = get_memory_store()
    if memory_store is None:
        return []
    results = memory_store.get(where={"user_id": user_id})
    documents = results.get("documents", [])
    return documents[:max_memories]


def delete_user_memories(user_id: str):
    """Delete all memories for a user (GDPR compliance)."""
    memory_store = get_memory_store()
    if memory_store is None:
        return 0
    try:
        memory_store.delete(where={"user_id": user_id})
        print(f"✅ Deleted all memories for user: {user_id}")
    except Exception as e:
        print(f"Error deleting memories: {e}")


# =============================================================
# MAIN — Demo multi-turn conversation with memory
# =============================================================

if __name__ == "__main__":
    if not CHROMA_AVAILABLE:
        print("Install: pip install langchain-community chromadb")
        exit(1)

    graph = build_memory_graph(checkpointer=MemorySaver())

    user_id = "user-demo-001"
    config  = {"configurable": {"thread_id": f"mem-{user_id}"}}

    print("=" * 60)
    print("LONG-TERM MEMORY AGENT — Multi-turn demo")
    print("=" * 60)

    conversation = [
        # Turn 1: Tell the agent about yourself
        "Hi! My name is Ahmed and I'm a data engineer with 5 years of experience.",
        # Turn 2: Add more facts
        "I love Python and LangGraph. I'm currently building an AI agent for my company.",
        # Turn 3: Add preferences
        "I prefer using local models to avoid API costs. I also like clean code and good documentation.",
        # Turn 4: Test recall ACROSS a simulated new session
        "What do you know about me? Give me a full summary.",
    ]

    for i, turn in enumerate(conversation, 1):
        print(f"\n[Turn {i}]")
        print(f"👤 You: {turn}")

        initial = {
            "messages": [HumanMessage(content=turn)],
            "user_id":  user_id,
            "loaded_memories": [],
            "memory_context":  ""
        }

        result = graph.invoke(initial, config=config)

        # Show response
        last_ai = next((m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)), "")
        print(f"🤖 AI: {last_ai[:200]}")

        # Show what was retrieved from memory (if any)
        if result.get("loaded_memories"):
            print(f"   📚 Retrieved memories: {result['loaded_memories'][:2]}")

    # Show all stored memories
    print("\n" + "=" * 60)
    print("ALL STORED MEMORIES FOR THIS USER:")
    print("=" * 60)
    memories = list_user_memories(user_id)
    if memories:
        for i, m in enumerate(memories, 1):
            print(f"  {i}. {m}")
    else:
        print("  No memories stored yet.")


# =============================================================
# EXERCISES:
#   1. Add memory categories (work/personal/preferences)
#      Store as metadata: {"user_id": "...", "category": "work"}
#      Retrieve by category based on question type
#
#   2. Add memory scoring — track how often each memory is retrieved
#      Increment a "hits" counter in metadata each time
#      After 100 memories, prune lowest-hit ones
#
#   3. Test cross-session memory:
#      - Start conversation, add facts
#      - Change thread_id (new session)
#      - Ask "what do you know about me?" — memories should carry over
#      (memories are tied to user_id not thread_id)
#
#   4. Implement memory contradiction detection:
#      Before saving "User dislikes Python", check if "User loves Python"
#      already exists → update/replace instead of adding duplicate
# =============================================================
