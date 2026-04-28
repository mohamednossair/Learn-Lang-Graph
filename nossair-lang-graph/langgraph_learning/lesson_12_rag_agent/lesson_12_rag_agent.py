# =============================================================
# LESSON 12 — RAG Agent (Retrieval-Augmented Generation)
# =============================================================
#
# WHAT YOU WILL LEARN:
#   1. What RAG is and why it matters
#   2. Embed documents and store in a local vector store (Chroma)
#   3. Build a retrieval tool the agent calls automatically
#   4. Agentic RAG: agent decides WHEN to retrieve
#   5. Self-correcting RAG: check relevance, re-query if needed
#
# INSTALL:
#   pip install langchain-community chromadb
#
# MENTAL MODEL:
#   User question
#      ↓
#   [agent] → retrieve_docs(query) → [vector store search]
#      ↑             ↓
#      └── reads retrieved context ──┘
#      ↓
#   [grounded answer based on real documents]
# =============================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model

import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️  Run: pip install langchain-community chromadb")


# =============================================================
# STEP 1 — Build a Knowledge Base
# Replace these with your own documents in production.
# =============================================================

KNOWLEDGE_BASE = [
    # LangGraph concepts
    "LangGraph is a library for building stateful, multi-step AI agent workflows as directed graphs.",
    "A StateGraph defines the workflow. Nodes are Python functions. Edges define execution order.",
    "The state is a TypedDict shared between all nodes. Reducers control how fields are updated.",
    "add_messages is a reducer that appends messages to a list instead of replacing them.",
    "compile() validates the graph and returns a runnable. invoke() runs the graph synchronously.",
    "stream() runs the graph and yields state snapshots after each node — used for real-time UIs.",

    # Tools and agents
    "The @tool decorator creates tools. The docstring tells the LLM when and how to use the tool.",
    "bind_tools() attaches tool schemas to the LLM so it can choose to call them.",
    "ToolNode executes tool calls from the LLM and returns ToolMessage results.",
    "The ReAct pattern: LLM reasons → calls tool → reads result → reasons again → final answer.",
    "should_continue() routing function: if last message has tool_calls, go to tools; else END.",

    # Memory and persistence
    "MemorySaver stores checkpoints in RAM. State is lost on process restart.",
    "SqliteSaver stores checkpoints on disk. State survives process restarts.",
    "thread_id identifies a conversation session. Each user should have their own thread_id.",
    "get_state_history() returns all past snapshots — enables time-travel debugging.",

    # Human-in-the-loop
    "interrupt() pauses the graph and waits for human input. Requires a checkpointer.",
    "Command(resume=value) resumes a paused graph, passing the value to the interrupt() call.",
    "Pattern A: approve/reject before dangerous actions. Pattern B: collect missing info. Pattern C: review output.",

    # Best practices
    "Always set recursion_limit in config to prevent infinite loops.",
    "Use with_structured_output(PydanticModel) to force the LLM to return valid structured data.",
    "Retry with exponential backoff: wait 2^attempt seconds between retries.",
    "Use Send() to run multiple tasks in parallel. Results merged by reducers.",
    "Log at the start of every node: node name, thread_id, message count.",

    # Multi-agent
    "The supervisor pattern: one LLM routes tasks to specialist agents, each with focused tools.",
    "Specialists report back to the supervisor after completing work.",
    "Use FINISH as a route target to signal the supervisor the task is complete.",

    # Database agents
    "Always call list_tables() and describe_table() before writing SQL queries.",
    "Only allow SELECT queries — reject INSERT, UPDATE, DELETE with a safety guard.",
    "Parameterized queries prevent SQL injection: cursor.execute('SELECT * FROM t WHERE id=?', (x,))",
]


def build_vector_store():
    """Chunk documents, embed them, store in Chroma."""
    if not CHROMA_AVAILABLE:
        return None

    print("Building vector store...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = [Document(page_content=text, metadata={"source": f"doc_{i}"})
            for i, text in enumerate(KNOWLEDGE_BASE)]
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=get_ollama_model())
    db_path = os.path.join(os.path.dirname(__file__), "rag_chroma_db")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"✅ Vector store built: {len(chunks)} chunks from {len(docs)} documents")
    return vectorstore


# =============================================================
# STEP 2 — Retrieval Tool
# =============================================================

_vectorstore = None  # lazy-loaded

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vector_store()
    return _vectorstore


@tool
def retrieve_documents(query: str) -> str:
    """
    Search the LangGraph knowledge base for relevant information.
    Use this for any question about LangGraph concepts, patterns, tools, or best practices.
    Always call this BEFORE answering to ensure your response is accurate and grounded.
    Returns the top 3 most relevant document chunks.
    """
    vs = get_vectorstore()
    if vs is None:
        return "Vector store not available. Please install: pip install langchain-community chromadb"

    results = vs.similarity_search(query, k=3)
    if not results:
        return "No relevant documents found for this query."

    formatted = []
    for i, doc in enumerate(results, 1):
        formatted.append(f"[Source {i}]: {doc.page_content}")
    return "\n\n".join(formatted)


@tool
def retrieve_with_score(query: str) -> str:
    """
    Search the knowledge base and return results WITH relevance scores (0-1, higher = more relevant).
    Use this when you need to assess how confident the knowledge base is about a topic.
    Score > 0.7 = highly relevant. Score < 0.4 = likely not in knowledge base.
    """
    vs = get_vectorstore()
    if vs is None:
        return "Vector store not available."

    results = vs.similarity_search_with_score(query, k=3)
    if not results:
        return "No results found."

    formatted = []
    for i, (doc, score) in enumerate(results, 1):
        relevance = "HIGH" if score > 0.7 else ("MEDIUM" if score > 0.4 else "LOW")
        formatted.append(f"[Source {i} | Relevance: {relevance} ({score:.2f})]: {doc.page_content}")
    return "\n\n".join(formatted)


# =============================================================
# STEP 3 — Basic RAG Agent
# =============================================================

class RAGState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model=get_ollama_model(), temperature=0)
rag_tools = [retrieve_documents]
rag_llm = llm.bind_tools(rag_tools)

SYSTEM_PROMPT = """You are a LangGraph expert assistant with access to a knowledge base.

ALWAYS follow this process:
1. Call retrieve_documents() with the user's question
2. Read the retrieved documents carefully
3. Base your answer ONLY on the retrieved documents
4. If the documents don't contain the answer, say "I don't have information about that in my knowledge base."
5. Cite which source chunk your answer comes from

Never answer from general knowledge alone — always retrieve first."""


def rag_agent_node(state: RAGState) -> dict:
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [rag_llm.invoke(msgs)]}


def should_retrieve(state: RAGState) -> str:
    last = state["messages"][-1]
    return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "end"


rag_builder = StateGraph(RAGState)
rag_builder.add_node("agent", rag_agent_node)
rag_builder.add_node("tools", ToolNode(rag_tools))
rag_builder.add_edge(START, "agent")
rag_builder.add_conditional_edges("agent", should_retrieve, {"tools": "tools", "end": END})
rag_builder.add_edge("tools", "agent")
rag_graph = rag_builder.compile()


# =============================================================
# STEP 4 — Self-Correcting RAG Agent
# If retrieved docs are not relevant enough, re-query.
# =============================================================

class SelfCorrectingRAGState(TypedDict):
    messages:         Annotated[list, add_messages]
    query:            str
    retrieved_docs:   str
    relevance_score:  float
    retry_count:      int
    final_answer:     str


def retrieve_node(state: SelfCorrectingRAGState) -> dict:
    results = retrieve_with_score.invoke({"query": state["query"]})
    return {"retrieved_docs": results}


def check_relevance(state: SelfCorrectingRAGState) -> dict:
    """Use LLM to score relevance of retrieved docs to query."""
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


def route_relevance(state: SelfCorrectingRAGState) -> str:
    if state["relevance_score"] >= 0.5 or state["retry_count"] >= 2:
        return "generate"
    return "refine_query"


def refine_query_node(state: SelfCorrectingRAGState) -> dict:
    """Ask LLM to rephrase the query to get better results."""
    refine_prompt = f"""The query '{state['query']}' did not retrieve relevant documents.
Rephrase it to use different keywords that might match better.
Reply with ONLY the new query."""
    resp = llm.invoke([HumanMessage(content=refine_prompt)])
    return {"query": resp.content.strip(), "retry_count": state["retry_count"] + 1}


def generate_answer(state: SelfCorrectingRAGState) -> dict:
    msgs = [
        SystemMessage(content="Answer based on the retrieved documents. Cite sources."),
        HumanMessage(content=f"Question: {state['query']}\n\nSources:\n{state['retrieved_docs']}")
    ]
    resp = llm.invoke(msgs)
    return {"final_answer": resp.content, "messages": [resp]}


sc_builder = StateGraph(SelfCorrectingRAGState)
sc_builder.add_node("retrieve",      retrieve_node)
sc_builder.add_node("check_rel",     check_relevance)
sc_builder.add_node("refine_query",  refine_query_node)
sc_builder.add_node("generate",      generate_answer)
sc_builder.add_edge(START,           "retrieve")
sc_builder.add_edge("retrieve",      "check_rel")
sc_builder.add_conditional_edges("check_rel", route_relevance, {"generate": "generate", "refine_query": "refine_query"})
sc_builder.add_edge("refine_query",  "retrieve")
sc_builder.add_edge("generate",      END)
self_correcting_rag = sc_builder.compile()


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    if not CHROMA_AVAILABLE:
        print("Install dependencies: pip install langchain-community chromadb")
        exit(1)

    print("=" * 60)
    print("BASIC RAG AGENT")
    print("=" * 60)

    questions = [
        "What is a StateGraph in LangGraph?",
        "How does interrupt() work and what does it require?",
        "What is the difference between MemorySaver and SqliteSaver?",
        "How do you prevent SQL injection in a database agent?",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        result = rag_graph.invoke({"messages": [HumanMessage(content=q)]})
        print(f"💬 {result['messages'][-1].content[:300]}")

    print("\n" + "=" * 60)
    print("SELF-CORRECTING RAG AGENT")
    print("=" * 60)
    result = self_correcting_rag.invoke({
        "messages": [],
        "query": "How do I handle parallel execution?",
        "retrieved_docs": "",
        "relevance_score": 0.0,
        "retry_count": 0,
        "final_answer": ""
    })
    print(f"Final answer:\n{result['final_answer'][:400]}")
    print(f"Retries used: {result['retry_count']}")


# =============================================================
# EXERCISES:
#   1. Add your own PDF/text file to the knowledge base
#      (use langchain_community.document_loaders.TextLoader)
#   2. Increase top-k from 3 to 5 — does answer quality improve?
#   3. Build a "multi-source" RAG with 2 separate Chroma collections
#   4. Track which source chunks were used most often (hit counter)
# =============================================================
