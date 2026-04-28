"""Task 12.2 — Source Attribution in RAG."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated
from typing_extensions import TypedDict

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# ============== KNOWLEDGE BASE ==============
KNOWLEDGE_BASE = [
    Document(page_content="Python was created by Guido van Rossum in 1991.", metadata={"source": "python_history.txt", "paragraph": 1}),
    Document(page_content="Python emphasizes code readability and simplicity.", metadata={"source": "python_history.txt", "paragraph": 2}),
    Document(page_content="Django is a Python web framework for rapid development.", metadata={"source": "django_guide.txt", "paragraph": 1}),
    Document(page_content="Django includes an ORM, admin interface, and templating engine.", metadata={"source": "django_guide.txt", "paragraph": 2}),
]

def build_vector_store():
    if not CHROMA_AVAILABLE:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(KNOWLEDGE_BASE)
    embeddings = OllamaEmbeddings(model=get_ollama_model())
    return Chroma.from_documents(documents=chunks, embedding=embeddings, 
                                  persist_directory=os.path.join(os.path.dirname(__file__), "task_12_attribution"))

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vector_store()
    return _vectorstore

@tool
def retrieve_with_attribution(query: str) -> str:
    """Search knowledge base and return results with source attribution."""
    vs = get_vectorstore()
    if not vs:
        return "Vector store not available"
    
    results = vs.similarity_search(query, k=3)
    if not results:
        return "No relevant documents found."
    
    formatted = []
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        para = doc.metadata.get("paragraph", "?")
        formatted.append(f"[Doc {i} from {src}, paragraph {para}]:\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# ============== AGENT WITH SOURCE ATTRIBUTION ==============
class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    answer_with_sources: str

llm = ChatOllama(model=get_ollama_model(), temperature=0)
rag_tools = [retrieve_with_attribution]
rag_llm = llm.bind_tools(rag_tools)

SYSTEM_PROMPT = """You are a helpful assistant with access to a knowledge base.

Instructions:
1. Always call retrieve_with_attribution() before answering
2. Base your answer ONLY on the retrieved documents
3. End EVERY answer with a Sources section like:

Sources: doc1.txt (paragraph 1), doc2.txt (paragraph 2)

Never answer from general knowledge alone."""

def agent_node(state: RAGState) -> dict:
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [rag_llm.invoke(msgs)]}

def should_retrieve(state: RAGState) -> str:
    last = state["messages"][-1]
    return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "end"

def format_answer_with_sources(state: RAGState) -> dict:
    """Ensure answer has proper source attribution."""
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return {"answer_with_sources": content}

builder = StateGraph(RAGState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(rag_tools))
builder.add_node("format", format_answer_with_sources)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_retrieve, {"tools": "tools", "end": "format"})
builder.add_edge("tools", "agent")
builder.add_edge("format", END)
graph = builder.compile()

if __name__ == "__main__":
    if not CHROMA_AVAILABLE:
        print("pip install langchain-community chromadb")
        exit(1)
    
    print("=" * 60)
    print("TASK 12.2 — SOURCE ATTRIBUTION")
    print("=" * 60)
    
    questions = [
        "Who created Python?",
        "What is Django?",
        "Tell me about Python and Django together",
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        result = graph.invoke({"messages": [HumanMessage(content=q)], "answer_with_sources": ""})
        answer = result.get("answer_with_sources", result["messages"][-1].content)
        print(f"A: {answer}\n")
        
        # Verify sources are present
        if "sources:" in answer.lower():
            print("✅ Source attribution present")
        else:
            print("❌ Missing source attribution")
