"""Task 12.3 — Multi-Collection RAG."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
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

# ============== TECHNICAL DOCS COLLECTION ==============
TECH_DOCS = [
    Document(page_content="Authentication uses JWT tokens stored in HTTP-only cookies.", 
             metadata={"source": "auth_system.txt", "type": "technical"}),
    Document(page_content="Database connection pooling prevents resource exhaustion.", 
             metadata={"source": "db_optimization.txt", "type": "technical"}),
    Document(page_content="API rate limiting prevents abuse and ensures fair usage.", 
             metadata={"source": "api_security.txt", "type": "technical"}),
    Document(page_content="OAuth 2.0 is the industry standard for authorization.", 
             metadata={"source": "oauth_guide.txt", "type": "technical"}),
]

# ============== HR POLICIES COLLECTION ==============
HR_DOCS = [
    Document(page_content="Employees receive 20 vacation days per year plus 10 sick days.", 
             metadata={"source": "vacation_policy.txt", "type": "hr"}),
    Document(page_content="Remote work is allowed up to 3 days per week with manager approval.", 
             metadata={"source": "remote_work.txt", "type": "hr"}),
    Document(page_content="Health insurance covers 90% of premiums for employees and dependents.", 
             metadata={"source": "benefits.txt", "type": "hr"}),
    Document(page_content="Performance reviews occur twice yearly in June and December.", 
             metadata={"source": "performance.txt", "type": "hr"}),
]

def build_collections():
    if not CHROMA_AVAILABLE:
        return None, None
    
    embeddings = OllamaEmbeddings(model=get_ollama_model())
    
    tech_path = os.path.join(os.path.dirname(__file__), "task_12_tech_docs")
    tech_vs = Chroma.from_documents(
        documents=TECH_DOCS, embedding=embeddings, 
        collection_name="technical_docs", persist_directory=tech_path
    )
    
    hr_path = os.path.join(os.path.dirname(__file__), "task_12_hr_policies")
    hr_vs = Chroma.from_documents(
        documents=HR_DOCS, embedding=embeddings,
        collection_name="hr_policies", persist_directory=hr_path
    )
    
    return tech_vs, hr_vs

_tech_vs = None
_hr_vs = None

def get_collections():
    global _tech_vs, _hr_vs
    if _tech_vs is None:
        _tech_vs, _hr_vs = build_collections()
    return _tech_vs, _hr_vs

# ============== CLASSIFICATION & RETRIEVAL TOOLS ==============
@tool
def classify_question(query: str) -> str:
    """Classify question as 'technical' or 'hr'. Reply with just the category."""
    llm = ChatOllama(model=get_ollama_model(), temperature=0)
    prompt = f"""Classify this question as either 'technical' or 'hr'. 
Technical questions are about code, APIs, databases, authentication, etc.
HR questions are about vacation, benefits, policies, performance reviews, etc.
Reply with ONLY 'technical' or 'hr'.

Question: {query}"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip().lower()

@tool
def search_appropriate_collection(query: str) -> str:
    """Search the appropriate collection based on question classification."""
    tech_vs, hr_vs = get_collections()
    if not tech_vs or not hr_vs:
        return "Vector stores not available"
    
    # Classify first
    llm_local = ChatOllama(model=get_ollama_model(), temperature=0)
    classify_prompt = f"""Classify as 'technical' or 'hr'. 
Technical: code, APIs, databases, authentication
HR: vacation, benefits, policies, remote work
Reply ONLY with category word.

Question: {query}"""
    category = llm_local.invoke([HumanMessage(content=classify_prompt)]).content.strip().lower()
    
    # Search appropriate collection
    if "hr" in category:
        results = hr_vs.similarity_search(query, k=2)
        prefix = "[HR Policies]"
    else:
        results = tech_vs.similarity_search(query, k=2)
        prefix = "[Technical Docs]"
    
    formatted = [prefix]
    for doc in results:
        src = doc.metadata.get("source", "unknown")
        formatted.append(f"[{src}]: {doc.page_content}")
    return "\n".join(formatted)

# ============== MULTI-COLLECTION RAG AGENT ==============
class MultiRAGState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model=get_ollama_model(), temperature=0)
tools = [classify_question, search_appropriate_collection]
llm_with_tools = llm.bind_tools(tools)

SYSTEM = """You are a company assistant with access to two knowledge bases:
- Technical documentation (APIs, databases, authentication)
- HR policies (vacation, benefits, remote work)

The system will automatically classify questions and search the right collection."""

def agent(state: MultiRAGState) -> dict:
    msgs = [SystemMessage(content=SYSTEM)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(msgs)]}

def should_continue(state: MultiRAGState) -> str:
    last = state["messages"][-1]
    return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "end"

builder = StateGraph(MultiRAGState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")
multi_rag = builder.compile()

if __name__ == "__main__":
    if not CHROMA_AVAILABLE:
        print("pip install langchain-community chromadb")
        exit(1)
    
    print("=" * 60)
    print("TASK 12.3 — MULTI-COLLECTION RAG")
    print("=" * 60)
    
    test_cases = [
        ("How does authentication work?", "technical"),
        ("What is the vacation policy?", "hr"),
        ("Tell me about database connection pooling", "technical"),
        ("How many sick days do I get?", "hr"),
    ]
    
    for question, expected in test_cases:
        print(f"\nQ: {question}")
        print(f"Expected category: {expected}")
        
        # Test classification directly
        cat_result = classify_question.invoke({"query": question})
        print(f"Classified as: {cat_result}")
        
        # Full agent response
        result = multi_rag.invoke({"messages": [HumanMessage(content=question)]})
        answer = result["messages"][-1].content
        print(f"Answer: {answer[:200]}...")
        
        if expected in cat_result.lower():
            print("✅ Correct classification")
        else:
            print("❌ Misclassification")
