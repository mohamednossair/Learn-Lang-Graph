"""Task 12.4 — Evaluate RAG Quality."""
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

# ============== EVALUATION DATASET ==============
EVAL_DATASET = [
    {
        "id": "Q01",
        "question": "Who created Python?",
        "expected_keywords": ["guido", "van rossum", "1991"],
        "description": "Python creator"
    },
    {
        "id": "Q02",
        "question": "What is Django?",
        "expected_keywords": ["python", "web", "framework"],
        "description": "Django basics"
    },
    {
        "id": "Q03",
        "question": "What does Docker do?",
        "expected_keywords": ["container", "application"],
        "description": "Docker purpose"
    },
    {
        "id": "Q04",
        "question": "What is PostgreSQL?",
        "expected_keywords": ["database", "relational", "open source"],
        "description": "PostgreSQL identity"
    },
    {
        "id": "Q05",
        "question": "What are SQL commands?",
        "expected_keywords": ["select", "insert", "delete"],
        "description": "SQL basics"
    },
    {
        "id": "Q06",
        "question": "What is REST?",
        "expected_keywords": ["api", "http", "architectural"],
        "description": "REST API"
    },
    {
        "id": "Q07",
        "question": "What is Python's standard library known for?",
        "expected_keywords": ["batteries", "included"],
        "description": "Python standard library"
    },
    {
        "id": "Q08",
        "question": "What does ACID stand for in databases?",
        "expected_keywords": ["atomicity", "consistency", "isolation", "durability"],
        "description": "ACID properties"
    },
    {
        "id": "Q09",
        "question": "Who invented Java?",  # Not in knowledge base
        "expected_keywords": ["don't have", "information", "not"],
        "description": "Out of scope question"
    },
    {
        "id": "Q10",
        "question": "What is Docker Compose used for?",
        "expected_keywords": ["multi-container", "services", "yaml"],
        "description": "Docker Compose"
    },
]

# ============== KNOWLEDGE BASE ==============
DOCS = [
    Document(page_content="Python was created by Guido van Rossum in 1991. It has a batteries included philosophy.", 
             metadata={"source": "python.txt"}),
    Document(page_content="Django is a Python web framework for rapid development with ORM and admin.", 
             metadata={"source": "django.txt"}),
    Document(page_content="Docker packages applications in containers for consistent deployment.", 
             metadata={"source": "docker.txt"}),
    Document(page_content="Docker Compose manages multi-container applications with YAML.", 
             metadata={"source": "docker.txt"}),
    Document(page_content="PostgreSQL is an open source object-relational database with ACID compliance.", 
             metadata={"source": "postgres.txt"}),
    Document(page_content="SQL commands include SELECT, INSERT, UPDATE, DELETE for data operations.", 
             metadata={"source": "sql.txt"}),
    Document(page_content="REST is an architectural style using HTTP for APIs, often with JSON.", 
             metadata={"source": "rest.txt"}),
]

def build_vectorstore():
    if not CHROMA_AVAILABLE:
        return None
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(DOCS)
    embeddings = OllamaEmbeddings(model=get_ollama_model())
    return Chroma.from_documents(documents=chunks, embedding=embeddings,
                                  persist_directory=os.path.join(os.path.dirname(__file__), "task_12_eval"))

_vs = None
def get_vs():
    global _vs
    if _vs is None:
        _vs = build_vectorstore()
    return _vs

@tool
def retrieve(query: str) -> str:
    """Search knowledge base."""
    vs = get_vs()
    if not vs:
        return "Vector store unavailable"
    results = vs.similarity_search(query, k=3)
    return "\n\n".join([f"[{r.metadata.get('source','?')}]: {r.page_content}" for r in results])

# ============== RAG AGENT ==============
class RAGState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model=get_ollama_model(), temperature=0)
tools = [retrieve]
llm_tools = llm.bind_tools(tools)

def agent(state: RAGState) -> dict:
    sys_msg = SystemMessage(content="Answer from retrieved documents only.")
    return {"messages": [llm_tools.invoke([sys_msg] + state["messages"])]}

def should_continue(state: RAGState) -> str:
    last = state["messages"][-1]
    return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "end"

builder = StateGraph(RAGState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")
graph = builder.compile()

# ============== EVALUATION ==============
def evaluate_answer(question, expected_keywords):
    """Score answer based on keyword presence."""
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content.lower()
    
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer)
    score = matched / len(expected_keywords)
    
    return {
        "answer": answer[:200],
        "matched": matched,
        "total": len(expected_keywords),
        "score": score,
        "passed": score >= 0.5
    }

def run_evaluation():
    results = {"passed": 0, "failed": 0, "details": []}
    
    for case in EVAL_DATASET:
        print(f"\n[{case['id']}] {case['description']}")
        print(f"Q: {case['question']}")
        
        eval_result = evaluate_answer(case["question"], case["expected_keywords"])
        
        status = "✅ PASS" if eval_result["passed"] else "❌ FAIL"
        print(f"{status} - Matched {eval_result['matched']}/{eval_result['total']} keywords")
        print(f"Answer: {eval_result['answer'][:100]}...")
        
        if eval_result["passed"]:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append({
                "id": case["id"],
                "question": case["question"],
                "expected": case["expected_keywords"],
                "got": eval_result["answer"]
            })
    
    total = len(EVAL_DATASET)
    accuracy = results["passed"] / total * 100
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Passed: {results['passed']}/{total}")
    print(f"Failed: {results['failed']}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print("🎯 Excellent RAG performance!")
    elif accuracy >= 60:
        print("⚠️ Acceptable but could improve")
    else:
        print("❌ RAG needs optimization - check chunk size/retrieval")
    
    return results

if __name__ == "__main__":
    if not CHROMA_AVAILABLE:
        print("pip install langchain-community chromadb")
        exit(1)
    
    print("=" * 60)
    print("TASK 12.4 — EVALUATE RAG QUALITY")
    print("=" * 60)
    
    run_evaluation()
