"""Task 11.2 — Document Processing with Parallel Subgraphs."""
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

llm = ChatOllama(model=get_ollama_model(), temperature=0)

# ============== ANALYSIS SUBGRAPH ==============
class DocAnalysisState(TypedDict):
    document: str
    word_count: int
    language: str
    keywords: list

def count_words(state: DocAnalysisState) -> dict:
    words = state["document"].split()
    return {"word_count": len(words)}

def detect_language(state: DocAnalysisState) -> dict:
    prompt = f"What language is this? Reply with just the language name: {state['document'][:100]}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"language": resp.content.strip()}

def extract_keywords(state: DocAnalysisState) -> dict:
    prompt = f"Extract top 3 keywords from this text. Reply as comma-separated list: {state['document'][:300]}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    keywords = [k.strip() for k in resp.content.split(",")][:3]
    return {"keywords": keywords}

analysis_builder = StateGraph(DocAnalysisState)
analysis_builder.add_node("count_words", count_words)
analysis_builder.add_node("detect_language", detect_language)
analysis_builder.add_node("extract_keywords", extract_keywords)
analysis_builder.add_edge(START, "count_words")
analysis_builder.add_edge("count_words", "detect_language")
analysis_builder.add_edge("detect_language", "extract_keywords")
analysis_builder.add_edge("extract_keywords", END)
analyze_subgraph = analysis_builder.compile()

# ============== PARALLEL PARENT GRAPH ==============
class ParallelDocsState(TypedDict):
    documents: list
    analyses: Annotated[list, lambda x, y: x + y]
    comparison_table: str

def fan_out_to_subgraphs(state: ParallelDocsState):
    return [Send("analyze_doc", {"document": doc}) for doc in state["documents"]]

def analyze_doc_node(state: DocAnalysisState) -> dict:
    result = analyze_subgraph.invoke(state)
    return {"analyses": [result]}

def combine_results(state: ParallelDocsState) -> dict:
    table = "| Doc # | Words | Language | Keywords |\n|-------|-------|----------|----------|\n"
    for i, analysis in enumerate(state["analyses"], 1):
        kw = ", ".join(analysis.get("keywords", []))
        table += f"| {i} | {analysis.get('word_count', 0)} | {analysis.get('language', 'unknown')} | {kw} |\n"
    return {"comparison_table": table}

parallel_builder = StateGraph(ParallelDocsState)
parallel_builder.add_node("analyze_doc", analyze_doc_node)
parallel_builder.add_node("combine", combine_results)
parallel_builder.add_conditional_edges(START, fan_out_to_subgraphs, ["analyze_doc"])
parallel_builder.add_edge("analyze_doc", "combine")
parallel_builder.add_edge("combine", END)
parallel_graph = parallel_builder.compile()

# ============== SEQUENTIAL VERSION FOR BENCHMARK ==============
def process_sequentially(documents):
    results = []
    for doc in documents:
        result = analyze_subgraph.invoke({"document": doc, "word_count": 0, "language": "", "keywords": []})
        results.append(result)
    return results

# ============== MAIN ==============
if __name__ == "__main__":
    test_docs = [
        "LangGraph is a library for building stateful AI agent workflows. It uses graphs to define execution.",
        "Python is a versatile programming language used for web development, data science, and automation.",
        "Machine learning enables computers to learn patterns from data without explicit programming.",
        "Docker containers package applications with their dependencies for consistent deployment.",
        "FastAPI is a modern web framework for building APIs with Python based on standard type hints.",
    ]
    
    print("=" * 60)
    print("PARALLEL PROCESSING")
    print("=" * 60)
    start = time.time()
    result = parallel_graph.invoke({"documents": test_docs, "analyses": [], "comparison_table": ""})
    parallel_time = time.time() - start
    print(f"Time: {parallel_time:.2f}s")
    print(result["comparison_table"])
    
    print("=" * 60)
    print("SEQUENTIAL PROCESSING")
    print("=" * 60)
    start = time.time()
    seq_results = process_sequentially(test_docs)
    seq_time = time.time() - start
    print(f"Time: {seq_time:.2f}s")
    
    print("=" * 60)
    print("BENCHMARK RESULT")
    print("=" * 60)
    speedup = seq_time / parallel_time if parallel_time > 0 else 1
    print(f"Parallel: {parallel_time:.2f}s")
    print(f"Sequential: {seq_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
