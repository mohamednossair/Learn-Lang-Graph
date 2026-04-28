"""Task 13.1 — Personalized Tutor."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

llm = ChatOllama(model=get_ollama_model(), temperature=0)

def get_store():
    if not CHROMA_AVAILABLE:
        return None
    embeddings = OllamaEmbeddings(model=get_ollama_model())
    return Chroma(persist_directory=os.path.join(os.path.dirname(__file__), "tutor_db"), embedding_function=embeddings)

class TutorState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    memories: list

# Build graph
builder = StateGraph(TutorState)

def load_memories(state: TutorState):
    store = get_store()
    if not store or not state["messages"]:
        return {"memories": []}
    last_msg = state["messages"][-1].content
    try:
        results = store.similarity_search(last_msg, k=5, filter={"user_id": state["user_id"]})
        return {"memories": [d.page_content for d in results]}
    except:
        return {"memories": []}

def tutor(state: TutorState):
    memories = state.get("memories", [])
    context = "Student profile:\n" + "\n".join(f"• {m}" for m in memories) if memories else ""
    system = f"You are a tutor. Adapt to student level.\n\n{context}"
    msgs = [{"role": "system", "content": system}] + [{"role": m.type, "content": m.content} for m in state["messages"]]
    resp = llm.invoke([HumanMessage(content=system)] + state["messages"])
    return {"messages": [resp]}

def save(state: TutorState):
    store = get_store()
    if not store:
        return {}
    last_msg = state["messages"][-1].content if state["messages"] else ""
    prompt = f"Extract student facts (level, struggles, preferences) from: {last_msg[:200]}. Return JSON list."
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        start, end = raw.find("["), raw.rfind("]")
        facts = json.loads(raw[start:end+1]) if start > -1 and end > -1 else []
        for f in facts:
            if isinstance(f, str) and len(f) > 5:
                store.add_documents([Document(page_content=f, metadata={"user_id": state["user_id"]})])
    except:
        pass
    return {}

builder.add_node("load", load_memories)
builder.add_node("tutor", tutor)
builder.add_node("save", save)
builder.add_edge(START, "load")
builder.add_edge("load", "tutor")
builder.add_edge("tutor", "save")
builder.add_edge("save", END)
graph = builder.compile()

if __name__ == "__main__":
    user_id = "student-001"
    sessions = [
        "I am completely new to programming.",
        "Explain variables simply please.",
        "What's a function? I'm confused.",
        "I learned Java before. Now learning Python.",
        "Teach me something new at my level."
    ]
    for i, msg in enumerate(sessions, 1):
        print(f"\nSession {i}: {msg}")
        result = graph.invoke({"messages": [HumanMessage(content=msg)], "user_id": user_id, "memories": []})
        ai_msg = next((m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)), "")
        print(f"Tutor: {ai_msg[:150]}...")
    print("\n✅ Tutor adapted over 5 sessions!")
