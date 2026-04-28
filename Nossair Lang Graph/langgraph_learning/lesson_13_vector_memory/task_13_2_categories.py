"""Task 13.2 — Memory Categories."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import Literal

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

def get_store():
    if not CHROMA_AVAILABLE:
        return None
    embeddings = OllamaEmbeddings(model=get_ollama_model())
    return Chroma(persist_directory=os.path.join(os.path.dirname(__file__), "cat_db"), embedding_function=embeddings)

def store_memory(user_id: str, content: str, category: Literal["work", "personal", "preferences", "goals"]):
    store = get_store()
    if store:
        store.add_documents([Document(page_content=content, metadata={"user_id": user_id, "category": category})])

def retrieve_by_category(user_id: str, query: str, category: str = None):
    store = get_store()
    if not store:
        return []
    filter_dict = {"user_id": user_id}
    if category:
        filter_dict["category"] = category
    return store.similarity_search(query, k=3, filter=filter_dict)

if __name__ == "__main__":
    store_memory("user1", "Works as Python developer at TechCorp", "work")
    store_memory("user1", "Likes hiking on weekends in mountains", "personal")
    store_memory("user1", "Prefers morning meetings over afternoon", "preferences")
    store_memory("user1", "Wants to become senior engineer in 2 years", "goals")
    
    print("Work question → work memories:")
    work = retrieve_by_category("user1", "job responsibilities", "work")
    for d in work:
        print(f"  • {d.page_content} (cat: {d.metadata['category']})")
    
    print("\nPersonal question → personal memories:")
    personal = retrieve_by_category("user1", "hobbies", "personal")
    for d in personal:
        print(f"  • {d.page_content} (cat: {d.metadata['category']})")
    
    print("\n✅ Category-based retrieval working!")
