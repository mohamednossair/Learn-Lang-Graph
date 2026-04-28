"""Task 13.4 — Memory Audit Tool."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from datetime import datetime

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

def get_store():
    if not CHROMA_AVAILABLE:
        return None
    embeddings = OllamaEmbeddings(model=get_ollama_model())
    return Chroma(persist_directory=os.path.join(os.path.dirname(__file__), "audit_db"), embedding_function=embeddings)

def audit_user_memories(user_id: str):
    store = get_store()
    if not store:
        return None
    
    # Get all memories for user
    results = store.get(where={"user_id": user_id})
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    
    audit = {
        "total_memories": len(docs),
        "categories": {},
        "duplicates": [],
        "outdated": [],
        "by_length": {"short": 0, "medium": 0, "long": 0}
    }
    
    seen = set()
    for doc, meta in zip(docs, metas):
        # Count by category
        cat = meta.get("category", "uncategorized")
        audit["categories"][cat] = audit["categories"].get(cat, 0) + 1
        
        # Check duplicates (simple exact match)
        if doc in seen:
            audit["duplicates"].append(doc[:50])
        seen.add(doc)
        
        # Length analysis
        length = len(doc)
        if length < 30:
            audit["by_length"]["short"] += 1
        elif length < 100:
            audit["by_length"]["medium"] += 1
        else:
            audit["by_length"]["long"] += 1
    
    return audit

def delete_duplicates(user_id: str):
    store = get_store()
    if not store:
        return 0
    
    results = store.get(where={"user_id": user_id})
    docs = results.get("documents", [])
    ids = results.get("ids", [])
    
    seen = {}
    to_delete = []
    for doc_id, doc in zip(ids, docs):
        if doc in seen:
            to_delete.append(doc_id)
        else:
            seen[doc] = doc_id
    
    if to_delete:
        store.delete(ids=to_delete)
    return len(to_delete)

if __name__ == "__main__":
    store = get_store()
    user_id = "user-audit-test"
    
    # Add test data with duplicates
    test_memories = [
        ("User works at TechCorp", {"user_id": user_id, "category": "work"}),
        ("User works at TechCorp", {"user_id": user_id, "category": "work"}),  # Duplicate
        ("User likes Python", {"user_id": user_id, "category": "preferences"}),
        ("User enjoys hiking", {"user_id": user_id, "category": "personal"}),
    ]
    
    for content, meta in test_memories:
        store.add_documents([Document(page_content=content, metadata=meta)])
    
    print("=" * 50)
    print("TASK 13.4 — MEMORY AUDIT TOOL")
    print("=" * 50)
    
    audit = audit_user_memories(user_id)
    print(f"\nTotal memories: {audit['total_memories']}")
    print(f"Categories: {audit['categories']}")
    print(f"Duplicates found: {len(audit['duplicates'])}")
    print(f"Length distribution: {audit['by_length']}")
    
    deleted = delete_duplicates(user_id)
    print(f"\nDeleted {deleted} duplicate(s)")
    
    audit_after = audit_user_memories(user_id)
    print(f"Remaining memories: {audit_after['total_memories']}")
    print("\n✅ Audit complete!")
