"""Task 13.3 — Memory Contradiction Detector."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

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
    return Chroma(persist_directory=os.path.join(os.path.dirname(__file__), "contradict_db"), embedding_function=embeddings)

def check_contradiction(user_id: str, new_fact: str):
    store = get_store()
    if not store:
        return False, "Store unavailable"
    
    similar = store.similarity_search(new_fact, k=3, filter={"user_id": user_id})
    if not similar:
        return False, None
    
    context = "\n".join([f"- {d.page_content}" for d in similar])
    prompt = f"""Does this new fact contradict any existing fact?
Existing facts:
{context}

New fact: {new_fact}

Reply: CONTRADICTION if they conflict, or NO if consistent.
Explain briefly."""
    
    resp = llm.invoke([HumanMessage(content=prompt)]).content
    is_contradiction = "CONTRADICTION" in resp.upper()
    return is_contradiction, resp

def add_or_update_memory(user_id: str, fact: str):
    store = get_store()
    if not store:
        return
    
    is_conflict, reason = check_contradiction(user_id, fact)
    if is_conflict:
        print(f"⚠️ Detected contradiction: {reason}")
        print(f"   Old fact will be kept. New fact rejected: {fact}")
        return False
    
    store.add_documents([Document(page_content=fact, metadata={"user_id": user_id})])
    return True

if __name__ == "__main__":
    user_id = "user-test"
    
    print("=" * 50)
    print("TASK 13.3 — CONTRADICTION DETECTOR")
    print("=" * 50)
    
    facts = [
        ("User loves Python programming", False),
        ("User dislikes Python and prefers Java", True),  # Should detect contradiction
        ("User enjoys hiking on weekends", False),
        ("User hates outdoor activities including hiking", True),  # Should detect
    ]
    
    for fact, should_conflict in facts:
        print(f"\nAdding: '{fact}'")
        success = add_or_update_memory(user_id, fact)
        if should_conflict and not success:
            print("✅ Correctly detected contradiction!")
        elif not should_conflict and success:
            print("✅ Successfully added fact!")
        else:
            print("❌ Unexpected result")
