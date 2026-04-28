"""Task 12.1 — Your Own Knowledge Base with test questions."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Run: pip install langchain-community chromadb")

# ============== KNOWLEDGE BASE DOCUMENTS ==============
KNOWLEDGE_BASE = [
    # Python documents
    Document(page_content="""
Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.
It emphasizes code readability with its use of significant whitespace.
Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
It has a comprehensive standard library and is known as a "batteries included" language.
    """, metadata={"source": "python_intro.txt", "paragraph": 1}),
    
    Document(page_content="""
Python's list comprehensions provide a concise way to create lists.
Common applications include web development, data analysis, artificial intelligence, and scientific computing.
Popular frameworks include Django for web, Pandas for data, and PyTorch for machine learning.
    """, metadata={"source": "python_intro.txt", "paragraph": 2}),
    
    # Database documents
    Document(page_content="""
PostgreSQL is a powerful, open source object-relational database system.
It has over 30 years of active development and a strong reputation for reliability and performance.
PostgreSQL supports advanced data types and has full ACID compliance.
    """, metadata={"source": "databases.txt", "paragraph": 1}),
    
    Document(page_content="""
SQL (Structured Query Language) is the standard language for relational database management systems.
SQL statements are used to perform tasks such as update data or retrieve data from a database.
Common SQL commands include SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, and DROP TABLE.
    """, metadata={"source": "databases.txt", "paragraph": 2}),
    
    # Docker documents
    Document(page_content="""
Docker is a platform for developing, shipping, and running applications in containers.
Containers are lightweight and contain everything needed to run the application.
Docker ensures your application runs the same regardless of the environment.
    """, metadata={"source": "docker_guide.txt", "paragraph": 1}),
    
    Document(page_content="""
Docker Compose is a tool for defining and running multi-container Docker applications.
With Compose, you use a YAML file to configure your application's services.
Then, with a single command, you create and start all the services.
    """, metadata={"source": "docker_guide.txt", "paragraph": 2}),
    
    # API documents
    Document(page_content="""
REST (Representational State Transfer) is an architectural style for designing networked applications.
REST APIs use HTTP requests to perform CRUD operations (Create, Read, Update, Delete).
JSON is the most common data format used in REST APIs.
    """, metadata={"source": "api_design.txt", "paragraph": 1}),
]

def build_knowledge_base():
    if not CHROMA_AVAILABLE:
        return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(KNOWLEDGE_BASE)
    
    embeddings = OllamaEmbeddings(model=get_ollama_model())
    db_path = os.path.join(os.path.dirname(__file__), "task_12_kb")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"Built knowledge base: {len(chunks)} chunks from {len(KNOWLEDGE_BASE)} documents")
    return vectorstore

def ask_question(vectorstore, question, llm):
    results = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([f"[From {doc.metadata['source']}, para {doc.metadata['paragraph']}]:\n{doc.page_content}" 
                          for doc in results])
    
    prompt = f"""Answer based ONLY on the following documents.
If the answer is not in the documents, say "I don't have information about that."

DOCUMENTS:
{context}

QUESTION: {question}"""
    
    answer = llm.invoke([SystemMessage(content="Answer from documents only."),
                         HumanMessage(content=prompt)])
    return answer.content, results

# ============== TEST QUESTIONS ==============
TEST_QUESTIONS = [
    ("Who created Python?", ["guido", "van rossum", "1991"]),
    ("What is PostgreSQL?", ["database", "open source", "relational"]),
    ("What does Docker do?", ["container", "application", "ship"]),
    ("What is REST?", ["api", "architectural", "http"]),
    ("What is Docker Compose?", ["multi-container", "yaml", "services"]),
    ("What are SQL commands?", ["select", "insert", "update", "delete"]),
    ("What is Python used for?", ["web", "data", "ai", "machine learning"]),
    ("What makes Python readable?", ["whitespace", "indentation"]),
    ("What is ACID compliance?", ["postgresql", "database"]),
    ("Who invented containers?", ["don't have information"]),  # This should fail - not in docs
]

if __name__ == "__main__":
    if not CHROMA_AVAILABLE:
        print("Install: pip install langchain-community chromadb")
        exit(1)
    
    print("=" * 60)
    print("TASK 12.1 — KNOWLEDGE BASE TESTING")
    print("=" * 60)
    
    vs = build_knowledge_base()
    llm = ChatOllama(model=get_ollama_model(), temperature=0)
    
    good_answers = 0
    poor_answers = 0
    
    for question, expected in TEST_QUESTIONS:
        print(f"\nQ: {question}")
        answer, chunks = ask_question(vs, question, llm)
        
        # Check if expected keywords are in answer
        answer_lower = answer.lower()
        matched = sum(1 for kw in expected if kw.lower() in answer_lower)
        score = matched / len(expected)
        
        print(f"Retrieved {len(chunks)} chunks")
        for c in chunks:
            print(f"  - {c.metadata['source']}, para {c.metadata['paragraph']}")
        print(f"Answer: {answer[:150]}...")
        
        if score >= 0.5:
            print(f"✅ Good answer (matched {matched}/{len(expected)} keywords)")
            good_answers += 1
        else:
            print(f"❌ Poor answer (matched only {matched}/{len(expected)} keywords)")
            poor_answers += 1
            print(f"   Expected keywords: {expected}")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {good_answers} good, {poor_answers} poor answers")
    print(f"Accuracy: {good_answers/len(TEST_QUESTIONS)*100:.1f}%")
    print("=" * 60)
