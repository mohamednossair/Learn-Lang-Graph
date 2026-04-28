"""Task 11.1 — Email Processing Pipeline with 3 subgraphs."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

llm = ChatOllama(model=get_ollama_model(), temperature=0)

# ============== SUBGRAPH 1: Spam Filter ==============
class SpamFilterState(TypedDict):
    sender: str
    subject: str
    body: str
    is_spam: bool
    reason: str

def check_sender(state: SpamFilterState) -> dict:
    suspicious_domains = ["spam.com", "fake.net", "scam.org"]
    domain = state["sender"].split("@")[-1] if "@" in state["sender"] else ""
    if domain in suspicious_domains:
        return {"is_spam": True, "reason": f"Suspicious sender domain: {domain}"}
    return {"is_spam": False}

def check_keywords(state: SpamFilterState) -> dict:
    spam_words = ["urgent", "winner", "lottery", "prize", "click here", "limited time"]
    text = (state["subject"] + " " + state["body"]).lower()
    found = [w for w in spam_words if w in text]
    if found:
        return {"is_spam": True, "reason": f"Spam keywords detected: {found}"}
    return {}

def check_formatting(state: SpamFilterState) -> dict:
    body = state["body"]
    if body.count("http") > 3 or body.count("!") > 5:
        return {"is_spam": True, "reason": "Excessive links or exclamation marks"}
    return {}

spam_builder = StateGraph(SpamFilterState)
spam_builder.add_node("check_sender", check_sender)
spam_builder.add_node("check_keywords", check_keywords)
spam_builder.add_node("check_formatting", check_formatting)
spam_builder.add_edge(START, "check_sender")
spam_builder.add_edge("check_sender", "check_keywords")
spam_builder.add_edge("check_keywords", "check_formatting")
spam_builder.add_edge("check_formatting", END)
spam_filter_subgraph = spam_builder.compile()

# ============== SUBGRAPH 2: Categorize ==============
class CategorizeState(TypedDict):
    subject: str
    body: str
    category: str

def categorize_email(state: CategorizeState) -> dict:
    text = (state["subject"] + " " + state["body"]).lower()
    promo_keywords = ["sale", "discount", "offer", "deal", "promotion", "% off"]
    update_keywords = ["newsletter", "update", "announcement", "release", "version"]
    
    if any(kw in text for kw in promo_keywords):
        return {"category": "promotions"}
    if any(kw in text for kw in update_keywords):
        return {"category": "updates"}
    return {"category": "inbox"}

cat_builder = StateGraph(CategorizeState)
cat_builder.add_node("categorize", categorize_email)
cat_builder.add_edge(START, "categorize")
cat_builder.add_edge("categorize", END)
categorize_subgraph = cat_builder.compile()

# ============== SUBGRAPH 3: Summarize ==============
class SummarizeState(TypedDict):
    subject: str
    body: str
    summary: str

def summarize_email(state: SummarizeState) -> dict:
    prompt = f"Summarize this email in one sentence. Subject: {state['subject']}\nBody: {state['body'][:500]}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": resp.content.strip()}

sum_builder = StateGraph(SummarizeState)
sum_builder.add_node("summarize", summarize_email)
sum_builder.add_edge(START, "summarize")
sum_builder.add_edge("summarize", END)
summarize_subgraph = sum_builder.compile()

# ============== PARENT GRAPH ==============
class EmailState(TypedDict):
    messages: Annotated[list, add_messages]
    sender: str
    subject: str
    body: str
    is_spam: bool
    reason: str
    category: str
    summary: str
    final_action: str

def spam_wrapper(state: EmailState) -> dict:
    result = spam_filter_subgraph.invoke({
        "sender": state["sender"],
        "subject": state["subject"],
        "body": state["body"],
        "is_spam": False, "reason": ""
    })
    return result

def cat_wrapper(state: EmailState) -> dict:
    if state.get("is_spam", False):
        return {"category": "spam"}
    result = categorize_subgraph.invoke({
        "subject": state["subject"],
        "body": state["body"],
        "category": ""
    })
    return result

def sum_wrapper(state: EmailState) -> dict:
    if state.get("is_spam", False):
        return {"summary": "SPAM - not summarized"}
    result = summarize_subgraph.invoke({
        "subject": state["subject"],
        "body": state["body"],
        "summary": ""
    })
    return result

def final_action_node(state: EmailState) -> dict:
    if state.get("is_spam", False):
        action = f"REJECTED (Spam: {state['reason']})"
    else:
        action = f"DELIVERED to {state['category']} | Summary: {state['summary'][:60]}..."
    return {"final_action": action}

builder = StateGraph(EmailState)
builder.add_node("spam_filter", spam_wrapper)
builder.add_node("categorize", cat_wrapper)
builder.add_node("summarize", sum_wrapper)
builder.add_node("final_action", final_action_node)
builder.add_edge(START, "spam_filter")
builder.add_edge("spam_filter", "categorize")
builder.add_edge("categorize", "summarize")
builder.add_edge("summarize", "final_action")
builder.add_edge("final_action", END)
email_graph = builder.compile()

# ============== TESTS ==============
def test_subgraphs_independently():
    print("=" * 60)
    print("TESTING SUBGRAPHS INDEPENDENTLY")
    print("=" * 60)
    
    # Test spam filter
    spam_test = spam_filter_subgraph.invoke({
        "sender": "bad@spam.com", "subject": "You won!", "body": "Click here to claim",
        "is_spam": False, "reason": ""
    })
    print(f"Spam test: {spam_test}")
    assert spam_test["is_spam"] == True
    
    # Test categorize
    cat_test = categorize_subgraph.invoke({
        "subject": "50% off sale today!", "body": "Big discount", "category": ""
    })
    print(f"Categorize test: {cat_test}")
    assert cat_test["category"] == "promotions"
    
    # Test summarize
    sum_test = summarize_subgraph.invoke({
        "subject": "Meeting tomorrow", "body": "We have a team meeting at 3pm.", "summary": ""
    })
    print(f"Summarize test: {sum_test}")
    assert len(sum_test["summary"]) > 0
    print("All subgraph tests passed!")

if __name__ == "__main__":
    test_subgraphs_independently()
    
    print("\n" + "=" * 60)
    print("EMAIL PROCESSING PIPELINE")
    print("=" * 60)
    
    test_emails = [
        {"sender": "boss@company.com", "subject": "Meeting tomorrow", "body": "We have a team meeting at 3pm to discuss the project."},
        {"sender": "spam@spam.com", "subject": "You won $1,000,000!", "body": "URGENT! Click here to claim your lottery prize now! Limited time!"},
        {"sender": "store@shop.com", "subject": "50% off everything!", "body": "Huge sale this weekend. Don't miss out!"},
    ]
    
    for email in test_emails:
        result = email_graph.invoke({
            "messages": [], "sender": email["sender"], "subject": email["subject"],
            "body": email["body"], "is_spam": False, "reason": "",
            "category": "", "summary": "", "final_action": ""
        })
        print(f"\nFrom: {email['sender']}")
        print(f"Subject: {email['subject']}")
        print(f"Action: {result['final_action']}")
