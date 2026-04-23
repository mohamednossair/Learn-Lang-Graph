# =============================================================
# LESSON 2 — Conditional Edges & Branching Logic
# =============================================================
#
# WHAT YOU WILL LEARN:
#   - add_conditional_edges() — route to different nodes
#   - Routing functions — decide the next node at runtime
#   - How to build if/else logic inside a graph
#
# MENTAL MODEL:
#   START → classify → (if positive) → handle_positive → END
#                    → (if negative) → handle_negative → END
#                    → (if neutral)  → handle_neutral  → END
# =============================================================

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# -------------------------------------------------------------
# STEP 1 — State
# -------------------------------------------------------------

class ReviewState(TypedDict):
    review: str        # user input review text
    sentiment: str     # filled by classify_node: positive/negative/neutral
    response: str      # filled by the handler node


# -------------------------------------------------------------
# STEP 2 — Nodes
# -------------------------------------------------------------

def classify_node(state: ReviewState) -> dict:
    """Classify sentiment of the review (simple keyword logic)."""
    text = state["review"].lower()
    if any(w in text for w in ["incredible", "outstanding", "phenomenal"]):
        sentiment = "very_positive"
    elif any(w in text for w in ["great", "love", "excellent", "amazing", "good"]):
        sentiment = "positive"
    elif any(w in text for w in ["bad", "terrible", "hate", "awful", "worst"]):
        sentiment = "negative"
    else:
        sentiment = "neutral"
    print(f"[classify] Sentiment detected: {sentiment}")
    return {"sentiment": sentiment}

def handle_very_positive(state: ReviewState) -> dict:
    print("[handle_very_positive] Generating very positive response...")
    return {"response": "Thank you so much! We're thrilled you had a great experience!"}


def handle_positive(state: ReviewState) -> dict:
    print("[handle_positive] Generating positive response...")
    return {"response": "Thank you so much! We're thrilled you had a great experience!"}


def handle_negative(state: ReviewState) -> dict:
    print("[handle_negative] Generating apology response...")
    return {"response": "We're sorry to hear that. We'll work hard to improve!"}


def handle_neutral(state: ReviewState) -> dict:
    print("[handle_neutral] Generating neutral response...")
    return {"response": "Thank you for your feedback. We appreciate your input!"}


# -------------------------------------------------------------
# STEP 3 — Routing Function
# This is the KEY to conditional edges.
# It receives the current state and returns the NAME of the
# next node to go to (as a string).
# The return type hint uses Literal to declare all options.
# -------------------------------------------------------------

def route_by_sentiment(state: ReviewState) -> Literal["handle_positive", "handle_negative", "handle_neutral","handle_very_positive"]:
    sentiment = state["sentiment"]
    if sentiment == "very_positive":
        return "handle_very_positive"
    elif sentiment == "positive":
        return "handle_positive"
    elif sentiment == "negative":
        return "handle_negative"
    else:
        return "handle_neutral"


# -------------------------------------------------------------
# STEP 4 — Build the Graph with Conditional Edges
# -------------------------------------------------------------

graph_builder = StateGraph(ReviewState)

graph_builder.add_node("classify", classify_node)
graph_builder.add_node("handle_very_positive", handle_very_positive)
graph_builder.add_node("handle_positive", handle_positive)
graph_builder.add_node("handle_negative", handle_negative)
graph_builder.add_node("handle_neutral", handle_neutral)

# Regular edge: START → classify
graph_builder.add_edge(START, "classify")

# CONDITIONAL edge: after classify, call route_by_sentiment()
# to decide which node to go to next.
# The dict maps return values → node names.
graph_builder.add_conditional_edges(
    "classify",                 # source node
    route_by_sentiment,         # routing function
    {
        # mapping: return value → node name
        "handle_very_positive": "handle_very_positive",
        "handle_positive": "handle_positive",
        "handle_negative": "handle_negative",
        "handle_neutral":  "handle_neutral"
    }
)

# All handler nodes go to END
graph_builder.add_edge("handle_very_positive", END)
graph_builder.add_edge("handle_positive", END)
graph_builder.add_edge("handle_negative", END)
graph_builder.add_edge("handle_neutral", END)

graph = graph_builder.compile()


# -------------------------------------------------------------
# STEP 5 — Test with different inputs
# -------------------------------------------------------------

if __name__ == "__main__":
    test_reviews = [
        "This product is outstanding amazing! I love it!",
        "This product is absolutely amazing! I love it!",
        "Terrible experience, worst purchase ever.",
        "It arrived on time. The package was okay.",
    ]

    for review in test_reviews:
        print("\n" + "=" * 55)
        print(f"Review: {review}")
        print("-" * 55)
        result = graph.invoke({"review": review, "sentiment": "", "response": ""})
        print(f"Response: {result['response']}")


# =============================================================
# EXERCISE:
#   1. Add a "very_positive" sentiment for reviews with words
#      like "incredible", "outstanding", "phenomenal"
#   2. Add a handle_very_positive node with an even better reply
#   3. Update route_by_sentiment() and add the new edge
# =============================================================
