# =============================================================
# TASK 7.2 — Content Moderation with HITL
# =============================================================
# Goal:
#   generator_node writes a social media post.
#   moderation_node checks for forbidden words.
#   If clean: auto-publish (mock).
#   If flagged: interrupt showing content to human.
#     Human edits → resumes with edited version → publish.
#
# Key concepts: interrupt(), Command(resume=edited_content)
# =============================================================

from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ── STEP 1: State ─────────────────────────────────────────────

class ModerationState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    draft: str
    flagged_words: list
    published: bool
    final_content: str


# ── STEP 2: Forbidden Words ───────────────────────────────────

FORBIDDEN = ["hate", "violent", "illegal", "spam", "scam", "fake"]


# ── STEP 3: Generator Node ────────────────────────────────────
# TODO: Use ChatOllama to write a short 2-sentence social media post about state["topic"]
# Return {"draft": "..."}

llm = ChatOllama(model="llama3", temperature=0.7)


def generator_node(state: ModerationState) -> dict:
    pass


# ── STEP 4: Moderation Node ───────────────────────────────────
# TODO:
#   1. Check draft for FORBIDDEN words (case-insensitive)
#   2. If clean: set published=True, final_content=draft
#   3. If flagged: interrupt({"draft": ..., "flagged": [...]})
#      Human returns edited version → set published=True, final_content=edited

def moderation_node(state: ModerationState) -> dict:
    draft = state["draft"]
    flagged = [w for w in FORBIDDEN if w in draft.lower()]

    if not flagged:
        print(f"[moderation] Clean ✓ — auto-publishing")
        return {"flagged_words": [], "published": True, "final_content": draft}

    print(f"[moderation] Flagged words: {flagged}")
    # TODO: interrupt here — ask human to edit
    pass


# ── STEP 5: Publish Node ──────────────────────────────────────

def publish_node(state: ModerationState) -> dict:
    print(f"[publish] Published: {state['final_content'][:80]}...")
    return {}


# ── STEP 6: Build Graph ───────────────────────────────────────

checkpointer = MemorySaver()

graph_builder = StateGraph(ModerationState)
graph_builder.add_node("generator", generator_node)
graph_builder.add_node("moderation", moderation_node)
graph_builder.add_node("publish", publish_node)

graph_builder.add_edge(START, "generator")
graph_builder.add_edge("generator", "moderation")
graph_builder.add_edge("moderation", "publish")
graph_builder.add_edge("publish", END)

graph = graph_builder.compile(checkpointer=checkpointer)


# ── STEP 7: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    topics = [
        "our new product launch this weekend",
        "why you should try our service today",
    ]

    for i, topic in enumerate(topics):
        config = {"configurable": {"thread_id": f"post-{i}"}}
        print(f"\n{'='*60}")
        print(f"Topic: {topic}")

        state = graph.invoke(
            {
                "messages": [], "topic": topic, "draft": "",
                "flagged_words": [], "published": False, "final_content": "",
            },
            config=config,
        )

        current = graph.get_state(config)
        if current.next:
            interrupt_data = current.tasks[0].interrupts[0].value
            print(f"  ⚠ FLAGGED — interrupt data: {interrupt_data}")
            # Simulate human editing: remove flagged words
            edited = interrupt_data["draft"]
            for word in interrupt_data.get("flagged", []):
                edited = edited.replace(word, "***")
            print(f"  Human edited: {edited[:80]}")
            state = graph.invoke(Command(resume=edited), config=config)

        print(f"  Published: {state.get('published')}")
        print(f"  Content  : {state.get('final_content', '')[:100]}")
