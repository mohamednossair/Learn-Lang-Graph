# =============================================================
# TASK 5.2 — Content Pipeline
# =============================================================
# Goal:
#   Supervisor routes a content request through a fixed pipeline:
#     researcher → writer → editor → seo_agent
#   Each specialist adds their contribution to state.
#   Supervisor drives the sequence — not the specialists.
#
# State: {messages, next, topic, research, draft, edited, seo_version}
# =============================================================

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_ollama_model


# ── STEP 1: State ─────────────────────────────────────────────

class ContentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str          # supervisor routing
    topic: str         # content topic
    research: str      # output from researcher
    draft: str         # output from writer
    edited: str        # output from editor
    seo_version: str   # final SEO-optimized version


# ── STEP 2: LLM ───────────────────────────────────────────────

llm = ChatOllama(model=get_ollama_model(), temperature=0.5)


# ── STEP 3: Supervisor Node ───────────────────────────────────
# TODO:
#   Supervisor knows the pipeline: researcher → writer → editor → seo_agent → FINISH
#   It checks which step was completed last and routes to the next one.
#   System prompt should describe the pipeline and the current state.

SUPERVISOR_PROMPT = """You are a content pipeline supervisor.
The pipeline steps in order are: researcher, writer, editor, seo_agent, FINISH.

Current state:
- research done: {research_done}
- draft done: {draft_done}
- editing done: {editing_done}
- seo done: {seo_done}

Route to the next step in the pipeline. Respond with ONLY one word:
researcher, writer, editor, seo_agent, or FINISH"""


def supervisor_node(state: ContentState) -> dict:
    # Determine which steps are completed
    research_done = bool(state.get("research"))
    draft_done = bool(state.get("draft"))
    editing_done = bool(state.get("edited"))
    seo_done = bool(state.get("seo_version"))

    # Build prompt with current state
    prompt = SUPERVISOR_PROMPT.format(
        research_done=research_done,
        draft_done=draft_done,
        editing_done=editing_done,
        seo_done=seo_done
    )

    messages = [SystemMessage(content=prompt)]
    response = llm.invoke(messages)

    next_step = response.content.strip().lower()
    print(f"[supervisor] Routing to: {next_step}")

    return {"next": next_step}


# ── STEP 4: Specialist Nodes ──────────────────────────────────

def researcher_node(state: ContentState) -> dict:
    prompt = [
        SystemMessage(content="You are a research specialist. Provide 3-5 key facts about the topic."),
        HumanMessage(content=f"Research the topic: {state['topic']}"),
    ]
    response = llm.invoke(prompt)
    print("[researcher] Done")
    return {"research": response.content, "messages": [response]}


def writer_node(state: ContentState) -> dict:
    prompt = [
        SystemMessage(content="You are a content writer. Write a short 150-word article based on the research."),
        HumanMessage(content=f"Topic: {state['topic']}\n\nResearch:\n{state['research']}"),
    ]
    response = llm.invoke(prompt)
    print("[writer] Done")
    return {"draft": response.content, "messages": [response]}


def editor_node(state: ContentState) -> dict:
    prompt = [
        SystemMessage(content="You are an editor. Improve clarity, fix grammar, and tighten the writing."),
        HumanMessage(content=f"Edit this draft:\n{state['draft']}"),
    ]
    response = llm.invoke(prompt)
    print("[editor] Done")
    return {"edited": response.content, "messages": [response]}


def seo_agent_node(state: ContentState) -> dict:
    prompt = [
        SystemMessage(content="""You are an SEO specialist. Optimize the content for search engines.
Add:
1. An SEO-optimized title (max 60 chars)
2. A meta description (max 160 chars)
3. 3-5 relevant keywords

Format the output as:
---
Title: [SEO title]
Meta: [meta description]
Keywords: [keyword1], [keyword2], [keyword3]

Content:
[the edited content]
---"""),
        HumanMessage(content=f"Topic: {state['topic']}\n\nEdited Content:\n{state['edited']}"),
    ]
    response = llm.invoke(prompt)
    print("[seo_agent] Done")
    return {"seo_version": response.content, "messages": [response]}


# ── STEP 5: Routing Function ──────────────────────────────────

def route_supervisor(state: ContentState) -> Literal["researcher", "writer", "editor", "seo_agent", "__end__"]:
    next_step = state.get("next", "").strip().lower()

    # Map "finish" to __end__, otherwise return the node name
    if next_step == "finish":
        return "__end__"
    elif next_step in ["researcher", "writer", "editor", "seo_agent"]:
        return next_step
    else:
        # Default to researcher if unknown/empty
        return "researcher"


# ── STEP 6: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(ContentState)

# Add all nodes
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("researcher", researcher_node)
graph_builder.add_node("writer", writer_node)
graph_builder.add_node("editor", editor_node)
graph_builder.add_node("seo_agent", seo_agent_node)

# START → supervisor
graph_builder.add_edge(START, "supervisor")

# Supervisor conditional routing
graph_builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    ["researcher", "writer", "editor", "seo_agent", "__end__"]
)

# Each specialist → supervisor (loop back)
graph_builder.add_edge("researcher", "supervisor")
graph_builder.add_edge("writer", "supervisor")
graph_builder.add_edge("editor", "supervisor")
graph_builder.add_edge("seo_agent", "supervisor")

graph = graph_builder.compile()


# ── STEP 7: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    topics = [
        "The benefits of using LangGraph for building AI agents",
        "Why Python is the best language for data science",
    ]

    for topic in topics:
        print("\n" + "=" * 65)
        print(f"Topic: {topic}")
        print("=" * 65)
        result = graph.invoke({
            "messages": [HumanMessage(content=f"Create content about: {topic}")],
            "next": "",
            "topic": topic,
            "research": "",
            "draft": "",
            "edited": "",
            "seo_version": "",
        })
        print(f"\nSEO Version:\n{result.get('seo_version', '(not completed yet)')[:400]}")
