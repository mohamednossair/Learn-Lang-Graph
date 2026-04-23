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

llm = ChatOllama(model="llama3", temperature=0.5)


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
    # TODO: determine next step based on what's been completed
    # Return {"next": "researcher"|"writer"|"editor"|"seo_agent"|"FINISH"}
    pass


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


# TODO: implement seo_agent_node
# def seo_agent_node(state: ContentState) -> dict:
#     # Add SEO title, meta description, and 3 keyword suggestions
#     pass


# ── STEP 5: Routing Function ──────────────────────────────────

def route_supervisor(state: ContentState) -> Literal["researcher", "writer", "editor", "seo_agent", "__end__"]:
    # TODO: map state["next"] → node name or "__end__"
    pass


# ── STEP 6: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(ContentState)

# TODO: add supervisor + all 4 specialist nodes
# TODO: START → supervisor
# TODO: conditional edges from supervisor
# TODO: each specialist → supervisor

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
