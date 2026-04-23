# =============================================================
# LESSON 7 — Human-in-the-Loop (HITL)
# =============================================================
#
# WHAT YOU WILL LEARN:
#   - interrupt() — pause the graph and wait for a human
#   - MemorySaver checkpointer — required for HITL (saves state)
#   - thread_id — how to identify a conversation session
#   - graph.invoke() with Command(resume=...) — resume after human input
#   - 3 HITL patterns:
#       Pattern A: Approve/Reject before tool execution
#       Pattern B: Human provides missing information
#       Pattern C: Human reviews and edits the agent's output
#
# WHY HITL MATTERS:
#   Agents can make mistakes. HITL lets you put humans at
#   critical checkpoints — approve actions, provide context,
#   or correct outputs — before they cause real-world effects.
# =============================================================

from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# =============================================================
# PATTERN A — Approve/Reject Before Tool Execution
# =============================================================
# The agent decides to call a tool, but BEFORE it runs,
# we pause and ask the human: "Is this okay?"
#
# FLOW:
#   agent → human_approval (INTERRUPT) → tools (if approved) → agent → END
# =============================================================

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient. This is a sensitive action requiring approval."""
    print(f"\n  [EMAIL SENT] To: {recipient} | Subject: {subject}")
    return f"Email successfully sent to {recipient} with subject: '{subject}'"


@tool
def delete_record(table: str, record_id: int) -> str:
    """Delete a record from the database. This is a dangerous irreversible action."""
    print(f"\n  [RECORD DELETED] Table: {table}, ID: {record_id}")
    return f"Record {record_id} deleted from {table}"


sensitive_tools = [send_email, delete_record]
safe_tools_names = set()
sensitive_tools_names = {t.name for t in sensitive_tools}


class ApprovalState(TypedDict):
    messages: Annotated[list, add_messages]


llm_a = ChatOllama(model="llama3", temperature=0).bind_tools(sensitive_tools)


def agent_a(state: ApprovalState) -> dict:
    system = SystemMessage(content="You are an assistant. Use tools when needed.")
    response = llm_a.invoke([system] + state["messages"])
    return {"messages": [response]}


def human_approval_node(state: ApprovalState) -> dict:
    """Pause here and ask the human to approve or reject the tool call."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tc in last_message.tool_calls:
            if tc["name"] in sensitive_tools_names:
                # Show what the agent wants to do
                print(f"\n{'='*55}")
                print(f"⚠️  HUMAN APPROVAL REQUIRED")
                print(f"  Tool: {tc['name']}")
                print(f"  Args: {tc['args']}")
                print(f"{'='*55}")

                # INTERRUPT — pause execution, hand control to human
                human_decision = interrupt({
                    "question": f"Agent wants to call '{tc['name']}' with args {tc['args']}. Approve? (yes/no)",
                    "tool_name": tc["name"],
                    "tool_args": tc["args"],
                })

                if human_decision.lower() != "yes":
                    # Human rejected — remove tool calls from last message
                    # and tell the agent it was rejected
                    return {
                        "messages": [AIMessage(content=f"Action '{tc['name']}' was rejected by the human. I will stop.")]
                    }
    return {}  # No change needed — proceed to tools


def route_after_approval(state: ApprovalState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# Build Pattern A graph
builder_a = StateGraph(ApprovalState)
builder_a.add_node("agent",          agent_a)
builder_a.add_node("human_approval", human_approval_node)
builder_a.add_node("tools",          ToolNode(sensitive_tools))
builder_a.add_edge(START,            "agent")
builder_a.add_conditional_edges(
    "agent",
    lambda s: "human_approval" if (
        hasattr(s["messages"][-1], "tool_calls") and s["messages"][-1].tool_calls
    ) else "end",
    {"human_approval": "human_approval", "end": END}
)
builder_a.add_conditional_edges("human_approval", route_after_approval, {"tools": "tools", "end": END})
builder_a.add_edge("tools", "agent")

# MemorySaver is REQUIRED for interrupt() to work
# It saves graph state so it can be resumed later
checkpointer_a = MemorySaver()
graph_a = builder_a.compile(checkpointer=checkpointer_a)


def run_pattern_a():
    print("\n" + "="*60)
    print("PATTERN A — Approve/Reject Before Tool Execution")
    print("="*60)

    config = {"configurable": {"thread_id": "thread-001"}}
    user_input = "Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'Please join at 3pm'"

    print(f"\nUser: {user_input}")

    # Run until interrupted
    result = graph_a.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    # Check if graph is interrupted (waiting for human)
    state = graph_a.get_state(config)
    if state.next:
        print(f"\nGraph interrupted at: {state.next}")
        # Get the interrupt value
        for task in state.tasks:
            if task.interrupts:
                print(f"Question: {task.interrupts[0].value['question']}")

        # Simulate human approving
        human_answer = input("\nYour decision (yes/no): ").strip().lower()

        # Resume the graph with human's answer
        result = graph_a.invoke(
            Command(resume=human_answer),
            config=config
        )

    print(f"\nFinal: {result['messages'][-1].content}")


# =============================================================
# PATTERN B — Human Provides Missing Information
# =============================================================
# The agent hits a point where it needs information it doesn't
# have. It pauses and asks the human directly.
#
# FLOW:
#   agent → needs_info (INTERRUPT) → agent (with new info) → END
# =============================================================

class InfoState(TypedDict):
    messages: Annotated[list, add_messages]
    user_info: dict    # filled by human during interrupt


llm_b = ChatOllama(model="llama3", temperature=0)


def agent_b(state: InfoState) -> dict:
    system = SystemMessage(content="""You are a personalized assistant.
    If you need the user's name or preferences to complete a task, ask for them.
    If user_info is available in the state, use it.""")
    messages = [system] + state["messages"]
    if state.get("user_info"):
        messages.append(SystemMessage(content=f"User info available: {state['user_info']}"))
    response = llm_b.invoke(messages)
    return {"messages": [response]}


def collect_user_info(state: InfoState) -> dict:
    """Ask human for missing information before proceeding."""
    print("\n⚠️  Agent needs more information from you.")

    # INTERRUPT — pause and ask for human input
    user_data = interrupt({
        "question": "Please provide your name and preferred language:",
        "fields": ["name", "preferred_language"]
    })

    # user_data comes back as a dict when resumed
    return {"user_info": user_data}


def should_collect_info(state: InfoState) -> str:
    """Check if we already have user info or need to collect it."""
    if not state.get("user_info"):
        return "collect_info"
    return "agent"


builder_b = StateGraph(InfoState)
builder_b.add_node("check_info",    lambda s: {})  # pass-through router node
builder_b.add_node("collect_info",  collect_user_info)
builder_b.add_node("agent",         agent_b)
builder_b.add_edge(START, "check_info")
builder_b.add_conditional_edges("check_info", should_collect_info, {"collect_info": "collect_info", "agent": "agent"})
builder_b.add_edge("collect_info", "agent")
builder_b.add_edge("agent", END)

checkpointer_b = MemorySaver()
graph_b = builder_b.compile(checkpointer=checkpointer_b)


def run_pattern_b():
    print("\n" + "="*60)
    print("PATTERN B — Human Provides Missing Information")
    print("="*60)

    config = {"configurable": {"thread_id": "thread-002"}}

    result = graph_b.invoke(
        {"messages": [HumanMessage(content="Create a personalized welcome message for me")], "user_info": {}},
        config=config
    )

    state = graph_b.get_state(config)
    if state.next:
        for task in state.tasks:
            if task.interrupts:
                print(f"\nAgent asks: {task.interrupts[0].value['question']}")

        name = input("Your name: ").strip()
        language = input("Preferred language: ").strip()

        result = graph_b.invoke(
            Command(resume={"name": name, "preferred_language": language}),
            config=config
        )

    print(f"\nFinal: {result['messages'][-1].content}")


# =============================================================
# PATTERN C — Human Reviews and Edits Output
# =============================================================
# The agent produces a draft. The human reviews it and can
# either approve it or provide corrections to improve it.
#
# FLOW:
#   agent (draft) → human_review (INTERRUPT) → if ok: END
#                                             → if edit: agent (revise) → ...
# =============================================================

class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    approved: bool
    revision_count: int


llm_c = ChatOllama(model="llama3", temperature=0.5)


def write_draft(state: ReviewState) -> dict:
    system = SystemMessage(content="You are a professional writer. Write concise, clear content.")
    response = llm_c.invoke([system] + state["messages"])
    print(f"\n[agent] Draft #{state['revision_count'] + 1} written.")
    return {"draft": response.content, "messages": [response], "revision_count": state["revision_count"] + 1}


def human_review_node(state: ReviewState) -> dict:
    print(f"\n{'='*55}")
    print(f"📝 DRAFT FOR REVIEW (version {state['revision_count']}):")
    print(f"{'='*55}")
    print(state["draft"])
    print(f"{'='*55}")

    # INTERRUPT — show draft and wait for human feedback
    feedback = interrupt({
        "question": "Review the draft above. Type 'approve' to accept, or type your feedback to request changes.",
        "draft": state["draft"],
        "version": state["revision_count"]
    })

    if feedback.lower() == "approve":
        return {"approved": True}
    else:
        # Add human feedback as a new message for the agent to revise
        return {
            "approved": False,
            "messages": [HumanMessage(content=f"Please revise the draft. Feedback: {feedback}")]
        }


def should_revise(state: ReviewState) -> str:
    if state.get("approved") or state.get("revision_count", 0) >= 3:
        return "end"
    return "write"


builder_c = StateGraph(ReviewState)
builder_c.add_node("write",  write_draft)
builder_c.add_node("review", human_review_node)
builder_c.add_edge(START,    "write")
builder_c.add_edge("write",  "review")
builder_c.add_conditional_edges("review", should_revise, {"write": "write", "end": END})

checkpointer_c = MemorySaver()
graph_c = builder_c.compile(checkpointer=checkpointer_c)


def run_pattern_c():
    print("\n" + "="*60)
    print("PATTERN C — Human Reviews and Edits Output")
    print("="*60)

    config = {"configurable": {"thread_id": "thread-003"}}

    result = graph_c.invoke(
        {"messages": [HumanMessage(content="Write a 3-sentence product description for a wireless ergonomic keyboard.")],
         "draft": "", "approved": False, "revision_count": 0},
        config=config
    )

    # Loop until approved or max revisions
    while True:
        state = graph_c.get_state(config)
        if not state.next:
            break

        for task in state.tasks:
            if task.interrupts:
                feedback = input("\nYour feedback (or 'approve'): ").strip()
                result = graph_c.invoke(Command(resume=feedback), config=config)
                break
        else:
            break

    print(f"\n✅ Approved content:\n{result.get('draft', result['messages'][-1].content)}")


# =============================================================
# MAIN — Run all three patterns
# =============================================================

if __name__ == "__main__":
    print("Which HITL pattern do you want to demo?")
    print("  A — Approve/Reject tool execution")
    print("  B — Human provides missing info")
    print("  C — Human reviews and edits output")
    choice = input("Choice (A/B/C): ").strip().upper()

    if choice == "A":
        run_pattern_a()
    elif choice == "B":
        run_pattern_b()
    elif choice == "C":
        run_pattern_c()
    else:
        print("Running all patterns...")
        run_pattern_a()
        run_pattern_b()
        run_pattern_c()


# =============================================================
# EXERCISE:
#   1. Build a "budget approval" workflow:
#      - Agent proposes a purchase (item, cost)
#      - If cost > $500 → require human approval (Pattern A)
#      - If cost <= $500 → auto-approve
#   2. Test with: "Buy a monitor for $450" and "Buy a server for $5000"
# =============================================================
