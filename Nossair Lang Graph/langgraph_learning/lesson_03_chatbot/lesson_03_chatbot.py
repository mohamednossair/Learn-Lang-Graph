# =============================================================
# LESSON 3 — Chatbot with Memory using MessagesState + Ollama
# =============================================================
#
# WHAT YOU WILL LEARN:
#   - MessagesState — LangGraph's built-in state for chat history
#   - How to connect a real LLM (Ollama) as a node
#   - How conversation history is preserved across turns
#   - How to build a multi-turn chatbot in ~30 lines
#
# PREREQUISITES:
#   - Ollama installed and running: https://ollama.com
#   - Model pulled: ollama pull llama3
#
# MENTAL MODEL:
#   START → chatbot_node → END
#           (LLM reads full message history, appends reply)
# =============================================================

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


# -------------------------------------------------------------
# STEP 1 — Define State using MessagesState pattern
#
# The 'messages' field uses Annotated + add_messages reducer.
# The reducer means: instead of replacing messages, it APPENDS.
# This is how chat history is preserved!
# -------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# -------------------------------------------------------------
# STEP 2 — Set up Ollama LLM
#
# ChatOllama connects to your local Ollama instance.
# Change model="llama3" to any model you have pulled.
# -------------------------------------------------------------

llm = ChatOllama(model="llama3", temperature=0.7)


# -------------------------------------------------------------
# STEP 3 — The Chatbot Node
#
# It receives the full message history in state["messages"],
# calls the LLM, and returns the AI response.
# add_messages reducer automatically appends it to history.
# -------------------------------------------------------------

def chatbot_node(state: ChatState) -> dict:
    print(f"[chatbot] Processing {len(state['messages'])} message(s)...")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# -------------------------------------------------------------
# STEP 4 — Build the Graph (simple: 1 node!)
# -------------------------------------------------------------

graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


# -------------------------------------------------------------
# STEP 5 — Run the Multi-turn Chatbot
# -------------------------------------------------------------

def chat(user_input: str, history: list) -> tuple[str, list]:
    """Send a message and get a reply. Maintains history."""
    history.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages": history})
    updated_history = result["messages"]
    ai_reply = updated_history[-1].content
    return ai_reply, updated_history


if __name__ == "__main__":
    print("=" * 55)
    print("LangGraph Chatbot with Ollama")
    print("Type 'quit' to exit, 'history' to see conversation")
    print("=" * 55)

    conversation_history = []

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "history":
            print("\n--- Conversation History ---")
            for msg in conversation_history:
                role = "You" if isinstance(msg, HumanMessage) else "AI"
                print(f"  {role}: {msg.content[:80]}...")
            continue

        print("AI: ", end="", flush=True)
        reply, conversation_history = chat(user_input, conversation_history)
        print(reply)


# =============================================================
# EXERCISE:
#   1. Add a "system_prompt" key to ChatState (str type)
#   2. Before calling the LLM, prepend a SystemMessage with
#      the system_prompt to the messages list
#   3. Test with: system_prompt = "You are a pirate. Always
#      respond in pirate language."
# =============================================================
