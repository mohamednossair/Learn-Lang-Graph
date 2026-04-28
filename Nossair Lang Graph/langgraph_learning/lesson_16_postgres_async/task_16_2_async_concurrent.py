"""Task 16.2 — Async Concurrent Execution."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_ollama_model
import asyncio, time
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model=get_ollama_model(), temperature=0)

async def async_chat_node(state: ChatState):
    """Async node — uses ainvoke instead of invoke."""
    resp = await llm.ainvoke(state["messages"])
    return {"messages": [resp]}

builder = StateGraph(ChatState)
builder.add_node("chat", async_chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
async_graph = builder.compile(checkpointer=MemorySaver())

async def sequential_demo():
    """Sequential: time_A + time_B + time_C."""
    print("--- Sequential (await one by one) ---")
    start = time.time()
    
    for i, q in enumerate(["What is Python?", "What is Docker?", "What is REST?"]):
        result = await async_graph.ainvoke(
            {"messages": [HumanMessage(content=q)]},
            config={"configurable": {"thread_id": f"seq-{i}"}}
        )
        print(f"  Q{i+1}: {result['messages'][-1].content[:60]}...")
    
    elapsed = time.time() - start
    print(f"  Sequential time: {elapsed:.2f}s")
    return elapsed

async def concurrent_demo():
    """Concurrent: max(time_A, time_B, time_C) via asyncio.gather."""
    print("\n--- Concurrent (asyncio.gather) ---")
    start = time.time()
    
    results = await asyncio.gather(
        async_graph.ainvoke(
            {"messages": [HumanMessage(content="What is Python?")]},
            config={"configurable": {"thread_id": "con-0"}}
        ),
        async_graph.ainvoke(
            {"messages": [HumanMessage(content="What is Docker?")]},
            config={"configurable": {"thread_id": "con-1"}}
        ),
        async_graph.ainvoke(
            {"messages": [HumanMessage(content="What is REST?")]},
            config={"configurable": {"thread_id": "con-2"}}
        ),
    )
    
    for i, r in enumerate(results):
        print(f"  Q{i+1}: {r['messages'][-1].content[:60]}...")
    
    elapsed = time.time() - start
    print(f"  Concurrent time: {elapsed:.2f}s")
    return elapsed

async def wrong_sync_in_async():
    """Anti-pattern: sync invoke() in async endpoint blocks event loop."""
    print("\n--- Anti-pattern: sync invoke in async context ---")
    print("  ❌ graph.invoke() blocks the event loop for 2-10s")
    print("  ✅ Use await graph.ainvoke() or asyncio.to_thread()")
    # Correct alternative:
    # result = await asyncio.to_thread(graph.invoke, state, config)

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 16.2 — ASYNC CONCURRENT EXECUTION")
    print("=" * 50)
    
    seq_time = asyncio.run(sequential_demo())
    con_time = asyncio.run(concurrent_demo())
    asyncio.run(wrong_sync_in_async())
    
    print(f"\n{'=' * 50}")
    print(f"Sequential: {seq_time:.2f}s")
    print(f"Concurrent: {con_time:.2f}s")
    if seq_time > 0:
        print(f"Speedup: {seq_time/con_time:.1f}x")
    print("✅ Async execution working!")
