"""Task 15.1 — FastAPI with LangGraph agent."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage
from lesson_04_tools_agent.lesson_04_tools_agent import graph

app = FastAPI(title="LangGraph Agent API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.get("/health")
def health():
    return {"status": "healthy", "service": "LangGraph Agent API"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.thread_id}}
        )
        response_text = result["messages"][-1].content
        return ChatResponse(response=response_text, thread_id=request.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting API server... Visit http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
