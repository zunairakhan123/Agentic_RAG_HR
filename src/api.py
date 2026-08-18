"""
FastAPI application layer supporting SSE response streaming and inbound email webhooks.
"""
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph import builder
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

hr_graph = None
_memory_context = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global hr_graph, _memory_context
    
    # Initialize the async SQLite checkpointer on startup
    _memory_context = AsyncSqliteSaver.from_conn_string("hr_agent_memory.db")
    memory = await _memory_context.__aenter__()
    await memory.setup()
    
    # The agent can now freely search PDFs and draft emails. 
    # Sending emails is protected by the LLM's system prompt guardrails.
    hr_graph = builder.compile(checkpointer=memory)
    yield
    
    # Clean up the DB connection on server shutdown
    await _memory_context.__aexit__(None, None, None)

# Pass the lifespan context manager to FastAPI
app = FastAPI(title="NextBridge Agentic HR System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class EmailApprovalRequest(BaseModel):
    thread_id: str
    approved: bool

class DepartmentReplyWebhook(BaseModel):
    thread_id: str
    department: str
    reply_body: str

async def event_generator(thread_id: str, message: str):
    config = {"configurable": {"thread_id": thread_id}}

    async for event in hr_graph.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        elif kind == "on_tool_start":
            tool_name = event["name"]
            tool_inputs = event["data"].get("input")
            yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name, 'input': tool_inputs})}\n\n"

        elif kind == "on_tool_end":
            output = event["data"].get("output")
            yield f"data: {json.dumps({'type': 'tool_end', 'output': str(output)})}\n\n"

    yield "data: [DONE]\n\n"

# In-memory queue to hold proactive agent messages for the frontend
pending_notifications: Dict[str, List[str]] = {}

@app.get("/api/notifications/{thread_id}")
async def get_notifications(thread_id: str):
    """Frontend polls this endpoint to fetch and clear unread webhook messages."""
    # .pop() ensures the message is pulled once and instantly cleared from the queue
    msgs = pending_notifications.pop(thread_id, [])
    return {"notifications": msgs}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        event_generator(request.thread_id, request.message),
        media_type="text/event-stream",
    )

@app.post("/api/approve-action")
async def approve_action_endpoint(request: EmailApprovalRequest):
    """
    Handles dynamic HITL authorization and injects the thread_id into the LLM's context.
    """
    if request.approved:
        # [FIX]: Dynamically inject the thread_id so the LLM can pass it to the SMTP tool
        auth_message = (
            f"I explicitly approve this draft. Please dispatch the email immediately "
            f"using the `send_department_email` tool. "
            f"CRITICAL: You must pass this exact string as the thread_id argument: {request.thread_id}"
        )
    else:
        auth_message = "I cancel this request. Do not send the email."

    return StreamingResponse(
        event_generator(request.thread_id, auth_message),
        media_type="text/event-stream",
    )

@app.post("/webhook/department-reply")
async def department_reply_webhook(data: DepartmentReplyWebhook):
    config = {"configurable": {"thread_id": data.thread_id}}

    # 1. Craft a system prompt forcing the LLM to process the external reply
    notification = (
        f"SYSTEM NOTIFICATION: The {data.department} department has replied to your request.\n"
        f"Reply Content: '{data.reply_body}'\n\n"
        f"Task: Notify the user about this response immediately. Ask if they want to send an acknowledgment email back."
    )

    # 2. Inject this into the LangGraph state (saves to SQLite)
    from langchain_core.messages import HumanMessage
    await hr_graph.aupdate_state(config, {"messages": [HumanMessage(content=notification)]})
    
    # 3. Trigger the LLM to generate a response based on the injected message
    result = await hr_graph.ainvoke(None, config)
    agent_msg = result["messages"][-1].content

    # 4. Push the generated response to the polling queue for the frontend to catch
    if data.thread_id not in pending_notifications:
        pending_notifications[data.thread_id] = []
    pending_notifications[data.thread_id].append(agent_msg)

    return {
        "status": "User notified successfully",
        "thread_id": data.thread_id
    }

# Replace your Pipecat WebSocket with this:
@app.websocket("/ws/voice/{thread_id}")
async def voice_websocket(websocket: WebSocket, thread_id: str):
    """Persistent bidirectional connection for Voice Mode."""
    await websocket.accept()
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        while True:
            # 1. Wait for transcribed text from the frontend
            user_text = await websocket.receive_text()
            print(f"[Voice Mode] Received: {user_text}")
            
            # 2. Execute LangGraph and stream tokens
            async for event in hr_graph.astream_events(
                {"messages": [HumanMessage(content=user_text)]}, 
                config=config, 
                version="v2"
            ):
                # Stream LLM tokens
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if isinstance(chunk, list) and len(chunk) > 0:
                        chunk = chunk[0].get("text", "")
                        
                    if chunk:
                        await websocket.send_text(json.dumps({"type": "token", "content": chunk}))
                
                # Stream Tool Execution status
                elif event["event"] == "on_tool_start":
                    tool_name = event["name"]
                    await websocket.send_text(json.dumps({"type": "status", "content": f"Checking {tool_name}..."}))
            
            # 3. Signal that the AI has finished its turn
            await websocket.send_text(json.dumps({"type": "done"}))
            
    except WebSocketDisconnect:
        print(f"[Voice Mode] Session {thread_id} disconnected normally.")
    except Exception as e:
        print(f"[Voice Mode] ERROR: {str(e)}")
        await websocket.send_text(json.dumps({"type": "error", "content": "An error occurred while processing your request."}))
# ==========================================
# Frontend Serving
# ==========================================
# 1. Serve the main index.html on the root URL
@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

# 2. (Optional) Mount the frontend directory if you add CSS/JS files later
# app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)