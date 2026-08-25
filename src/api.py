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
from src.graph import supervisor_builder
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.cache import check_semantic_cache,init_frequency_db
import uvicorn
import sys
import asyncio
import warnings
# Suppress harmless Pydantic V2 serialization warnings from LangChain structured outputs
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

hr_graph = None
_memory_context = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global hr_graph, _memory_context
    
    # 1. Initialize the Persistent Frequency Cache DB
    await init_frequency_db()
    
    # 2. Initialize the async SQLite checkpointer on startup
    _memory_context = AsyncSqliteSaver.from_conn_string("hr_agent_memory.db")
    memory = await _memory_context.__aenter__()
    await memory.setup()
    
    # 3. Compile the graph
    hr_graph = supervisor_builder.compile(checkpointer=memory)
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
    full_response = "" 

    async for event in hr_graph.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        version="v2",
    ):
        kind = event["event"]
        # Determine exactly which node generated this event
        node_name = event.get("metadata", {}).get("langgraph_node", "")

        # 1. Stream ONLY the actual user-facing LLM nodes (hides internal JSON)
        if kind == "on_chat_model_stream" and node_name in ["generation", "agent"]:
            chunk = event["data"]["chunk"]
            content = chunk.content if hasattr(chunk, "content") else chunk
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get("text", "")
            if content:
                full_response += content 
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
        # [NEW FIX]: Trigger UI wipe when Self-RAG starts a new draft
        elif kind == "on_chat_model_start" and node_name == "generation":
            yield f"data: {json.dumps({'type': 'clear'})}\n\n"

        # 1b. Catch the Guardrail rejection and stream the polite message
        elif kind == "on_chain_end" and node_name == "guardrail":
            output = event["data"].get("output", {})
            if isinstance(output, dict) and output.get("final_status") == "guardrail_blocked":
                msg = output["messages"][0].content
                full_response += msg
                yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"

        # 2. Stream ReAct Tool executions
        elif kind == "on_tool_start":
            tool_name = event["name"]
            tool_inputs = event["data"].get("input")
            yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name, 'input': tool_inputs})}\n\n"

        elif kind == "on_tool_end":
            output = event["data"].get("output")
            yield f"data: {json.dumps({'type': 'tool_end', 'output': str(output)})}\n\n"
            
        # 3. Stream Node Transitions for UI Status
        elif kind == "on_chain_start":
            tracked_nodes = [
                "guardrail", "router", "rag_wrapper", "agent", 
                "rag_router", "retrieve", "grader", "rewrite", "reflection"
            ]
            if node_name in tracked_nodes:
                # Map internal names to professional UI labels
                pretty_names = {
                    "guardrail": "Input Guardrail",
                    "router": "Supervisor Routing",
                    "rag_wrapper": "Initializing RAG",
                    "agent": "ReAct Agent",
                    "rag_router": "Analyzing Complexity",
                    "retrieve": "Retrieving Documents",
                    "grader": "Grading Context",
                    "rewrite": "Optimizing Query",
                    "reflection": "Validating Answer"
                }
                display_name = pretty_names.get(node_name, node_name)
                status_msg = f"Executing {display_name}..."
                yield f"data: {json.dumps({'type': 'status', 'content': status_msg})}\n\n"

    # [FIX]: Removed the old save_to_semantic_cache() block!
    # Caching is now handled entirely in the background by track_and_promote.
    yield "data: [DONE]\n\n"

# In-memory queue to hold proactive agent messages for the frontend
pending_notifications: Dict[str, List[str]] = {}

async def fake_stream_generator(text: str):
    """Simulates the SSE stream to instantly render cached responses."""
    yield f"data: {json.dumps({'type': 'status', 'content': '⚡ Served instantly from Semantic Cache'})}\n\n"
    await asyncio.sleep(0.1) # Tiny pause for UI to catch the status
    yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"
    yield "data: [DONE]\n\n"

# ==========================================
# Endpoints for Frontend & Webhooks
# ==========================================

@app.get("/api/notifications/{thread_id}")
async def get_notifications(thread_id: str):
    """Frontend polls this endpoint to fetch and clear unread webhook messages."""
    # .pop() ensures the message is pulled once and instantly cleared from the queue
    msgs = pending_notifications.pop(thread_id, [])
    return {"notifications": msgs}


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Check cache first
    cached_answer = check_semantic_cache(request.message)
    if cached_answer:
        # If cache hit, stream the cached answer instantly and skip LangGraph
        return StreamingResponse(
            fake_stream_generator(cached_answer),
            media_type="text/event-stream"
        )
    
    # 2. If no cache, run the LangGraph event generator
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
                
                # Stream Tool Execution status (Emails, Web)
                elif event["event"] == "on_tool_start":
                    tool_name = event["name"]
                    await websocket.send_text(json.dumps({"type": "status", "content": f"Checking {tool_name}..."}))
                    
                # [NEW] Stream CRAG Node Execution status (RAG)
                elif event["event"] == "on_chain_start":
                    node_name = event["name"]
                    if node_name in ["retrieval", "grader", "rewrite", "reflection"]:
                        await websocket.send_text(json.dumps({"type": "status", "content": f"Running {node_name} phase..."}))
            
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


if __name__ == "__main__":
    
    # Windows asyncio policy fix to prevent EventLoop errors with SQLite/FastAPI
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)