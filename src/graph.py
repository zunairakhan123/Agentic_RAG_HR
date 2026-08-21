"""
LangGraph StateGraph for the NextBridge HR Agent.
Unites the ReAct Tool-Calling Agent (Emails/Web) with the CRAG Pipeline (Policy Retrieval).
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, trim_messages
from src.state import AgentState
from src.tools import (
    guardrailed_web_search,
    draft_department_email,
    send_department_email,
)
from src.nodes import (
    input_guardrail_node,
    adaptive_router_node,
    retrieval_node,
    document_grader_node,
    rewrite_query_node,
    generation_node,
    reflection_node
)

# ==========================================
# 1. Initialize LLM & Tools (ReAct Agent)
# ==========================================

llm = ChatOpenAI(
    base_url="https://relation-creature-tap-bradley.trycloudflare.com/v1",
    api_key="not-needed", 
    model="qwen3:30b",  
    temperature=0.1, 
    max_retries=5,       
    timeout=500.0        
)

# Notice: RAG is removed from tools. It is handled by the graph now.
tools = [
    guardrailed_web_search,
    draft_department_email,
    send_department_email,
]

llm_with_tools = llm.bind_tools(tools).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True 
)

# ==========================================
# 2. ReAct Agent Node (Conversations & Emails)
# ==========================================

async def agent_node(state: AgentState):
    """Primary reasoning node for conversations, emails, and web search."""
    system_instruction = SystemMessage(
        content=(
            "You are the official NextBridge HR AI Assistant.\n\n"
            "OPERATIONAL RULES:\n"
            "1. EMAIL WORKFLOWS (Leaves, Complaints, Meals):\n"
            "   - If the user requests to send an email, call `draft_department_email`.\n"
            "   - Present the exact drafted email (To, Subject, Body) to the user and ask for explicit verification.\n"
            "   - DO NOT call `send_department_email` until the user explicitly confirms.\n"
            "2. DEPARTMENT REPLIES:\n"
            "   - When notified of a response from a department, summarize the response for the user.\n"
        )
    )
    
    trimmed_messages = trim_messages(
        state["messages"],
        max_tokens=10, 
        strategy="last",
        token_counter=len, 
        allow_partial=False,
        start_on="human",
    )
    messages = [system_instruction] + trimmed_messages
    
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


# ==========================================
# 3. Conditional Edge Functions
# ==========================================

def route_after_guardrail(state: AgentState) -> str:
    if state.get("final_status") == "guardrail_blocked":
        return "end"
    return "router"

def route_after_router(state: AgentState) -> str:
    """Traffic Cop: Directs to CRAG (RAG) or Agent (Emails/Chat)"""
    query_type = state.get("query_type", "complex")
    
    # [FIX]: Both 'email' and casual 'chat' bypass the heavy CRAG retrieval loop
    if query_type in ["email", "chat"]:
        return "agent" # Send to fast ReAct conversational pathway
        
    return "retrieve"  # Send to heavy CRAG policy pathway

def route_tools(state: AgentState) -> str:
    """Determines whether the ReAct agent called a tool."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"

def route_after_grading(state: AgentState) -> str:
    if state.get("documents_relevant"):
        return "generate"
    if state.get("retrieval_attempts", 0) >= 3:
        return "generate"
    return "rewrite"

def route_after_reflection(state: AgentState) -> str:
    """Self-RAG logic with advanced error-type routing."""
    # 1. If grounded, we are done
    if state.get("answer_grounded"):
        return "end"
    
    # 2. Check limits to prevent infinite loops
    if state.get("generation_attempts", 0) >= 3 or state.get("retrieval_attempts", 0) >= 3:
        return "end"
    
    # 3. Dynamic Routing based on error type
    error_type = state.get("reflection_error_type", "wording_problem")
    
    if error_type == "missing_evidence":
        # The documents were actually bad. Kick back to retrieval via Rewrite.
        return "rewrite"
    else:
        # The documents are fine, the LLM just phrased it poorly/hallucinated.
        return "regenerate"

# ==========================================
# 4. Build the Unified Super-Graph
# ==========================================

builder = StateGraph(AgentState)

# Add all nodes
builder.add_node("guardrail", input_guardrail_node)
builder.add_node("router", adaptive_router_node)
builder.add_node("retrieve", retrieval_node)
builder.add_node("grader", document_grader_node)
builder.add_node("rewrite", rewrite_query_node)
builder.add_node("generation", generation_node)
builder.add_node("reflection", reflection_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

# Entry point is always the Guardrail
builder.set_entry_point("guardrail")

# Guardrail -> Router
builder.add_conditional_edges("guardrail", route_after_guardrail, {"end": END, "router": "router"})

# Router -> splits between RAG and Agent
builder.add_conditional_edges("router", route_after_router, {
    "agent": "agent",
    "retrieve": "retrieve"
})

# --- The ReAct (Agent) Pathway ---
builder.add_conditional_edges("agent", route_tools, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

# --- The CRAG (RAG) Pathway ---
builder.add_edge("retrieve", "grader")
builder.add_conditional_edges("grader", route_after_grading, {
    "generate": "generation",
    "rewrite": "rewrite"
})
builder.add_edge("rewrite", "retrieve") # Cyclic back to retrieval

builder.add_edge("generation", "reflection")
builder.add_conditional_edges(
    "reflection",
    route_after_reflection,
    {
        "end": END,
        "regenerate": "generation", # Loop back to Generation
        "rewrite": "rewrite"        # Deep loop back to Rewrite -> Retrieval
    }
)

# Compile and export
graph = builder.compile()