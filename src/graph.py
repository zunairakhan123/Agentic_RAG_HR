
"""
Multi-Agent Hierarchical StateGraph for the NextBridge HR Agent.
"""
import os
import asyncio
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, trim_messages, AIMessage
from langchain_core.runnables import RunnableConfig

from src.state import SupervisorState, RAGState
from langchain_groq import ChatGroq
from src.tools import guardrailed_web_search, draft_department_email, send_department_email
from src.cache import track_and_promote
from src.nodes import (
    input_guardrail_node, adaptive_router_node,
    retrieval_node, document_grader_node, rewrite_query_node,
    generation_node, reflection_node,rag_router_node
)

load_dotenv()


# ==========================================
#  Initialize Primary LLM (For ReAct Agent)
# ==========================================

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0.1
)
tools = [guardrailed_web_search, draft_department_email, send_department_email]
llm_with_tools = llm.bind_tools(tools).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

# ==========================================
# 1. ReAct Agent Node (For Emails/Web Search/Chat)
# ==========================================
async def agent_node(state: SupervisorState):
    query_type = state.get("query_type", "chat")
    user_message = state["messages"][-1].content
    
    # 1. CIRCUIT BREAKER: Count how many times the tool has returned data
    tool_executions = sum(1 for msg in state["messages"] if getattr(msg, "type", "") == "tool")
    
    active_tools = []
    mode_instruction = ""
    
    if query_type == "web":
        if tool_executions >= 2:
            active_tools = []
            mode_instruction = "You have searched the web enough. Summarize the final answer based on the conversation history."
        else:
            active_tools = [guardrailed_web_search]
            mode_instruction = (
                "Use the `guardrailed_web_search` tool to find external company information , current news, personnel listings ,products and services. "
                "CRITICAL GROUNDING RULES: "
                "1. You MUST ONLY use facts provided in the tool's return snippet. "
                "2. If the tool snippet does not contain the exact name or answer, DO NOT guess or use your internal knowledge. "
                "3. We are Nextbridge Pvt. Ltd. (a software/IT company). Ignore any search results about oil, hydrocarbons, or global conglomerates. "
                "4. Once search results are returned, answer immediately and DO NOT call the tool again."
                "5. Don't use your inner knowledge or hallucinate. Only use the tool's output."
            )
            
    # [PRODUCTION FIX]: Stop Tool Shortcut Hallucination
    elif query_type == "email" or query_type == "execute_pending_action":
        
        # Condition 1: UI Button clicked (injects explicit phrase) OR State Validator passed ("yes")
        if "I explicitly approve this draft" in user_message or query_type == "execute_pending_action":
            active_tools = [send_department_email]
            mode_instruction = "The user has approved the email. You MUST call `send_department_email` immediately to dispatch it. Do not draft anything."
        
        # Condition 2: Initial user request (even if they use the word "send")
        else:
            active_tools = [draft_department_email]
            mode_instruction = "Draft the email using `draft_department_email`. YOU DO NOT HAVE PERMISSION TO SEND IT YET. Draft it, present it to the user, and wait for confirmation."
            
    else: # "chat"
        active_tools = []
        mode_instruction = "You are in pure chat mode. NO TOOLS ARE AVAILABLE."

    system_instruction = SystemMessage(
        content=f"You are the official NextBridge HR AI Assistant.\nCURRENT MODE: {mode_instruction}\n"
    )
    
    trimmed = trim_messages(state["messages"], max_tokens=12, strategy="last", token_counter=len, start_on="human")
    messages = [system_instruction] + trimmed
    
    if active_tools:
        specialized_llm = llm.bind_tools(active_tools).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
    else:
        specialized_llm = llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
        
    # [PRODUCTION FIX]: Graceful Fallback for Inference Crashes
    try:
        response = await specialized_llm.ainvoke(messages)
    except Exception as e:
        print(f"\n[Agent Warning] Inference or Tool Validation Error: {e}")
        
        # If Groq crashes because the LLM tried to hallucinate the send tool, 
        # gracefully return a text message guiding the user back to the UI.
        from langchain_core.messages import AIMessage
        response = AIMessage(
            content="I am not authorized to send the email directly based on that command. "
                    "Please review the draft above and use the **Approve & Send** button to dispatch it securely."
        )
    # [NEW STATE MANAGEMENT]: Tell the graph to wait for approval if a draft was just generated
    new_workflow_state = "idle"
    
    # Check if the agent just successfully used the draft tool in recent history
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "draft_department_email":
            new_workflow_state = "awaiting_approval"
            break
            
    # If the user just explicitly approved it (or button was clicked), reset back to idle
    if "I explicitly approve this draft" in user_message or query_type == "execute_pending_action":
        new_workflow_state = "idle"
        
    return {
        "messages": [response],
        "workflow_state": new_workflow_state
    }

# ==========================================
# 2. Compile the CRAG Subgraph (Child Graph)
# ==========================================
def route_after_retrieval(state: RAGState) -> str:
    """If simple, skip the Grader and go straight to Generation."""
    if state.get("query_type") == "simple":
        return "generation"
    return "grader"

def route_after_grading(state: RAGState) -> str:
    if state.get("documents_relevant") or state.get("retrieval_attempts", 0) >= 3:
        return "generation"
    return "rewrite"

def route_after_generation(state: RAGState) -> str:
    """If simple, we bypass the heavy Reflection guardrail for speed."""
    if state.get("query_type") == "simple":
        return "end"
    return "reflection"

def route_after_reflection(state: RAGState) -> str:
    if state.get("answer_grounded") or state.get("generation_attempts", 0) >= 3 or state.get("retrieval_attempts", 0) >= 3:
        return "end"
    if state.get("reflection_error_type") == "missing_evidence":
        return "rewrite"
    return "regenerate"

# ----------------------------------------
# 3. Build the RAG Subgraph
# ----------------------------------------
from src.nodes import rag_router_node # Ensure you import the new node

rag_builder = StateGraph(RAGState)
rag_builder.add_node("rag_router", rag_router_node)
rag_builder.add_node("retrieve", retrieval_node)
rag_builder.add_node("grader", document_grader_node)
rag_builder.add_node("rewrite", rewrite_query_node)
rag_builder.add_node("generation", generation_node)
rag_builder.add_node("reflection", reflection_node)

# Entry point is now the internal RAG router
rag_builder.set_entry_point("rag_router")

rag_builder.add_edge("rag_router", "retrieve")
rag_builder.add_conditional_edges("retrieve", route_after_retrieval, 
                                  {"generation": "generation",
                                    "grader": "grader"})

rag_builder.add_conditional_edges("grader", route_after_grading, 
                                  {"generation": "generation",
                                    "rewrite": "rewrite"})
rag_builder.add_edge("rewrite", "retrieve")

rag_builder.add_conditional_edges("generation", route_after_generation,
                                   {"end": END, 
                                    "reflection": "reflection"})
rag_builder.add_conditional_edges("reflection", route_after_reflection, 
                                  {"end": END, 
                                   "regenerate": "generation",
                                    "rewrite": "rewrite"})

rag_graph = rag_builder.compile()

# ==========================================
# 3. The Wrapper Node (State Translator)
# ==========================================
async def rag_wrapper_node(state: SupervisorState, config: RunnableConfig) -> dict:
    """Translates states, executes subgraph, and fires background cache promotion."""
    user_query = state["messages"][-1].content
    query_type = state.get("query_type", "rag") # Default to RAG if not specified
    retriever_strategy = state.get("retriever_strategy") # <--- 1. Extract

    # 1. Translate DOWN to child state
    initial_child_state = {
        "query": user_query,
        "query_type": query_type,
        "retriever_strategy": retriever_strategy,        # <--- 2. Inject
        "rewritten_query": "",
        "documents": [],
        "retrieval_attempts": 0,
        "generation_attempts": 0,
        "answer_grounded": False,
        "generation": ""
    }
    
    # 2. Execute Subgraph (Passing config ensures SSE streams continue working!)
    result_state = await rag_graph.ainvoke(initial_child_state, config)
    
    final_answer = result_state.get("generation", "I could not find a grounded answer.")
    
    # 3. Fire-and-forget Late Cache Promotion
    asyncio.create_task(track_and_promote(user_query, final_answer, result_state))
    
    # 4. Translate UP to parent state
    return {"messages": [AIMessage(content=final_answer)]}


# ==========================================
# 4. Compile the Supervisor Graph (Parent Graph)
# ==========================================
def route_after_guardrail(state: SupervisorState) -> str:
    if state.get("final_status") == "guardrail_blocked":
        return "end"
    return "router"

def route_after_router(state: SupervisorState) -> str:
    # [UPDATE]: Added "web" so it routes to the ReAct agent
    if state.get("query_type") in ["email", "chat", "web"]:
        return "agent"
    return "rag_wrapper"

def route_tools(state: SupervisorState) -> str:
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tools"
    return "end"

supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("guardrail", input_guardrail_node)
supervisor_builder.add_node("router", adaptive_router_node)
supervisor_builder.add_node("agent", agent_node)
supervisor_builder.add_node("tools", ToolNode(tools))
supervisor_builder.add_node("rag_wrapper", rag_wrapper_node) # The nested graph

supervisor_builder.set_entry_point("guardrail")
supervisor_builder.add_conditional_edges("guardrail", route_after_guardrail, {"end": END, "router": "router"})
supervisor_builder.add_conditional_edges("router", route_after_router, {"agent": "agent", "rag_wrapper": "rag_wrapper"})
supervisor_builder.add_conditional_edges("agent", route_tools, {"tools": "tools", "end": END})
supervisor_builder.add_edge("tools", "agent")
supervisor_builder.add_edge("rag_wrapper", END)

graph = supervisor_builder.compile()