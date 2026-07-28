"""
LangGraph state machine topology and nodes.
"""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from src.state import AgentState
from src.tools import (
    search_hr_documents,
    guardrailed_web_search,
    draft_department_email,
    send_department_email,
)

# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

tools = [
    search_hr_documents,
    guardrailed_web_search,
    draft_department_email,
    send_department_email,
]
llm_with_tools = llm.bind_tools(tools)

# 1. Add 'async' to the function definition
async def agent_node(state: AgentState):
    """Primary reasoning node applying guardrails and state decisions."""
    system_instruction = SystemMessage(
        content=(
            "You are the official NextBridge HR AI Assistant.\n\n"
            "OPERATIONAL RULES:\n"
            "1. HR POLICIES: Always check local HR documents first using `search_hr_documents`.\n"
            "2. EMAIL WORKFLOWS (Leaves, Complaints, Meals):\n"
            "   - First, answer the user's inquiry using policy documents.\n"
            "   - If the user requests to send an email, call `draft_department_email`.\n"
            "   - Present the exact drafted email (To, Subject, Body) to the user and ask for explicit verification.\n"
            "   - DO NOT call `send_department_email` until the user explicitly confirms.\n"
            "3. DEPARTMENT REPLIES:\n"
            "   - When notified of a response from a department, summarize the response for the user.\n"
            "   - Ask the user if they wish to send an acknowledgment email.\n"
            "4. GUARDRAILS:\n"
            "   - You ONLY discuss NextBridge, HR policy, engineering, and work contexts.\n"
            "   - For non-work queries (e.g., fashion, sports), politely decline."
        )
    )
    messages = [system_instruction] + state["messages"]
    
    response = await llm_with_tools.ainvoke(messages)
    
    return {"messages": [response]}

def route_tools(state: AgentState) -> str:
    """Determines whether to route to a tool or end the interaction."""
    last_message = state["messages"][-1]
    
    # [FIX]: Safely check for tool_calls to handle both AIMessage and HumanMessage types
    if getattr(last_message, "tool_calls", None):
        return "tools"
    
    return "__end__"

# Export the uncompiled builder
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", route_tools, ["tools", END])
builder.add_edge("tools", "agent")