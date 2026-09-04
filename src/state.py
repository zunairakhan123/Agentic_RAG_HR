"""
LangGraph state schemas for the NextBridge HR Agent.
Implements the Multi-Agent Hierarchical Pattern by isolating 
lightweight orchestration state from heavy RAG logic state.
"""

from typing import Annotated, TypedDict, Optional, List, Dict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


# =========================================================
# 1. Parent Graph State (Supervisor)
# =========================================================
class SupervisorState(TypedDict):
    """Ultra-lightweight state for orchestration, conversation history, and HITL."""
    
    messages: Annotated[List[BaseMessage], add_messages]

    # Email / HITL State (Must persist across checkpoints)
    email_draft: Optional[Dict[str, str]]
    awaiting_approval: bool

    # Routing Status
    query_type: Optional[str]  # e.g., "simple", "complex", "email", "chat"
    final_status: Optional[str] # e.g., "success", "guardrail_blocked"

    retriever_strategy: Optional[str]

# =========================================================
# 2. Child Graph State (CRAG Subgraph)
# =========================================================
class RAGState(TypedDict):
    """Heavy, ephemeral state strictly for the CRAG loop. Destroyed after execution to save memory."""
    
    query: str
    query_type: str 
    intent_key: Optional[str]  
    query_variants: List[str]

    # Retrieval State
    documents: List[Document]
    retrieval_attempts: int
    retriever_strategy: Optional[str]
    retrieval_failure_reason: Optional[str]
    # [FIX] Use a dedicated string to overwrite drafts, NOT a list!
    final_generation: str

    # CRAG Grading State
    documents_relevant: bool
    retrieval_grade_reason: Optional[str]

    # Query Correction State
    rewritten_query: Optional[str]
    correction_reason: Optional[str]

    # Generation State
    generation: Optional[str]
    generation_attempts: int

    # Self-RAG Reflection State
    answer_grounded: bool
    reflection_error_type: Optional[str]
    reflection_reason: Optional[str]