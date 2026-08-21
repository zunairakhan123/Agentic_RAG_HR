"""
LangGraph state schema for the NextBridge HR Agent.

Tracks:
- Conversation/tool state
- Adaptive query routing
- CRAG retrieval and grading
- Query rewriting
- Self-RAG generation/reflection
- Correction-loop telemetry
"""

from typing import Annotated, TypedDict, Optional, List, Dict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state for the HR Agent LangGraph workflow."""

    # =========================================================
    # 1. Conversation & Existing Tool State
    # =========================================================

    messages: Annotated[List[BaseMessage], add_messages]

    email_draft: Optional[Dict[str, str]]
    awaiting_approval: bool

    # =========================================================
    # 2. Query / Routing State
    # =========================================================

    query: str
    query_type: Optional[str]
    # Examples:
    # "simple"
    # "complex"
    # "email"
    query_variants: List[str]

    # =========================================================
    # 3. Retrieval State
    # =========================================================

    documents: List[Document]

    retrieval_attempts: int

    # Identifies why the current retrieval was rejected.
    retrieval_failure_reason: Optional[str]

    # =========================================================
    # 4. CRAG Grading State
    # =========================================================

    documents_relevant: bool

    # Optional textual explanation from the grader.
    retrieval_grade_reason: Optional[str]

    # =========================================================
    # 5. Query Correction State
    # =========================================================

    rewritten_query: Optional[str]

    correction_reason: Optional[str]

    # =========================================================
    # 6. Generation State
    # =========================================================

    generation: Optional[str]

    generation_attempts: int

    # =========================================================
    # 7. Self-RAG Reflection State
    # =========================================================

    answer_grounded: bool

    reflection_reason: Optional[str]

    # =========================================================
    # 8. Final Routing / Status
    # =========================================================

    final_status: Optional[str]
    # Examples:
    # "success"
    # "retrieval_failed"
    # "generation_failed"
    # "guardrail_blocked"