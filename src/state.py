"""
LangGraph state schema for tracking message history and pending action states.
"""

from typing import Annotated, TypedDict, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Represents the shared state of the HR Agent conversation thread."""

    messages: Annotated[List[BaseMessage], add_messages]  # When a new node outputs a message, do not overwrite the existing message list. Append the new message to the end of the list.
    email_draft: Optional[Dict[str, str]]  # Stores {'department': ..., 'to': ..., 'subject': ..., 'body': ...}
    awaiting_approval: bool                # Halts state when True until user confirms