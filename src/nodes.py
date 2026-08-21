"""
LangGraph nodes for the NextBridge HR Agent.
Contains the Guardrail and Adaptive Router logic using Pydantic structured outputs.
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from src.state import AgentState
from src.retrievers import get_simple_retriever, get_complex_retriever
from langchain_core.messages import AIMessage

# Initialize local LLM
llm = ChatOpenAI(
    base_url="https://relation-creature-tap-bradley.trycloudflare.com/v1",
    api_key="not-needed",
    model="qwen3:30b",
    temperature=0.1
)

# ==========================================
# Pydantic Schemas for Structured Output
# ==========================================

class GuardrailResult(BaseModel):
    is_valid: bool = Field(description="True if the query is related to work, HR, software, or internal emails. False if off-topic.")
    reason: str = Field(description="Brief explanation for why the query was accepted or rejected.")

class RouterResult(BaseModel):
    category: Literal["email", "simple", "complex", "chat"] = Field(
        description=(
            "'email': User explicitly wants to draft or send an internal email. "
            "'simple': Direct factual policy lookup. "
            "'complex': Ambiguous, multi-step, scenario-based, multi-part, or requires reasoning across multiple policy sections. "
            "'chat': The user is just saying a general greeting (e.g., hi, hello, thanks , how are you etc.) "
        )
    )

class DocumentGraderResult(BaseModel):
    is_relevant: bool = Field(
        description="True if the documents contain sufficient evidence to answer the query. False if irrelevant or missing key information."
    )
    reason: str = Field(
        description="Brief explanation of what evidence was found or what is missing."
    )

class RewriteResult(BaseModel):
    rewritten_query: str = Field(
        description="A better, more optimized search query based on the failure of the previous one."
    )

class ReflectionResult(BaseModel):
    is_grounded: bool = Field(
        description="True if every claim in the generated answer is directly supported by the context. False if it hallucinates."
    )
    error_type: Literal["none", "missing_evidence", "wording_problem"] = Field(
        description="If is_grounded is False, classify the error. 'missing_evidence' if the context lacks the facts to answer the user. 'wording_problem' if the context has the facts but the generator added unprompted external knowledge."
    )
    reason: str = Field(description="Explanation of the evaluation.")

# ==========================================
# Node Implementations
# ==========================================

async def input_guardrail_node(state: AgentState) -> dict:
    """
    Part 11: Input Guardrail.
    Returns a state update dictionary. Appends a polite rejection if off-topic.
    """
    user_query = state["messages"][-1].content
    
    prompt = f"""You are a strict Input Guardrail for the NextBridge HR AI Agent.
Evaluate the following user query.

Valid topics: General greetings (e.g., "hi", "hello", "who are you"), NextBridge HR policies, leaves, benefits, payroll, drafting/sending internal emails, software engineering context.
Invalid topics: Coding help, general knowledge, pop culture, creative writing, fashion, etc.

Query: "{user_query}"
"""
    
    structured_llm = llm.with_structured_output(GuardrailResult)
    
    try:
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        if not result.is_valid:
            # FIX: Create a polite rejection message for the user
            fallback = f"I am strictly a NextBridge HR Assistant. I cannot assist with this query. (Reason: {result.reason})"
            return {
                "query": user_query,
                "retrieval_attempts": 0,
                "generation_attempts": 0,
                "final_status": "guardrail_blocked",
                "correction_reason": result.reason,
                "messages": [AIMessage(content=fallback)] # <--- Saves to memory
            }
        else:
            return {
                "query": user_query,
                "retrieval_attempts": 0,
                "generation_attempts": 0,
                "final_status": "processing",
                "correction_reason": None
            }
            
    except Exception as e:
        print(f"[Guardrail Error] Defaulting to valid. Error: {e}")
        return {
            "query": user_query,
            "retrieval_attempts": 0,
            "generation_attempts": 0,
            "final_status": "processing"
        }


async def adaptive_router_node(state: AgentState) -> dict:
    """
    Adaptive Query Routing.
    Returns a state update dictionary containing only the classified query_type.
    """
    # If guardrail blocked, do not update the routing state
    if state.get("final_status") == "guardrail_blocked":
        return {}

    user_query = state.get("query", state["messages"][-1].content)

    prompt = f"""You are a Query Router for an HR Agent.
Classify the following query into exactly ONE of these categories:
- "email": The user explicitly wants to draft or send an email.
- "simple": A direct, single-fact policy lookup (e.g., "How many sick leaves do I get?").
- "complex": Ambiguous, multi-part, or requires scenario analysis (e.g., "Do I lose my PTO if I don't use it?").
- "chat": The user is just saying a general greeting (e.g., "hi", "hello", "thanks").

Query: "{user_query}"
"""
    
    # Bind the Pydantic schema to the LLM
    structured_llm = llm.with_structured_output(RouterResult)
    
    try:
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        return {"query_type": result.category}
        
    except Exception as e:
        print(f"[Router Error] Defaulting to 'complex'. Error: {e}")
        return {"query_type": "complex"}

async def retrieval_node(state: AgentState) -> dict:
    """
    Executes adaptive retrieval and tracks telemetry.
    Returns the retrieved documents and increments the attempt counter.
    """
    # 1. Determine which query to use (original vs. rewritten)
    active_query = state.get("rewritten_query") or state.get("query")
    query_type = state.get("query_type", "complex")
    
    # 2. Track loop telemetry
    current_attempts = state.get("retrieval_attempts", 0) + 1
    
    # 3. Handle bypass conditions
    if state.get("final_status") == "guardrail_blocked":
        return {}
    
    if query_type == "email":
        # Email queries generally don't need policy retrieval
        return {
            "documents": [],
            "retrieval_attempts": current_attempts,
            "retrieval_failure_reason": "Skipped (Email Routing)"
        }

    # 4. Adaptive Retrieval Execution
    try:
        # If we are in a correction loop (attempts > 1), force the complex retriever
        if query_type == "simple" and current_attempts == 1:
            retriever = get_simple_retriever()
        else:
            retriever = get_complex_retriever()
            
        # We use invoke (synchronous block) because HuggingFaceCrossEncoder is CPU bound
        # In a fully async production environment, this should be wrapped in run_in_executor
        docs = retriever.invoke(active_query)
        
        return {
            "documents": docs,
            "retrieval_attempts": current_attempts,
            "retrieval_failure_reason": None
        }
        
    except Exception as e:
        print(f"[Retrieval Error] Failed to retrieve context: {e}")
        return {
            "documents": [],
            "retrieval_attempts": current_attempts,
            "retrieval_failure_reason": f"System Error: {str(e)}"
        }

# ==========================================
# CRAG Nodes
# ==========================================

async def document_grader_node(state: AgentState) -> dict:
    """
    Corrective RAG (CRAG) Grader.
    Evaluates whether the retrieved documents actually answer the user's query.
    """
    active_query = state.get("rewritten_query") or state.get("query")
    documents = state.get("documents", [])
    
    # Fast-fail: If retrieval found absolutely nothing
    if not documents:
        return {
            "documents_relevant": False,
            "retrieval_grade_reason": "Retrieval returned 0 chunks."
        }

    # Format the context for the LLM judge
    context = "\n\n".join([f"--- Chunk {i+1} ---\n{doc.page_content}" for i, doc in enumerate(documents)])
    
    prompt = f"""You are a strict grading evaluator for an HR system.
Your job is to determine if the retrieved context contains sufficient information to answer the query.

Query: "{active_query}"

Retrieved Context:
{context}

If the context contains the answer (even partially), mark is_relevant as true.
If the context is completely unrelated or insufficient, mark is_relevant as false.
"""

    structured_llm = llm.with_structured_output(DocumentGraderResult)
    
    try:
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        return {
            "documents_relevant": result.is_relevant,
            "retrieval_grade_reason": result.reason
        }
    except Exception as e:
        print(f"[Grader Error] {e}")
        # Fail-open if the LLM crashes so we don't get stuck in a loop
        return {
            "documents_relevant": True, 
            "retrieval_grade_reason": "Grader failed, assuming relevant to proceed."
        }


async def rewrite_query_node(state: AgentState) -> dict:
    """
    CRAG Query Rewriter.
    If the documents were insufficient, this node rewrites the query for a better retrieval attempt.
    """
    original_query = state.get("query")
    previous_rewrites = state.get("rewritten_query", "None")
    failure_reason = state.get("retrieval_grade_reason", "Insufficient context.")
    
    prompt = f"""You are an expert search query optimizer for an HR vector database.
The previous search failed to find the right documents.

Original User Query: "{original_query}"
Previous Search Query Used: "{previous_rewrites}"
Why it failed: "{failure_reason}"

Rewrite the query to be highly optimized for a vector database. Use formal corporate HR terminology. Do not answer the question, just provide the new search string.
"""

    structured_llm = llm.with_structured_output(RewriteResult)
    
    try:
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        return {
            "rewritten_query": result.rewritten_query,
            "correction_reason": f"Rewrote query because: {failure_reason}"
        }
    except Exception as e:
        print(f"[Rewrite Error] {e}")
        return {
            "rewritten_query": original_query,
            "correction_reason": "Rewrite failed, falling back to original query."
        }
    
# ==========================================
# Self-Reflection Nodes
# ==========================================

async def generation_node(state: AgentState) -> dict:
    """
    Generation Node.
    Synthesizes the answer and appends it to the chat messages.
    """
    query = state.get("rewritten_query") or state.get("query")
    docs = state.get("documents", [])
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are a helpful HR assistant. Answer the user query using ONLY the provided context.
If the answer is not in the context, state that clearly. Do not use external knowledge.

Query: "{query}"

Context:
{context}

Answer:"""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    return {
        "generation": response.content,
        "generation_attempts": state.get("generation_attempts", 0) + 1,
        "messages": [AIMessage(content=response.content)] # <--- ADDED THIS LINE
    }


async def reflection_node(state: AgentState) -> dict:
    """
    Self-RAG Reflection (Upgraded).
    Critiques the generated answer and determines the exact root cause of hallucinations.
    """
    active_query = state.get("rewritten_query") or state.get("query")
    generation = state.get("generation")
    docs = state.get("documents", [])
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are a strict hallucination grader. 
Evaluate the generated answer against the retrieved context.

Query: "{active_query}"
Answer: "{generation}"

Context:
{context}

RULES:
1. If the answer is completely supported by the Context, set is_grounded=True and error_type="none".
2. If the answer contains facts NOT in the context, set is_grounded=False.
3. If is_grounded=False, determine WHY:
   - If the Context actually lacks the information needed to answer the query, set error_type="missing_evidence".
   - If the Context HAS the information, but the Answer just brought in extra outside knowledge, set error_type="wording_problem".
"""
    
    structured_llm = llm.with_structured_output(ReflectionResult)
    
    try:
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        return {
            "answer_grounded": result.is_grounded,
            "reflection_error_type": result.error_type,
            "reflection_reason": result.reason
        }
    except Exception as e:
        return {
            "answer_grounded": True, 
            "reflection_error_type": "none",
            "reflection_reason": f"Reflection failed: {e}"
        }