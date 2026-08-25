"""
Core external action tools for the NextBridge HR Agent.
(Note: RAG Retrieval is handled natively by LangGraph nodes in src/nodes.py, not as a tool).
"""

import os
from typing import Dict
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

DEPARTMENT_DIRECTORY: Dict[str, str] = {
    "MIS": "zunairahawar7@gmail.com",
    "HR": "zunairahawar7@gmail.com",
    "MEAL": "zunairahawar7@gmail.com",
    "ADMIN": "zunairahawar7@gmail.com",
}

# =========================================================================
# External Action Tools
# =========================================================================

@tool
def guardrailed_web_search(query: str) -> str:
    """
    Searches the web for Nextbridge company information.
    """
    try:
        # [PRODUCTION FIX]: Programmatic Entity Scoping
        # Force the search engine to look for the software company, not the oil company.
        scoped_query = f"{query} Nextbridge software IT company Pakistan Lahore"
        
        search_engine = TavilySearchResults(max_results=4)
        results = search_engine.invoke(scoped_query)
        
        if not results or len(results) == 0:
            return "Search completed, but no relevant public information was found."
            
        return str(results)
        
    except Exception as err:
        print(f"[Tavily Search Error] {err}")
        return f"Web search failed: {str(err)}"

@tool
def draft_department_email(department: str, subject: str, body: str) -> str:
    """Drafts an email to a specific NextBridge department (MIS, HR, MEAL, ADMIN) for user verification."""
    dept_clean = department.upper().strip()
    
    # Map common LLM variations to our exact keys
    if "HR" in dept_clean or "HUMAN" in dept_clean:
        dept_key = "HR"
    elif "MIS" in dept_clean or "IT" in dept_clean or "TECH" in dept_clean:
        dept_key = "MIS"
    elif "MEAL" in dept_clean or "FOOD" in dept_clean:
        dept_key = "MEAL"
    else:
        dept_key = "ADMIN"

    # Force the recipient to your active testing email
    recipient = DEPARTMENT_DIRECTORY.get(dept_key, "zunairahawar7@gmail.com")

    return (
        f"DRAFT_GENERATED\n"
        f"Department: {dept_key}\n"
        f"To: {recipient}\n"
        f"Subject: {subject}\n"
        f"Body:\n{body}"
    )


@tool
def send_department_email(to_email: str, subject: str, body: str, thread_id: str) -> str:
    """Dispatches the approved email to the destination department via SMTP."""
    sender_email = os.getenv("SMTP_EMAIL")
    sender_password = os.getenv("SMTP_PASSWORD")
    
    # SECURITY OVERRIDE: Prevent LLM Hallucinations
    valid_recipients = list(DEPARTMENT_DIRECTORY.values())
    if to_email not in valid_recipients:
        print(f"[SECURITY] LLM hallucinated email '{to_email}'. Rerouting to safe directory.")
        to_email = "zunairahawar7@gmail.com"  # Fallback to testing email

    if not sender_email or not sender_password:
        print("[SMTP ERROR] Credentials missing in .env")
        return "ERROR: Email credentials are not configured on the server."

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = f"{subject} [Ref: {thread_id}]"
    msg['From'] = sender_email
    msg['To'] = to_email

    try:
        print(f"\n[SMTP] Attempting to send email to {to_email}...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        
        print(f"[SMTP SUCCESS] Email sent to {to_email}")
        return (
            f"SUCCESS: Email successfully dispatched to {to_email}. "
            f"The thread tracking reference is [{thread_id}]. "
            f"I will notify you immediately once {to_email} sends a response."
        )
    except Exception as e:
        error_msg = f"FAILED to send email due to SMTP error: {str(e)}"
        print(f"[SMTP ERROR] {error_msg}")
        return error_msg