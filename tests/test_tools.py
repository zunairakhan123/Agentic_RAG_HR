import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools import search_hr_documents, guardrailed_web_search, draft_department_email

def test_tools():
    print("--- 1. Testing HR Retrieval Tool (Cold Start) ---")
    t0 = time.time()
    res1 = search_hr_documents.invoke({"query": "casual leave policy"})
    t1 = time.time()
    print(f"Cold execution time: {t1 - t0:.2f}s")
    assert "Excerpt" in res1, "Expected context excerpts in response"

    print("\n--- 2. Testing HR Retrieval Tool (Warm / Cached) ---")
    t2 = time.time()
    res2 = search_hr_documents.invoke({"query": "medical insurance coverage"})
    t3 = time.time()
    print(f"Warm execution time: {t3 - t2:.2f}s (Should be significantly faster)")

    print("\n--- 3. Testing Guardrailed Web Search ---")
    blocked = guardrailed_web_search.invoke({"query": "latest summer fashion trends"})
    assert "GUARDRAIL_BLOCKED" in blocked, "Guardrail failed to block irrelevant topic"
    print("✓ Guardrail successfully blocked non-work query.")

    print("\n--- 4. Testing Email Draft Generation ---")
    draft = draft_department_email.invoke({
        "department": "HR",
        "subject": "Leave Application",
        "body": "Requesting 2 days leave."
    })
    assert "DRAFT_GENERATED" in draft
    print("✓ Email draft tool formatted payload correctly.")

if __name__ == "__main__":
    test_tools()