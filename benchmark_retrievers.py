# """
# Automated Empirical Retrieval Benchmarking Harness.
# Executes registered retrieval strategies across a curated evaluation dataset,
# measures runtime latency, records retrieved chunks, pushes LangSmith traces,
# and exports comparison reports to evaluation/results/.
# """

# import os
# import sys
# import time
# import json
# from pathlib import Path
# from typing import List, Dict, Any

# from dotenv import load_dotenv

# # Ensure the root project directory is in the Python path
# ROOT_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(ROOT_DIR))

# load_dotenv()

# from langchain_core.tracers.context import collect_runs
# from src.retrievers import RETRIEVER_REGISTRY, get_retriever

# # Output Paths
# RESULTS_DIR = ROOT_DIR / "evaluation" / "results"
# REPORT_JSON_PATH = RESULTS_DIR / "retrieval_benchmark_report.json"
# REPORT_MD_PATH = RESULTS_DIR / "retrieval_benchmark_report.md"

# # # =====================================================================
# # # 1. Curated Benchmark Queries Covering All Document Archetypes
# # # =====================================================================
# # BENCHMARK_QUERIES = [
# #     {
# #         "id": "Q1_SINGLE_HOP",
# #         "archetype": "Hierarchical Policy",
# #         "query": "How many casual leaves are allowed per year for permanent employees?",
# #         "expected_source": "handbook-final-v6.0_2_45702763540.pdf",
# #     },
# #     {
# #         "id": "Q2_TABULAR_DIR",
# #         "archetype": "Tabular Directory",
# #         "query": "What is the discount percentage for Chughtais Lahore Lab at Jail Road?",
# #         "expected_source": "aicl_discount_center_list-updated_209015612052.pdf",
# #     },
# #     {
# #         "id": "Q3_ATOMIC_FORM",
# #         "archetype": "Claim Form",
# #         "query": "What documents and details are required to submit an OPD claim form?",
# #         "expected_source": "nxb-hr-t-07-private_opd_claim_form",
# #     },
# #     {
# #         "id": "Q4_POLICY_CLAUSE",
# #         "archetype": "Regulatory Rules",
# #         "query": "What percentage of salary is contributed by the employee to the Provident Fund?",
# #         "expected_source": "draft_nb_employees_provident_fund_rules",
# #     },
# #     {
# #         "id": "Q5_COMPOUND_MULTI",
# #         "archetype": "Multi-Intent Compound",
# #         "query": "Compare OPD and IPD claim procedures and explain what medical expenses are excluded under the policy.",
# #         "expected_source": "exclusions_1195870934.pdf",
# #     },
# # ]
# # =====================================================================
# # 1. Curated Stress-Level Benchmark Queries
# # =====================================================================
# BENCHMARK_QUERIES = [
#     {
#         "id": "Q1_TABULAR_NEEDLE_IN_HAYSTACK",
#         "archetype": "Tabular Directory",
#         "query": "I am in Lahore. What is the exact discount percentage I get at Salman Chughtais Lahore Lab at the Johar Town branch, and what is their phone number?",
#         "expected_source": "aicl_discount_center_list-updated_209015612052.pdf",
#         "stress_test": "Requires the retriever to isolate a specific city, lab name, and branch from a dense table without returning the Islamabad or Karachi branches."
#     },
#     {
#         "id": "Q2_CONDITIONAL_EDGE_CASE",
#         "archetype": "Hierarchical Policy",
#         "query": "I am a female employee in my 3rd trimester of pregnancy. How many WFH days am I allowed per month now, and how many WFH days will I get when I return from my 3-month maternity leave?",
#         "expected_source": "handbook-final-v6.0_2_45702763540.pdf",
#         "stress_test": "Tests deep hierarchical chunking. The retriever must find two separate conditional rules (3rd trimester vs. post-maternity return) under the Remote Work Policy."
#     },
#     {
#         "id": "Q3_CROSS_DOCUMENT_CONFLICT",
#         "archetype": "Multi-Document Compound",
#         "query": "I have a medical emergency. Can I request an advance salary of 90%, or should I take a temporary withdrawal from my Provident Fund? Explain the maximum limits for both.",
#         "expected_source": "multiple", # Should pull from both handbook and pf_rules
#         "stress_test": "Forces Map-Reduce to decompose the query and fetch the 80% advance salary limit from the Handbook AND the 75% balance/6-months salary limit from the PF Rules."
#     },
#     {
#         "id": "Q4_SEMANTIC_EXCLUSION_TRAP",
#         "archetype": "Regulatory Exclusions",
#         "query": "Will my health insurance cover emergency dental surgery after a car accident, and does it cover my wife's normal childbirth delivery?",
#         "expected_source": "exclusions_1195870934.pdf",
#         "stress_test": "Tests the Cross-Encoder's precision. Normal dental and childbirth are excluded, BUT 'Emergency Accidental Dental' is explicitly an exception to the exclusion. Naive vectors usually fail this."
#     },
#     {
#         "id": "Q5_TIME_CALCULATION_LOGIC",
#         "archetype": "General Policy",
#         "query": "If I take a leave on Friday and Monday, how many annual leaves will be deducted? Also, what happens if I clock in but only work for 6 hours on Tuesday?",
#         "expected_source": "handbook-final-v6.0_2_45702763540.pdf",
#         "stress_test": "Requires retrieving the 'Sandwich Rule' (4 days deducted) and the minimum attendance threshold (7 hours 1 minute to avoid absence)."
#     },
#     {
#         "id": "Q6_HIERARCHICAL_PF_WITHDRAWAL",
#         "archetype": "Hierarchical Policy",
#         "query": "What is the difference between a permanent withdrawal for purchasing a house and a temporary withdrawal for Hajj in terms of repayment and limits?",
#         "expected_source": "draft_nb_employees_provident_fund_rules-edited_version-_final-2_164444031339.pdf",
#         "stress_test": "Tests retrieval across different tables/clauses in the PF document to contrast permanent (no repayment, 36/24 months) vs. temporary (36 months repayment, 6 months salary) rules."
#     }
# ]
# # Strategies to Benchmark
# STRATEGIES_TO_EVALUATE = [
#     "dense",
#     "hybrid",
#     "hybrid_rerank",
#     "main",
#     "main_self_query",
#     "main_parent_doc",
#     "main_multi_query",
#     "main_rag_fusion",
#     "main_hyde",
# ]


# # =====================================================================
# # 2. Benchmarking Engine
# # =====================================================================
# def run_benchmark():
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
#     print("=" * 80)
#     print("🚀 Starting Empirical Retrieval Strategy Benchmarking")
#     print(f"📊 Evaluating {len(STRATEGIES_TO_EVALUATE)} strategies across {len(BENCHMARK_QUERIES)} benchmark queries")
#     print("=" * 80)

#     summary_records = []
#     raw_evaluation_log = {}

#     for strat_name in STRATEGIES_TO_EVALUATE:
#         print(f"\n========================================================")
#         print(f"🔬 BENCHMARKING STRATEGY: '{strat_name.upper()}'")
#         print(f"========================================================")

#         try:
#             retriever = get_retriever(strat_name)
#         except Exception as e:
#             print(f"[!] Could not initialize '{strat_name}': {e}")
#             continue

#         strategy_latencies = []
#         strategy_doc_counts = []
#         strategy_query_logs = []

#         for q_item in BENCHMARK_QUERIES:
#             q_id = q_item["id"]
#             query_text = q_item["query"]
#             expected_src = q_item["expected_source"].lower()

#             print(f"\n  [{q_id}] Query: \"{query_text}\"")

#             # Execute with latency timer and LangSmith run collector
#             start_time = time.perf_counter()
#             run_id = None

#             try:
#                 with collect_runs() as cb:
#                     # Pass config tags so LangSmith indexes each strategy cleanly
#                     docs = retriever.invoke(
#                         query_text,
#                         config={
#                             "tags": ["retrieval_benchmark", f"strategy:{strat_name}", f"qid:{q_id}"],
#                             "metadata": {
#                                 "strategy": strat_name,
#                                 "query_id": q_id,
#                                 "archetype": q_item["archetype"]
#                             }
#                         }
#                     )
#                     if cb.traced_runs:
#                         run_id = str(cb.traced_runs[0].id)
#             except Exception as e:
#                 print(f"    ❌ Execution failed: {e}")
#                 docs = []

#             elapsed_ms = (time.perf_counter() - start_time) * 1000.0
#             strategy_latencies.append(elapsed_ms)
#             strategy_doc_counts.append(len(docs))

#             # Inspect retrieved chunks for expected source match
#             matched_source = False
#             top_sources = []
#             chunk_previews = []

#             for d in docs:
#                 src = str(d.metadata.get("source_file", "")).lower()
#                 top_sources.append(src)
#                 if expected_src in src:
#                     matched_source = True
#                 chunk_previews.append({
#                     "source": d.metadata.get("source_file", "unknown"),
#                     "page": d.metadata.get("page_number", 0),
#                     "parent_section": d.metadata.get("parent_section", ""),
#                     "preview": d.page_content[:150].replace("\n", " ") + "..."
#                 })

#             hit_indicator = "🎯 HIT" if matched_source else "⚠️ MISS"
#             print(f"    -> Latency: {elapsed_ms:.1f}ms | Docs: {len(docs)} | Top Source: {top_sources[:2]} | {hit_indicator}")

#             strategy_query_logs.append({
#                 "query_id": q_id,
#                 "query": query_text,
#                 "archetype": q_item["archetype"],
#                 "latency_ms": round(elapsed_ms, 2),
#                 "docs_retrieved": len(docs),
#                 "target_matched": matched_source,
#                 "langsmith_run_id": run_id,
#                 "retrieved_chunks": chunk_previews
#             })

#         # Calculate strategy metrics
#         avg_latency = sum(strategy_latencies) / len(strategy_latencies) if strategy_latencies else 0
#         sorted_latencies = sorted(strategy_latencies)
#         p95_idx = int(len(sorted_latencies) * 0.95)
#         p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)] if sorted_latencies else 0
#         avg_docs = sum(strategy_doc_counts) / len(strategy_doc_counts) if strategy_doc_counts else 0
#         accuracy = sum(1 for q in strategy_query_logs if q["target_matched"]) / len(strategy_query_logs) if strategy_query_logs else 0

#         summary_records.append({
#             "strategy": strat_name,
#             "avg_latency_ms": round(avg_latency, 2),
#             "p95_latency_ms": round(p95_latency, 2),
#             "avg_docs_returned": round(avg_docs, 1),
#             "target_recall_accuracy": f"{accuracy * 100:.1f}%"
#         })

#         raw_evaluation_log[strat_name] = {
#             "summary": summary_records[-1],
#             "details": strategy_query_logs
#         }

#     # =====================================================================
#     # 3. Export Reports (JSON & Markdown)
#     # =====================================================================
#     with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
#         json.dump(raw_evaluation_log, f, indent=2, ensure_ascii=False)
#     print(f"\n💾 Full JSON evaluation report exported to: {REPORT_JSON_PATH}")

#     # Generate Markdown comparison table
#     md_content = "# 📊 NextBridge HR Agent: Retrieval Strategy Benchmark Report\n\n"
#     md_content += f"*Generated from empirical fresh runs against {len(BENCHMARK_QUERIES)} multi-archetype evaluation queries.*\n\n"
#     md_content += "| Strategy | Avg Latency (ms) | p95 Latency (ms) | Avg Docs Returned | Target Recall Accuracy |\n"
#     md_content += "|---|---|---|---|---|\n"

#     for r in summary_records:
#         md_content += f"| **{r['strategy']}** | {r['avg_latency_ms']} ms | {r['p95_latency_ms']} ms | {r['avg_docs_returned']} | {r['target_recall_accuracy']} |\n"

#     md_content += "\n\n## 📝 Query Archetypes Evaluated\n"
#     for q in BENCHMARK_QUERIES:
#         md_content += f"- **{q['id']} ({q['archetype']})**: \"{q['query']}\"\n"

#     with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
#         f.write(md_content)
#     print(f"📄 Markdown leaderboard exported to: {REPORT_MD_PATH}")

#     # Print Leaderboard directly to console
#     print("\n" + "=" * 80)
#     print("🏆 FINAL BENCHMARK LEADERBOARD")
#     print("=" * 80)
#     print(f"{'Strategy':<26} | {'Avg Latency':<12} | {'p95 Latency':<12} | {'Avg Docs':<10} | {'Recall'}")
#     print("-" * 80)
#     for r in summary_records:
#         print(f"{r['strategy']:<26} | {str(r['avg_latency_ms']) + ' ms':<12} | {str(r['p95_latency_ms']) + ' ms':<12} | {str(r['avg_docs_returned']):<10} | {r['target_recall_accuracy']}")
#     print("=" * 80)


# if __name__ == "__main__":
#     run_benchmark()

"""
Production-Grade Retrieval & Evaluation Benchmarking Suite.
- Safe path resolution (outputs strictly to evaluation/results/benchmarks/).
- Robust Tenacity retry/jitter on all Cloudflare LLM calls.
- Strict token budgeting to prevent context overflow while preserving evaluation depth.
- Multi-source hit detection (resolves compound queries like Q3).
- LangSmith dual-tagging (supports filtering by 'main' or 'strategy:main').
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import json
import textwrap
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# =====================================================================
# 1. Dynamic Root & Output Directory Resolution
# =====================================================================
CURRENT_FILE = Path(__file__).resolve()
# Auto-detect whether running from root or from scripts/
if CURRENT_FILE.parent.name == "scripts":
    ROOT_DIR = CURRENT_FILE.parent.parent
else:
    ROOT_DIR = CURRENT_FILE.parent

sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

import httpx
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tracers.context import collect_runs
from langchain_core.documents import Document
from langsmith import Client
from langchain_core.runnables import RunnableLambda

from src.retrievers import (
    get_retriever,
    get_cloudflare_llm,
    CLOUDFLARE_BASE_URL,
    CLOUDFLARE_MODEL_NAME
)
from langsmith import traceable

# Output directory: always strictly inside nextbridge_hr_agent/evaluation/results/benchmarks/
RESULTS_DIR = ROOT_DIR / "evaluation" / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON_PATH = RESULTS_DIR / "retrieval_benchmark_report.json"
REPORT_MD_PATH = RESULTS_DIR / "retrieval_benchmark_report.md"

# LangSmith Client
ls_client = Client()

# Evaluation Model: Default to qwen3:30b
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "qwen3:30b")


# =====================================================================
# 2. Production Resilience: Tenacity Wrapper
# =====================================================================
@retry(
    wait=wait_exponential_jitter(initial=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, Exception)),
    reraise=True
)
def robust_eval_call(llm, messages: list):
    """Executes an LLM call with exponential backoff & jitter to handle tunnel drops."""
    return llm.invoke(messages)


# =====================================================================
# 3. Stress Evaluation Dataset
# =====================================================================
BENCHMARK_DATASET = [

    {
        "id": "Q1_TABULAR_NEEDLE",
        "archetype": "Tabular Directory",
        "query": (
            "I am in Lahore. What discount do I get at Salman Chughtais "
            "Lahore Lab at the Johar Town location, and what phone numbers "
            "are listed for that location?"
        ),
        "expected_sources": [
            "aicl_discount_center_list-updated_209015612052.pdf"
        ],
        "ground_truth": (
            "The listed discount is 30% to 40%. The Johar Town contact "
            "numbers are 0300 460 5193 and 0321 400 0188."
        ),
    },

    {
        "id": "Q2_CONDITIONAL_POLICY",
        "archetype": "Hierarchical Policy",
        "query": (
            "I am a female employee in my 3rd trimester of pregnancy. "
            "How many WFH days am I allowed per month now, and how many "
            "WFH days will I get when I return from my 3-month maternity leave?"
        ),
        "expected_sources": [
            "handbook-final-v6.0_2_45702763540.pdf"
        ],
        "ground_truth": (
            "A female employee in the 3rd trimester is allowed 12 WFH days "
            "per month. After returning from 3 months of maternity leave, "
            "she is allowed 12 WFH days per month for the next 3 months, "
            "until the baby is 6 months old."
        ),
    },

    {
        "id": "Q3_CROSS_DOC_CONFLICT",
        "archetype": "Multi-Document Compound",
        "query": (
            "I have a medical emergency. Can I request an advance salary "
            "of 90%, or should I take a temporary withdrawal from my "
            "Provident Fund? Explain the maximum limits for both."
        ),
        "expected_sources": [
            "handbook-final-v6.0_2_45702763540.pdf",
            "draft_nb_employees_provident_fund_rules-edited_version-_final-2_164444031339.pdf"
        ],
        "ground_truth": (
            "An advance salary cannot be 90%; the maximum is 80% of monthly "
            "gross pay. For illness expenses, a temporary Provident Fund "
            "withdrawal is limited to 6 months of basic salary or 75% of "
            "the member's total account balance, whichever is less. The "
            "temporary withdrawal must be repaid within 36 months."
        ),
    },

    {
        "id": "Q4_SEMANTIC_EXCLUSION",
        "archetype": "Exclusions Annexure",
        "query": (
            "Will my health insurance exclude emergency dental treatment "
            "after a car accident, and is my wife's normal childbirth "
            "delivery excluded?"
        ),
        "expected_sources": [
            "exclusions_1195870934.pdf"
        ],
        "ground_truth": (
            "Emergency Accidental Dental Treatment is an exception to the "
            "general dental-treatment exclusion. Pregnancy and childbirth, "
            "including normal delivery, are excluded unless specifically "
            "covered under a separate rider."
        ),
    },

    {
        "id": "Q5_TIME_CALCULATION",
        "archetype": "General Policy",
        "query": (
            "If I take leave on Friday and Monday, how many days will be "
            "counted under the Sandwich Rule? Also, what happens if I "
            "clock in but only work for 6 hours on Tuesday?"
        ),
        "expected_sources": [
            "handbook-final-v6.0_2_45702763540.pdf"
        ],
        "ground_truth": (
            "Under the Sandwich Rule, Friday, Saturday, Sunday, and Monday "
            "are counted as 4 days unless the leave was planned and "
            "pre-approved. An employee must complete at least 7 hours and "
            "1 minute in a day to be marked present, so working only 6 hours "
            "does not satisfy the minimum attendance threshold."
        ),
    },

    {
        "id": "Q6_HIERARCHICAL_PF",
        "archetype": "Provident Fund Rules",
        "query": (
            "What is the difference between a permanent withdrawal for "
            "purchasing a house and a temporary withdrawal for Hajj in "
            "terms of repayment and limits?"
        ),
        "expected_sources": [
            "draft_nb_employees_provident_fund_rules-edited_version-_final-2_164444031339.pdf"
        ],
        "ground_truth": (
            "A permanent Provident Fund withdrawal for purchasing or "
            "building a house does not require repayment. The general "
            "limit is 36 months of basic salary or the member's own "
            "contribution balance plus profit, whichever is less. A "
            "temporary withdrawal for Hajj must be repaid within 36 months "
            "and is limited to 6 months of basic salary or 75% of the "
            "member's total account balance, whichever is less."
        ),
    },

]

STRATEGIES_TO_EVALUATE = [
    "dense",
    "hybrid",
    "hybrid_rerank",
    "main",
    "main_self_query",
    "main_parent_doc",
    "main_multi_query",
    "main_rag_fusion",
    "main_hyde",
]


# =====================================================================
# 4. Context Budgeting & LLM Judge
# =====================================================================

def safe_truncate(text: str, max_chars: int) -> str:
    """Safely truncates text to the nearest word boundary to prevent severed tokens."""
    if len(text) <= max_chars:
        return text
    return textwrap.shorten(text, width=max_chars, placeholder="... [TRUNCATED]")

@traceable(run_type="tool", name="Build_Budgeted_Context")
def build_budgeted_context(docs: List[Document], max_docs: int = 4, max_chars_per_doc: int = 1500) -> str:
    """
    Constructs a controlled, token-safe context window.
    """
    context_blocks = []
    for d in docs[:max_docs]:
        src = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page_number", 0)
        content = safe_truncate(d.page_content.strip(), max_chars_per_doc)
        context_blocks.append(f"[[SOURCE: {src} | PAGE: {page}]]\n{content}")
    return "\n\n---\n\n".join(context_blocks)

@traceable(run_type="chain", name="LLM_Judge_Evaluation")
def judge_response(
    judge_llm,
    query: str,
    ground_truth: str,
    context_text: str,
    generated_answer: str
) -> Optional[Dict[str, float]]:
    """
    Evaluates RAG performance using Chain-of-Thought reasoning.
    Returns normalized floats, or None if the evaluation fails (prevents metric skewing).
    """
    
    # Safely bound inputs to prevent context overflow in the judge prompt
    safe_context = safe_truncate(context_text, 4000)
    safe_answer = safe_truncate(generated_answer, 1500)

    prompt = f"""You are a strict, deterministic evaluator for a Retrieval-Augmented Generation (RAG) pipeline.

Evaluate the following three dimensions:
1. CONTEXT RELEVANCE: Does the context contain the facts needed?
2. GROUNDEDNESS: Is the generated answer supported by the context without hallucinations?
3. ANSWER CORRECTNESS: Does the generated answer match the ground truth?

EVALUATION PROCEDURE
Step 1: Identify the atomic factual requirements in the user query.
Step 2: Check if the retrieved context contains explicit evidence supporting them.
Step 3: Check every factual claim in the generated answer against the retrieved context.
Step 4: Compare the generated answer against the ground-truth fact-by-fact.

SCORING RUBRIC (0.0 to 1.0)
- 1.0: Perfect match / Fully supported / All evidence present.
- 0.8: Minor omission or harmless paraphrasing.
- 0.5: Partially correct / Some claims unsupported / Missing critical evidence.
- 0.2: Mostly incorrect / Weakly related.
- 0.0: Completely irrelevant / Hallucinated / Contradicts ground truth.

OUTPUT FORMAT
Return ONLY valid JSON. You MUST provide your step-by-step reasoning before outputting the scores.

{{
  "reasoning": "<Write your step-by-step evaluation here based on the procedure>",
  "context_relevance": 0.0,
  "groundedness": 0.0,
  "answer_correctness": 0.0
}}

User Query: "{query}"
Target Ground Truth: "{ground_truth}"

Retrieved Context:
\"\"\"{safe_context}\"\"\"

Generated Answer:
\"\"\"{safe_answer}\"\"\"
"""
    try:
        # Assuming robust_eval_call includes Tenacity retries
        res = robust_eval_call(
            judge_llm,
            [
                SystemMessage(content="You are an automated evaluation judge. Output valid JSON only."),
                HumanMessage(content=prompt)
            ]
        )
        content = res.content.strip()
        
        # Robust JSON extraction
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(content[start : end + 1])
            return {
                "context_relevance": max(0.0, min(1.0, float(data.get("context_relevance", 0.0)))),
                "groundedness": max(0.0, min(1.0, float(data.get("groundedness", 0.0)))),
                "answer_correctness": max(0.0, min(1.0, float(data.get("answer_correctness", 0.0)))),
            }
    except Exception as e:
        print(f"    ⚠️ [Judge Notice] Evaluation failed and will be excluded from metrics: {e}")

    # Return None instead of 0.5 to prevent poisoning the dataset averages
    return None

# =====================================================================
# 5. Main Execution Engine
# =====================================================================
def run_benchmark():
    eval_llm = get_cloudflare_llm()

    print("=" * 80)
    print("🚀 NextBridge HR Agent: Empirical Multi-Strategy Benchmarking")
    print(f"📁 Reports Destination: {RESULTS_DIR}")
    print(f"🤖 Evaluation LLM: {EVAL_MODEL_NAME} via Cloudflare Tunnel")
    print(f"📊 Strategies: {len(STRATEGIES_TO_EVALUATE)} | Queries: {len(BENCHMARK_DATASET)}")
    print("=" * 80)

    leaderboard = []
    full_log = {}

    for strat in STRATEGIES_TO_EVALUATE:
        print(f"\n========================================================")
        print(f"🔬 EVALUATING STRATEGY: '{strat.upper()}'")
        print(f"========================================================")

        retriever = get_retriever(strat)
        strat_latencies = []
        strat_hits = []
        strat_relevance = []
        strat_groundedness = []
        strat_correctness = []

        query_records = []

        for item in BENCHMARK_DATASET:
            qid = item["id"]
            query = item["query"]
            expected_sources = item["expected_sources"]
            ground_truth = item["ground_truth"]

            print(f"\n  [{qid}] \"{query[:75]}...\"")

            start_t = time.perf_counter()
            run_id = None

            # [CRITICAL FIX]: Wrap the custom method to enforce LangSmith config propagation
            traced_retriever = RunnableLambda(retriever.invoke).with_config(
                run_name=f"Eval_{strat.upper()}"
            )

            # 1. Execute Retrieval with LangSmith Tracing
            with collect_runs() as cb:
                docs = traced_retriever.invoke(
                    query,
                    config={
                        "tags": [
                            "retrieval_benchmark",
                            strat,                  
                            f"strategy:{strat}",    
                            qid,
                            f"qid:{qid}"
                        ],
                        "metadata": {
                            "strategy": strat,
                            "query_id": qid,
                            "archetype": item["archetype"]
                        }
                    }
                )
                if cb.traced_runs:
                    run_id = str(cb.traced_runs[0].id)

            latency_ms = (time.perf_counter() - start_t) * 1000.0
            strat_latencies.append(latency_ms)

            # 2. Multi-Source Hit Matching (Fixes the Q3 bug)
            retrieved_sources = [str(d.metadata.get("source_file", "")).lower() for d in docs]
            all_sources_hit = all(
                any(exp.lower() in src for src in retrieved_sources)
                for exp in expected_sources
            )
            strat_hits.append(1 if all_sources_hit else 0)

            # 3. Budgeted Context Windowing
            budgeted_context = build_budgeted_context(docs, max_docs=4, max_chars_per_doc=1500)

            # 4. Generate Answer via Tunnel LLM
            gen_prompt = (
                f"You are the official NextBridge HR assistant. "
                f"Answer the user query based ONLY on the context below. If facts are absent, state so clearly.\n\n"
                f"Context:\n{budgeted_context}\n\n"
                f"Query: {query}\nAnswer:"
            )
            gen_res = robust_eval_call(eval_llm, [HumanMessage(content=gen_prompt)])
            generated_ans = gen_res.content.strip()

            # 5. Judge Context Relevance, Groundedness, and Accuracy
            scores = judge_response(eval_llm, query, ground_truth, budgeted_context, generated_ans)
            
            if scores:
                strat_relevance.append(scores["context_relevance"])
                strat_groundedness.append(scores["groundedness"])
                strat_correctness.append(scores["answer_correctness"])
                
                hit_symbol = "🎯 HIT" if all_sources_hit else "⚠️ MISS"
                print(f"    -> Latency: {latency_ms:.0f}ms | Source: {hit_symbol} | Relevance: {scores['context_relevance']:.2f} | Grounded: {scores['groundedness']:.2f} | Accuracy: {scores['answer_correctness']:.2f}")

                # 6. Push Feedback directly to LangSmith
                if run_id:
                    try:
                        ls_client.create_feedback(run_id, key="context_relevance", score=scores["context_relevance"])
                        ls_client.create_feedback(run_id, key="groundedness", score=scores["groundedness"])
                        ls_client.create_feedback(run_id, key="answer_correctness", score=scores["answer_correctness"])
                    except Exception:
                        pass
            else:
                hit_symbol = "🎯 HIT" if all_sources_hit else "⚠️ MISS"
                print(f"    -> Latency: {latency_ms:.0f}ms | Source: {hit_symbol} | ⚠️ EVALUATION FAILED (Excluded from averages)")
                scores = {"context_relevance": None, "groundedness": None, "answer_correctness": None}

            query_records.append({
                "qid": qid,
                "latency_ms": round(latency_ms, 1),
                "source_hit": all_sources_hit,
                "scores": scores,
                "langsmith_run_id": run_id
            })

    # =====================================================================
    # 6. Export Reports (JSON & Markdown)
    # =====================================================================
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(full_log, f, indent=2)
    print(f"\n💾 Full JSON report saved to: {REPORT_JSON_PATH}")

    md_table = "# 🏆 NextBridge HR Agent: Retrieval & Groundedness Benchmark\n\n"
    md_table += f"*Evaluated against {len(BENCHMARK_DATASET)} stress queries across all document archetypes using `{EVAL_MODEL_NAME}`.*\n\n"
    md_table += "| Strategy | Latency (ms) | Source Recall | Context Relevance | Groundedness | Answer Accuracy |\n"
    md_table += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for r in leaderboard:
        md_table += f"| **{r['strategy']}** | {r['avg_latency_ms']} ms | {r['source_recall']} | {r['context_relevance']} | {r['groundedness']} | {r['answer_accuracy']} |\n"

    md_table += "\n\n## 📝 Query Breakdown\n"
    for q in BENCHMARK_DATASET:
        md_table += f"- **{q['id']} ({q['archetype']})**: \"{q['query']}\"\n"

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"📄 Markdown leaderboard saved to: {REPORT_MD_PATH}")

    # Display final console table
    print("\n" + "=" * 90)
    print("🏆 FINAL EMPIRICAL BENCHMARK LEADERBOARD")
    print("=" * 90)
    print(f"{'Strategy':<24} | {'Latency':<10} | {'Recall':<8} | {'Relevance':<10} | {'Grounded':<9} | {'Accuracy'}")
    print("-" * 90)
    for r in leaderboard:
        print(f"{r['strategy']:<24} | {str(r['avg_latency_ms']) + ' ms':<10} | {r['source_recall']:<8} | {r['context_relevance']:<10} | {r['groundedness']:<9} | {r['answer_accuracy']}")
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()