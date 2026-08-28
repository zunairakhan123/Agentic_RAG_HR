# """
# Final Agentic RAGAS Evaluation
# Benchmarks the new CRAG LangGraph against the Golden Dataset using identical custom metrics.
# """

# import sys
# import os

# # Add the parent root directory to Python's path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import re
# import json
# import asyncio
# import httpx
# import pandas as pd
# from dotenv import load_dotenv
# from datasets import Dataset

# from ragas import evaluate
# from ragas.metrics import (
#     ResponseRelevancy, 
#     LLMContextPrecisionWithReference, 
#     LLMContextRecall
# )
# from ragas.llms import llm_factory
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from ragas.run_config import RunConfig
# from openai import AsyncOpenAI
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_huggingface import HuggingFaceEmbeddings
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# # Import the NEW Agentic Graph
# from src.graph import builder

# load_dotenv()

# # Pointing to a new LangSmith project for the Final Architecture
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "NextBridge_HR_Agent_Final"

# COMPANY_BASE_URL = "https://relation-creature-tap-bradley.trycloudflare.com/v1"
# MODEL_NAME = "qwen3:30b"

# # -------------------------------------------------------------------------
# # Custom Lightweight Faithfulness Evaluator (Identical to Baseline)
# # -------------------------------------------------------------------------
# async def evaluate_single_faithfulness(eval_llm: ChatOpenAI, response: str, contexts: list) -> float:
#     """Evaluates faithfulness by extracting and verifying claims in a single LLM pass."""
#     if not contexts or contexts == ["No context retrieved"]:
#         return 0.0

#     context_str = "\n---\n".join(contexts)
#     prompt = f"""You are an unbiased AI auditor evaluating RAG faithfulness.
# Your task: Determine whether statements in the RESPONSE are supported by the provided CONTEXT.

# CONTEXT:
# {context_str}

# RESPONSE:
# {response}

# INSTRUCTIONS:
# 1. Extract distinct factual claims made in the RESPONSE.
# 2. For each claim, check if it is directly supported by the CONTEXT.
# 3. Return ONLY a valid JSON object matching this schema:
# {{
#   "claims": [
#     {{"claim": "statement text", "supported": true}}
#   ],
#   "faithfulness_score": 1.0
# }}

# If there are no factual claims, return "faithfulness_score": 1.0.
# Do not output markdown codeblocks (no ```json). Output raw JSON only."""

#     try:
#         res = await eval_llm.ainvoke([
#             SystemMessage(content="You are a strict JSON-only evaluation judge."),
#             HumanMessage(content=prompt)
#         ])
#         content = res.content.strip()
        
#         match = re.search(r"\{.*\}", content, re.DOTALL)
#         if match:
#             parsed = json.loads(match.group(0))
#             return float(parsed.get("faithfulness_score", 0.0))
#         return 0.0
#     except Exception as e:
#         print(f"[Warning] Custom faithfulness evaluation failed for item: {e}")
#         return 0.0


# async def run_custom_faithfulness_batch(eval_llm: ChatOpenAI, responses: list, contexts_list: list) -> list:
#     """Runs faithfulness scoring with concurrency control."""
#     semaphore = asyncio.Semaphore(2) 

#     async def sem_eval(resp, ctx):
#         async with semaphore:
#             return await evaluate_single_faithfulness(eval_llm, resp, ctx)

#     tasks = [sem_eval(resp, ctx) for resp, ctx in zip(responses, contexts_list)]
#     return await asyncio.gather(*tasks)


# # -------------------------------------------------------------------------
# # Agent Generation (Updated for CRAG State Schema)
# # -------------------------------------------------------------------------
# def extract_contexts_from_crag_state(state):
#     """Extracts chunks directly from the new LangGraph state schema."""
#     docs = state.get("documents", [])
#     if docs:
#         return [doc.page_content for doc in docs]
#     return ["No context retrieved"]

# async def generate_predictions(golden_data):
#     questions, ground_truths, answers, contexts_list = [], [], [], []

#     # Using an isolated DB for evaluation to avoid corrupting production memory
#     async with AsyncSqliteSaver.from_conn_string("hr_agent_memory_eval.db") as memory:
#         await memory.setup()
#         hr_graph = builder.compile(checkpointer=memory)

#         for i, item in enumerate(golden_data):
#             print(f"Processing query {i+1}/{len(golden_data)}: {item['question']}")
#             config = {"configurable": {"thread_id": f"eval_final_{i}"}}
            
#             result_state = await hr_graph.ainvoke(
#                 {"messages": [HumanMessage(content=item['question'])]}, 
#                 config=config
#             )
            
#             answers.append(result_state["messages"][-1].content)
#             contexts_list.append(extract_contexts_from_crag_state(result_state))
#             questions.append(item['question'])
#             ground_truths.append(item['ground_truth'])
            
#             # Print loop telemetry for visibility
#             print(f"  -> Loops: Retrieval({result_state.get('retrieval_attempts', 0)}), Gen({result_state.get('generation_attempts', 0)})")

#     return questions, ground_truths, answers, contexts_list


# # -------------------------------------------------------------------------
# # Main Pipeline
# # -------------------------------------------------------------------------
# def run_final_eval():
#     dataset_path = "evaluation/datasets/golden_v1.json"
#     if not os.path.exists(dataset_path):
#         print(f"ERROR: Could not find {dataset_path}.")
#         return

#     with open(dataset_path, "r", encoding="utf-8") as f:
#         golden_data = json.load(f)

#     # 1. Run Graph Inference
#     questions, ground_truths, answers, contexts_list = asyncio.run(generate_predictions(golden_data))

#     # 2. Build Dataset for standard Ragas metrics
#     data = {
#         "user_input": questions,        
#         "reference": ground_truths,     
#         "response": answers,            
#         "retrieved_contexts": contexts_list 
#     }
#     dataset = Dataset.from_dict(data)

#     print("\nInitializing NextBridge Evaluators...")
#     company_client = AsyncOpenAI(
#         base_url=COMPANY_BASE_URL,
#         api_key="not-needed",
#         max_retries=5,
#         http_client=httpx.AsyncClient(
#             timeout=500.0,
#             limits=httpx.Limits(max_connections=2, max_keepalive_connections=2)
#         )
#     )
    
#     ragas_llm = llm_factory(MODEL_NAME, client=company_client)

#     eval_embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"}
#     )
#     ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

#     # 3. Standard Ragas Metrics
#     metrics = [
#         ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
#         LLMContextPrecisionWithReference(llm=ragas_llm),
#         LLMContextRecall(llm=ragas_llm)
#     ]

#     ragas_config = RunConfig(max_retries=5, max_wait=60, max_workers=1)

#     print("Running RAGAS Core Metrics...")
#     result = evaluate(
#         dataset=dataset,
#         metrics=metrics,
#         run_config=ragas_config
#     )

#     df = result.to_pandas()

#     # 4. Run Custom Faithfulness Evaluation
#     print("Running Custom Faithfulness Evaluation...")
#     judge_llm = ChatOpenAI(
#         base_url=COMPANY_BASE_URL,
#         api_key="not-needed",
#         model=MODEL_NAME,
#         temperature=0.0,
#         max_retries=5,
#         timeout=500.0
#     )
#     faithfulness_scores = asyncio.run(run_custom_faithfulness_batch(judge_llm, answers, contexts_list))
#     df["faithfulness"] = faithfulness_scores

#     # 5. Export JSON and CSV to the FINAL directory
#     os.makedirs("evaluation/results/final", exist_ok=True)
#     df.to_csv("evaluation/results/final/per_question.csv", index=False)
#     df.to_json("evaluation/results/final/per_question.json", orient="records", indent=4)
    
#     summary = df.mean(numeric_only=True).to_dict()
#     summary["total_evaluated"] = len(df)
    
#     with open("evaluation/results/final/summary.json", "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=4)

#     print("\n=== FINAL ARCHITECTURE EVALUATION COMPLETE ===")
#     print(json.dumps(summary, indent=4))
#     print("\nCompare these results in evaluation/results/final/ against evaluation/results/baseline/")

# if __name__ == "__main__":
#     import platform
#     if platform.system() == 'Windows':
#         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#     run_final_eval()
"""
Final Agentic RAGAS Evaluation
Benchmarks the new CRAG LangGraph against the Golden Dataset using isolated, resumable stages,
bounded async requests, and deterministic faithfulness calculations.
"""

import sys
import os
import re
import json
import asyncio
import traceback
import httpx
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ragas import evaluate
from ragas.metrics import (
    ResponseRelevancy, 
    LLMContextPrecisionWithReference, 
    LLMContextRecall
)
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Import the isolated RAG Subgraph builder
from src.graph import rag_builder

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "NextBridge_HR_Agent_Final_V2"

COMPANY_BASE_URL = "https://relation-creature-tap-bradley.trycloudflare.com/v1"
MODEL_NAME = "qwen3:30b"

# --- CONFIGURABLE BOUNDS ---
FAITHFULNESS_TIMEOUT = 120
FAITHFULNESS_CONCURRENCY = 1
MAX_RETRIES = 3

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def robust_json_parse(text: str) -> dict:
    """Extracts and parses JSON, ignoring markdown fences."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in response.")
    parsed = json.loads(match.group(0))
    if "claims" not in parsed or not isinstance(parsed["claims"], list):
        raise ValueError("JSON missing 'claims' array.")
    return parsed

def extract_contexts_from_crag_state(state):
    """Extracts chunks directly from the new LangGraph state schema."""
    docs = state.get("documents", [])
    if docs:
        return [doc.page_content for doc in docs]
    return ["No context retrieved"]

# -------------------------------------------------------------------------
# STAGE 1: LangGraph Generation
# -------------------------------------------------------------------------
async def stage_1_generation(golden_data: list, checkpoint_file: str) -> list:
    """Generates predictions directly through the isolated RAG subgraph."""
    predictions = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    
    start_idx = len(predictions)
    if start_idx >= len(golden_data):
        print("[STAGE 1] Generation already complete.")
        return predictions

    async with AsyncSqliteSaver.from_conn_string("hr_agent_memory_eval.db") as memory:
        await memory.setup()
        
        # Compile ONLY the RAG subgraph
        rag_graph = rag_builder.compile(checkpointer=memory)

        for i in range(start_idx, len(golden_data)):
            item = golden_data[i]
            print(f"[GENERATION] Processing query {i+1}/{len(golden_data)}: {item['question']}")
            config = {"configurable": {"thread_id": f"eval_rag_only_{i}"}}
            
            try:
                # Pass the exact state schema expected by RAGState
                result_state = await rag_graph.ainvoke(
                    {
                        "query": item['question'],
                        "query_type": "rag",
                        "documents": [],
                        "generation_attempts": 0,
                        "retrieval_attempts": 0
                    }, 
                    config=config
                )
                
                prediction = {
                    "question": item['question'],
                    "ground_truth": item['ground_truth'],
                    # Extract the specific string we defined in RAGState earlier
                    "response": result_state.get("generation", "No generation produced."),
                    "contexts": extract_contexts_from_crag_state(result_state)
                }
                predictions.append(prediction)
                
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(predictions, f, indent=4)
                    
                print(f"  -> Loops: Retrieval({result_state.get('retrieval_attempts', 0)}), Gen({result_state.get('generation_attempts', 0)})")
            except Exception as e:
                print(f"[GENERATION] FATAL ERROR on query {i+1}: {e}")
                raise e
                
    return predictions

# -------------------------------------------------------------------------
# STAGE 2: RAGAS Metrics
# -------------------------------------------------------------------------
def stage_2_ragas(predictions: list, checkpoint_file: str) -> list:
    """Runs standard RAGAS evaluation with isolation."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            ragas_results = json.load(f)
        if len(ragas_results) == len(predictions):
            print("[STAGE 2] RAGAS evaluation already complete.")
            return ragas_results

    dataset = Dataset.from_dict({
        "user_input": [p["question"] for p in predictions],
        "reference": [p["ground_truth"] for p in predictions],
        "response": [p["response"] for p in predictions],
        "retrieved_contexts": [p["contexts"] for p in predictions]
    })

    company_client = AsyncOpenAI(
        base_url=COMPANY_BASE_URL,
        api_key="not-needed",
        max_retries=3,
        http_client=httpx.AsyncClient(timeout=180.0, limits=httpx.Limits(max_connections=2))
    )
    ragas_llm = llm_factory(MODEL_NAME, client=company_client)
    ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    ))

    metrics = [
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm)
    ]
    
    ragas_config = RunConfig(max_retries=3, max_wait=30, max_workers=1)

    print("[STAGE 2] Running RAGAS Core Metrics...")
    try:
        result = evaluate(dataset=dataset, metrics=metrics, run_config=ragas_config)
        ragas_results = result.to_pandas().to_dict(orient="records")
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(ragas_results, f, indent=4)
        return ragas_results
    except Exception as e:
        print(f"[STAGE 2] RAGAS Encountered an error: {e}. Checkpoint not saved.")
        return []

# -------------------------------------------------------------------------
# STAGE 3: Custom Faithfulness Evaluation
# -------------------------------------------------------------------------
async def evaluate_single_faithfulness(eval_llm: ChatOpenAI, response: str, contexts: list, q_index: int, total_q: int) -> dict:
    """Evaluates faithfulness by extracting claims via LLM, scored deterministically by Python."""
    if not contexts or contexts == ["No context retrieved"]:
        return {
            "faithfulness": 1.0,
            "faithfulness_status": "success",
            "faithfulness_claims_total": 0,
            "faithfulness_claims_supported": 0,
            "faithfulness_claims": [],
            "faithfulness_error": None
        }

    context_str = "\n---\n".join(contexts)
    prompt = f"""You are an AI auditor. Your ONLY task is to extract factual claims from the RESPONSE and verify if they are supported by the CONTEXT.

CONTEXT:
{context_str}

RESPONSE:
{response}

INSTRUCTIONS:
1. Extract distinct factual claims made in the RESPONSE. Ignore greetings and opinions.
2. For each claim, check if it is directly supported by the CONTEXT.
3. Return ONLY a valid JSON object matching this exact schema:
{{
  "claims": [
    {{"claim": "extracted statement text", "supported": true/false}}
  ]
}}
Do not return markdown formatting. Do not return a numerical score."""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[FAITHFULNESS] Question {q_index}/{total_q}")
        print(f"[FAITHFULNESS] Attempt {attempt}/{MAX_RETRIES}")
        
        try:
            res = await asyncio.wait_for(
                eval_llm.ainvoke([
                    SystemMessage(content="You are a strict JSON-only evaluation judge."),
                    HumanMessage(content=prompt)
                ]),
                timeout=FAITHFULNESS_TIMEOUT
            )
            
            parsed = robust_json_parse(res.content.strip())
            claims = parsed["claims"]
            total_claims = len(claims)
            
            if total_claims == 0:
                score = 1.0
                supported = 0
            else:
                supported = sum(1 for c in claims if c.get("supported") is True)
                score = supported / total_claims

            print(f"[FAITHFULNESS] Success")
            print(f"[FAITHFULNESS] Claims: {total_claims}")
            print(f"[FAITHFULNESS] Supported: {supported}")
            print(f"[FAITHFULNESS] Score: {score:.4f}")
            
            return {
                "faithfulness": score,
                "faithfulness_status": "success",
                "faithfulness_claims_total": total_claims,
                "faithfulness_claims_supported": supported,
                "faithfulness_claims": claims,
                "faithfulness_error": None
            }
            
        except asyncio.TimeoutError:
            print(f"[FAITHFULNESS] Timeout")
            error_status = "timeout"
            err_msg = "Exceeded FAITHFULNESS_TIMEOUT"
        except Exception as e:
            err_str = str(e).lower()
            if "524" in err_str or "cloudflare" in err_str:
                print(f"[FAITHFULNESS] Cloudflare 524")
                error_status = "cloudflare_524"
            elif "json" in err_str:
                print(f"[FAITHFULNESS] Invalid JSON structure")
                error_status = "invalid_json"
            else:
                print(f"[FAITHFULNESS] Connection/Server Error")
                error_status = "server_error"
            err_msg = str(e)

        if attempt < MAX_RETRIES:
            wait_time = 2 ** attempt
            print(f"[FAITHFULNESS] Retrying in {wait_time}s")
            await asyncio.sleep(wait_time)
            
    print(f"[FAITHFULNESS] FAILED")
    print(f"[FAITHFULNESS] Score: null")
    return {
        "faithfulness": None,
        "faithfulness_status": error_status,
        "faithfulness_claims_total": None,
        "faithfulness_claims_supported": None,
        "faithfulness_claims": [],
        "faithfulness_error": err_msg
    }

async def stage_3_faithfulness(predictions: list, checkpoint_file: str) -> list:
    """Runs bounded, isolated faithfulness scoring with checkpoints."""
    results = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            results = json.load(f)

    # Convert checkpoint to dictionary for quick lookup
    completed = {i: res for i, res in enumerate(results) if res.get("faithfulness_status") == "success"}
    
    # Pre-fill results array to match predictions length
    while len(results) < len(predictions):
        results.append({})

    judge_llm = ChatOpenAI(
        base_url=COMPANY_BASE_URL,
        api_key="not-needed",
        model=MODEL_NAME,
        temperature=0.0,
        max_retries=0, # Handled by our custom loop
        timeout=FAITHFULNESS_TIMEOUT
    )

    # Run sequentially (Concurrency=1) to ensure extreme stability
    for i, pred in enumerate(predictions):
        if i in completed:
            continue # Skip successful evaluations
            
        result_dict = await evaluate_single_faithfulness(
            judge_llm, 
            pred["response"], 
            pred["contexts"], 
            i + 1, 
            len(predictions)
        )
        
        results[i] = result_dict
        
        # Checkpoint after EVERY question
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"[FAITHFULNESS] Checkpoint saved\n[FAITHFULNESS] Continuing to question {i+2 if i+1 < len(predictions) else 'end'}")

    return results

# -------------------------------------------------------------------------
# STAGE 4: Final Merge and Summary
# -------------------------------------------------------------------------
def stage_4_merge_and_summarize(predictions, ragas_results, faithfulness_results, output_dir):
    """Combines all isolated states into final CSV and computes correct statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge
    merged_data = []
    for i in range(len(predictions)):
        row = {**predictions[i]}
        if i < len(ragas_results):
            # Extract standard RAGAS scores, ignoring duplicate input columns
            for k, v in ragas_results[i].items():
                if k not in row:
                    row[k] = v
        if i < len(faithfulness_results):
            row.update(faithfulness_results[i])
        merged_data.append(row)

    df = pd.DataFrame(merged_data)
    df.to_csv(os.path.join(output_dir, "per_question_v2.csv"), index=False)
    df.to_json(os.path.join(output_dir, "per_question_v2.json"), orient="records", indent=4)

    # Compute Summary
    total_q = len(df)
    faith_evaluated = df['faithfulness'].notnull().sum()
    
    summary = {
        "total_questions": total_q,
        "faithfulness_evaluated": int(faith_evaluated),
        "faithfulness_failed": int(total_q - faith_evaluated),
        "faithfulness_coverage": float(faith_evaluated / total_q) if total_q > 0 else 0,
        "faithfulness_mean": float(df['faithfulness'].mean(skipna=True)) if faith_evaluated > 0 else None
    }
    
    # Add RAGAS metric means
    for col in ["answer_relevancy", "llm_context_precision_with_reference", "context_recall"]:
        if col in df.columns:
            summary[f"{col}_mean"] = float(df[col].mean(skipna=True))

    with open(os.path.join(output_dir, "summary_v2.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n=== FINAL ARCHITECTURE EVALUATION COMPLETE ===")
    print(json.dumps(summary, indent=4))

# -------------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------------
def run_final_eval():
    output_dir = "evaluation/results/final"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_path = "evaluation/datasets/golden_v1.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    # STAGE 1
    pred_file = os.path.join(output_dir, "predictions_checkpoint.json")
    predictions = asyncio.run(stage_1_generation(golden_data, pred_file))

    # STAGE 2
    ragas_file = os.path.join(output_dir, "ragas_checkpoint.json")
    ragas_results = stage_2_ragas(predictions, ragas_file)

    # STAGE 3
    faith_file = os.path.join(output_dir, "faithfulness_checkpoint.json")
    faithfulness_results = asyncio.run(stage_3_faithfulness(predictions, faith_file))

    # STAGE 4
    stage_4_merge_and_summarize(predictions, ragas_results, faithfulness_results, output_dir)

if __name__ == "__main__":
    import platform
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    run_final_eval()