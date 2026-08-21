"""
Final Agentic RAGAS Evaluation
Benchmarks the new CRAG LangGraph against the Golden Dataset using identical custom metrics.
"""

import sys
import os

# Add the parent root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
import asyncio
import httpx
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

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

# Import the NEW Agentic Graph
from src.graph import builder

load_dotenv()

# Pointing to a new LangSmith project for the Final Architecture
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "NextBridge_HR_Agent_Final"

COMPANY_BASE_URL = "https://relation-creature-tap-bradley.trycloudflare.com/v1"
MODEL_NAME = "qwen3:30b"

# -------------------------------------------------------------------------
# Custom Lightweight Faithfulness Evaluator (Identical to Baseline)
# -------------------------------------------------------------------------
async def evaluate_single_faithfulness(eval_llm: ChatOpenAI, response: str, contexts: list) -> float:
    """Evaluates faithfulness by extracting and verifying claims in a single LLM pass."""
    if not contexts or contexts == ["No context retrieved"]:
        return 0.0

    context_str = "\n---\n".join(contexts)
    prompt = f"""You are an unbiased AI auditor evaluating RAG faithfulness.
Your task: Determine whether statements in the RESPONSE are supported by the provided CONTEXT.

CONTEXT:
{context_str}

RESPONSE:
{response}

INSTRUCTIONS:
1. Extract distinct factual claims made in the RESPONSE.
2. For each claim, check if it is directly supported by the CONTEXT.
3. Return ONLY a valid JSON object matching this schema:
{{
  "claims": [
    {{"claim": "statement text", "supported": true}}
  ],
  "faithfulness_score": 1.0
}}

If there are no factual claims, return "faithfulness_score": 1.0.
Do not output markdown codeblocks (no ```json). Output raw JSON only."""

    try:
        res = await eval_llm.ainvoke([
            SystemMessage(content="You are a strict JSON-only evaluation judge."),
            HumanMessage(content=prompt)
        ])
        content = res.content.strip()
        
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return float(parsed.get("faithfulness_score", 0.0))
        return 0.0
    except Exception as e:
        print(f"[Warning] Custom faithfulness evaluation failed for item: {e}")
        return 0.0


async def run_custom_faithfulness_batch(eval_llm: ChatOpenAI, responses: list, contexts_list: list) -> list:
    """Runs faithfulness scoring with concurrency control."""
    semaphore = asyncio.Semaphore(2) 

    async def sem_eval(resp, ctx):
        async with semaphore:
            return await evaluate_single_faithfulness(eval_llm, resp, ctx)

    tasks = [sem_eval(resp, ctx) for resp, ctx in zip(responses, contexts_list)]
    return await asyncio.gather(*tasks)


# -------------------------------------------------------------------------
# Agent Generation (Updated for CRAG State Schema)
# -------------------------------------------------------------------------
def extract_contexts_from_crag_state(state):
    """Extracts chunks directly from the new LangGraph state schema."""
    docs = state.get("documents", [])
    if docs:
        return [doc.page_content for doc in docs]
    return ["No context retrieved"]

async def generate_predictions(golden_data):
    questions, ground_truths, answers, contexts_list = [], [], [], []

    # Using an isolated DB for evaluation to avoid corrupting production memory
    async with AsyncSqliteSaver.from_conn_string("hr_agent_memory_eval.db") as memory:
        await memory.setup()
        hr_graph = builder.compile(checkpointer=memory)

        for i, item in enumerate(golden_data):
            print(f"Processing query {i+1}/{len(golden_data)}: {item['question']}")
            config = {"configurable": {"thread_id": f"eval_final_{i}"}}
            
            result_state = await hr_graph.ainvoke(
                {"messages": [HumanMessage(content=item['question'])]}, 
                config=config
            )
            
            answers.append(result_state["messages"][-1].content)
            contexts_list.append(extract_contexts_from_crag_state(result_state))
            questions.append(item['question'])
            ground_truths.append(item['ground_truth'])
            
            # Print loop telemetry for visibility
            print(f"  -> Loops: Retrieval({result_state.get('retrieval_attempts', 0)}), Gen({result_state.get('generation_attempts', 0)})")

    return questions, ground_truths, answers, contexts_list


# -------------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------------
def run_final_eval():
    dataset_path = "evaluation/datasets/golden_v1.json"
    if not os.path.exists(dataset_path):
        print(f"ERROR: Could not find {dataset_path}.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    # 1. Run Graph Inference
    questions, ground_truths, answers, contexts_list = asyncio.run(generate_predictions(golden_data))

    # 2. Build Dataset for standard Ragas metrics
    data = {
        "user_input": questions,        
        "reference": ground_truths,     
        "response": answers,            
        "retrieved_contexts": contexts_list 
    }
    dataset = Dataset.from_dict(data)

    print("\nInitializing NextBridge Evaluators...")
    company_client = AsyncOpenAI(
        base_url=COMPANY_BASE_URL,
        api_key="not-needed",
        max_retries=5,
        http_client=httpx.AsyncClient(
            timeout=500.0,
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=2)
        )
    )
    
    ragas_llm = llm_factory(MODEL_NAME, client=company_client)

    eval_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

    # 3. Standard Ragas Metrics
    metrics = [
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm)
    ]

    ragas_config = RunConfig(max_retries=5, max_wait=60, max_workers=1)

    print("Running RAGAS Core Metrics...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=ragas_config
    )

    df = result.to_pandas()

    # 4. Run Custom Faithfulness Evaluation
    print("Running Custom Faithfulness Evaluation...")
    judge_llm = ChatOpenAI(
        base_url=COMPANY_BASE_URL,
        api_key="not-needed",
        model=MODEL_NAME,
        temperature=0.0,
        max_retries=5,
        timeout=500.0
    )
    faithfulness_scores = asyncio.run(run_custom_faithfulness_batch(judge_llm, answers, contexts_list))
    df["faithfulness"] = faithfulness_scores

    # 5. Export JSON and CSV to the FINAL directory
    os.makedirs("evaluation/results/final", exist_ok=True)
    df.to_csv("evaluation/results/final/per_question.csv", index=False)
    df.to_json("evaluation/results/final/per_question.json", orient="records", indent=4)
    
    summary = df.mean(numeric_only=True).to_dict()
    summary["total_evaluated"] = len(df)
    
    with open("evaluation/results/final/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n=== FINAL ARCHITECTURE EVALUATION COMPLETE ===")
    print(json.dumps(summary, indent=4))
    print("\nCompare these results in evaluation/results/final/ against evaluation/results/baseline/")

if __name__ == "__main__":
    import platform
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    run_final_eval()