import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    MultiQueryRetriever,
)
from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker,
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import ChatOpenAI
from src.ingestion import get_embedding_function, VECTOR_DB_DIR, BM25_PATH

def run_experiment(query: str):
    print(f"\n{'='*60}\nEXPERIMENT QUERY: '{query}'\n{'='*60}")
    
    # 1. SETUP MODELS
    vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=get_embedding_function())
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    mq_llm = ChatOpenAI(
        base_url="https://relation-creature-tap-bradley.trycloudflare.com/v1",
        api_key="not-needed",
        model="qwen3:30b",
        temperature=0.0
    )

    # 2. BASELINE DENSE
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    t0 = time.time()
    baseline_docs = dense_retriever.invoke(query)
    t_baseline = time.time() - t0
    
    baseline_ids = [doc.metadata.get("chunk_id", "UNKNOWN")[:8] for doc in baseline_docs]

    # 3. HYBRID ONLY (No LLM)
    bm25.k = 3
    hybrid = EnsembleRetriever(retrievers=[bm25, vectorstore.as_retriever(search_kwargs={"k": 3})], weights=[0.4, 0.6])
    
    t0 = time.time()
    hybrid_docs = hybrid.invoke(query)
    t_hybrid = time.time() - t0
    
    hybrid_ids = [doc.metadata.get("chunk_id", "UNKNOWN")[:8] for doc in hybrid_docs]

    # 4. FINAL PIPELINE (MultiQuery + Hybrid + Reranker)
    bm25.k = 5
    big_hybrid = EnsembleRetriever(retrievers=[bm25, vectorstore.as_retriever(search_kwargs={"k": 5})], weights=[0.4, 0.6])
    mq = MultiQueryRetriever.from_llm(retriever=big_hybrid, llm=mq_llm)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    final_pipeline = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=mq)
    
    t0 = time.time()
    final_docs = final_pipeline.invoke(query)
    t_final = time.time() - t0
    
    final_ids = [doc.metadata.get("chunk_id", "UNKNOWN")[:8] for doc in final_docs]

    # 5. REPORT
    print("\n--- LATENCY ---")
    print(f"Baseline Dense: {t_baseline:.2f}s")
    print(f"Hybrid Only:    {t_hybrid:.2f}s")
    print(f"Final Pipeline: {t_final:.2f}s")

    print("\n--- CHUNK IDs DISCOVERED ---")
    print(f"Baseline Dense: {baseline_ids}")
    print(f"Hybrid Only:    {hybrid_ids}")
    print(f"Final Pipeline: {final_ids}")

    print("\n--- FINAL PIPELINE TOP CHUNKS ---")
    for i, doc in enumerate(final_docs, 1):
        print(f"[{i}] (ID: {doc.metadata.get('chunk_id', 'UNKNOWN')[:8]}) {doc.page_content[:150].strip()}...")
        print("-" * 40)

if __name__ == "__main__":
    # This query uses informal vocabulary ("PTO", "carry over") to test if the MultiQuery LLM 
    # successfully translates it into formal policy terms ("Annual Leave", "Encashment").
    run_experiment("Do I lose my PTO if I don't use it by December, or does it carry over?")
    #A query where expansion does not help or introduces noise
    run_experiment("What is the exact email address for the HR department?")
