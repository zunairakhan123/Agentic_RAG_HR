"""
Singleton factories for the adaptive retrieval pipelines.
"""

import os
import re
import json
import pickle
import concurrent.futures
from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker,
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.ingestion import get_embedding_function, VECTOR_DB_DIR, BM25_PATH

_SIMPLE_RETRIEVER_CACHE = None
_COMPLEX_RETRIEVER_CACHE = None

def _get_base_ensemble():
    """Initializes the base Dense + Sparse ensemble."""
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=get_embedding_function(),
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 5

    return EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], weights=[0.4, 0.6])

def get_simple_retriever():
    """Returns the fast Hybrid + Cross-Encoder pipeline."""
    global _SIMPLE_RETRIEVER_CACHE
    if _SIMPLE_RETRIEVER_CACHE is not None:
        return _SIMPLE_RETRIEVER_CACHE

    print("[SYSTEM] Booting Simple Retriever...")
    ensemble = _get_base_ensemble()
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=4)
    
    _SIMPLE_RETRIEVER_CACHE = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble
    )
    return _SIMPLE_RETRIEVER_CACHE


# --- NEW: Custom Class to handle Map-Reduce Retrieval seamlessly ---
class SubQueryComplexRetriever:
    def __init__(self, ensemble, compressor, llm):
        self.ensemble = ensemble
        self.compressor = compressor
        self.llm = llm

    def invoke(self, query: str) -> list:
        # 1. Decompose the multi-part question
        prompt = f"""Break down the user's input into separate, standalone search queries for an HR database.
        - Split compound questions into separate standalone queries (1 query per distinct topic).
        - Preserve acronyms and entities exactly as provided by the user (do not arbitrarily change or translate acronyms).
        - Use broad, descriptive domain terminology where appropriate to maximize search hits.  
        - If it is a single question, return a list with just that one query.
        - Maximum 10 sub-queries. Fix any typos (e.g. 'anual' -> 'annual').

        User Query: "{query}"
        
        Output ONLY valid JSON matching this schema:
        {{"sub_queries": ["query 1", "query 2"]}}"""
        
        try:
            res = self.llm.invoke([SystemMessage(content="Output JSON only."), HumanMessage(content=prompt)])
            
            # [FIX]: Robust JSON extraction that ignores markdown fences
            content = res.content.strip()
            # Find the first '{' and the last '}'
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx+1]
                sub_queries = json.loads(json_str).get("sub_queries", [query])
            else:
                sub_queries = [query]
                print("[COMPLEX RETRIEVER] Warning: No JSON object found in LLM response.")
                
        except Exception as e:
            print(f"[COMPLEX RETRIEVER] Decomposition failed: {e}")
            sub_queries = [query]

        print(f"\n[COMPLEX RETRIEVER] Sub-Queries Generated ({len(sub_queries)}):")
        for idx, sq in enumerate(sub_queries, 1):
            print(f"  {idx}. '{sq}'")

        # 2. Parallel execution: Retrieve and Re-rank for EACH sub-query individually
        def process_sq(sq):
            docs = self.ensemble.invoke(sq)
            if not docs: 
                return []
            # Compress specifically against the sub-query!
            compressed = self.compressor.compress_documents(docs, sq)
            print(f"  -> Sub-Query '{sq}' yielded {len(compressed)} chunks after re-ranking.")
            return compressed
            
        final_docs = []
        seen = set()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_sq, sub_queries))
            
        # 3. Combine and Deduplicate
        for res_list in results:
            for doc in res_list:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    final_docs.append(doc)
                    
        print(f"[COMPLEX RETRIEVER] Total unique chunks delivered: {len(final_docs)}")
        return final_docs


def get_complex_retriever():
    """Returns the thorough Map-Reduce Sub-Query + Hybrid + Cross-Encoder pipeline."""
    global _COMPLEX_RETRIEVER_CACHE
    if _COMPLEX_RETRIEVER_CACHE is not None:
        return _COMPLEX_RETRIEVER_CACHE

    print("[SYSTEM] Booting Complex Retriever...")
    ensemble = _get_base_ensemble()
    
    mq_llm = ChatOpenAI(
        base_url="https://relation-creature-tap-bradley.trycloudflare.com/v1",
        api_key="not-needed",
        model="qwen3:30b",
        temperature=0.0
    )
    
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Take top 5 documents PER sub-query (Total max 50 docs for a 10-part question)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

    # Return our custom wrapper class that perfectly mimics LangChain's .invoke()
    _COMPLEX_RETRIEVER_CACHE = SubQueryComplexRetriever(
        ensemble=ensemble,
        compressor=compressor,
        llm=mq_llm
    )
    
    return _COMPLEX_RETRIEVER_CACHE