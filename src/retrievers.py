# """
# Singleton factories for the adaptive retrieval pipelines.
# """

# import os
# import re
# import json
# import pickle
# import concurrent.futures
# from langchain_chroma import Chroma
# from langchain_classic.retrievers import (
#     EnsembleRetriever,
#     ContextualCompressionRetriever,
# )
# from langchain_classic.retrievers.document_compressors import (
#     CrossEncoderReranker,
# )
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage, HumanMessage
# from src.ingestion import get_embedding_function, VECTOR_DB_DIR, BM25_PATH

# _SIMPLE_RETRIEVER_CACHE = None
# _COMPLEX_RETRIEVER_CACHE = None

# def _get_base_ensemble():
#     """Initializes the base Dense + Sparse ensemble."""
#     vectorstore = Chroma(
#         persist_directory=VECTOR_DB_DIR,
#         embedding_function=get_embedding_function(),
#     )
#     dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

#     with open(BM25_PATH, "rb") as f:
#         bm25_retriever = pickle.load(f)
#     bm25_retriever.k = 5

#     return EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], weights=[0.4, 0.6])

# def get_simple_retriever():
#     """Returns the fast Hybrid + Cross-Encoder pipeline."""
#     global _SIMPLE_RETRIEVER_CACHE
#     if _SIMPLE_RETRIEVER_CACHE is not None:
#         return _SIMPLE_RETRIEVER_CACHE

#     print("[SYSTEM] Booting Simple Retriever...")
#     ensemble = _get_base_ensemble()
#     cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
#     compressor = CrossEncoderReranker(model=cross_encoder, top_n=4)
    
#     _SIMPLE_RETRIEVER_CACHE = ContextualCompressionRetriever(
#         base_compressor=compressor, 
#         base_retriever=ensemble
#     )
#     return _SIMPLE_RETRIEVER_CACHE


# # --- NEW: Custom Class to handle Map-Reduce Retrieval seamlessly ---
# class SubQueryComplexRetriever:
#     def __init__(self, ensemble, compressor, llm):
#         self.ensemble = ensemble
#         self.compressor = compressor
#         self.llm = llm

#     def invoke(self, query: str) -> list:
#         # 1. Decompose the multi-part question
#         prompt = f"""Break down the user's input into separate, standalone search queries for an HR database.
#         - Split compound questions into separate standalone queries (1 query per distinct topic).
#         - Preserve acronyms and entities exactly as provided by the user (do not arbitrarily change or translate acronyms).
#         - Use broad, descriptive domain terminology where appropriate to maximize search hits.  
#         - If it is a single question, return a list with just that one query.
#         - Maximum 10 sub-queries. Fix any typos (e.g. 'anual' -> 'annual').

#         User Query: "{query}"
        
#         Output ONLY valid JSON matching this schema:
#         {{"sub_queries": ["query 1", "query 2"]}}"""
        
#         try:
#             res = self.llm.invoke([SystemMessage(content="Output JSON only."), HumanMessage(content=prompt)])
            
#             # [FIX]: Robust JSON extraction that ignores markdown fences
#             content = res.content.strip()
#             # Find the first '{' and the last '}'
#             start_idx = content.find('{')
#             end_idx = content.rfind('}')
            
#             if start_idx != -1 and end_idx != -1:
#                 json_str = content[start_idx:end_idx+1]
#                 sub_queries = json.loads(json_str).get("sub_queries", [query])
#             else:
#                 sub_queries = [query]
#                 print("[COMPLEX RETRIEVER] Warning: No JSON object found in LLM response.")
                
#         except Exception as e:
#             print(f"[COMPLEX RETRIEVER] Decomposition failed: {e}")
#             sub_queries = [query]

#         print(f"\n[COMPLEX RETRIEVER] Sub-Queries Generated ({len(sub_queries)}):")
#         for idx, sq in enumerate(sub_queries, 1):
#             print(f"  {idx}. '{sq}'")

#         # 2. Parallel execution: Retrieve and Re-rank for EACH sub-query individually
#         def process_sq(sq):
#             docs = self.ensemble.invoke(sq)
#             if not docs: 
#                 return []
#             # Compress specifically against the sub-query!
#             compressed = self.compressor.compress_documents(docs, sq)
#             print(f"  -> Sub-Query '{sq}' yielded {len(compressed)} chunks after re-ranking.")
#             return compressed
            
#         final_docs = []
#         seen = set()
        
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             results = list(executor.map(process_sq, sub_queries))
            
#         # 3. Combine and Deduplicate
#         for res_list in results:
#             for doc in res_list:
#                 if doc.page_content not in seen:
#                     seen.add(doc.page_content)
#                     final_docs.append(doc)
                    
#         print(f"[COMPLEX RETRIEVER] Total unique chunks delivered: {len(final_docs)}")
#         return final_docs


# def get_complex_retriever():
#     """Returns the thorough Map-Reduce Sub-Query + Hybrid + Cross-Encoder pipeline."""
#     global _COMPLEX_RETRIEVER_CACHE
#     if _COMPLEX_RETRIEVER_CACHE is not None:
#         return _COMPLEX_RETRIEVER_CACHE

#     print("[SYSTEM] Booting Complex Retriever...")
#     ensemble = _get_base_ensemble()
    
#     mq_llm = ChatOpenAI(
#         base_url="https://relation-creature-tap-bradley.trycloudflare.com/v1",
#         api_key="not-needed",
#         model="qwen3:30b",
#         temperature=0.0
#     )
    
#     cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
#     # Take top 5 documents PER sub-query (Total max 50 docs for a 10-part question)
#     compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

#     # Return our custom wrapper class that perfectly mimics LangChain's .invoke()
#     _COMPLEX_RETRIEVER_CACHE = SubQueryComplexRetriever(
#         ensemble=ensemble,
#         compressor=compressor,
#         llm=mq_llm
#     )
    
#     return _COMPLEX_RETRIEVER_CACHE

"""
Unified Retriever Registry & Strategy Switchboard.
Implements Atomic Baselines -> Champion 'Main' -> Main-Extended Combinations.
Uses local Cloudflare Tunnel (qwen3:30b) for all decomposition, HyDE, and extraction.
Features Singleton RAM Caching and Execution Telemetry.
"""

import os
import re
import json
import pickle
import concurrent.futures
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_openai import ChatOpenAI

from src.ingestion import (
    get_embedding_function,
    VECTOR_DB_DIR,
    BM25_PATH,
    PARENT_STORE_DIR
)

# --- NEW IMPORTS FOR PRODUCTION RESILIENCE ---
import httpx
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type
from langsmith import traceable

# =====================================================================
# 1. Singletons, Caches & Cloudflare Tunnel Configuration
# =====================================================================
CLOUDFLARE_BASE_URL = os.getenv(
    "CLOUDFLARE_LLM_URL",
    "https://relation-creature-tap-bradley.trycloudflare.com/v1"
)
CLOUDFLARE_MODEL_NAME = os.getenv("CLOUDFLARE_LLM_MODEL", "qwen3:30b")

_CHROMA_DB = None
_BM25_RETRIEVER = None
_CROSS_ENCODER = None
_CLOUDFLARE_LLM = None
_PARENT_STORE = None

# RAM Cache for initialized retriever instances to ensure zero overhead on switching
_RETRIEVER_INSTANCE_CACHE: Dict[str, Any] = {}

# =====================================================================
# 1. Resilient LLM Invocation Wrapper
# =====================================================================
@retry(
    # Wait 2^x seconds, add jitter, up to 10s max between retries
    wait=wait_exponential_jitter(initial=2, max=10),
    # Stop after 5 total attempts
    stop=stop_after_attempt(5),
    # Catch typical connection drops and timeouts
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, Exception)),
    reraise=True
)
def robust_llm_invoke(llm: ChatOpenAI, messages: list):
    """
    Production-grade LLM execution with exponential backoff and jitter.
    Prevents pipeline failure during transient network drops or rate limits.
    """
    return llm.invoke(messages)

def get_cloudflare_llm() -> ChatOpenAI:
    """Singleton factory for the local Cloudflare tunnel LLM."""
    global _CLOUDFLARE_LLM
    if _CLOUDFLARE_LLM is None:
        _CLOUDFLARE_LLM = ChatOpenAI(
            base_url=CLOUDFLARE_BASE_URL,
            api_key=os.getenv("CLOUDFLARE_API_KEY", "not-needed"),
            model=CLOUDFLARE_MODEL_NAME,
            temperature=0.0
        )
    return _CLOUDFLARE_LLM


def get_chroma_db() -> Chroma:
    """Loads and caches the persisted ChromaDB vector database."""
    global _CHROMA_DB
    if _CHROMA_DB is None:
        print("[SYSTEM CACHE] Loading ChromaDB vector store into RAM...")
        _CHROMA_DB = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=get_embedding_function(),
            collection_metadata={"hnsw:space": "cosine"}
        )
    return _CHROMA_DB


def get_bm25_retriever() -> BM25Retriever:
    """Loads and caches the BM25 sparse keyword retriever."""
    global _BM25_RETRIEVER
    if _BM25_RETRIEVER is None:
        if not os.path.exists(BM25_PATH):
            raise FileNotFoundError(f"BM25 index missing at {BM25_PATH}. Run python -m src.ingestion first.")
        print("[SYSTEM CACHE] Loading BM25 sparse index into RAM...")
        with open(BM25_PATH, "rb") as f:
            _BM25_RETRIEVER = pickle.load(f)
        _BM25_RETRIEVER.k = 5  # Default top-k for BM25 retrieval
    return _BM25_RETRIEVER


def get_base_ensemble(dense_k: int = 10, bm25_k: int = 10) -> EnsembleRetriever:
    """Returns the foundational Hybrid (BM25 40% + Dense 60%) retriever."""
    vectorstore = get_chroma_db()
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": dense_k})
    bm25 = get_bm25_retriever()
    bm25.k = bm25_k
    return EnsembleRetriever(retrievers=[bm25, dense_retriever], weights=[0.4, 0.6])


def get_cross_encoder_reranker(top_n: int = 6) -> CrossEncoderReranker:
    """Loads and caches the HuggingFace Cross-Encoder re-ranker."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        print("[SYSTEM CACHE] Loading HuggingFace Cross-Encoder model into RAM...")
        _CROSS_ENCODER = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoderReranker(model=_CROSS_ENCODER, top_n=top_n)


def get_parent_store() -> Dict[str, Dict[str, Any]]:
    """Loads full parent documents saved during layout-aware ingestion."""
    global _PARENT_STORE
    if _PARENT_STORE is None:
        parent_pkl = PARENT_STORE_DIR / "parent_documents.pkl"
        if not parent_pkl.exists():
            return {}
        with open(parent_pkl, "rb") as f:
            _PARENT_STORE = pickle.load(f)
    return _PARENT_STORE


def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """Preserves order while removing duplicate documents."""
    seen = set()
    unique = []
    for d in docs:
        content_key = d.metadata.get("chunk_id") or d.page_content.strip()
        if content_key not in seen:
            seen.add(content_key)
            unique.append(d)
    return unique


# =====================================================================
# 2. Strategy Implementations with State Telemetry
# =====================================================================

class DenseRetriever:
    def __init__(self, k: int = 8):
        self.k = k
        self.db = get_chroma_db()

    def invoke(self, query: str, **kwargs) -> List[Document]:
        print(f"\n [DENSE RETRIEVER] Searching top {self.k} vector matches for: '{query}'")
        docs = self.db.similarity_search(query, k=self.k)
        print(f" [DENSE RETRIEVER] Delivered {len(docs)} chunks.")
        return docs


class HybridRetriever:
    def __init__(self, dense_k: int = 10, bm25_k: int = 10):
        self.ensemble = get_base_ensemble(dense_k=dense_k, bm25_k=bm25_k)

    def invoke(self, query: str, **kwargs) -> List[Document]:
        print(f"\n [HYBRID RETRIEVER] Executing BM25 (40%) + Dense (60%) for: '{query}'")
        docs = self.ensemble.invoke(query)
        print(f" [HYBRID RETRIEVER] Delivered {len(docs)} combined chunks.")
        return docs


class HybridRerankRetriever:
    def __init__(self, top_n: int = 6):
        self.ensemble = get_base_ensemble(dense_k=10, bm25_k=10)
        self.compressor = get_cross_encoder_reranker(top_n=top_n)
        self.pipeline = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.ensemble
        )

    @traceable(run_type="retriever", name="Hybrid_Reranker_retriever")
    def invoke(self, query: str, **kwargs) -> List[Document]:
        print(f"\n [HYBRID + RERANK] Pulling candidates and scoring with Cross-Encoder for: '{query}'")
        docs = self.pipeline.invoke(query)
        print(f" [HYBRID + RERANK] Cross-Encoder selected top {len(docs)} high-precision chunks.")
        return docs


class MainMapReduceRetriever:
    def __init__(self, top_n_per_subquery: int = 6):
        self.llm = get_cloudflare_llm()
        self.ensemble = get_base_ensemble(dense_k=10, bm25_k=10)
        self.compressor = get_cross_encoder_reranker(top_n=top_n_per_subquery)

    @traceable(run_type="chain", name="Decompose_Query")
    def decompose(self, query: str) -> List[str]:
        prompt = f"""Break down the user's input into separate, standalone search queries for an HR database.
- Split compound questions into separate standalone queries (1 query per distinct topic).
- Preserve acronyms and entities exactly as provided by the user (e.g. OPD, IPD, PF).
- If it is a single question, return a list with just that one query.
- Maximum 6 sub-queries. Fix any obvious typos.

User Query: "{query}"

Output ONLY valid JSON matching this schema:
{{"sub_queries": ["query 1", "query 2"]}}"""
        try:
            # [UPDATED]: Use the resilient wrapper instead of direct invocation
            res = robust_llm_invoke(
                self.llm,
                [
                    SystemMessage(content="Output valid JSON object only."),
                    HumanMessage(content=prompt)
                ]
            )
            content = res.content.strip()
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(content[start : end + 1])
                return data.get("sub_queries", [query])
        except Exception as e:
            print(f"  [Main Retriever] Decomposition fallback: {e}")
        return [query]

    @traceable(run_type="retriever", name="MapReduce_SubQuery_Search")
    def invoke(self, query: str, **kwargs) -> List[Document]:
        print(f"\n🧩 [MAIN CHAMPION] Decomposing complex query: '{query[:100]}...'")
        sub_queries = self.decompose(query)
        print(f"   -> Generated {len(sub_queries)} sub-queries:")
        for idx, sq in enumerate(sub_queries, 1):
            print(f"     {idx}. '{sq}'")

        per_query_docs = []
        
        # STRICTLY SEQUENTIAL EXECUTION: Collect top chunks per sub-query independently
        for sq in sub_queries:
            print(f"   -> [SUB-QUERY SEARCH] Executing hybrid retrieval for: '{sq}'")
            docs = self.ensemble.invoke(sq)
            
            if docs:
                compressed = self.compressor.compress_documents(docs, sq)
                print(f"   -> [CROSS-ENCODER] Sub-query '{sq}' narrowed to {len(compressed)} chunks.")
                # Guarantee each sub-query contributes up to 2 top chunks to prevent starvation
                per_query_docs.extend(compressed[:2])

        unique = deduplicate_documents(per_query_docs)
        
        print(f"✅ [MAIN CHAMPION] Delivered {len(unique)} balanced parent pages across {len(sub_queries)} sub-queries.")
        return unique


class SelfQueryFilter(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="One of: 'health_insurance', 'provident_fund', 'leaves_attendance', 'travel_allowance', 'general_policy', or null."
    )
    benefit_type: Optional[str] = Field(
        default=None,
        description="One of: 'opd', 'ipd', 'discount_center', 'exclusions', 'provident_fund', or null."
    )
    has_form: Optional[bool] = Field(default=None, description="True if query explicitly asks for a form/checklist/template.")


class MainSelfQueryRetriever:
    def __init__(self):
        self.main_retriever = MainMapReduceRetriever()
        self.llm = get_cloudflare_llm()
        self.db = get_chroma_db()

    @traceable(run_type="chain", name="Extract_Metadata_Filter")
    def _extract_filter(self, query: str) -> Optional[Dict[str, Any]]:
        prompt = f"""Extract metadata filters for this HR question.
Question: "{query}"

Available Categories: health_insurance, provident_fund, leaves_attendance, travel_allowance, general_policy
Benefit Types: opd, ipd, discount_center, exclusions, provident_fund

Output ONLY valid JSON matching this schema:
{{"category": "...", "benefit_type": "...", "has_form": false}} (Use null for unmentioned fields)"""
        try:
            res = robust_llm_invoke(
                self.llm,
                [SystemMessage(content="Output JSON only."), HumanMessage(content=prompt)]
            )
            content = res.content.strip()
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(content[start : end + 1])
                conds = []
                if data.get("category"): conds.append({"category": data["category"]})
                if data.get("benefit_type"): conds.append({"benefit_type": data["benefit_type"]})
                if data.get("has_form") is not None: conds.append({"has_form": data["has_form"]})
                if len(conds) == 1: return conds[0]
                if len(conds) > 1: return {"$and": conds}
        except Exception:
            pass
        return None
    
    @traceable(run_type="retriever", name="SelfQuery_Execution")
    def invoke(self, query: str, **kwargs) -> List[Document]:
        print(f"\n [SELF QUERY] Extracting structured metadata constraints...")
        where_filter = self._extract_filter(query)
        if where_filter:
            print(f"  -> [Filter Applied]: {where_filter}")
            filtered_docs = self.db.similarity_search(query, k=4, filter=where_filter)
            main_docs = self.main_retriever.invoke(query)
            return deduplicate_documents(filtered_docs + main_docs)
        print(f"  -> [No Strict Filter Detected] Executing base main retrieval.")
        return self.main_retriever.invoke(query)


class MainParentDocumentRetriever:
    def __init__(self):
        self.main_retriever = MainMapReduceRetriever()
        self.parent_store = get_parent_store()

    @traceable(run_type="retriever", name="Parent_Doc_Hydration")
    def invoke(self, query: str, **kwargs) -> List[Document]:
        print(f"\n📂 [PARENT DOC] Initializing mapped retrieval...")
        child_docs = self.main_retriever.invoke(query)
        hydrated_docs = []

        print(f"  -> Hydrating {len(child_docs)} child chunks into full parent pages...")
        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id in self.parent_store:
                p_data = self.parent_store[parent_id]
                hydrated_docs.append(
                    Document(
                        page_content=f"# {p_data['doc_title']} (Page {p_data['page_number']})\n\n{p_data['full_text']}",
                        metadata={**doc.metadata, "is_parent_hydrated": True}
                    )
                )
            else:
                hydrated_docs.append(doc)

        unique = deduplicate_documents(hydrated_docs)
        print(f" [PARENT DOC] Delivered {len(unique)} fully hydrated pages.")
        return unique


class MainMultiQueryRetriever:
    def __init__(self):
        self.main_retriever = MainMapReduceRetriever()
        self.llm = get_cloudflare_llm()

    @traceable(run_type="retriever", name="MultiQuery_Expansion")
    def invoke(self, query: str, **kwargs) -> List[Document]:
        print(f"\n [MULTI-QUERY] Expanding perspectives...")
        prompt = f"""Generate 2 alternative phrasing variations for this HR query to improve vector retrieval.
Query: "{query}"
Output ONLY a JSON array of strings:
{{"variations": ["variation 1", "variation 2"]}}"""
        variations = [query]
        try:
            res = robust_llm_invoke(
                self.llm,
                [SystemMessage(content="Output JSON only."), HumanMessage(content=prompt)]
            )
            content = res.content.strip()
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1:
                variations += json.loads(content[start : end + 1]).get("variations", [])
        except Exception:
            pass

        print(f"  -> Exploring {len(variations)} perspectives.")
        all_docs = []
        for q_var in variations[:3]:
            all_docs.extend(self.main_retriever.invoke(q_var))
        
        unique = deduplicate_documents(all_docs)
        print(f" [MULTI-QUERY] Consolidated {len(unique)} chunks across perspectives.")
        return unique


class MainRAGFusionRetriever:
    def __init__(self, rrf_constant: int = 60, top_n: int = 5):
        self.main_retriever = MainMapReduceRetriever()
        self.compressor = get_cross_encoder_reranker(top_n=top_n)
        self.llm = get_cloudflare_llm()
        self.rrf_k = rrf_constant

    @traceable(run_type="retriever", name="RAG_Fusion_RRF_Scoring")
    def invoke(self, query: str, **kwargs) -> List[Document]:
        print(f"\n [RAG FUSION] Decomposing query and computing Reciprocal Rank Fusion...")
        sub_queries = self.main_retriever.decompose(query)
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for sq in sub_queries:
            ranked_list = self.main_retriever.ensemble.invoke(sq)
            for rank, doc in enumerate(ranked_list):
                doc_id = doc.metadata.get("chunk_id") or doc.page_content[:100]
                doc_map[doc_id] = doc
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (rank + self.rrf_k))

        sorted_ids = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        top_candidates = [doc_map[did] for did in sorted_ids[:10]]
        print(f" [RAG FUSION] Scored {len(doc_map)} total unique candidates via RRF formula.")
        
        for did in sorted_ids[:3]:
            print(f"   -> Top RRF Score: {doc_scores[did]:.4f}")

        if top_candidates:
            print(f" [RAG FUSION] Passing top {len(top_candidates)} candidates to Cross-Encoder...")
            final_docs = self.compressor.compress_documents(top_candidates, query)
            print(f" [RAG FUSION] Final delivered chunks: {len(final_docs)}")
            return final_docs
        return []


class MainHyDERetriever:
    def __init__(self):
        self.main_retriever = MainMapReduceRetriever()
        self.llm = get_cloudflare_llm()

    @traceable(run_type="retriever", name="HyDE_Retrieval")
    def invoke(self, query: str, **kwargs) -> List[Document]:
        print(f"\n [HyDE] Generating hypothetical HR policy for query...")
        prompt = f"""

You are a NextBridge company assistant.
 
Write short, authoritative, factual, policy-document-style passage that would
plausibly answer the question below, as if it were an excerpt from
an official NextBridge HR or company document.
 
Rules:
 
1. Write it as a declarative statement, not a question.
2. Do not say "I don't know" or hedge — invent plausible specific
   details in the style of a real policy document even if you are
   unsure of the exact figures.
3. Keep it to 2-4 sentences.
4. Return only the passage, nothing else.

Question: "{query}"
Policy Excerpt:"""
        try:
            res = robust_llm_invoke(
                self.llm,
                [
                    SystemMessage(content="Write formal HR policy text."),
                    HumanMessage(content=prompt)
                ]
            )
            hypothetical_passage = res.content.strip()
            print(f" [HyDE] Drafted Passage Preview:\n   \"{hypothetical_passage[:350]}...\"")
            
            print(f" [HyDE] Executing semantic search using hypothetical vector...")
            augmented_query = f"{query}\n{hypothetical_passage[:500]}"
            docs = self.main_retriever.invoke(augmented_query)
            print(f" [HyDE] Delivered {len(docs)} chunks mapped from hypothetical space.")
            return docs
        except Exception:
            return self.main_retriever.invoke(query)


# =====================================================================
# 3. Central Factory Switchboard & Singleton Logic
# =====================================================================
RETRIEVER_REGISTRY: Dict[str, Any] = {
    "dense": DenseRetriever,
    "hybrid": HybridRetriever,
    "hybrid_rerank": HybridRerankRetriever,
    "main": MainMapReduceRetriever,
    "hybrid_multi_sub_rerank": MainMapReduceRetriever,
    "main_self_query": MainSelfQueryRetriever,
    "main_parent_doc": MainParentDocumentRetriever,
    "main_multi_query": MainMultiQueryRetriever,
    "main_rag_fusion": MainRAGFusionRetriever,
    "main_hyde": MainHyDERetriever,
}


def get_retriever(strategy_name: Optional[str] = None):
    """
    Factory function with Singleton Caching.
    Boots the retriever strategy once into RAM, then serves instances instantly.
    """
    selected = strategy_name or os.getenv("ACTIVE_RETRIEVER", "main")
    selected = selected.lower().strip()

    if selected not in RETRIEVER_REGISTRY:
        available = list(RETRIEVER_REGISTRY.keys())
        raise ValueError(f"Unknown retriever strategy '{selected}'. Available: {available}")

    # Return from memory cache if already initialized
    if selected in _RETRIEVER_INSTANCE_CACHE:
        return _RETRIEVER_INSTANCE_CACHE[selected]

    # Initialize once and cache in RAM
    print(f"\n[SYSTEM CACHE] Initializing strategy '{selected.upper()}' for the first time...")
    factory = RETRIEVER_REGISTRY[selected]
    instance = factory() if callable(factory) else factory
    _RETRIEVER_INSTANCE_CACHE[selected] = instance
    return instance