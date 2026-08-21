"""
Singleton factories for the adaptive retrieval pipelines.
"""

import os
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
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    
    _SIMPLE_RETRIEVER_CACHE = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble
    )
    return _SIMPLE_RETRIEVER_CACHE

def get_complex_retriever():
    """Returns the thorough MultiQuery + Hybrid + Cross-Encoder pipeline."""
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
    mq_retriever = MultiQueryRetriever.from_llm(retriever=ensemble, llm=mq_llm)
    
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)

    _COMPLEX_RETRIEVER_CACHE = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=mq_retriever
    )
    return _COMPLEX_RETRIEVER_CACHE