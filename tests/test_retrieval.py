import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker,
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from src.ingestion import get_embedding_function, VECTOR_DB_DIR, BM25_PATH

def test_retrieval_pipeline(query: str):
    print(f"\n================ QUERY: '{query}' ================")

    # 1. Dense (Chroma)
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=get_embedding_function()
    )
    dense_docs = vectorstore.similarity_search(query, k=3)
    print("\n--- [A] Pure Dense (Chroma Top 3) ---")
    for i, doc in enumerate(dense_docs, 1):
        print(f"[{i}] {doc.page_content[:140].strip()}...\n")

    # 2. Sparse (BM25)
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    bm25.k = 3
    sparse_docs = bm25.invoke(query)
    print("--- [B] Pure Sparse (BM25 Top 3) ---")
    for i, doc in enumerate(sparse_docs, 1):
        print(f"[{i}] {doc.page_content[:140].strip()}...\n")

    # 3. Hybrid + Cross-Encoder Reranked
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    bm25.k = 8
    ensemble = EnsembleRetriever(
        retrievers=[bm25, dense_retriever],
        weights=[0.4, 0.6]
    )
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    hybrid_reranker = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble
    )

    reranked_docs = hybrid_reranker.invoke(query)
    print("--- [C] Hybrid Ensemble + Cross-Encoder (Final Top 3) ---")
    for i, doc in enumerate(reranked_docs, 1):
        print(f"[{i}] {doc.page_content[:140].strip()}...\n")

if __name__ == "__main__":
    # Test with both a semantic and an exact keyword/policy query
    test_retrieval_pipeline("What is the probation period duration and evaluation criteria?")
    test_retrieval_pipeline("Provident Fund contribution percentage")