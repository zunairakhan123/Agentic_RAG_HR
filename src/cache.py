"""
Lightweight Semantic Cache to prevent redundant graph executions for near-duplicate queries.
"""
import os
from langchain_chroma import Chroma
from src.ingestion import get_embedding_function

CACHE_DIR = os.path.join("data", "semantic_cache")

# Initialize a dedicated Chroma DB just for caching query-to-answer pairs
query_cache = Chroma(
    persist_directory=CACHE_DIR,
    embedding_function=get_embedding_function()
)

def check_semantic_cache(query: str, similarity_threshold: float = 0.15) -> str:
    """
    Checks if a semantically similar query was recently answered.
    Chroma uses L2 distance by default (closer to 0.0 means more similar).
    """
    results = query_cache.similarity_search_with_score(query, k=1)
    
    if results:
        best_match, distance = results[0]
        # If the distance is below the threshold, it's a semantic match
        if distance <= similarity_threshold:
            print(f"[CACHE HIT] Found similar query: '{best_match.page_content}' (Dist: {distance:.2f})")
            return best_match.metadata.get("answer")
            
    return None

def save_to_semantic_cache(query: str, answer: str):
    """Saves a successfully grounded answer to the cache."""
    query_cache.add_texts(
        texts=[query],
        metadatas=[{"answer": answer}]
    )
    print(f"[CACHE SAVED] Cached response for query: '{query}'")