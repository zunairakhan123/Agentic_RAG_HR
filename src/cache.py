"""
Frequency-Based Semantic Cache and CI/CD Data Pipeline Generator.
"""
import os
import json
import asyncio
import hashlib
import aiosqlite
from langchain_chroma import Chroma
from src.ingestion import get_embedding_function

# ==========================================
# 1. Constants & Paths
# ==========================================
CACHE_DIR = os.path.join("data", "semantic_cache")
DB_PATH = os.path.join("data", "semantic_cache", "frequency.db")
EVAL_DATASET_PATH = os.path.join("evaluation", "datasets", "promoted_cache_queries.json")
PROMOTION_THRESHOLD = 3

# ==========================================
# 2. Database Initializations
# ==========================================
query_cache = Chroma(
    persist_directory=CACHE_DIR,
    embedding_function=get_embedding_function()
)

async def init_frequency_db():
    """Initializes the persistent SQLite tracker."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS query_frequency (hash TEXT PRIMARY KEY, count INTEGER)"
        )
        await db.commit()

# ==========================================
# 3. Helper Functions
# ==========================================
def get_query_hash(query: str) -> str:
    """Normalizes and hashes the query for exact-match frequency tracking."""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()

def check_semantic_cache(query: str, similarity_threshold: float = 0.15) -> str:
    """Fast-path Read Phase."""
    results = query_cache.similarity_search_with_score(query, k=1)
    if results:
        best_match, distance = results[0]
        if distance <= similarity_threshold:
            print(f"[CACHE HIT] Dist: {distance:.2f}")
            return best_match.metadata.get("answer")
    return None

def save_to_semantic_cache(query: str, answer: str):
    """Saves to ChromaDB and exports to the CI/CD JSON evaluation pipeline."""
    query_cache.add_texts(texts=[query], metadatas=[{"answer": answer}])
    
    os.makedirs(os.path.dirname(EVAL_DATASET_PATH), exist_ok=True)
    eval_entry = {"question": query, "ground_truth": answer}
    
    try:
        data = []
        if os.path.exists(EVAL_DATASET_PATH):
            with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        data.append(eval_entry)
        
        with open(EVAL_DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[Pipeline Error] Failed to write eval JSON: {e}")

# ==========================================
# 4. Main Promotion Logic
# ==========================================
async def track_and_promote(query: str, answer: str, rag_state: dict):
    retrieval_attempts = rag_state.get("retrieval_attempts", 0)
    is_grounded = rag_state.get("answer_grounded", False)
    is_simple = rag_state.get("query_type") == "simple"

    if retrieval_attempts == 0 or not (is_grounded or is_simple):
        return

    # [THE UPGRADE]: Use the LLM's normalized intent instead of the raw query!
    semantic_intent = rag_state.get("intent_key", query)
    
    # Hash the normalized intent (e.g., "sick_leave_policy" always hashes to the same ID)
    q_hash = get_query_hash(semantic_intent)
    
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT count FROM query_frequency WHERE hash = ?", (q_hash,)) as cursor:
            row = await cursor.fetchone()
            current_count = row[0] if row else 0
            
        new_count = current_count + 1
        
        await db.execute(
            "INSERT OR REPLACE INTO query_frequency (hash, count) VALUES (?, ?)", 
            (q_hash, new_count)
        )
        await db.commit()
    
    print(f"  -> [Telemetry] Query '{query}' frequency: {new_count}/{PROMOTION_THRESHOLD}")
    
    if new_count == PROMOTION_THRESHOLD:
        save_to_semantic_cache(query, answer)
        print("  -> [CACHE PROMOTED] Pushed to Chroma and Eval Pipeline.")