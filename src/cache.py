"""
Frequency-Based Semantic Cache and CI/CD Data Pipeline Generator.
Uses Two-Tiered LLM-Evaluated Caching (The Production Standard).
"""
import os
import json
import uuid
import aiosqlite
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from src.ingestion import get_embedding_function

# ==========================================
# 1. Constants & Setup
# ==========================================
CACHE_DIR = os.path.join("data", "semantic_cache")
TRACKER_DIR = os.path.join("data", "semantic_tracker") 
DB_PATH = os.path.join("data", "semantic_cache", "frequency.db")
EVAL_DATASET_PATH = os.path.join("evaluation", "datasets", "promoted_cache_queries.json")
PROMOTION_THRESHOLD = 3

embed_fn = get_embedding_function()
collection_config = {"hnsw:space": "cosine"}

query_cache = Chroma(persist_directory=CACHE_DIR, embedding_function=embed_fn, collection_metadata=collection_config)
tracker_cache = Chroma(persist_directory=TRACKER_DIR, embedding_function=embed_fn, collection_metadata=collection_config)

# Initialize Fast LLM for the Cache Judge
fast_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0.0
)

class CacheEquivalence(BaseModel):
    is_same: bool = Field(description="True if the queries ask for the exact same core information, False otherwise.")

async def init_frequency_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("CREATE TABLE IF NOT EXISTS query_frequency (hash TEXT PRIMARY KEY, count INTEGER)")
        await db.commit()

# ==========================================
# 2. Helper Functions
# ==========================================
def check_semantic_cache(query: str) -> str:
    """Read phase: Strict match required for instant retrieval."""
    results = query_cache.similarity_search_with_score(query, k=1)
    if results and (1.0 - results[0][1]) >= 0.90:  # 90% threshold for instant delivery
        return results[0][0].metadata.get("answer")
    return None

def save_to_semantic_cache(query: str, answer: str):
    query_cache.add_texts(texts=[query], metadatas=[{"answer": answer}])
    os.makedirs(os.path.dirname(EVAL_DATASET_PATH), exist_ok=True)
    
    try:
        data = []
        if os.path.exists(EVAL_DATASET_PATH):
            with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        data.append({"question": query, "ground_truth": answer})
        
        with open(EVAL_DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[Pipeline Error] Failed to write eval JSON: {e}")

# ==========================================
# 3. Main Promotion Logic (LLM Judge)
# ==========================================
async def track_and_promote(query: str, answer: str, rag_state: dict):
    if rag_state.get("retrieval_attempts", 0) == 0 or not (rag_state.get("answer_grounded") or rag_state.get("query_type") == "simple"):
        return

    # STEP 1: Broad Vector Search (Relaxed to 65% to catch acronyms and rewrites)
    results = tracker_cache.similarity_search_with_score(query, k=1)
    
    q_hash = None
    if results:
        closest_q = results[0][0].page_content
        cosine_similarity = 1.0 - results[0][1]
        
        if cosine_similarity >= 0.65:
            # STEP 2: The LLM Judge determines true equivalence
            prompt = f"""Are these two user queries asking for the exact same HR information?
            Query 1: "{query}"
            Query 2: "{closest_q}"
            
            Ignore typos, acronyms (like OPD vs Outpatient), or filler words. Focus only on the core intent.
            """
            try:
                structured_llm = fast_llm.with_structured_output(CacheEquivalence, method="json_schema")
                judge_result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
                
                if judge_result.is_same:
                    q_hash = results[0][0].metadata.get("tracking_id")
                    print(f"  -> [TRACKER HIT] LLM verified equivalence with: '{closest_q}'")
                else:
                    print(f"  -> [TRACKER MISS] LLM rejected equivalence with: '{closest_q}'")
            except Exception as e:
                print(f"  -> [TRACKER WARNING] LLM Judge failed: {e}")

    # STEP 3: Assign new hash if no match was found/verified
    if not q_hash:
        q_hash = str(uuid.uuid4())
        tracker_cache.add_texts(texts=[query], metadatas=[{"tracking_id": q_hash}])

    # Update SQLite Tracker
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT count FROM query_frequency WHERE hash = ?", (q_hash,)) as cursor:
            row = await cursor.fetchone()
            current_count = row[0] if row else 0
            
        new_count = current_count + 1
        await db.execute("INSERT OR REPLACE INTO query_frequency (hash, count) VALUES (?, ?)", (q_hash, new_count))
        await db.commit()
    
    print(f"  -> [Telemetry] Intent Frequency: {new_count}/{PROMOTION_THRESHOLD}")
    
    if new_count == PROMOTION_THRESHOLD:
        save_to_semantic_cache(query, answer)
        print("  -> [CACHE PROMOTED] Pushed to Chroma and Eval Pipeline.")