import os
import pickle

BM25_PATH = os.path.join("data", "bm25_retriever.pkl")
CHUNKS_PATH = os.path.join("data", "chunks.pkl")
VECTOR_DB_DIR = os.path.join("data", "vector_db")

def test_artifacts():
    print("--- 1. Checking File Existence ---")
    assert os.path.exists(BM25_PATH), f"Missing {BM25_PATH}"
    assert os.path.exists(CHUNKS_PATH), f"Missing {CHUNKS_PATH}"
    assert os.path.exists(VECTOR_DB_DIR), f"Missing {VECTOR_DB_DIR}"
    print("✓ All artifact directories and files exist.")

    print("\n--- 2. Validating Canonical Chunks ---")
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"✓ Loaded {len(chunks)} canonical chunks.")
    
    # Verify metadata schema
    sample = chunks[0]
    assert "chunk_id" in sample.metadata, "chunk_id missing in chunk metadata!"
    print(f"✓ Sample Chunk ID: {sample.metadata['chunk_id']}")
    print(f"✓ Sample Content Preview: {sample.page_content[:120]}...")

    print("\n--- 3. Validating BM25 Retriever Artifact ---")
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    test_docs = bm25.invoke("probation period")
    print(f"✓ BM25 functional. Returned {len(test_docs)} documents for test query.")

if __name__ == "__main__":
    test_artifacts()