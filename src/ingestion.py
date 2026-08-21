"""
Optimized Unified Ingestion Pipeline for NextBridge HR PDFs.
Generates Canonical Chunks, Sparse (BM25) artifacts, and Dense (Chroma) embeddings.
"""

import os
import pickle
import hashlib
import shutil

# CRITICAL FIX for Windows: Prevents the tokenizer from deadlocking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

load_dotenv()

DOCS_DIR = os.path.join("data", "HRM_docs")
VECTOR_DB_DIR = os.path.join("data", "vector_db")
BM25_PATH = os.path.join("data", "bm25_retriever.pkl")
CHUNKS_PATH = os.path.join("data", "chunks.pkl")


def get_embedding_function():
    """Returns local BGE-Small model (optimized for CPU)."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def generate_deterministic_id(text: str, metadata: dict) -> str:
    """Generates a stable MD5 hash ID based on chunk content and source."""
    unique_string = f"{metadata.get('source', '')}_{metadata.get('page', '')}_{text}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()


def ingest_all_documents():
    """Loads PDFs, chunks text, applies IDs, and builds both BM25 and Chroma indices."""
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"[!] Directory '{DOCS_DIR}' created. Please place your PDFs there.")
        return

    print(f"[*] Loading PDF documents from '{DOCS_DIR}'...")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()

    if not documents:
        print(f"[!] No PDFs found in '{DOCS_DIR}'.")
        return

    print(f"[*] Loaded {len(documents)} pages. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    
    print("[*] Applying stable chunk_id metadata...")
    for chunk in chunks:
        chunk.metadata["chunk_id"] = generate_deterministic_id(chunk.page_content, chunk.metadata)

    total_chunks = len(chunks)
    print(f"[*] Total chunks to process: {total_chunks}")

    # 1. Serialize Canonical Chunks
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[✓] Canonical chunks saved to '{CHUNKS_PATH}'")

    # 2. Build BM25 Sparse Index
    print("[*] Generating BM25 Sparse Index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"[✓] BM25 Index saved to '{BM25_PATH}'")

    # 3. Clean legacy Chroma DB to prevent duplicates or mismatched IDs
    if os.path.exists(VECTOR_DB_DIR):
        try:
            shutil.rmtree(VECTOR_DB_DIR)
            print(f"[*] Cleared old Vector DB at '{VECTOR_DB_DIR}'")
        except PermissionError:
            print(f"[!] Warning: Could not delete {VECTOR_DB_DIR}. A process might be locking it.")
            print("[!] Please delete it manually in File Explorer before continuing.")
            return

    # 4. Build new Dense Chroma Index
    print("[*] Initializing Vector DB and generating embeddings...")
    embedding_fn = get_embedding_function()
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embedding_fn
    )

    batch_size = 50
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"    -> Processed {min(i + batch_size, total_chunks)} / {total_chunks} chunks...")

    print("\n[✓] FULL PIPELINE COMPLETE! Both indices are now perfectly aligned.")


if __name__ == "__main__":
    ingest_all_documents()