"""
Optimized Document ingestion pipeline for NextBridge HR PDFs.
Uses BGE-Small for fast CPU inference and prevents Windows deadlocks.
"""

import os
# CRITICAL FIX for Windows: Prevents the tokenizer from deadlocking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DOCS_DIR = os.path.join("data", "HRM_docs")
VECTOR_DB_DIR = os.path.join("data", "vector_db")


def get_embedding_function():
    """Returns local BGE-Small model (optimized for CPU)."""
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",  # Switched to small for fast CPU processing
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def ingest_hr_documents():
    """Loads PDFs, chunks text, and batches them into ChromaDB."""
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
    total_chunks = len(chunks)
    print(f"[*] Total chunks to process: {total_chunks}")

    print("[*] Downloading model and initializing Vector DB...")
    embedding_fn = get_embedding_function()
    
    # Initialize empty Chroma DB
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embedding_fn
    )

    print("[*] Generating embeddings in batches...")
    batch_size = 50  # Small batches prevent RAM spikes
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"    -> Processed {min(i + batch_size, total_chunks)} / {total_chunks} chunks...")

    print(f"[✓] Vector DB successfully created at '{VECTOR_DB_DIR}'!")


if __name__ == "__main__":
    ingest_hr_documents()