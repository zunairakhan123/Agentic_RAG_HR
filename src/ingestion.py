"""
Generalized Layout-Aware and Metadata-Enriched Ingestion Pipeline.
Dynamically handles all 21+ NextBridge HR PDFs:
- Preserves Tabular Directories & Discount Lists
- Keeps Claim & Checklist Forms Intact
- Adds Breadcrumb Hierarchies to Handbooks & Trust Rules
- Exports data/inspected_chunks.json for transparent quality auditing
"""

import os
import re
import json
import pickle
import shutil
import hashlib
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Disable HuggingFace tokenizer deadlock warnings on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pypdf
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================================================
# 1. Directory Configurations & Paths
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "HRM_docs"
VECTOR_DB_DIR = str(DATA_DIR / "vector_db")
BM25_PATH = str(DATA_DIR / "bm25_retriever.pkl")
CHUNKS_PKL_PATH = str(DATA_DIR / "chunks.pkl")
PARENT_STORE_DIR = DATA_DIR / "parent_store"
INSPECTED_CHUNKS_PATH = DATA_DIR / "inspected_chunks.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_EMBEDDING_FN = None


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Singleton factory for the BGE embedding model."""
    global _EMBEDDING_FN
    if _EMBEDDING_FN is None:
        _EMBEDDING_FN = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDING_FN


# =====================================================================
# 2. Dynamic Classifier (Works for All 21 Documents)
# =====================================================================
def detect_document_metadata(filename: str, sample_text: str) -> Dict[str, Any]:
    """
    Dynamically infers category, document type, and processing archetype
    from filename patterns and sample page contents.
    """
    name_clean = filename.lower().replace("-", " ").replace("_", " ")
    text_sample = sample_text[:1500].lower()

    # 1. Identify Category
    if any(k in name_clean or k in text_sample for k in ["provident", "fund", "pf"]):
        category = "provident_fund"
    elif any(k in name_clean or k in text_sample for k in ["opd", "ipd", "health", "hospital", "discount", "claim", "medical", "insurance", "exclusion"]):
        category = "health_insurance"
    elif any(k in name_clean or k in text_sample for k in ["leave", "attendance", "holiday"]):
        category = "leaves_attendance"
    elif any(k in name_clean or k in text_sample for k in ["travel", "allowance", "vehicle", "fuel"]):
        category = "travel_allowance"
    else:
        category = "general_policy"

    # 2. Identify Benefit Type
    if "opd" in name_clean or "opd" in text_sample:
        benefit_type = "opd"
    elif "ipd" in name_clean or "ipd" in text_sample:
        benefit_type = "ipd"
    elif "discount" in name_clean:
        benefit_type = "discount_center"
    elif "exclusion" in name_clean:
        benefit_type = "exclusions"
    elif "provident" in name_clean:
        benefit_type = "provident_fund"
    else:
        benefit_type = "general"

    # 3. Identify Structural Archetype & Doc Type
    # Is it a Form?
    is_form = any(k in name_clean for k in ["form", "checklist"]) or any(
        k in text_sample for k in ["signature of employee", "patient name", "claim amount", "verified by hr", "date:"]
    )
    # Is it a Tabular Directory?
    is_table = any(k in name_clean for k in ["discount", "rate", "list", "directory"]) or (
        text_sample.count("\n") > 30 and ("%" in text_sample or "hospital" in text_sample or "city" in text_sample)
    )
    # Is it a Hierarchical Rule/Handbook?
    is_hierarchy = any(k in name_clean for k in ["handbook", "rules", "policy", "manual"]) or bool(
        re.search(r"\bsection\b|\brule\b|\bclause\b", text_sample)
    )

    if is_form:
        archetype = "form"
        doc_type = "form"
    elif is_table:
        archetype = "tabular"
        doc_type = "directory"
    elif is_hierarchy:
        archetype = "hierarchical_policy"
        doc_type = "policy_rules"
    else:
        archetype = "standard_prose"
        doc_type = "policy"

    # Generate a clean readable title
    clean_title = Path(filename).stem.replace("-", " ").replace("_", " ")
    clean_title = re.sub(r"\b\d{6,}\b", "", clean_title).strip().title()

    return {
        "doc_title": clean_title,
        "category": category,
        "benefit_type": benefit_type,
        "doc_type": doc_type,
        "archetype": archetype,
    }


# =====================================================================
# 3. Text Extraction & Cleaning
# =====================================================================
def extract_pdf_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """Loads a PDF and extracts text per page with whitespace cleanup."""
    pages = []
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        for idx, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
            cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
            if cleaned:
                pages.append({"page_number": idx + 1, "text": cleaned})
    except Exception as e:
        print(f"  [!] Failed to read PDF {pdf_path.name}: {e}")
    return pages


# =====================================================================
# 4. Archetype-Specific Chunking Engines
# =====================================================================
def chunk_form(page_text: str, meta: Dict[str, Any], page_num: int, filename: str) -> List[Document]:
    """Preserves entire form pages as cohesive atomic blocks."""
    content = f"# {meta['doc_title']} (Page {page_num})\n"
    content += f"**Category**: {meta['category']} | **Type**: {meta['doc_type']}\n\n"
    content += page_text

    chunk_meta = {
        **meta,
        "source_file": filename,
        "page_number": page_num,
        "has_table": bool(re.search(r"(table|sr\s*#|amount|relation|patient|charges)", page_text, re.I)),
        "has_form": True,
        "parent_section": f"{meta['doc_title']} - Page {page_num} Form Block",
    }
    return [Document(page_content=content, metadata=chunk_meta)]


def chunk_table_rows(page_text: str, meta: Dict[str, Any], page_num: int, filename: str) -> List[Document]:
    """Chunks tabular directories into row blocks while retaining column header context."""
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    chunks = []
    
    header_context = lines[0] if lines else "Directory Listing"
    block_size = 8  # Group 8 rows together to maintain address, contact, and discount relationship

    for i in range(0, len(lines), block_size):
        sub_lines = lines[i : i + block_size]
        content = f"### {meta['doc_title']} (Page {page_num})\n"
        content += f"**Listing Context**: {header_context}\n\n"
        content += "\n".join(f"- {line}" for line in sub_lines)

        chunk_meta = {
            **meta,
            "source_file": filename,
            "page_number": page_num,
            "has_table": True,
            "has_form": False,
            "parent_section": f"Page {page_num} Entries {i+1} to {i+len(sub_lines)}",
        }
        chunks.append(Document(page_content=content, metadata=chunk_meta))
    return chunks


def chunk_hierarchical(page_text: str, meta: Dict[str, Any], page_num: int, filename: str) -> List[Document]:
    """Splits policies along section/clause headings and prepends breadcrumbs."""
    pattern = r"(?=(\n(?:[0-9]{1,2}\.|[a-z]\)|[ivxLCDM]+\.|\bSection\b|\bRule\b|\bClause\b)\s+[A-Z]))"
    sections = re.split(pattern, page_text, flags=re.IGNORECASE)
    sections = [s.strip() for s in sections if s and len(s.strip()) > 35]

    if not sections:
        sections = [page_text]

    chunks = []
    for sec in sections:
        first_line = sec.splitlines()[0][:80]
        content = f"[{meta['doc_title']} > Page {page_num} > {first_line}]\n\n{sec}"

        chunk_meta = {
            **meta,
            "source_file": filename,
            "page_number": page_num,
            "has_table": bool(re.search(r"(table|schedule|deduction|percentage|%|entitlement)", sec, re.I)),
            "has_form": False,
            "parent_section": first_line,
        }
        chunks.append(Document(page_content=content, metadata=chunk_meta))
    return chunks


def chunk_standard_prose(page_text: str, meta: Dict[str, Any], page_num: int, filename: str) -> List[Document]:
    """Fallback chunker for standard prose that splits cleanly on paragraph boundaries."""
    paragraphs = page_text.split("\n\n")
    current_block = []
    current_len = 0
    chunks = []

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > 800 and current_block:
            combined = "\n\n".join(current_block)
            content = f"[{meta['doc_title']} > Page {page_num}]\n\n{combined}"
            chunks.append(Document(
                page_content=content,
                metadata={**meta, "source_file": filename, "page_number": page_num, "has_table": False, "has_form": False}
            ))
            current_block = [para]
            current_len = para_len
        else:
            current_block.append(para)
            current_len += para_len

    if current_block:
        combined = "\n\n".join(current_block)
        content = f"[{meta['doc_title']} > Page {page_num}]\n\n{combined}"
        chunks.append(Document(
            page_content=content,
            metadata={**meta, "source_file": filename, "page_number": page_num, "has_table": False, "has_form": False}
        ))
    return chunks


# =====================================================================
# 5. Master Orchestrator
# =====================================================================
def ingest_all_documents() -> List[Document]:
    """Iterates through all PDFs in HRM_docs, applies dynamic chunking, and maps parents."""
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[!] Created directory '{DOCS_DIR}'. Please place your 21 PDFs there.")
        return []

    PARENT_STORE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(list(DOCS_DIR.glob("*.pdf")))

    if not pdf_files:
        print(f"[!] No PDFs found in '{DOCS_DIR}'.")
        return []

    print(f"\n[*] Found {len(pdf_files)} HR policy documents. Starting ingestion...\n")

    all_chunks: List[Document] = []
    parent_docs_map: Dict[str, Dict[str, Any]] = {}

    for pdf_path in pdf_files:
        filename = pdf_path.name
        pages = extract_pdf_pages(pdf_path)
        if not pages:
            continue

        # Dynamic classification based on file name and first page sample
        sample_text = pages[0]["text"]
        meta = detect_document_metadata(filename, sample_text)
        print(f"  -> [{meta['archetype'].upper()}] '{filename}' | Category: {meta['category']} | Pages: {len(pages)}")

        for p_data in pages:
            p_num = p_data["page_number"]
            p_text = p_data["text"]

            # 1. Build and store Parent Page Document (for ParentDocumentRetriever)
            parent_id = f"{hashlib.md5(filename.encode()).hexdigest()[:8]}_P{p_num}"
            parent_docs_map[parent_id] = {
                "parent_id": parent_id,
                "source_file": filename,
                "doc_title": meta["doc_title"],
                "category": meta["category"],
                "page_number": p_num,
                "full_text": p_text,
            }

            # 2. Select appropriate structure-aware chunking strategy
            if meta["archetype"] == "form":
                doc_chunks = chunk_form(p_text, meta, p_num, filename)
            elif meta["archetype"] == "tabular":
                doc_chunks = chunk_table_rows(p_text, meta, p_num, filename)
            elif meta["archetype"] == "hierarchical_policy":
                doc_chunks = chunk_hierarchical(p_text, meta, p_num, filename)
            else:
                doc_chunks = chunk_standard_prose(p_text, meta, p_num, filename)

            # 3. Attach parent link and unique ID to each child chunk
            for c in doc_chunks:
                c.metadata["parent_id"] = parent_id
                c.metadata["chunk_id"] = hashlib.md5(f"{parent_id}_{c.page_content[:100]}".encode()).hexdigest()
                all_chunks.append(c)

    # Serialize Parent Store artifacts
    with open(PARENT_STORE_DIR / "parent_documents.pkl", "wb") as f:
        pickle.dump(parent_docs_map, f)
    with open(PARENT_STORE_DIR / "parent_documents.json", "w", encoding="utf-8") as f:
        json.dump(parent_docs_map, f, indent=2)

    # Serialize Canonical Chunks (matching your previous chunks.pkl workflow)
    with open(CHUNKS_PKL_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"\n[✓] Serialized {len(all_chunks)} chunks to '{CHUNKS_PKL_PATH}'")

    return all_chunks


def export_inspected_chunks(chunks: List[Document]):
    """Exports structured JSON file for auditing chunk boundaries and metadata."""
    audit_records = []
    for idx, c in enumerate(chunks):
        audit_records.append({
            "chunk_index": idx,
            "chunk_id": c.metadata.get("chunk_id", ""),
            "parent_id": c.metadata.get("parent_id", ""),
            "source_file": c.metadata.get("source_file", ""),
            "doc_title": c.metadata.get("doc_title", ""),
            "category": c.metadata.get("category", ""),
            "benefit_type": c.metadata.get("benefit_type", ""),
            "doc_type": c.metadata.get("doc_type", ""),
            "page_number": c.metadata.get("page_number", 0),
            "has_table": c.metadata.get("has_table", False),
            "has_form": c.metadata.get("has_form", False),
            "parent_section": c.metadata.get("parent_section", ""),
            "char_count": len(c.page_content),
            "preview": c.page_content[:180].replace("\n", " ") + "...",
            "full_content": c.page_content,
        })

    with open(INSPECTED_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2, ensure_ascii=False)
    print(f"[✓] Auditable inspection log saved to '{INSPECTED_CHUNKS_PATH}'")


def build_indices(chunks: List[Document]):
    """Builds both Sparse (BM25) and Dense (ChromaDB with Cosine) indices."""
    # 1. Build Sparse BM25
    print("[*] Generating BM25 Sparse Index...")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"[✓] BM25 Index saved to '{BM25_PATH}'")

    # 2. Build Dense ChromaDB
    if os.path.exists(VECTOR_DB_DIR):
        try:
            shutil.rmtree(VECTOR_DB_DIR)
            print(f"[*] Cleared old Vector DB at '{VECTOR_DB_DIR}'")
        except PermissionError:
            print(f"[!] Warning: Could not delete '{VECTOR_DB_DIR}' because a process locked it.")
            print("[!] Please stop any running uvicorn/python process and re-run.")
            return

    print("[*] Building ChromaDB with Cosine Metric (bge-base-en-v1.5)...")
    embed_fn = get_embedding_function()
    
    # Explicitly configure Chroma's HNSW index to use Cosine space
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embed_fn,
        persist_directory=VECTOR_DB_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"[✓] ChromaDB built with {vectorstore._collection.count()} vectors in Cosine space.")


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 NextBridge HR Agent: Dynamic Layout-Aware Ingestion Pipeline")
    print("=" * 70)

    all_chunks = ingest_all_documents()
    if all_chunks:
        export_inspected_chunks(all_chunks)
        build_indices(all_chunks)
        print("\n🎉 Full Ingestion, Serialization, and Audit Complete!")