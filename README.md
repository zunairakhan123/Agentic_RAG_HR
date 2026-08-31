# 🤖 NextBridge Agentic HR Microservice (CRAG, Self-RAG & HITL Architecture)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B%20Stateful-purple?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-red?style=flat-square)
![LangSmith](https://img.shields.io/badge/LangSmith-Traced-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A production-grade, event-driven Agentic AI microservice that orchestrates corporate HR workflows. Built on a hierarchical LangGraph state machine, the system integrates Corrective RAG (CRAG) with Self-RAG reflection, programmatic entity-scoped tool execution, dynamic tool sandboxing for Human-in-the-Loop (HITL) actions, two-tier semantic caching, and full duplex voice streaming.

---

##  Table of Contents

- [Architectural Highlights](#-architectural-highlights)
- [System Architecture (Super-Graph)](#-system-architecture-super-graph)
- [Core Engineering Mechanisms](#-core-engineering-mechanisms)
  - [1. Corrective RAG & Self-RAG Subgraph](#1-corrective-rag--self-rag-subgraph)
  - [2. Programmatic Entity Scoping (Anti-Hallucination)](#2-programmatic-entity-scoping-anti-hallucination)
  - [3. Dynamic Tool Sandboxing (HITL Security)](#3-dynamic-tool-sandboxing-hitl-security)
  - [4. Real-Time SSE Buffer Invalidation](#4-real-time-sse-buffer-invalidation)
  - [5. Two-Tier Semantic & Frequency Caching](#5-two-tier-semantic--frequency-caching)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Environment](#-prerequisites--environment)
- [Installation & Local Setup](#-installation--local-setup)
- [Running the Application](#-running-the-application)
- [Evaluation & Benchmarking](#-evaluation--benchmarking)
- [Security & Guardrails Overview](#-security--guardrails-overview)
- [License](#-license)

---

##  Architectural Highlights

- **Hierarchical State Orchestration:** Decouples lightweight conversational routing from compute-heavy CRAG loops, eliminating tool hallucination and non-deterministic routing traps.
- **Resilient Self-Correction:** Evaluates retrieved chunk relevance with a structured grader, rewrites poor retrieval queries, and self-checks generated responses against context using an output reflection guardrail.
- **Zero-Trust Tool Sandboxing:** Strips write/dispatch capabilities (`send_department_email`) from the agent's runtime execution environment until strict Human-in-the-Loop verification conditions are satisfied.
- **Programmatic Entity Guarding:** Enforces deterministic query transformations at the tool level to prevent external entity collisions across similarly named corporate entities.
- **Smooth UI Invalidation Protocol:** Emits granular Server-Sent Events (`clear`, `token`, `status`, `tool_start`, `tool_end`) to deliver real-time token streaming and transparent reflection wipes without client-side text-stacking glitches.
- **Map-Reduce Sub-Query Retrieval:** Decomposes complex, multi-intent questions into isolated parallel vector searches and localized cross-encoder re-ranking. This completely eliminates keyword starvation, cross-encoder penalties, and LLM "generator amnesia" on compound queries.
- **LLM-Evaluated Semantic Caching:** Replaces naive string hashing with a two-tier Cosine Similarity and LLM-as-a-Judge caching system. Accurately merges acronyms and rephrasings (e.g., "OPD" vs "Outpatient") for tracking, while maintaining strict 90% semantic thresholds for instant cache delivery.


---

## System Architecture (Super-Graph)
```
                          ┌─────────────────────────┐
                          │     Frontend / Client   │
                          │ (Chat UI / Voice Mode)  │
                          └───────────┬─────────────┘
                                      │
                                      ▼
                          ┌─────────────────────────┐
                          │  Semantic Cache Layer   │───[ 90% Cosine Hit ]───► Instant Stream
                          │ (ChromaDB + LLM Judge)  │                            (< 150ms)
                          └───────────┬─────────────┘
                                      │ [ Cache Miss ]
                                      ▼
                          ┌─────────────────────────┐
                          │  Input Guardrail Node   │───[ Blocked ]─────► [ END ]
                          │ (Whitelist Scope Gate)  │
                          └───────────┬─────────────┘
                                      │ [ Pass ]
                                      ▼
                          ┌─────────────────────────┐
                          │  Adaptive Router Node   │
                          └──────┬───────────┬──────┘
                                 │           │
            ┌───[ email / web / chat ]       └───[ rag: simple / complex ]───┐
            │                                                                │
            ▼                                                                ▼
┌─────────────────────────┐                                      ┌─────────────────────────┐
│    ReAct Agent Node     │◄──┐                                  │   RAG Child Subgraph    │
│ (Dynamic Tool Sandboxed)│   │                                  └───────────┬─────────────┘
└───────────┬─────────────┘   │                                              │
            │                 │                                              ▼
            ├───────────────┐ │                                  ┌─────────────────────────┐
            ▼               ▼ │                         ┌───────►│   Adaptive Retrieval    │
  ┌──────────────────┐ ┌──────────────────┐             │        └───────────┬─────────────┘
  │ Scoped Web Search│ │  Draft Email     │             │                    │
  └──────────────────┘ └────────┬─────────┘             │         ┌──────────┴──────────┐
                                 │                      │     [Complex]             [Simple]
                                 ▼                      │         │                     │
                       ┌───────────────────┐            │         ▼                     │
                       │ UI Approval Card  │            │  ┌─────────────┐              │
                       │ (Human-in-the-Loop)│           │  │ Grader Node │              │
                       └─────────┬─────────┘            │  └──┬───────┬──┘              │
                                 │ [ Approved ]         │     │       │                 │
                                 ▼                      │ (bad docs)  │(relevant)       │
                       ┌───────────────────┐            │     ▼       │                 │
                       │ Dispatch Email    │            │ ┌───────┐   │                 │
                       │   (SMTP Tool)     │            └─┤Rewrite│   │                 │
                       └─────────┬─────────┘          ┌──►└───────┘   ▼                 ▼
                                 │                    │            ┌───────────────────────┐
                                 ▼                    │            │    Generation Node    │◄──┐
                              [ END ]                 │            └─────────┬─────────────┘   │
                                                      │                      │                 │
                                                      │            ┌─────────┴─────────┐       │
                                                      │       [Complex]            [Simple]    │
                                                      │            │                   │       │
                                                      │            ▼                   │       │
                                                      │  ┌───────────────────┐         │       │
                                                      │  │  Reflection Node  │         │       │
                                                      │  └────┬─────────┬────┘         │       │
                                                      │       │         │              │       │
                                              (missing│       │(retry)  │(grounded)    │       │
                                              evidence)       ▼         ▼              ▼       │
                                                      │ [Regenerate]                 [ END ]   │
                                                      │       │                                │
                                                      └───────┴────────────────────────────────┘

```

---

## ⚙️ Core Engineering Mechanisms

### 1. Corrective RAG & Self-RAG Subgraph

Retrieval is split into an isolated, cyclic child graph featuring an adaptive dual-pipeline:

- **Map-Reduce Parallel Retrieval:** The complex pathway dynamically breaks multi-intent questions into semantic sub-queries. It executes concurrent hybrid searches and isolates Cross-Encoder re-ranking per topic, preventing chunk starvation on compound inputs.
- **Resilient Self-Correction:** Evaluates retrieved chunk relevance, rewrites poor retrieval queries, and self-checks generated answers against context using a strict output reflection guardrail.

### 2. Programmatic Entity Scoping (Anti-Hallucination)

To eliminate entity collision in external search indexes (e.g., confusing Nextbridge Pvt. Ltd. with unrelated global entities like Next Bridge Hydrocarbons), the tool execution layer enforces automatic query decoration:

```python
# Programmatic context bounding inside the tool boundary
scoped_query = f"{query} Nextbridge software IT company Pakistan Lahore"
```

This guarantees grounding in the software firm's corporate footprint without relying on the LLM to recall contextual modifiers.

### 3. Dynamic Tool Sandboxing (HITL Security)

To enforce Human-in-the-Loop safety and prevent Tool Shortcut Hallucinations (where the model calls execution tools prematurely):

- **State Masking:** The `agent_node` dynamically inspects conversation state. During standard generation, only `draft_department_email` is bound to the LLM.
- **Physical Revocation:** The destructive `send_department_email` tool is omitted from the model's schema until the client explicitly submits an approval payload from the UI verification card.

### 4. Real-Time SSE Buffer Invalidation

When the Self-RAG subgraph detects a factual drift and rewrites a response, sending incremental tokens over standard SSE can cause draft stacking in client UIs.

- **The `clear` Protocol:** On `on_chat_model_start` for the generation node, the backend emits `data: {"type": "clear"}`.
- **Client Buffer Invalidation:** The frontend intercepts this signal, flushes the accumulated markdown buffer, renders a non-blocking refinement indicator, and cleanly streams the verified answer.

### 5. Two-Tier Semantic & Frequency Caching

Avoids redundant graph execution and LLM overhead using an intelligent Cosine Similarity engine:

- **Write Phase (LLM-as-a-Judge):** Uses a relaxed vector threshold (65% Cosine Similarity) to fetch potential historical queries, handing them to an ultra-fast LPU-based LLM Judge to verify intent equivalence. This natively handles typos, acronyms, and semantic overlaps without false positives.
- **Read Phase (Instant Delivery):** Uses a strict 90% Cosine Similarity gate to intercept identical queries before they hit the LangGraph workflow, streaming cached verified answers to the frontend in < 150ms.

---

##  Technology Stack

| Component | Technology | Role |
|---|---|---|
| Orchestration | LangGraph (>=0.2.0) | Stateful cyclic graph execution, child subgraphs, checkpointers |
| Backend Framework | FastAPI | Async SSE token streaming, WebSockets, REST webhooks |
| State Persistence | AsyncSqliteSaver (aiosqlite) | Async thread checkpointing and session state persistence |
| Inference Engines | Groq Cloud / Local Qwen Models | Routing, ReAct reasoning, extraction, and generation |
| Embeddings & Search | ChromaDB + HuggingFace | Dense vector store + BAAI/bge-base-en-v1.5 embeddings |
| Re-Ranking | Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` for high-precision filtering |
| Observability | LangSmith | Graph execution tracing, latency breakdown, node monitoring |
| Evaluation | RAGAS | Context Precision, Recall, Faithfulness, and Answer Relevancy |

---

## 📂 Repository Structure

```
nextbridge_hr_agent/
├── data/
│   ├── HRM_docs/                 # Primary HR policy source PDFs
│   ├── semantic_cache/           # ChromaDB persistent directory for cached responses
│   └── vector_db/                # ChromaDB persistent store for chunk embeddings
├── src/
│   ├── api.py                    # FastAPI server (SSE, WebSockets, notification polling)
│   ├── cache.py                  # Two-tier semantic caching & frequency tracking logic
│   ├── email_listener.py         # Background IMAP worker for thread-based reply intake
│   ├── graph.py                  # Hierarchical Super-Graph (ReAct + CRAG Subgraph)
│   ├── ingestion.py              # PDF parsing, semantic chunking, and ChromaDB indexing
│   ├── nodes.py                  # Node logic: Guardrail, Grader, Rewrite, Reflection, Router
│   ├── state.py                  # TypedDict schemas for Parent and Subgraph states
│   └── tools.py                  # Scoped web search & sandboxed SMTP dispatch tools
├── evaluation/
│   ├── datasets/                 # Gold-standard evaluation Q&A pairs
│   └── evaluate_final.py         # End-to-end RAGAS scoring pipeline
├── frontend/
│   └── index.html                # Unified UI: SSE parsing, dynamic HITL cards, STT/TTS
├── tests/                        # Component integration and unit test suites
├── requirements.txt              # Production dependency specifications
├── .env.example                  # Environment variable configuration template
└── README.md                     # Technical architecture and setup documentation
```

---

##  Prerequisites & Environment

- **Python:** 3.11 or higher
- **Inference Access:** Valid Groq API Key OR Local Ollama/vLLM instance
- **Observability:** LangSmith API Key (optional, for tracing)
- **Email Service:** Gmail / Corporate SMTP credentials with App Password enabled

### Environment Variables Configuration

Create a `.env` file in the project root:

```ini
# --- LLM & Inference ---
GROQ_API_KEY="gsk_..."

# --- Observability (LangSmith) ---
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="NextBridge_Agentic_HR_System"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="lsv2_pt_..."

# --- External Search Tools ---
TAVILY_API_KEY="tvly-..."

# --- SMTP / IMAP Configuration ---
SMTP_EMAIL="your_service_account@gmail.com"
SMTP_PASSWORD="your_app_password"
IMAP_SERVER="imap.gmail.com"
SMTP_SERVER="smtp.gmail.com"
```

---

##  Installation & Local Setup

**1. Clone the Repository & Set Up Virtual Environment**

```bash
git clone https://github.com/your-org/nextbridge_hr_agent.git
cd nextbridge_hr_agent

python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**2. Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Ingest HR Policy Documents**

Place your HR policy PDFs into `data/HRM_docs/` and run the vector ingestion pipeline:

```bash
python -m src.ingestion
```

---

##  Running the Application

The system requires two concurrent runtime processes.

**Step 1: Start the FastAPI API Server (Terminal 1)**

```bash
python -m src.api
```

- The API runs on `http://localhost:8000`.
- Interactive API documentation is available at `http://localhost:8000/docs`.

**Step 2: Start the Background IMAP Listener (Terminal 2)**

```bash
python -m src.email_listener
```

Continuously polls for incoming department replies and routes updates back to active session threads.

**Step 3: Access the Frontend Interface**

Navigate to `http://localhost:8000` in your web browser.

---

## 📊 Evaluation & Benchmarking

The architecture undergoes automated evaluation using the RAGAS framework against synthetic and golden dataset distributions to ensure compliance before production promotion.

| Metric | Score | Primary Architectural Driver |
|---|---|---|
| Answer Relevancy | 0.904 | Adaptive Query Router & Map-Reduce Decomposer |
| Context Precision | 0.818 | Isolated Cross-Encoder Re-Ranking Pipeline |
| Context Recall | 0.800 | Chunking Strategy & Document Grading Node |
| Faithfulness | 0.980 | Self-RAG Output Reflection Guardrail |

To execute the evaluation pipeline locally:

```bash
python -m evaluation.evaluate_final
```

---

##  Security & Guardrails Overview

- **Input Security Gateway:** Evaluates incoming user inputs against a domain-specific whitelist, dropping non-operational queries, script injections, and jailbreak attempts before invoking backend graph paths.
- **Hallucination Quarantine:** Ensures that generative RAG drafts undergo mandatory factual verification against ground-truth document chunks before streaming completion.
- **Controlled Side-Effects:** Prevents external actions (e.g., email dispatch, database writes) from running purely autonomously by requiring a verifiable approval handshake from the client interface.

---

##  License

This project is licensed under the MIT License. See the `LICENSE` file for complete details.