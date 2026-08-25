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

## 📑 Table of Contents

- [Architectural Highlights](#-architectural-highlights)
- [System Architecture (Super-Graph)](#-system-architecture-super-graph)
- [Core Engineering Mechanisms](#-core-engineering-mechanisms)
  - [1. Corrective RAG & Self-RAG Subgraph](#1-corrective-rag--self-rag-subgraph)
  - [2. Programmatic Entity Scoping (Anti-Hallucination)](#2-programmatic-entity-scoping-anti-hallucination)
  - [3. Dynamic Tool Sandboxing (HITL Security)](#3-dynamic-tool-sandboxing-hitl-security)
  - [4. Real-Time SSE Buffer Invalidation](#4-real-time-sse-buffer-invalidation)
  - [5. Inference Hardening & Provider Compatibility](#5-inference-hardening--provider-compatibility)
  - [6. Two-Tier Semantic & Frequency Caching](#6-two-tier-semantic--frequency-caching)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Environment](#-prerequisites--environment)
- [Installation & Local Setup](#-installation--local-setup)
- [Running the Application](#-running-the-application)
- [Evaluation & Benchmarking](#-evaluation--benchmarking)
- [Security & Guardrails Overview](#-security--guardrails-overview)
- [License](#-license)

---

## 🚀 Architectural Highlights

- **Hierarchical State Orchestration:** Decouples lightweight conversational routing from compute-heavy CRAG loops, eliminating tool hallucination and non-deterministic routing traps.
- **Resilient Self-Correction:** Evaluates retrieved chunk relevance with a structured grader, rewrites poor retrieval queries, and self-checks generated responses against context using an output reflection guardrail.
- **Zero-Trust Tool Sandboxing:** Strips write/dispatch capabilities (`send_department_email`) from the agent's runtime execution environment until strict Human-in-the-Loop verification conditions are satisfied.
- **Programmatic Entity Guarding:** Enforces deterministic query transformations at the tool level to prevent external entity collisions across similarly named corporate entities.
- **Smooth UI Invalidation Protocol:** Emits granular Server-Sent Events (`clear`, `token`, `status`, `tool_start`, `tool_end`) to deliver real-time token streaming and transparent reflection wipes without client-side text-stacking glitches.

---

## 🏛 System Architecture (Super-Graph)

```
                          ┌─────────────────────────┐
                          │    Frontend / Client    │
                          │ (Chat UI / Voice Mode)  │
                          └───────────┬─────────────┘
                                      │
                                      ▼
                          ┌─────────────────────────┐
                          │  Semantic Cache Layer   │───[ Cache Hit ]───► Instant Stream
                          │  (ChromaDB + SQLite DB) │                     (< 200ms)
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
            ┌───[ email / web / chat ]      └───[ rag: simple / complex ]───┐
            │                                                               │
            ▼                                                               ▼
┌─────────────────────────┐                                     ┌─────────────────────────┐
│    ReAct Agent Node     │◄──┐                                 │   RAG Child Subgraph    │
│ (Dynamic Tool Sandboxed)│   │                                 │ (CRAG + Self-RAG Loop)  │
└───────────┬─────────────┘   │                                 └───────────┬─────────────┘
            │                 │                                             │
            ├───────────────┐ │                                             ▼
            ▼               ▼ │                                 ┌─────────────────────────┐
  ┌──────────────────┐ ┌──────────────────┐                     │    Vector Retrieval     │
  │ Scoped Web Search│ │  Draft Email     │                     │  (BGE + Cross-Encoder)  │
  └──────────────────┘ └────────┬─────────┘                     └───────────┬─────────────┘
                                 │                                           │
                                 ▼                                           ▼
                       ┌───────────────────┐                     ┌─────────────────────────┐
                       │ UI Approval Card  │                     │      Grader Node        │
                       │ (Human-in-the-Loop)│                    └─────┬─────────────┬─────┘
                       └─────────┬─────────┘                          │             │
                                 │ [ Approved ]              (bad docs)│             │ (relevant)
                                 ▼                                     ▼             ▼
                       ┌───────────────────┐                 ┌───────────┐   ┌───────────────┐
                       │ Dispatch Email    │                 │  Rewrite  │   │  Generation   │◄─┐
                       │   (SMTP Tool)     │                 │   Node    │   │     Node      │  │
                       └─────────┬─────────┘                 └─────┬─────┘   └───────┬───────┘  │
                                 │                                 │                 │          │
                                 ▼                                 └────►[Retrying]   ▼          │
                              [ END ]                                          ┌───────────────┐ │
                                                                                │ Reflection    │ │
                                                                                │ Node          ├─┘
                                                                                │(Hallucination │ (retry)
                                                                                │  Check)       │
                                                                                └───────┬───────┘
                                                                                        │ (grounded)
                                                                                        ▼
                                                                                     [ END ]
```

---

## ⚙️ Core Engineering Mechanisms

### 1. Corrective RAG & Self-RAG Subgraph

Retrieval is split into an isolated, cyclic child graph:

- **Retrieval & Re-ranking:** Queries are embedded using BGE embeddings and re-ranked via a HuggingFace Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
- **Document Grader:** A structured output model evaluates retrieved chunks for strict semantic relevance. If context is insufficient, it routes to `rewrite_query_node`.
- **Reflection Guardrail:** The generated answer is passed through an output reflection node to grade factual support against retrieved context. Hallucinated answers loop back for structured regeneration.

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

### 5. Inference Hardening & Provider Compatibility

When utilizing providers with strict tool-choice validation rules (e.g., Groq API), removing tools dynamically can cause runtime exceptions (`Tool choice is none, but model called a tool`). The agent node handles this via:

- **Context Scrubbing:** Automatically converts historical `tool_calls` into plain `AIMessage` content and purges orphaned tool messages before dispatching toolless requests.
- **Inference Fallback:** Catches provider-level validation errors inside an async try/except wrapper and triggers a clean, tool-free conversational fallback, preventing thread deadlocks.

### 6. Two-Tier Semantic & Frequency Caching

Avoids redundant graph execution and LLM overhead:

- **SQLite Frequency Ledger:** Normalizes semantic intent and tracks query hit frequencies.
- **ChromaDB Semantic Cache:** Once a query surpasses the promotion threshold, its verified answer is embedded and indexed for sub-millisecond retrieval.

---

## 🛠 Technology Stack

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

## 📋 Prerequisites & Environment

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

## 💻 Installation & Local Setup

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

## 🚦 Running the Application

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
| Answer Relevancy | 0.904 | Adaptive Query Router & Query Rewriting Loops |
| Context Precision | 0.818 | Dense Cross-Encoder Re-Ranking Pipeline |
| Context Recall | 0.800 | Chunking Strategy & Document Grading Node |
| Faithfulness | 0.800 | Self-RAG Output Reflection Guardrail |

To execute the evaluation pipeline locally:

```bash
python -m evaluation.evaluate_final
```

---

## 🔒 Security & Guardrails Overview

- **Input Security Gateway:** Evaluates incoming user inputs against a domain-specific whitelist, dropping non-operational queries, script injections, and jailbreak attempts before invoking backend graph paths.
- **Hallucination Quarantine:** Ensures that generative RAG drafts undergo mandatory factual verification against ground-truth document chunks before streaming completion.
- **Controlled Side-Effects:** Prevents external actions (e.g., email dispatch, database writes) from running purely autonomously by requiring a verifiable approval handshake from the client interface.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for complete details.