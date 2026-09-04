# NextBridge Agentic HR Microservice (CRAG, Self-RAG & HITL Architecture)

A production-grade, event-driven Agentic AI microservice that orchestrates corporate HR workflows. Built on a hierarchical LangGraph state machine, the system integrates Corrective RAG (CRAG) with Self-RAG reflection, programmatic entity-scoped tool execution, dynamic tool sandboxing for Human-in-the-Loop (HITL) actions, two-tier semantic caching, and full duplex voice streaming.

## Table of Contents

- [Architectural Highlights](#architectural-highlights)
- [System Architecture (Super-Graph)](#system-architecture-super-graph)
- [Core Engineering Mechanisms](#core-engineering-mechanisms)
  - [1. Corrective RAG & Self-RAG Subgraph](#1-corrective-rag--self-rag-subgraph)
  - [2. Programmatic Entity Scoping (Anti-Hallucination)](#2-programmatic-entity-scoping-anti-hallucination)
  - [3. Dynamic Tool Sandboxing (HITL Security)](#3-dynamic-tool-sandboxing-hitl-security)
  - [4. Real-Time SSE Buffer Invalidation](#4-real-time-sse-buffer-invalidation)
  - [5. Two-Tier Semantic & Frequency Caching](#5-two-tier-semantic--frequency-caching)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites & Environment](#prerequisites--environment)
- [Installation & Local Setup](#installation--local-setup)
- [Running the Application](#running-the-application)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Security & Guardrails Overview](#security--guardrails-overview)
- [License](#license)

## Architectural Highlights

- **Hierarchical State Orchestration**: Decouples lightweight conversational routing from compute-heavy CRAG loops, eliminating tool hallucination and non-deterministic routing traps.
- **Resilient Self-Correction**: Evaluates retrieved chunk relevance with a structured grader, rewrites poor retrieval queries, and self-checks generated responses against context using an output reflection guardrail with semantic leniency.
- **Zero-Trust Tool Sandboxing**: Strips write/dispatch capabilities (send_department_email) from the agent's runtime execution environment until strict Human-in-the-Loop verification conditions are satisfied.
- **Programmatic Entity Guarding**: Enforces deterministic query transformations at the tool level to prevent external entity collisions across similarly named corporate entities.
- **Smooth UI Invalidation Protocol**: Emits granular Server-Sent Events (clear, token, status, tool_start, tool_end) to deliver real-time token streaming and transparent single-wipe initialization without client-side text-stacking glitches.
- **Fair-Share Map-Reduce Retrieval**: Decomposes complex, multi-intent questions into isolated parallel vector searches with per-sub-query fair-share chunk distribution (up to 2 chunks per branch) and context capping (6 pages max). This eliminates keyword starvation, cross-encoder penalties, and LLM context exhaustion on compound queries.
- **LLM-Evaluated Semantic Caching**: Replaces naive string hashing with a two-tier Cosine Similarity and LLM-as-a-Judge caching system. Accurately merges acronyms and rephrasings (e.g., "OPD" vs "Outpatient") for tracking, while maintaining strict 90% semantic thresholds for instant cache delivery.
- **Retrievers & Aliasing**: Integrated a full suite of modular retrieval strategies into RETRIEVER_REGISTRY (dense, hybrid, hybrid_rerank, main[multi_sub,hybridrerank], main_self_query, main_parent_doc, champion, main_multi_query, main_rag_fusion, main_hyde), with "champion" aliased directly to MainParentDocumentRetriever.
- **Adaptive Routing & Strategy Switch**: Configured the rag_router_node and retrieval_node to dynamically switch strategies based on query complexity—routing simple, direct queries to the low-latency hybrid_rerank strategy and complex, multi-part, or comparative inquiries to the main_parent_doc (Champion Map-Reduce) architecture, while supporting manual API payload overrides.
- **Fair-Share Allocation & Budget Capping**: Upgraded MainMapReduceRetriever with independent per-sub-query chunk retrieval (compressed[:2] fair-share distribution across sub-query branches) and strict parent hydration limits (capped at 6 high-density pages max) to completely eliminate keyword starvation and context window fatigue.

## System Architecture (Super-Graph)

```
                              +---------------+
                              |   __start__   |
                              +-------+-------+
                                      |
                                      v
                              +---------------+
                              |   guardrail   |
                              +-------+-------+
                     [end]            |
            +----------------------+  |
            |                      |  v
            |                      | +---------------+
            |                      | |     router    |
            |                      | +---+-------+---+
            |          [email/web/chat] |       | [rag: simple/complex]
            |                      |    |       |
            |                      v    |       v
            |            +-------------+|  +----------------------------------------+
            |     +----->|    agent    ||  |             rag_wrapper                |
            |     |      +--+-------+--+|  |                                        |
            |     | [tool   |       |   |  |          +-------------+               |
            |     |  call]  |    [end]  |  |          | rag_router  |               |
            |     |         v       |   |  |          +------+------+               |
            |     |    +--------+   |   |  |                 v                      |
            |     |    | tools  |   |   |  |          +-------------+               |
            |     |    +--+--+--+   |   |  |    +---->|  retrieve   |               |
            |     |       |  |      |   |  |    |     +--+-------+--+               |
            | [resume]    | [end]   |   |  | [complex] |       | [simple]           |
            |     |       |  |      |   |  |    |      v       |                    |
            |     +-------+  |      |   |  |    | +-----------+|                    |
            |                |      |   |  |    | |  grader   ||                    |
            |                |      |   |  |    | +--+-----+--+|                    |
            |                |      |   |  |    | [bad] | [relevant]                |
            |                |      |   |  |    |[docs] v       |    |               |
            |                |      |   |  |    | +---------+   |    |               |
            |                |      |   |  |    +-| rewrite |   |    |               |
            |                |      |   |  |      +---------+   |    |               |
            |                |      |   |  |                    v    v               |
            |                |      |   |  |             +-------------+             |
            |                |      |   |  |             | generation  |             |
            |                |      |   |  |             +--+-------+--+             |
            |                |      |   |  |     [regenerate] |    | [grounded/end]  |
            |                |      |   |  |             v          |                |
            |                |      |   |  |     +-------------+    |                |
            |                |      |   |  |     | reflection  |    |                |
            |                |      |   |  |     +--+-------+--+    |                |
            |                |      |   |  |  [missing]| [grounded/end]              |
            |                |      |   |  | [evidence]|      |     |                |
            |                |      |   |  |     +-----+      v     v                |
            |                |      |   |  |                +-----------+            |
            |                |      |   |  |                | __end__   |            |
            |                |      |   |  |                +-----+-----+            |
            |                |      |   |  +----------------------|------------------+
            |                |      |   |                         v
            |                |      |   |                  +-----------+
            |                |      +---+----------------->|   tools   |
            |                |          |                  +-----+-----+
            |                |          |                        |
            |                +----------+------------------------+
            |                           |
            |                           v
            |                     +-----------+
            +-------------------->|  __end__  |
                                   +-----------+
```

## Core Engineering Mechanisms

### 1. Corrective RAG & Self-RAG Subgraph

Retrieval is split into an isolated, cyclic child graph featuring an adaptive dual-pipeline:

- **Fair-Share Map-Reduce Retrieval**: Dynamically decomposes multi-intent questions into semantic sub-queries. It executes sequential hybrid searches, applies localized Cross-Encoder re-ranking, and uses a fair-share chunk distribution (up to 2 chunks per sub-query branch) capped at 6 total pages to prevent keyword starvation and context flooding.
- **Resilient Self-Correction**: Evaluates retrieved chunk relevance, rewrites poor retrieval queries, and self-checks generated answers against context using an output reflection guardrail configured with a semantic leniency clause for minor phrasing variants.

### 2. Programmatic Entity Scoping (Anti-Hallucination)

To eliminate entity collision in external search indexes (e.g., confusing Nextbridge Pvt. Ltd. with unrelated global entities like Next Bridge Hydrocarbons), the tool execution layer enforces automatic query decoration:

```python
# Programmatic context bounding inside the tool boundary
scoped_query = f"{query} Nextbridge software IT company Pakistan Lahore"
```

This guarantees grounding in the software firm's corporate footprint without relying on the LLM to recall contextual modifiers.

### 3. Dynamic Tool Sandboxing (HITL Security)

To enforce Human-in-the-Loop safety and prevent Tool Shortcut Hallucinations (where the model calls execution tools prematurely):

- **State Masking**: The agent_node dynamically inspects conversation state. During standard generation, only draft_department_email is bound to the LLM.
- **Physical Revocation**: The destructive send_department_email tool is omitted from the model's schema until the client explicitly submits an approval payload from the UI verification card.

### 4. Real-Time SSE Buffer Invalidation

When the Self-RAG subgraph detects a factual drift and rewrites a response, sending incremental tokens over standard SSE can cause draft stacking in client UIs.

- **The Single-Wipe Protocol**: On on_chat_model_start for the generation node, a request-scoped local flag emits data: {"type": "clear"} exactly once on Attempt #1, ignoring subsequent reflection/regeneration correction loops.
- **Client Buffer Invalidation**: The frontend intercepts this signal, flushes the accumulated markdown buffer, and cleanly streams the final verified answer.

### 5. Two-Tier Semantic & Frequency Caching

Avoids redundant graph execution and LLM overhead using an intelligent Cosine Similarity engine:

- **Write Phase (LLM-as-a-Judge)**: Uses a relaxed vector threshold (65% Cosine Similarity) to fetch potential historical queries, handing them to an ultra-fast LPU-based LLM Judge to verify intent equivalence. This natively handles typos, acronyms, and semantic overlaps without false positives.
- **Read Phase (Instant Delivery)**: Uses a strict 90% Cosine Similarity gate to intercept identical queries before they hit the LangGraph workflow, streaming cached verified answers to the frontend in < 150ms.

## Technology Stack

| Component | Technology | Role |
|---|---|---|
| Orchestration | LangGraph (>=0.2.0) | Stateful cyclic graph execution, child subgraphs, checkpointers |
| Backend Framework | FastAPI | Async SSE token streaming, WebSockets, REST webhooks |
| State Persistence | AsyncSqliteSaver (aiosqlite) | Async thread checkpointing and session state persistence |
| Inference Engines | Groq Cloud / Local Qwen Models | Routing, ReAct reasoning, extraction, and generation |
| Embeddings & Search | ChromaDB + HuggingFace | Dense vector store + BAAI/bge-base-en-v1.5 embeddings |
| Re-Ranking | Cross-Encoder | cross-encoder/ms-marco-MiniLM-L-6-v2 for high-precision filtering |
| Observability | LangSmith | Isolated project tracing (nextbridge-hr-agent-production), latency breakdown |
| Transport Resilience | Tenacity | Exponential backoff & jitter wrappers for Cloudflare tunnel transport |
| Evaluation | RAGAS | Context Precision, Recall, Faithfulness, and Answer Relevancy |

## Repository Structure

```
nextbridge_hr_agent/
├── data/
│   ├── HRM_docs/                 # Primary HR policy source PDFs
│   ├── semantic_cache/           # ChromaDB persistent directory for cached responses
│   └── vector_db/                # ChromaDB persistent store for chunk embeddings
├── src/
│   ├── api.py                    # FastAPI server (SSE, WebSockets, isolated LangSmith tracing)
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

## Prerequisites & Environment

- **Python**: 3.11 or higher
- **Inference Access**: Valid Groq API Key OR Local Ollama/vLLM instance via Cloudflare Tunnel
- **Observability**: LangSmith API Key (routing to nextbridge-hr-agent-production)
- **Email Service**: Gmail / Corporate SMTP credentials with App Password enabled

### Environment Variables Configuration

Create a .env file in the project root:

```ini
# --- LLM & Inference ---
GROQ_API_KEY="gsk_..."

# --- Observability (LangSmith Production Isolation) ---
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="nextbridge-hr-agent-production"
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

## Installation & Local Setup

### 1. Clone the Repository & Set Up Virtual Environment

```bash
git clone https://github.com/your-org/nextbridge_hr_agent.git
cd nextbridge_hr_agent

python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ingest HR Policy Documents

Place your HR policy PDFs into data/HRM_docs/ and run the vector ingestion pipeline:

```bash
python -m src.ingestion
```

## Running the Application

The system requires two concurrent runtime processes.

**Step 1: Start the FastAPI API Server (Terminal 1)**

```bash
python -m src.api
```

The API runs on http://localhost:8000.
Interactive API documentation is available at http://localhost:8000/docs.

**Step 2: Start the Background IMAP Listener (Terminal 2)**

```bash
python -m src.email_listener
```

Continuously polls for incoming department replies and routes updates back to active session threads.

**Step 3: Access the Frontend Interface**

Navigate to http://localhost:8000 in your web browser.

## Evaluation & Benchmarking

The architecture undergoes automated evaluation using the RAGAS framework against synthetic and golden dataset distributions to ensure compliance before production promotion.

| Metric | Score | Primary Architectural Driver |
|---|---|---|
| Answer Relevancy | 0.904 | Adaptive Query Router & Map-Reduce Decomposer |
| Context Precision | 0.818 | Fair-Share Cross-Encoder Re-Ranking Pipeline |
| Context Recall | 0.800 | Multi-Sub-Query Distribution & Document Grading Node |
| Faithfulness | 0.980 | Self-RAG Output Reflection Guardrail with Leniency Clause |

To execute the evaluation pipeline locally:

```bash
python -m evaluation.evaluate_final
```

## Security & Guardrails Overview

- **Input Security Gateway**: Evaluates incoming user inputs against a domain-specific whitelist, dropping non-operational queries, script injections, and jailbreak attempts before invoking backend graph paths.
- **Hallucination Quarantine**: Ensures that generative RAG drafts undergo mandatory factual verification against ground-truth document chunks before streaming completion.
- **Controlled Side-Effects**: Prevents external actions (e.g., email dispatch, database writes) from running purely autonomously by requiring a verifiable approval handshake from the client interface.

## License

This project is licensed under the MIT License. See the LICENSE file for complete details.