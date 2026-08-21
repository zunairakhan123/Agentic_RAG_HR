# 🤖 Agentic HR Assistant (CRAG & Self-RAG Architecture)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Async-teal)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Agent-purple)
![LangSmith](https://img.shields.io/badge/LangSmith-traces-blue)
![Qwen3](https://img.shields.io/badge/Qwen3-30B%20(Local)-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Prototype--to--Production-yellow)

A production-grade, event-driven Agentic AI microservice that automates corporate HR workflows. This system moves beyond linear RAG by implementing a Corrective RAG (CRAG) state machine, Self-RAG Reflection (Output Guardrails), Semantic Caching, and a Human-in-the-Loop (HITL) framework for secure asynchronous email dispatch.

## Table of Contents

- [Overview](#overview)
- [Core Architecture (Super-Graph)](#core-architecture-super-graph)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Running the System](#running-the-system)
- [Evaluation & Observability](#evaluation--observability)
- [Security & Guardrails](#security--guardrails)
- [Production Deployment Roadmap](#production-deployment-roadmap)
- [License](#license)

## Overview

This project implements a cyclic, state-driven autonomous agent. Unlike standard conversational chatbots, this architecture evaluates its own retrieved context, rewrites its own search queries if the context is poor, and checks its own generated answers for hallucinations before returning them to the user.

**Key Capabilities:**

- **Super-Graph Routing:** An adaptive traffic cop dynamically routes basic interactions to a fast ReAct conversational agent, and policy queries to a heavy CRAG retrieval loop.
- **Agentic CRAG & Self-RAG:** Implements a Document Grader and Query Rewriter to ensure high-quality context, capped with a Reflection node acting as an output guardrail against hallucinations.
- **Semantic Caching:** Bypasses expensive graph executions for near-duplicate queries using a lightweight ChromaDB cache, dropping latency from ~15s to <0.2s.
- **State Persistence & HITL:** Uses LangGraph with an asynchronous SQLite checkpointer to freeze agent execution before sensitive tool calls (e.g., sending an email), awaiting explicit human approval.
- **Continuous Voice Mode:** Features a bidirectional WebSocket connection with continuous speech-to-text (STT) and barge-in text-to-speech (TTS) streaming.
- **Asynchronous Webhooks:** A standalone IMAP worker continuously polls for external department replies, parses MIME headers, and injects responses directly into the agent's memory graph.

## Core Architecture (Super-Graph)

```
                               ┌─────────────────────────┐
                               │    Frontend / FastAPI   │
                               │  (Semantic Cache Check) │
                               └───────────┬─────────────┘
                                           │
                               ┌───────────▼─────────────┐
                               │  Input Guardrail Node   │──(Blocked)──► [ END ]
                               └───────────┬─────────────┘
                                           │ (Pass)
                               ┌───────────▼─────────────┐
                               │  Adaptive Router Node   │
                               └──────┬───────────┬──────┘
                  ┌──(email/chat)─────┘           └──(simple/complex)─────┐
                  │                                                       │
        ┌─────────▼─────────┐                                   ┌─────────▼─────────┐
    ┌──►│ ReAct Agent Node  │                               ┌──►│  Retrieval Node   │
    │   └─────────┬─────────┘                               │   └─────────┬─────────┘
    │             │ (tools)                                 │             │
    │   ┌─────────▼─────────┐                               │   ┌─────────▼─────────┐
    └───┤     Tool Node     │                               │   │    Grader Node    │
        │ (Web Search, SMTP)│                               │   └─────┬───────┬─────┘
        └───────────────────┘                               │ (rewrite) │       │ (generate)
                  │ (end)                                   │           │       │
                  │                               ┌─────────┴────────┐  │  ┌────▼────────────┐
                  │                               │   Rewrite Node   │◄─┘  │  Generate Node  │◄─┐
                  │                               └─────────▲────────┘     └────┬────────────┘  │
                  │                                         │                   │               │
                  │                                     (rewrite)               ▼               │ 
                  │                                         │        ┌───────────────────┐      │
                  │                                         └────────┤  Reflection Node  ├──────┘
                  │                                                  └──────────┬────────┘   
                  |                                                             |
                  │                                                             │ (end: grounded)
                  └─────────────────────────────────────────────────────────────▼────────► [END ]
```

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Orchestration | LangGraph | State machine, CRAG loops, and HITL execution freezing |
| Inference | Qwen3:30b (Local) | Heavy reasoning, query routing, and RAG evaluation |
| Backend API | FastAPI | Async SSE streaming, WebSockets, and webhooks |
| Vector Store | ChromaDB | BGE embeddings for hybrid retrieval and semantic caching |
| Re-Ranking | HuggingFace Cross-Encoder | ms-marco-MiniLM-L-6-v2 for dense precision ranking |
| State DB | SQLite (aiosqlite) | Persistent checkpointer for LangGraph thread memory |
| Observability | LangSmith | End-to-end node latency, loop tracing, and token usage |
| Evaluation | RAGAS | Mathematical benchmarking for Context Precision & Relevancy |

## Repository Structure

```
nextbridge_hr_agent/
├── data/
│   ├── HRM_docs/              # HR PDF source documents
│   ├── semantic_cache/        # ChromaDB cache for duplicate queries
│   └── vector_db/             # Persisted ChromaDB embeddings
├── src/
│   ├── api.py                 # FastAPI server (SSE, WebSockets, Webhooks)
│   ├── cache.py               # Semantic query caching logic
│   ├── email_listener.py      # Background IMAP worker for thread ID extraction
│   ├── graph.py               # Super-Graph definition (CRAG + ReAct)
│   ├── ingestion.py           # ETL pipeline (Chunking, BGE Embeddings)
│   ├── nodes.py               # Agentic node definitions (Grader, Rewriter, Router)
│   ├── state.py               # LangGraph TypedDict state schema
│   └── tools.py               # Web search & SMTP dispatch tools
├── tests/                     # Each component tests
├── evaluation/
│   ├── datasets/              # Golden dataset (JSON)
│   └── evaluate_final.py      # RAGAS evaluation pipeline
├── frontend/
│   └── index.html             # UI (Short-polling, TTS, STT, HITL Approval)
└── hr_agent_memory.db         # Persistent SQLite checkpointer DB (auto-generated)
```

## Prerequisites

- Python 3.11+
- Local LLM endpoint (e.g., Cloudflare tunnel pointing to Ollama/vLLM hosting Qwen3:30b)
- LangSmith Account (for tracing)
- Gmail / Corporate SMTP account with an App Password

### Environment Variables (.env)

```ini
# LangSmith Observability
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="NextBridge_Agentic_HR_System"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="lsv2_pt_..."

# Email Configurations
SMTP_EMAIL="your_testing_email@gmail.com"
SMTP_PASSWORD="your_16_char_app_password"

# Tool APIs
TAVILY_API_KEY="tvly-..."
```

## Running the System

The architecture requires two parallel processes.

**1. FastAPI Backend (Terminal 1):**

```bash
python -m src.api
```

**2. IMAP Background Worker (Terminal 2):**

```bash
python -m src.email_listener
```

**Accessing the Client:**

- Open `http://localhost:8000` (or `frontend/index.html` directly) in a modern browser.
- Type a query to use standard SSE streaming.
- Click "Enter Voice Mode" to test the continuous WebSocket STT/TTS loop.

## Evaluation & Observability

This architecture was strictly benchmarked against a golden dataset using the RAGAS framework, replacing standard linear RAG with the CRAG loops.

**Final Architecture Scorecard:**

- Answer Relevancy: 0.904 (Driven by the Query Rewriter & Router)
- Context Precision: 0.818 (Driven by Cross-Encoder Re-ranking)
- Context Recall: 0.800 (Driven by Hybrid Search & Chunking strategy)
- Faithfulness: 0.800 (Driven by the Self-RAG Output Guardrail)

All node executions, loop iterations, and token counts are streamed dynamically to the LangSmith dashboard for real-time CI/CD monitoring.

## Security & Guardrails

- **Input Guardrail:** `input_guardrail_node` mathematically restricts queries to HR policies, leaves, payroll, and engineering, preventing prompt injection or off-topic abuse.
- **Output Guardrail:** `reflection_node` runs a strict hallucination check against retrieved context before outputting to the user.
- **HITL by Design:** The `send_department_email` tool can never execute autonomously. LangGraph safely suspends the state until explicit frontend POST approval.

## Production Deployment Roadmap

To scale this to enterprise production environments:

- **Containerization:** Wrap FastAPI and the IMAP worker in `docker-compose`.
- **Database Migration:** Upgrade the `aiosqlite` checkpointer to PostgreSQL (`langgraph-checkpoint-postgres`) to support horizontal scaling. Upgrade local Chroma to Pinecone/Milvus.
- **Queueing:** Replace the local IMAP loop with an event-driven Kafka or RabbitMQ queue for high-throughput webhook handling.
- **CI/CD:** Integrate `evaluation/evaluate_final.py` into GitHub Actions to block PRs that degrade RAGAS metrics below 0.80.

## License

Distributed under the MIT License. See `LICENSE` for details.

        