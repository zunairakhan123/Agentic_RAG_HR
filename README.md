# 🤖 Asynchronous Agentic HR Assistant

> A decoupled, event-driven Agentic AI microservice that automates corporate HR workflows using LLMs, Retrieval-Augmented Generation (RAG), and a Human-in-the-Loop (HITL) state machine for secure email dispatch and asynchronous webhook processing.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Async-teal)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Agent-purple)
![Groq](https://img.shields.io/badge/Groq-Llama%203.1%208B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Prototype--to--Production-yellow)

---

## Table of Contents

- [Overview](#overview)
- [Core Architecture](#core-architecture)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Environment Variables](#environment-variables)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Security Considerations](#security-considerations)
- [Testing](#testing)
- [Production Deployment Roadmap](#production-deployment-roadmap)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project moves beyond linear, single-turn conversational chatbots by implementing a **cyclic, state-driven autonomous agent**. It bridges internal HTTP network requests with external corporate SMTP/IMAP servers to answer HR policy questions and — when necessary — draft and send emails to the relevant department on a human's explicit authorization.

**Key capabilities:**

- **Agentic RAG** — Dynamically searches a local ChromaDB vector store to ground responses in verified company HR policy documents, reducing hallucination.
- **State Persistence & HITL** — Uses LangGraph with a SQLite checkpointer to freeze agent execution before sensitive tool calls (e.g., sending an email), awaiting explicit human approval via the frontend.
- **Asynchronous Webhooks** — A standalone IMAP background worker continuously polls for department replies, parses Base64/MIME headers, and injects external responses directly back into the agent's memory graph.
- **Decoupled Short-Polling Frontend** — A stateless JavaScript/HTML interface that stays in sync with the FastAPI backend without requiring persistent WebSocket connections.

---

## Core Architecture

```
                        ┌─────────────────────┐
                        │   Frontend (HTML/JS)│
                        │Short-polling+TTS+STT│
                        └──────────┬──────────┘
                                   │ SSE / REST
                                   ▼
                        ┌─────────────────────┐
                        │   FastAPI Backend   │
                        │  (api.py)           │
                        │  - SSE streaming    │
                        │  - Webhook endpoint │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼                              ▼
          ┌───────────────────┐         ┌───────────────────┐
          │   LangGraph Agent │◄───────►│  SQLite Checkpoint│
          │   (graph.py)      │         │  (thread memory)  │
          │  - Guardrails     │         └───────────────────┘
          │  - HITL freeze/   |
          |    resume         │
          └─────────┬─────────┘
                    │
        ┌───────────┼───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────────┐
│  ChromaDB RAG │       │  Tools (tools.py) │
│ (ingestion.py)│       │  - Retrieval      │
└───────────────┘       │  - Web search     │
                        │  - Email dispatch │
                        └──────────┬────────┘
                                   │ SMTP
                                   ▼
                        ┌─────────────────────┐
                        │  Corporate Mailbox  │
                        └──────────┬──────────┘
                                   │ IMAP (polling)
                                   ▼
                        ┌─────────────────────┐
                        │  email_listener.py  │
                        │  Background worker  │
                        │ → injects replies in│
                        │ to the agent graph  │
                        └─────────────────────┘
```

**Flow summary:**
1. A user asks an HR policy question through the frontend.
2. The LangGraph agent retrieves grounded context from ChromaDB and reasons via Groq's Llama 3.1 8B model.
3. If the query requires escalation (e.g., "email the finance department"), the agent **freezes execution** at the email tool call and waits for human approval.
4. Once approved, the email is dispatched via SMTP and the thread ID is tracked.
5. The IMAP worker polls for a reply, parses it, and resumes the frozen graph with the new information — closing the loop without any manual re-prompting , then notification appears on the frontend by short polling.

---

## Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | LangGraph | Cyclic state machine, routing, and HITL execution freezing |
| **Integration** | LangChain | Tool definition schemas and vector database wrappers |
| **Inference** | Groq (Llama 3.1 8B) | High-throughput, low-latency function calling and reasoning |
| **Backend API** | FastAPI (Python, async) | SSE streaming and webhook endpoint management |
| **Vector Store** | ChromaDB | Local embedding storage for document retrieval |
| **State DB** | SQLite | Persistent checkpointer for LangGraph thread memory |
| **Networking** | SMTP / IMAP | Dispatching drafts and listening for external department replies |
| **Frontend** | HTML / JavaScript | Short-polling UI with SSE streaming and browser TTS |

---

## Repository Structure

```
nextbridge_hr_agent/
├── data/
│   ├── HRM_docs/               # Place your 23–25 HR PDF documents here
│   └── vector_db/               # Persisted ChromaDB embeddings (auto-generated)
├── src/
│   ├── __init__.py
│   ├── email_listener.py        # Background IMAP worker for thread ID extraction
│   ├── ingestion.py              # PDF loading, semantic chunking, BGE embedding
│   ├── state.py                  # LangGraph state schema definition
│   ├── tools.py                  # RAG retrieval, guardrailed web search, & email tools
│   ├── graph.py                  # LangGraph state machine, guardrails, & checkpointing
│   └── api.py                    # FastAPI server with SSE streaming & email webhooks
├── frontend/
│   └── index.html                 # Web UI (SSE streaming, HITL approval, browser TTS)
├── .env.example                    # Template for required environment variables
├── hr_agent_memory.db                # Local SQLite DB for graph persistence (auto-generated)
└── requirements.txt                     # Python dependencies
```

---

## Prerequisites

- Python 3.11+
- pip / venv
- A Groq API key ([console.groq.com](https://console.groq.com))
- An email account with **app password** access (Gmail, Outlook, or corporate SMTP/IMAP with app-password support)
- (Optional) Docker & Docker Compose for containerized deployment

---

## Local Development Setup

### 1. Clone and install dependencies

```bash
git clone <repository_url>
cd nextbridge_hr_agent
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example file and fill in your own credentials:

```bash
cp .env.example .env
```

See [Environment Variables](#environment-variables) below for the full list and descriptions.

> ⚠️ **Never commit your `.env` file.** Confirm `.env` is listed in `.gitignore` before your first commit. If you've already pushed real credentials, rotate them immediately — removing the file later does not remove it from git history.

### 3. Initialize the vector database

Run the ETL pipeline to chunk your HR documents and generate embeddings:

```bash
python -m src.ingestion
```

This reads all PDFs from `data/HRM_docs/` and persists embeddings to `data/vector_db/`.

---

## Environment Variables

| Variable | Required | Description |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | ✅ | API key for Groq LLM inference (Llama 3.1 8B) |
| `SMTP_EMAIL` | ✅ | Sending mailbox address for outbound HR emails |
| `SMTP_PASSWORD` | ✅ | App password (16-char) for the SMTP account — **not your regular login password** |
| `TAVILY_API_KEY` | ✅ | Internet Search Provider |

Example `.env.example`:

```ini
GROQ_API_KEY=our_groq_key
SMTP_EMAIL=your_testing_email@gmail.com
SMTP_PASSWORD=your_16_char_app_password
TAVILY_API_KEY=your_key
```

---

## Running the System

The architecture requires **two parallel processes**.

**Terminal 1 — FastAPI backend:**

```bash
python -m src.api
```

**Terminal 2 — IMAP background worker:**

```bash
python -m src.email_listener
```

**Launch the client:**

Open `start frontend/index.html` in any modern browser. The frontend short-polls the FastAPI backend and subscribes to SSE for streamed agent responses.

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/chat` | Submit a user query; returns an SSE stream of the agent's reasoning/response |
| `GET` | `/status/{thread_id}` | Poll the current state of a given conversation thread |
| `POST` | `/approve/{thread_id}` | Human approval endpoint — resumes a frozen HITL graph and dispatches the pending email |
| `POST` | `/reject/{thread_id}` | Rejects a pending action and unfreezes the graph without sending |
| `POST` | `/webhook/email` | Internal endpoint used by `email_listener.py` to inject parsed replies into the graph |

> Interactive Swagger docs are available at `/docs` once the FastAPI server is running.

---

## How It Works

1. **Ingestion** — HR policy PDFs are chunked and embedded (BGE embeddings) into ChromaDB.
2. **Retrieval** — Incoming questions are matched against the vector store to ground the LLM's answer in actual policy text.
3. **Reasoning** — Groq's Llama 3.1 8B performs function calling to decide whether to answer directly, search the web (guardrailed), or escalate via email.
4. **HITL Freeze** — Before any email is sent, LangGraph pauses execution and persists state to SQLite, waiting for a human to approve or reject via the frontend.
5. **Dispatch & Listen** — On approval, the email is sent via SMTP; `email_listener.py` polls IMAP in the background and resumes the graph automatically when a reply arrives.

---

## Security Considerations

- **Credentials:** Use app passwords, never primary account passwords, for `SMTP_PASSWORD`.
- **`.env` hygiene:** Confirm `.env` is git-ignored before pushing. Rotate any credential that was ever committed, even briefly.
- **Guardrailed web search:** The web search tool is scoped and filtered — review `tools.py` before relaxing its guardrails in production.
- **HITL by design:** Any outbound email requires explicit human approval; the agent cannot autonomously send email.
- **PII in vector store:** HR documents may contain sensitive data — restrict filesystem and network access to `data/vector_db/` accordingly.

---

## Production Deployment Roadmap

To scale this prototype into a production-grade deployment, the following upgrades are recommended:

- **Containerization** — Wrap the FastAPI server and the IMAP listener in isolated Docker containers via `docker-compose` to ensure environment parity across staging and production.
- **Database Migration** — Replace the local SQLite checkpointer with a managed PostgreSQL instance for concurrent state tracking, and migrate ChromaDB to Pinecone for scalable vector retrieval.
- **Workflow Automation** — Implement GitHub Actions to automate linting, testing, and container registry pushes on merge to `main`.
- **Message Broker** — Transition the IMAP polling script to push events to a Redis or RabbitMQ queue, allowing the FastAPI server to process incoming webhooks asynchronously under high load.
- **Observability** — Add structured logging and tracing (e.g., LangSmith) around graph transitions for debugging HITL flows in production.
- **Secrets Management** — Move from `.env` files to a managed secrets store (AWS Secrets Manager, HashiCorp Vault, etc.) for production credentials.

## Troubleshooting

| Issue | Likely Cause | Fix |
| :--- | :--- | :--- |
| `email_listener.py` never picks up replies | IMAP credentials or app password incorrect | Verify app password and IMAP host; check firewall/port access |
| Agent responses aren't grounded in policy docs | Vector DB not initialized or stale | Re-run `python -m src.ingestion` |
| HITL approval doesn't resume the graph | Thread ID mismatch between frontend and backend | Confirm `thread_id` is persisted client-side and passed on `/approve` |
| SMTP `Authentication failed` | Using account password instead of app password | Generate and use a dedicated app password |

---

## Contributing

1. Fork the repository and create a feature branch.
2. Follow existing code style (`black`, `ruff` recommended).
3. Add or update tests for any new behavior.
4. Open a pull request with a clear description of the change.

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

*Engineered for scalability, automation, and complex real-world HR workflows.*