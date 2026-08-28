## System Architecture (Super-Graph)

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
                                                   [Rewrite]       ▼                   │       │
                                                      |  ┌───────────────────┐         │       │
                                                      └─ │  Reflection Node  │         │       │
                                                         └────┬─────────┬────┘         │       │
                                                              │         │              │       │
                                                       (retry)│         │(grounded)    │       │
                                                              ▼         ▼              ▼       │
                                                        [Regenerate]                 [ END ]   │
                                                              │                                │
                                                              └────────────────────────────────┘