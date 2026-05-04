# Helix SROP — Arnab Nath

## Setup

```bash
git clone <your-repo>
cd helix-srop
uv sync
cp .env.example .env  # fill in GOOGLE_API_KEY
uv run python -m app.rag.ingest --path docs/
uv run uvicorn app.main:app --reload
```

## Quick Test

```bash
# 1. Create a session
SESSION=$(curl -s -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' | jq -r .session_id)

# 2. Chat
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | jq .
```

## Architecture

The system uses a **multi-agent orchestration** pattern powered by the Google ADK.

- **Root Orchestrator**: Routes user intent to specialist sub-agents.
- **Knowledge Agent**: Handles RAG retrieval using semantic search + LLM reranking.
- **Account Agent**: Interacts with mock account/build systems.
- **Support Agent**: Handles escalation by creating tickets in a persistent database.

## Design Decisions

### State persistence (Pattern 3)

I used **Pattern 3 (Dynamic Agent Injection)** because it allows the AI to be fully stateful while the backend remains stateless. By injecting the user context (`user_id`, `plan_tier`) and recent conversation history directly into the system instruction each turn, we avoid complex custom session service implementations while maintaining excellent multi-turn coherence.

### Chunking strategy

I used a **hybrid heading-aware and sentence-aware** chunking strategy. By splitting documents based on Markdown headers first, we preserve the semantic grouping of sections. Within large sections, we use sentence-based overlap to ensure context is not lost at the boundaries.

### Vector store choice

I chose **ChromaDB** because of its native support for embedding functions and lightweight persistent storage, which is ideal for a self-contained submission.

## Extensions Completed

- [X] **E1: Idempotency** — Implemented `Idempotency-Key` header support. Responses are cached in the DB to prevent re-running the pipeline on replay.
- [X] **E2: Escalation agent** — Added a `SupportAgent` and a `create_ticket` tool that writes to a persistent `tickets` table.
- [X] **E3: Streaming SSE** — `POST /chat/{id}` supports `Accept: text/event-stream` for real-time token streaming.
- [X] **E4: Reranking** — Implemented an **LLM-as-judge reranker** in the documentation search tool. It retrieves the top 10 candidates and uses Gemini-1.5-Flash to select the most relevant top 5.
- [X] **E5: Guardrails** — Implemented refusal on out-of-scope queries and automated PII redaction (Emails/API Keys) in the structured logs. Added `tests/test_guardrails.py`.
- [X] **E6: Docker** — Provided `Dockerfile` and `docker-compose.yml` for easy deployment.

## Known Limitations

- Reranking adds approximately 1-2 seconds of latency to the initial knowledge retrieval.
- Free-tier Gemini API quotas are highly restrictive (limit: 20-60 RPM).
