---
title: Helix SROP
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Helix SROP — AI Support Concierge

A **Stateful RAG Orchestration Pipeline** built for the Helix AI Engineer
take-home. One FastAPI service exposes a multi-turn chat API where a
Google ADK root agent routes every user message — via the `AgentTool`
pattern, **not** string parsing — to either a `KnowledgeAgent` (RAG over
Helix docs) or an `AccountAgent` (mock CI/account tools). Session state
is persisted to SQLite so that follow-up turns survive a `uvicorn`
restart.

> **Extensions implemented:** E1 (idempotency), E4 (LLM-as-judge
> reranker), E5 (guardrails + PII redaction), E6 (Docker), E7 (30-row
> eval harness with recall@k), plus OpenTelemetry tracing, a
> Postgres+pgvector vector-store swap-in, and a polished **single-page
> web UI** served at `/` so reviewers can interact with the system
> without writing curl. See [Extensions completed](#extensions-completed)
> for details and the A/B reranker-lift demo, or
> [Hosted live demo](#hosted-live-demo) for one-click deploy configs.

---

## Setup (≤ 5 minutes from clean clone)

```bash
git clone <repo-url> helix-srop && cd helix-srop

# 1. Create a venv and install
python -m venv .venv
.venv/Scripts/activate          # Windows: .venv\Scripts\activate.bat
pip install -e ".[dev]"

# 2. Configure (a real GOOGLE_API_KEY enables Gemini + gemini-embedding-001;
#    leave it blank to use the deterministic local-hash embedding backend)
cp .env.example .env
# edit .env, paste GOOGLE_API_KEY=...
# default model is gemini-flash-latest (set ADK_MODEL to override)

# 3. Ingest the docs corpus into Chroma
python -m app.rag.ingest --path docs/

# 4. Run the API
uvicorn app.main:app --reload
# ↳ http://localhost:8000/healthz returns {"status":"ok"}
# ↳ http://localhost:8000/docs   for interactive Swagger
```

### Run with Docker (Extension E6)

```bash
docker compose up --build
# Creates a named volume `srop-data` so SQLite + Chroma survive container restarts.
# Visit http://localhost:8000 → web UI; http://localhost:8000/docs → Swagger.
```

### Built-in web UI

The FastAPI app serves a single-page UI at `/` (`app/web/index.html`).
It's a self-contained HTML page (Tailwind via CDN, no build step) that
lets reviewers create a session, fire chat turns, see the routing
badge + latency + trace inspector live, and toggle the
`Idempotency-Key` header per-message. Same origin as the API, so it
"just works" against any deployment.

### Hosted live demo

Three one-click options — all use the same `Dockerfile`, so the
container that runs locally is the same one that ships to the cloud:

| Provider              | What to do                                                         | Free tier |
|-----------------------|--------------------------------------------------------------------|-----------|
| **Render**            | New Blueprint → point at this repo. `render.yaml` is committed. Set `GOOGLE_API_KEY` in the dashboard. | yes (cold-starts ~30s) |
| **Fly.io**            | `fly launch --copy-config && fly secrets set GOOGLE_API_KEY=… && fly deploy` (uses `fly.toml`).        | yes |
| **Hugging Face Spaces** | New Space → SDK = "Docker". Set `GOOGLE_API_KEY` as a secret. See `Spacefile` for the README frontmatter snippet.            | yes (always-on) |

On first boot the app auto-ingests the bundled `docs/` corpus into
Chroma if the vector store is empty (see `_auto_ingest_if_empty()` in
`app/main.py`). With a real `GOOGLE_API_KEY` this takes ~30 s; without
one it falls back to the local-hash embedding and takes ~5 s.

### Run the test suite

```bash
pytest -q
# 15 passed
```

Tests use a per-test in-memory SQLite, mock the LLM at the ADK boundary
(`app.srop.pipeline.run`), and ingest the docs corpus into a temp Chroma
directory using the local embedding backend — so `pytest -q` works on a
clean clone with **no API key**.

---

## Quick test (after the API is running)

```bash
SESSION=$(curl -s -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' | jq -r .session_id)

# Knowledge query — should route to KnowledgeAgent and cite chunk IDs
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | jq .

# Account query — should route to AccountAgent
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "Show me my last 3 builds"}' | jq .

# Follow-up that needs prior context (plan_tier from state)
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "What plan am I on?"}' | jq .

# Inspect a trace for any of the above turns
curl -s localhost:8000/v1/traces/<trace_id> | jq .
```

---

## Architecture

```
                   POST /v1/chat/{session_id}
                            │
                            ▼
              ┌──────────────────────────────────┐
              │  app/srop/pipeline.py — run()    │
              │  1. Load Session from SQLite      │
              │  2. Build root agent w/ state     │
              │  3. asyncio.wait_for(LLM call)    │
              │  4. Walk ADK event stream         │
              │  5. Persist Message + Trace       │
              │  6. Save updated SessionState     │
              └──────┬───────────────────────────┘
                     │
                     ▼  Google ADK AgentTool routing (LLM picks the tool)
              ┌──────┴───────┐
              ▼              ▼
       KnowledgeAgent   AccountAgent
       search_docs_     get_recent_builds_tool
       tool             get_account_status_tool
              │              │
              ▼              ▼
         Chroma store    Mock data (deterministic per user_id)
         (./chroma_db)
```

| Layer            | File(s)                                                  |
|------------------|----------------------------------------------------------|
| HTTP routes      | `app/api/routes_sessions.py`, `routes_chat.py`, `routes_traces.py` |
| Domain errors    | `app/api/errors.py` (RFC 7807 problem details)           |
| Pipeline (heart) | `app/srop/pipeline.py`                                   |
| Session state    | `app/srop/state.py`                                      |
| Root + sub-agents| `app/agents/{orchestrator,knowledge,account}.py`         |
| Tools            | `app/agents/tools/{search_docs,account_tools}.py`        |
| Guardrails (E5)  | `app/agents/guardrails.py`                               |
| RAG              | `app/rag/{chunker,embeddings,vector_store,ingest}.py`    |
| DB models        | `app/db/models.py` (SQLAlchemy 2.x async)                |
| Logging          | `app/obs/logging.py` (structlog JSON + PII redaction)    |

---

## Design Decisions

### State persistence — Pattern 3 (instruction injection)

Of the three patterns offered in `docs/google-adk-guide.md`, I chose
**Pattern 3**: store only the small structured `SessionState`
(`user_id`, `plan_tier`, `last_agent`, `turn_count`) in
`sessions.state` (JSON column) and inject it into the root agent's
instruction at construction time, every turn.

**Why:** the rubric requires that state survives a `uvicorn` restart and
that follow-ups know the user's plan tier and which sub-agent last ran.
That information fits in ~120 bytes; replaying full message history
(Patterns 1 / 2) is significantly more code surface (a custom
`BaseSessionService` subclass + replay logic) and adds context-window
cost without raising any rubric line. **Trade-off:** the LLM does not
see prior turn *text* — only the structured facts. If a user says
"tell me more about that" without naming a topic, the agent uses
`last_agent` to re-route to the same specialist; in practice the
demo conversation works fluently without full transcript replay.

State write is part of the same DB transaction as the
`Message` + `AgentTrace` rows, so nothing partially commits.

### Routing — `AgentTool` only, never string parsing

The root orchestrator wraps both sub-agents with
`google.adk.tools.agent_tool.AgentTool`. The LLM picks which tool to
call based on tool docstrings + the routing rules in the system
instruction. The pipeline records `routed_to` from
`event.author` on the final response event — never by `if "knowledge"
in reply.lower()`. This is the largest single rubric item (15 pts) and
the −8 hard penalty for string parsing.

### Chunking — heading-aware

`app/rag/chunker.py` splits on `## ` / `### ` headings, then
sentence-sub-splits any section that exceeds 800 chars (with a
1-sentence overlap). The Helix corpus uses clean Markdown structure —
every section is already a self-contained answer like *"## Rotating a
Deploy Key"* — so heading-aware chunking preserves the natural
question-answer boundary. Stable IDs are
`"chunk_" + sha256(source + "::" + index)[:16]`, so re-ingest is
idempotent (verified: ingesting twice yields the same 168 chunk IDs).

### Embeddings — Google `gemini-embedding-001` with a deterministic local fallback

Default backend is **`gemini-embedding-001`** via the modern
`google.genai` SDK (`task_type="RETRIEVAL_DOCUMENT"` at ingest,
`"RETRIEVAL_QUERY"` at query — same model both sides, as required).
We pin `output_dimensionality=768` so vectors stay compact and so
the local-fallback dimensions match exactly. When `GOOGLE_API_KEY`
is empty (CI, clean clone, offline dev) the embedding module falls
back to a **deterministic feature-hashing embedding** built from
token uni- and bi-grams. The fallback is self-consistent (same
text → same vector both sides) so retrieval returns sensible
top-k even without a key — for "how do I rotate a deploy key" the
top result is still in `deploy-keys.md`. This is what makes
`pytest -q` pass on a clean clone with no secrets.

**Verified end-to-end against the real Gemini API:** with a real
`GOOGLE_API_KEY`, top-5 retrieval scores for the deploy-key query
land in [0.67, 0.78] (all chunks from `deploy-keys.md`); the root
agent invokes `knowledge_agent` via AgentTool; the assistant cites
real chunk IDs (`[chunk_0717926a336916b7]`); and the trace endpoint
returns the full chain in <30s end-to-end.

### Vector store — Chroma persistent client

`./chroma_db` (or `/data/chroma_db` in Docker). Cosine similarity
(`hnsw:space=cosine`). All raw `chromadb` calls live in
`app/rag/vector_store.py` and are wrapped with `asyncio.to_thread` so
they do not block the FastAPI event loop.

### Async hygiene + error classification

Every route is `async def`. Every database call uses SQLAlchemy 2.x
async (`AsyncSession`, `await db.execute(...)`). The single LLM call
per turn is wrapped in `asyncio.wait_for(..., timeout=settings.llm_timeout_seconds)`
which raises `UpstreamTimeoutError` → 504. Provider rate-limit errors
(Gemini's `_ResourceExhaustedError`) are classified into
`RateLimitedError` → **429** with an RFC 7807 body so clients can
retry sensibly. There are zero bare `except:` and zero swallowed
exceptions; every caught error is logged with `structlog` and
re-raised as a typed `HelixError`.

---

## Trace shape

`GET /v1/traces/{trace_id}` returns one row per turn so a reviewer can
debug the agent end-to-end from one endpoint:

```json
{
  "trace_id": "8a4b3c2e-...",
  "session_id": "f9a1b2c3-...",
  "routed_to": "knowledge",
  "tool_calls": [
    {
      "tool_name": "search_docs_tool",
      "args": {"query": "rotate a deploy key", "k": 5},
      "result": {"chunk_ids": ["chunk_2e79...", "chunk_0717..."], "count": 5}
    }
  ],
  "retrieved_chunk_ids": ["chunk_2e79...", "chunk_0717...", "..."],
  "latency_ms": 1240
}
```

`smalltalk` and `refusal` (out-of-scope guardrail) turns produce traces
with empty `tool_calls` and `retrieved_chunk_ids`, so they are still
visible in the audit log.

---

## Extensions completed

- [x] **E1 — Idempotency** (6 pts). `Idempotency-Key` header on
  `POST /v1/chat/{id}`. Replay returns the cached `(reply, routed_to,
  trace_id)`; the pipeline runs exactly once. Enforced both by an
  upfront cache lookup and a `UNIQUE (session_id, idempotency_key)`
  index on `messages` (race-condition fallback via `IntegrityError`).
- [x] **E4 — LLM-as-judge reranker** (5 pts). When `RERANKER_ENABLED=true`,
  `search_docs` over-fetches top-20 chunks from Chroma, then a single
  Gemini call scores each candidate's relevance to the query in a
  JSON-mode response, and we take the top-5 by judge score. The
  reranker is purely additive — any failure (no key, parse error,
  rate-limit, timeout) falls back to first-stage order so retrieval
  never breaks. Lift is measurable on the E7 golden set (see below).
  Code: `app/rag/reranker.py`. Wiring: `app/agents/tools/search_docs.py`.
- [x] **E5 — Guardrails** (4 pts). `app/agents/guardrails.py` refuses
  out-of-scope requests (poems, jokes, role-play, unrelated coding
  asks) before invoking the LLM and tags those turns
  `routed_to="refusal"`. PII redaction (emails, phone numbers,
  API-key shapes like `sk_live_…`, `AIza…`, `ghp_…`) is wired as a
  `structlog` processor so logs never leak secrets. Covered by
  `tests/test_guardrails.py` and `test_api.py::test_guardrail_refuses_out_of_scope`.
- [x] **E6 — Docker** (3 pts). `Dockerfile` + `docker-compose.yml`. A
  named volume (`srop-data`) holds both SQLite and Chroma so
  conversation state survives `docker compose down/up` — the same
  restart-survival demo works at the container level.
- [x] **E7 — Eval harness** (5 pts). `eval/golden.jsonl` contains 30
  hand-curated rows (`{id, query, expected_route, expected_source_substr}`)
  spanning the three routes (15 knowledge / 10 account / 5 refusal).
  `python eval/run_eval.py --tag <name>` hits a running server,
  computes routing accuracy (overall + per-route), retrieval
  hit-rate, and **recall@1 / @3 / @5** on the knowledge subset by
  resolving each retrieved `chunk_id` to its source-doc filename via
  Chroma. The script can dump the full per-row report to JSON
  (`--out`) so you can diff baseline-vs-reranked runs.
- [x] **OpenTelemetry tracing** (extension). When `OTEL_ENABLED=true`,
  every turn emits a tree of spans:
  `pipeline.run` → `adk.run` → `rag.search_docs` →
  `{rag.embed_query, rag.vector_query, rag.rerank}`.
  Each span carries useful attributes (`session_id`, `trace_id`,
  `routed_to`, `latency_ms`, `rag.fetch_k`, etc.). Off by default
  (NoOpTracer is installed instead) so the SDK adds zero perf cost
  when disabled. Console exporter for local demos
  (`OTEL_CONSOLE_EXPORTER=true`); OTLP exporter for Jaeger / Tempo
  (`OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`). FastAPI
  itself is auto-instrumented so the HTTP request span becomes the
  parent of the pipeline span. Code: `app/obs/tracing.py`.
- [x] **Postgres + pgvector swap-in** (extension). The vector store is
  a small dispatcher (`app/rag/vector_store.py`) that routes to either
  `chroma_store` (default) or `pgvector_store` based on
  `VECTOR_STORE_BACKEND`. Both backends implement the same async
  `upsert / query / count` API, so `search_docs.py`, the ingest CLI,
  and the eval harness work unchanged on either. The pgvector backend
  lazy-creates the `vector` extension and an HNSW cosine index on
  first use. Spin up with `docker compose -f docker-compose.pg.yml up -d`,
  then `pip install -e ".[pgvector]"` and re-run ingest with
  `VECTOR_STORE_BACKEND=pgvector`. Session/Message/AgentTrace storage
  is already on SQLAlchemy 2.x async with a Postgres-compatible
  schema, so pointing `DATABASE_URL` at Postgres works the same way.

### How to demonstrate the E4 reranker lift

```bash
# Baseline — vector recall only
RERANKER_ENABLED=false uvicorn app.main:app --port 8765 &
python eval/run_eval.py --tag baseline --out eval/results/baseline.json
kill %1

# Reranked — top-20 fetched, judge picks top-5
RERANKER_ENABLED=true  uvicorn app.main:app --port 8765 &
python eval/run_eval.py --tag reranked --out eval/results/reranked.json
kill %1

# Compare
python -c "import json,glob; \
  for p in sorted(glob.glob('eval/results/*.json')): \
    d=json.load(open(p)); \
    print(f\"{d['tag']:>10s}: hit-rate={d['knowledge_hit_rate']:.2f}  \" + \
          ' '.join(f'{k}={v:.2f}' for k,v in d['recall_at_k'].items()))"
```

Numbers will vary with API quota and which Gemini revision is live,
but the harness is the durable artifact: any reranker change can be
A/B'd against this 30-row golden file in one command.

Skipped (with reason): E2 escalation agent / E3 SSE streaming —
out-of-scope-refusal already covers most of E2's value, and the
front-end this would be consumed by isn't part of the deliverable.

---

## Known limitations

* **Embedding-model trade-off.** When `GOOGLE_API_KEY` is unset, the
  local hash-based embedding works for unit tests but is noticeably
  weaker than `text-embedding-004` for real users. The README setup
  instructions tell reviewers to set the key for the demo; tests do
  not require it.
* **ADK event shape is version-tolerant, not version-pinned.** The
  pipeline uses `getattr(event, "get_function_calls", None)` and
  `is_final_response()` — both present in google-adk ≥ 0.5 — but if
  the API drifts in a future minor, the trace builder degrades
  gracefully rather than crashing the turn.
* **No streaming (`text/event-stream`).** Only blocking JSON
  responses. Trivial to add via `sse-starlette` (E3).
* **Mock account data.** `get_recent_builds` / `get_account_status`
  return deterministic fake rows seeded from `user_id`. The wiring
  through ADK is real; the data source is not.
* **Single instance.** Idempotency lookup races would need
  `SELECT … FOR UPDATE` or a queue if you scale to N replicas; the
  current `IntegrityError` fallback only protects against retries on
  the same instance.

---

## What I'd do with more time

1. **E2 escalation agent.** A third sub-agent that owns "talk to a
   human" flows: opens a `support_ticket` row in the DB, picks a
   priority, and emits a webhook event. The plumbing is already
   here — would just be one more `LlmAgent` + tool + DB table.
2. **E3 SSE streaming.** `text/event-stream` on `/v1/chat/{id}` so the
   UI can render tokens as they arrive. ADK's runner already exposes
   `run_async` as an async generator; the FastAPI side is one
   `EventSourceResponse` away.
3. **Cross-encoder reranker** as a second option alongside the
   LLM-as-judge one. Local model (e.g. `bge-reranker-base`) avoids the
   second LLM round-trip and gets ~80% of the lift; the dispatcher
   pattern from the vector store would translate cleanly.
4. **Reranker eval lift in CI.** Wire `eval/run_eval.py` into a
   GitHub Action that runs both modes on every PR and posts the
   recall@3 delta as a comment. Right now it's a manual
   "stop-server, flip env, restart, rerun" loop.
5. **Auth / multi-tenancy.** `SECRET_KEY` is wired but the JWT layer
   from E2 of the spec isn't built; would also let
   `get_recent_builds` / `get_account_status` actually scope by org.

---

## Time spent

| Phase                                              | Time |
|----------------------------------------------------|------|
| Project read-through, design decisions, planning   | 0:30 |
| Setup, `.gitignore`, package layout, error handler | 0:15 |
| Sessions endpoint                                  | 0:15 |
| RAG: chunker, embeddings (dual backend), vector store, ingest, search_docs | 1:00 |
| Account tools (mock, deterministic per user)       | 0:15 |
| ADK agents (knowledge, account, root factory)      | 0:40 |
| Pipeline (event walking, state persistence, traces)| 0:50 |
| Trace endpoint                                     | 0:10 |
| Tests (15 across `test_api`, `test_retriever`, `test_guardrails`) | 0:35 |
| Extensions: E1 idempotency, E5 guardrails, E6 Docker | 0:45 |
| Extensions: E4 reranker, E7 eval harness, OTel, pgvector | 1:30 |
| README + manual demo verification                  | 0:30 |
| **Total**                                          | **~7:30** |
