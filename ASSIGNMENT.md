# AI Engineer Take-Home Assignment: Stateful RAG Orchestration Pipeline (SROP)

**Role:** AI Engineer (Backend focus)
**Level:** 1–3 years experience
**Estimated time:** 3–4 hours
**Stack:** Python 3.11+, FastAPI, Google ADK (`google-adk`), SQLite, vector store of your choice

**Reference docs in `docs/`:**
- `rag-guide.md` — RAG concepts, chunking, embeddings, vector store code examples
- `google-adk-guide.md` — ADK agents, tools, `AgentTool` pattern, session persistence
- `fastapi-async-guide.md` — async patterns, SQLAlchemy, Pydantic v2
- `*.md` files (10 files) — Helix product documentation (the RAG corpus)

You should not need external resources to complete the core requirements.

---

## 1. The Scenario

**Helix** is a B2B SaaS dev-tools platform. The support team wants an **AI Support Concierge** that handles two workflows in a single ongoing conversation:

1. **Knowledge questions** — "How do I rotate a deploy key?" → answer from product docs via RAG.
2. **Account lookups** — "Show me my last 3 failed builds" → query internal tools.

The agent must remember context across turns: the user's plan tier, which sub-agent last ran, and recent results. That memory must survive a process restart.

---

## 2. What You Are Building

```
POST /v1/chat/{session_id}
         │
         ▼
┌─────────────────────────┐
│  SROP Pipeline          │
│  1. Load session state  │
│  2. Run ADK orchestrator│
│  3. Save updated state  │
│  4. Write trace to DB   │
└────────────┬────────────┘
             │ routes via ADK AgentTool
       ┌─────┴──────┐
       ▼            ▼
 KnowledgeAgent  AccountAgent
 (RAG + search)  (DB tools)
       │
  Vector store    App DB
  (doc chunks)  (sessions, traces)
```

---

## 3. Core Requirements (graded out of 70 points)

### 3.1 REST API — 3 endpoints (8 pts)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/sessions` | Create a session. Body: `{user_id, plan_tier}`. Returns `{session_id}`. |
| `POST` | `/v1/chat/{session_id}` | Send a message. Returns `{reply, routed_to, trace_id}`. |
| `GET`  | `/v1/traces/{trace_id}` | Return the structured trace for one turn. |

Requirements:
- All handlers `async def`. No blocking I/O on the event loop.
- Pydantic v2 request/response models.
- `SESSION_NOT_FOUND` (404) when session doesn't exist.
- `UPSTREAM_TIMEOUT` (504) when LLM doesn't respond in time.
- `/healthz` (bonus — takes 2 minutes, adds 2 points to quality score).

### 3.2 Persistence (8 pts)

- SQLite via SQLAlchemy 2.x async. No sync calls inside async handlers.
- `create_all` on startup is fine — no migrations required.
- Tables you need: `sessions`, `messages`, `agent_traces`.
- A vector store for RAG (Chroma recommended in `rag-guide.md`, others fine).

### 3.3 Google ADK Agent (15 pts)

Build with the `google-adk` package:

**Root orchestrator** — routes to sub-agents using **ADK's `AgentTool` pattern**. This is the core requirement. Do not route by string-parsing LLM output.

**KnowledgeAgent** — calls `search_docs(query: str, k: int)` → hits your vector store → answer cites chunk IDs.

**AccountAgent** — exposes `get_recent_builds(user_id, limit)` and `get_account_status(user_id)`. Mock data is fine; the wiring is evaluated.

See `docs/google-adk-guide.md` for the `AgentTool` pattern and how to extract routing decisions from ADK events.

```python
# The pattern you must use (see guide for full detail)
from google.adk.tools.agent_tool import AgentTool

root_agent = LlmAgent(
    name="srop_root",
    model="gemini-2.0-flash",
    instruction="Route to the correct specialist tool...",
    tools=[
        AgentTool(agent=knowledge_agent),
        AgentTool(agent=account_agent),
    ],
)
```

### 3.4 Stateful Session Management (15 pts)

**This is the most weighted section.**

After turn 1, turns 2, 3, ... must know:
- `user_id` and `plan_tier` — do not re-ask
- Which sub-agent last ran — so follow-ups route correctly
- Turn count

**Hard requirement: state must survive a process restart.** Kill and restart `uvicorn`. The next message must work with the prior context. In-memory state fails this requirement. Demonstrate in your Loom.

Three valid implementation patterns are in `docs/google-adk-guide.md`. Pick one and document your choice.

### 3.5 RAG (12 pts)

1. **`ingest.py` CLI:** `python -m app.rag.ingest --path docs/` — chunks + embeds + upserts to vector store.
   - Chunk strategy: your choice. Justify in README (1–2 sentences).
   - Stable chunk IDs (deterministic — re-ingest must not duplicate).
   - Extract frontmatter metadata (`product_area`, `title`).

2. **`search_docs` tool:** top-k similarity search, returns chunk IDs + scores (required for citations and traces).

3. **Citations:** KnowledgeAgent answers must reference chunk IDs (e.g. "According to [chunk_abc123]..."). The system prompt must enforce this.

See `docs/rag-guide.md` for chunking code, embedding API examples, and Chroma usage.

### 3.6 Trace (6 pts)

One `agent_traces` row per turn containing:
- `trace_id`, `session_id`
- `routed_to` — which sub-agent handled this turn
- `tool_calls` — list of `{tool_name, args, result}` (JSON)
- `retrieved_chunk_ids` — list of chunk IDs returned by `search_docs`
- `latency_ms` — total turn latency

`GET /v1/traces/{trace_id}` returns this as JSON. A reviewer should be able to debug a misbehaving agent from this endpoint alone.

### 3.7 Async + Errors (6 pts)

- Timeout every LLM call (`asyncio.wait_for`). Raise `UpstreamTimeoutError` on timeout.
- No bare `except:`. No swallowed exceptions. Every caught exception is logged or re-raised.
- Type hints on all functions.

### 3.8 Tests (required — 2 tests) (0 pts deducted if missing → but graded under "tests" rubric line)

1. **Integration test** — POST two messages to the same session with LLM mocked at the ADK boundary. Assert turn 1 routes to the correct sub-agent. Assert turn 2 has access to context set in turn 1.

2. **Unit test** — `search_docs("rotate deploy key", k=3)` returns results with non-empty chunk IDs and scores in [0, 1].

`pytest -q` must pass from a clean clone.

---

## 4. Extensions (graded out of 30 points)

| Ext | Points | Description |
|-----|--------|-------------|
| E1 | 6 | **Idempotency** — `Idempotency-Key` header. Replay returns original response; pipeline runs once. |
| E2 | 5 | **Escalation agent** — third sub-agent: `create_ticket(user_id, summary, priority)` → writes to `tickets` table → returns ticket ID. Ticket ID stored in session state and available in follow-ups. |
| E3 | 5 | **Streaming SSE** — `POST /chat/{id}` supports `Accept: text/event-stream`. |
| E4 | 4 | **Reranking** — LLM-as-judge reranker on top-k retrieval. Show before/after on 5 queries. |
| E5 | 4 | **Guardrails** — refusal on out-of-scope queries (e.g. "write me a poem"), PII redaction in logs. One test asserting the refusal works. |
| E6 | 3 | **Docker** — `Dockerfile` + `docker-compose.yml`. `docker compose up` → smoke test passes. |
| E7 | 3 | **Eval harness** — run `python eval/run_eval.py` against your server. Report routing accuracy in README. |

---

## 5. Evaluation Rubric

| Area | Points | What "great" looks like |
|------|--------|------------------------|
| API design | 8 | 3 endpoints work, error responses have codes, Pydantic v2 models tight |
| Persistence | 8 | Fully async SQLAlchemy, vector store integrated |
| ADK agent | 15 | Sub-agents wired as `AgentTool`, routing via LLM tool selection — not string parsing |
| State management | 15 | Persists across restart, schema explicit, follow-ups use prior context |
| RAG | 12 | Chunking justified, stable IDs, chunk scores returned, citations in answers |
| Async + errors | 6 | LLM timeout, no bare `except:`, type hints |
| Trace quality | 6 | `GET /traces/{id}` is genuinely useful for debugging |
| Tests | 4 | Pytest passes, integration test mocks at ADK boundary |
| Code quality | 4 | Type hints, ruff clean, module boundaries sensible |
| README | 4 | Setup < 5 min, honest tradeoffs, time breakdown |
| Extensions | 30 | Per Section 4 |

**Total: 70 core + 30 extensions = 100. Hire bar: 65.**

**Hard penalties:**
- Does not run from clean clone: **−15**
- State lost on process restart: **−10**
- Routing via string parsing instead of `AgentTool`: **−8**
- Bare `except:` or swallowed errors: **−4 each, max −12**
- Sync I/O inside async handlers: **−4 each, max −8**
- Real API keys committed: **automatic reject**

---

## 6. Suggested Time Split

| Task | Time |
|------|------|
| Env setup + DB schema + FastAPI boilerplate | 30 min |
| `ingest.py` — chunk + embed + upsert | 30 min |
| `search_docs` tool + vector store wiring | 20 min |
| ADK agents — orchestrator + 2 sub-agents | 40 min |
| `pipeline.py` — state in/out of ADK, trace write | 40 min |
| Routes — session create + chat + trace | 20 min |
| State persistence (the hard part) | 20 min |
| Tests + README | 20 min |
| **Total** | **~3h 20min** |

---

## 7. Submission

- GitHub repo (public, or invite reviewer)
- `README.md` with: setup steps (<5 min), architecture diagram, design decision for state persistence (which pattern + why), chunking strategy justification, known limitations, time spent
- Loom (≤4 min): demo multi-turn conversation crossing both sub-agents, restart `uvicorn` mid-demo to prove state survives
- `pytest -q` passes from clean clone
- No secrets committed

**Questions:** `assignment@helix.example`
