"""
LLM-as-judge reranker — Extension E4.

The first-stage retriever (Chroma + ``gemini-embedding-001``) is good but
order-imperfect: a chunk that happens to share many query tokens often
beats a more topical chunk on raw cosine similarity. We fix that by
fetching ``top_n`` from the vector store and asking a small LLM to score
*each* chunk's relevance to the query in [0, 1], then re-ordering by
that score and keeping ``top_k``.

Design notes
------------

* **Single LLM call** — we send all candidates in one prompt and ask
  for a JSON array of ``{chunk_id, score}``. That's one round-trip,
  not N. Cheap.
* **Robust to provider failure** — if the rerank call fails for any
  reason (rate-limit, parse failure, missing key), we **fall back to
  the original first-stage order**. The retrieval pipeline keeps
  working — the reranker is purely additive.
* **Toggleable** — controlled by ``settings.reranker_enabled``. Tests
  run with it off so they remain hermetic.
* **Truncated chunks** — we truncate each chunk's text to a few hundred
  chars before sending it to the judge. Reranker doesn't need full
  text to decide ordering, and small prompts = lower latency + cost.

The judge's score is *only* used for ordering inside ``search_docs``;
we keep the original retrieval cosine score on the returned ``DocChunk``
so traces remain meaningful.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import structlog

from app.api.api_key import active_google_api_key
from app.settings import settings

if TYPE_CHECKING:
    from app.agents.tools.search_docs import DocChunk

log = structlog.get_logger()

_MAX_CHUNK_CHARS = 600
_JSON_RE = re.compile(r"\[.*\]", re.DOTALL)


def _build_prompt(query: str, chunks: list[DocChunk]) -> str:
    """Build the rerank prompt. One LLM call scores N chunks."""
    parts = [
        "You are a retrieval reranker for a CI/CD product's support docs.",
        "Score how directly each chunk answers the user's question on a 0.0-1.0 scale:",
        "  - 1.0 = chunk directly answers the question",
        "  - 0.5 = chunk is on-topic but does not answer it",
        "  - 0.0 = chunk is unrelated",
        "",
        f"USER QUESTION:\n{query}",
        "",
        "CANDIDATE CHUNKS:",
    ]
    for c in chunks:
        body = (c.content or "").strip().replace("\n", " ")
        if len(body) > _MAX_CHUNK_CHARS:
            body = body[:_MAX_CHUNK_CHARS] + "..."
        source = c.metadata.get("source", "unknown") if c.metadata else "unknown"
        parts.append(f"[{c.chunk_id}] (source={source}) {body}")
    parts.extend(
        [
            "",
            "Respond with ONLY a JSON array, no prose, like:",
            '[{"chunk_id":"chunk_xxxx","score":0.92}, ...]',
            "Include every chunk_id from the candidates, exactly once.",
        ]
    )
    return "\n".join(parts)


def _parse_scores(text: str, valid_ids: set[str]) -> dict[str, float]:
    """Best-effort JSON extraction. Returns ``{}`` on any failure."""
    if not text:
        return {}
    match = _JSON_RE.search(text)
    payload = match.group(0) if match else text
    try:
        arr = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if not isinstance(arr, list):
        return {}
    out: dict[str, float] = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("chunk_id", "")).strip()
        if cid not in valid_ids:
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        out[cid] = max(0.0, min(1.0, score))
    return out


def _call_judge_sync(prompt: str) -> str:
    """Sync Gemini call; caller wraps with ``asyncio.to_thread``.

    Uses the same ``google.genai`` client the embeddings module uses, so
    we reuse its API key plumbing and don't introduce a second SDK.
    """
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=active_google_api_key())
    resp = client.models.generate_content(
        model=settings.reranker_model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=1024,
        ),
    )
    return getattr(resp, "text", "") or ""


async def rerank_chunks(
    query: str, candidates: list[DocChunk], top_k: int
) -> list[DocChunk]:
    """Rerank ``candidates`` with an LLM judge; return top-k by judge score.

    On *any* failure (no API key, parse error, rate-limit, etc.) we log
    a warning and return the original first-stage ordering, sliced to
    ``top_k``. This makes the reranker purely additive — turning it on
    can only ever help retrieval, never break it.
    """
    if not candidates:
        return []
    if not active_google_api_key().strip():
        log.warning("reranker_skipped_no_api_key")
        return candidates[:top_k]

    import asyncio

    prompt = _build_prompt(query, candidates)
    valid_ids = {c.chunk_id for c in candidates}
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_call_judge_sync, prompt),
            timeout=max(5, settings.llm_timeout_seconds // 3),
        )
    except (TimeoutError, Exception) as exc:  # noqa: BLE001 - reranker must never break retrieval
        log.warning(
            "reranker_call_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:160],
        )
        return candidates[:top_k]

    scores = _parse_scores(raw, valid_ids)
    if not scores:
        log.warning("reranker_parse_empty", raw_len=len(raw))
        return candidates[:top_k]

    def sort_key(c: DocChunk) -> tuple[float, float]:
        return (scores.get(c.chunk_id, 0.0), c.score)

    ordered = sorted(candidates, key=sort_key, reverse=True)
    log.info(
        "reranker_applied",
        candidates=len(candidates),
        scored=len(scores),
        top_k=top_k,
    )
    return ordered[:top_k]
