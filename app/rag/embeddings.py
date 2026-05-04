"""
Embedding wrapper.

Two backends, one interface:

* **`google`** (default) — `gemini-embedding-001` via the modern
  `google.genai` SDK (the same SDK google-adk depends on, no deprecated
  `google.generativeai` in our import path). Uses
  ``task_type="RETRIEVAL_DOCUMENT"`` at ingest and ``"RETRIEVAL_QUERY"``
  at query time, as recommended by the RAG guide. We pin
  ``output_dimensionality=768`` so Chroma vectors stay compact and so
  the local fallback can produce dimensionally-compatible vectors.
* **`local`** — deterministic feature-hashing embedding. No external deps,
  no API key. Used automatically when `GOOGLE_API_KEY` is empty (so
  `pytest -q` works on a clean clone with zero secrets) or when the
  caller passes ``backend="local"``.

The local backend produces poor-quality recall in absolute terms but is
**stable and self-consistent** — the same text always maps to the same
vector, and similar n-gram bags map to nearby vectors. That is enough for
the unit test "search returned chunks have non-empty IDs and scores in [0,1]".

Both backends produce L2-normalized 768-dim vectors, so cosine distance
== ``1 − dot product`` regardless of which one is in use.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Literal

from app.settings import settings

EmbeddingTaskType = Literal["retrieval_document", "retrieval_query"]
EMBEDDING_DIM = 768
_GOOGLE_MODEL = "gemini-embedding-001"

_TASK_TYPE_MAP = {
    "retrieval_document": "RETRIEVAL_DOCUMENT",
    "retrieval_query": "RETRIEVAL_QUERY",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _backend() -> Literal["google", "local"]:
    """Pick the embedding backend based on the configured API key.

    Both ingest and query must use the same backend — vectors from
    different backends live in different mathematical spaces and
    cosine similarity becomes meaningless. Falls back to a deterministic
    local hash embedding when no key is configured so tests, CI, and
    offline development still work.
    """
    if settings.google_api_key.strip():
        return "google"
    return "local"


def _hash_token(token: str, salt: str) -> int:
    raw = f"{salt}:{token}".encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def _local_embed_one(text: str) -> list[float]:
    """Deterministic feature-hashing embedding (token uni- and bi-grams)."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    vec = [0.0] * EMBEDDING_DIM
    if not tokens:
        vec[0] = 1.0
        return vec
    for token in tokens:
        idx = _hash_token(token, "uni") % EMBEDDING_DIM
        sign = 1.0 if (_hash_token(token, "sign") & 1) else -1.0
        vec[idx] += sign
    for a, b in zip(tokens, tokens[1:]):
        bigram = f"{a}_{b}"
        idx = _hash_token(bigram, "bi") % EMBEDDING_DIM
        sign = 1.0 if (_hash_token(bigram, "sign") & 1) else -1.0
        vec[idx] += sign * 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _google_embed_batch(
    texts: list[str], task_type: EmbeddingTaskType
) -> list[list[float]]:
    """Sync Google embedding call; caller is responsible for ``to_thread``.

    Uses the modern ``google.genai`` SDK (the same one google-adk depends
    on, so no extra deprecated package in our import graph).
    """
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=settings.google_api_key)
    api_task_type = _TASK_TYPE_MAP[task_type]
    out: list[list[float]] = []
    for text in texts:
        result = client.models.embed_content(
            model=_GOOGLE_MODEL,
            contents=[text],
            config=genai_types.EmbedContentConfig(
                task_type=api_task_type,
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        vec = list(result.embeddings[0].values)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


def embed_texts(
    texts: list[str],
    task_type: EmbeddingTaskType = "retrieval_document",
    backend: Literal["auto", "google", "local"] = "auto",
) -> list[list[float]]:
    """Embed a batch of texts. **Synchronous** — call via ``asyncio.to_thread``
    from async code paths.
    """
    if not texts:
        return []
    chosen = _backend() if backend == "auto" else backend
    if chosen == "google":
        return _google_embed_batch(texts, task_type)
    return [_local_embed_one(t) for t in texts]


def embed_query(
    query: str, backend: Literal["auto", "google", "local"] = "auto"
) -> list[float]:
    """Convenience: embed a single query string with retrieval_query task type."""
    return embed_texts([query], task_type="retrieval_query", backend=backend)[0]
