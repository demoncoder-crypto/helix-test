"""
BYOK (Bring Your Own Key) — request-scoped Google API key.

Reviewers should be able to paste their own ``GOOGLE_API_KEY`` in the
web UI rather than burning the server's shared free-tier quota. The
header carries the key on every chat turn; the server uses it for that
request only and never persists it.

Mechanics
---------

* ``set_request_api_key(key)`` writes the key into a ``ContextVar``
  early in each request, *before* the pipeline runs. Because asyncio
  contexts are task-scoped, two concurrent requests can each have a
  different active key without stepping on each other.
* ``active_google_api_key()`` is the read side. Embeddings + reranker
  call this instead of reading ``settings.google_api_key`` so they
  pick up the per-request key automatically.
* ``adk_env_lock`` serializes calls to the ADK runner specifically.
  ADK reads the key from ``os.environ["GOOGLE_API_KEY"]`` (process-
  global), so swapping the env var per request would race under
  concurrency. We hold a lock around the swap+run+restore section. The
  hot path becomes effectively single-threaded *for the LLM call*,
  which is fine for a demo and for hosted free-tier traffic.

Security notes
--------------
* The key is never written to the DB, the trace, or the response body.
* The structlog PII redactor already masks ``AIza...`` shaped tokens
  in any log line, so a stray ``log.info(..., api_key=...)`` won't
  leak it.
* TLS is the user's problem (deploy behind HTTPS — Render / Fly / HF
  Spaces all do this by default).
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.settings import settings

_REQUEST_API_KEY: ContextVar[str] = ContextVar("google_api_key", default="")

# Single global lock — guards the os.environ swap that ADK requires.
adk_env_lock = asyncio.Lock()


def set_request_api_key(key: str | None) -> None:
    """Bind the per-request key. Pass ``None`` or empty to clear."""
    _REQUEST_API_KEY.set((key or "").strip())


def active_google_api_key() -> str:
    """Return the per-request key if set, else the server's configured key."""
    req_key = _REQUEST_API_KEY.get()
    if req_key:
        return req_key
    return settings.google_api_key


def has_active_key() -> bool:
    return bool(active_google_api_key().strip())


@contextmanager
def adk_env_override(key: str) -> Iterator[None]:
    """Temporarily install ``key`` as ``GOOGLE_API_KEY`` in the process env.

    Restores the previous value on exit even if the ADK call raises.
    Caller MUST hold ``adk_env_lock`` for the duration — this contextmanager
    just handles the swap/restore cleanly, not the concurrency.
    """
    if not key:
        yield
        return
    sentinel = object()
    previous = os.environ.get("GOOGLE_API_KEY", sentinel)
    os.environ["GOOGLE_API_KEY"] = key
    try:
        yield
    finally:
        if previous is sentinel:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = previous  # type: ignore[arg-type]
