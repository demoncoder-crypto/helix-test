"""
POST /v1/chat/{session_id} — send a user message, get an assistant reply.

Supports the ``Idempotency-Key`` header (Extension E1): if the same
``(session_id, idempotency_key)`` was previously processed, we return the
cached assistant reply without re-running the pipeline. The pipeline
itself enforces the constraint via a UNIQUE index on
``messages(session_id, idempotency_key)``.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.api_key import set_request_api_key
from app.db.models import Message
from app.db.session import get_db
from app.srop import pipeline

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    reply: str
    routed_to: str
    trace_id: str


async def _lookup_idempotent_reply(
    session_id: str, idempotency_key: str | None, db: AsyncSession
) -> ChatResponse | None:
    """Return a cached assistant reply if this idempotency key was already
    processed for this session, else None."""
    if not idempotency_key:
        return None
    result = await db.execute(
        select(Message).where(
            Message.session_id == session_id,
            Message.idempotency_key == idempotency_key,
            Message.role == "assistant",
        )
    )
    cached = result.scalar_one_or_none()
    if cached is None:
        return None
    return ChatResponse(
        reply=cached.content,
        routed_to=cached.routed_to or "smalltalk",
        trace_id=cached.trace_id or "",
    )


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    google_api_key: str | None = Header(default=None, alias="X-Google-Api-Key"),
) -> ChatResponse:
    """Run one turn of the SROP pipeline.

    Headers:
    - ``Idempotency-Key`` (optional) — replay-safe key per (session, key).
    - ``X-Google-Api-Key`` (optional) — BYOK. When present, this key is
      used for embeddings + reranker + the ADK LLM call **for this
      request only**. Falls back to the server's configured key if the
      header is absent or empty.

    Error cases:
    - Session not found → 404 ``SESSION_NOT_FOUND``
    - LLM timeout → 504 ``UPSTREAM_TIMEOUT``
    - LLM rate-limited → 429 ``RATE_LIMITED``
    """
    set_request_api_key(google_api_key)

    cached = await _lookup_idempotent_reply(session_id, idempotency_key, db)
    if cached is not None:
        return cached

    try:
        result = await pipeline.run(
            session_id=session_id,
            user_message=body.content,
            db=db,
            idempotency_key=idempotency_key,
        )
    except IntegrityError:
        await db.rollback()
        cached = await _lookup_idempotent_reply(session_id, idempotency_key, db)
        if cached is not None:
            return cached
        raise

    return ChatResponse(
        reply=result.content, routed_to=result.routed_to, trace_id=result.trace_id
    )
