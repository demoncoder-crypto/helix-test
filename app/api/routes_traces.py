"""
GET /v1/traces/{trace_id} — return the structured trace for one pipeline turn.

Reviewers (and on-call engineers) should be able to debug a misbehaving
agent from this endpoint alone — that's why the response carries every
tool call (name, args, result), the chunk IDs that were retrieved for
the answer, the sub-agent that handled the turn, and end-to-end latency.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import TraceNotFoundError
from app.db.models import AgentTrace
from app.db.session import get_db

router = APIRouter(tags=["traces"])


class ToolCallRecord(BaseModel):
    tool_name: str
    args: dict[str, Any] = {}
    result: dict[str, Any] | list[Any] | str | int | float | bool | None = None


class TraceResponse(BaseModel):
    trace_id: str
    session_id: str
    routed_to: str
    tool_calls: list[ToolCallRecord]
    retrieved_chunk_ids: list[str]
    latency_ms: int


@router.get("/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(
    trace_id: str,
    db: AsyncSession = Depends(get_db),
) -> TraceResponse:
    """Return the trace for a single turn or 404 if unknown."""
    result = await db.execute(
        select(AgentTrace).where(AgentTrace.trace_id == trace_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise TraceNotFoundError(f"Trace {trace_id} does not exist")

    return TraceResponse(
        trace_id=row.trace_id,
        session_id=row.session_id,
        routed_to=row.routed_to,
        tool_calls=[ToolCallRecord(**tc) for tc in (row.tool_calls or [])],
        retrieved_chunk_ids=list(row.retrieved_chunk_ids or []),
        latency_ms=row.latency_ms,
    )
