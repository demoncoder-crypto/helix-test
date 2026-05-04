"""
SROP entrypoint — orchestrates one turn of the Helix Support Concierge.

Steps:
1. Load Session row + ``SessionState`` from SQLite.
2. Build the root ADK agent with state injected into the instruction
   (Pattern 3 from the ADK guide).
3. Run the agent under ``asyncio.wait_for`` (LLM timeout → ``UpstreamTimeoutError``).
4. Walk the ADK event stream to extract:
   - which sub-agent produced the final response (``routed_to``),
   - the ordered list of tool calls + results,
   - the chunk IDs returned by ``search_docs_tool`` (for citations).
5. Persist user + assistant ``Message`` rows.
6. Persist an ``AgentTrace`` row.
7. Update ``SessionState`` (turn_count, last_agent) and write back to DB.
8. Return a ``PipelineResult`` to the route.

Out-of-scope guardrail (Extension E5) short-circuits before the LLM call.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.agents.guardrails import is_out_of_scope, refusal_message
from app.agents.orchestrator import build_root_agent
from app.api.api_key import active_google_api_key, adk_env_lock, adk_env_override
from app.api.errors import (
    HelixError,
    RateLimitedError,
    SessionNotFoundError,
    UpstreamTimeoutError,
)
from app.db.models import AgentTrace, Message
from app.db.models import Session as SessionModel
from app.obs.tracing import get_tracer
from app.settings import settings
from app.srop.state import SessionState

log = structlog.get_logger()

_APP_NAME = "helix_srop"
_KNOWLEDGE_AGENT_NAME = "knowledge_agent"
_ACCOUNT_AGENT_NAME = "account_agent"
_KNOWLEDGE_TOOL_NAME = "search_docs_tool"
_CHUNK_ID_RE = re.compile(r"chunk_[A-Fa-f0-9]{16}")


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


@dataclass
class _TurnTrace:
    """Mutable accumulator filled in while we walk ADK events."""

    final_text: str = ""
    routed_to: str = "smalltalk"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)


def _author_to_route(author: str | None) -> str:
    if author == _KNOWLEDGE_AGENT_NAME:
        return "knowledge"
    if author == _ACCOUNT_AGENT_NAME:
        return "account"
    return "smalltalk"


def _route_from_tool_calls(tool_calls: list[dict[str, Any]]) -> str | None:
    """Decide ``routed_to`` from which AgentTool was invoked.

    With the ``AgentTool`` pattern the final event's ``author`` is always
    the root orchestrator, even when it called a specialist. The actual
    routing signal is which tool the LLM picked. We check in this order
    so a turn that calls *both* sub-agents resolves to "knowledge"
    (which is what we cite to the user).
    """
    seen = {tc.get("tool_name") for tc in tool_calls}
    if _KNOWLEDGE_AGENT_NAME in seen or _KNOWLEDGE_TOOL_NAME in seen:
        return "knowledge"
    if _ACCOUNT_AGENT_NAME in seen or any(
        n in seen for n in ("get_recent_builds_tool", "get_account_status_tool")
    ):
        return "account"
    return None


def _extract_chunk_ids(value: Any) -> list[str]:
    """Recursively walk a JSON-ish value pulling out ``chunk_<hex16>`` IDs."""
    found: list[str] = []
    if value is None:
        return found
    if isinstance(value, str):
        found.extend(_CHUNK_ID_RE.findall(value))
    elif isinstance(value, dict):
        for v in value.values():
            found.extend(_extract_chunk_ids(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            found.extend(_extract_chunk_ids(v))
    return found


def _coerce_jsonable(value: Any) -> Any:
    """Best-effort conversion of ADK objects to JSON-serializable shapes."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # noqa: BLE001 - logged + fallback
            log.warning("adk_event_model_dump_failed", type=type(value).__name__)
    if hasattr(value, "__dict__"):
        return {k: _coerce_jsonable(v) for k, v in value.__dict__.items() if not k.startswith("_")}
    return str(value)


def _extract_tool_call(event: Any) -> list[dict[str, Any]]:
    """Pull function-call objects off an event in a version-tolerant way."""
    out: list[dict[str, Any]] = []
    getter = getattr(event, "get_function_calls", None)
    if callable(getter):
        try:
            calls = getter() or []
        except Exception:  # noqa: BLE001 - logged + safe fallback
            log.warning("adk_get_function_calls_failed")
            calls = []
        for fc in calls:
            out.append(
                {
                    "tool_name": getattr(fc, "name", None) or "unknown",
                    "args": _coerce_jsonable(getattr(fc, "args", {}) or {}),
                    "result": None,
                }
            )
    return out


def _extract_tool_results(event: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    getter = getattr(event, "get_function_responses", None)
    if callable(getter):
        try:
            responses = getter() or []
        except Exception:  # noqa: BLE001 - logged + safe fallback
            log.warning("adk_get_function_responses_failed")
            responses = []
        for fr in responses:
            out.append(
                {
                    "tool_name": getattr(fr, "name", None) or "unknown",
                    "result": _coerce_jsonable(getattr(fr, "response", None)),
                }
            )
    return out


def _final_text_from_event(event: Any) -> str:
    content = getattr(event, "content", None)
    if not content:
        return ""
    parts = getattr(content, "parts", None) or []
    chunks: list[str] = []
    for p in parts:
        text = getattr(p, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks)


async def _consume_events(stream: Any, accum: _TurnTrace) -> None:
    """Iterate the ADK async event stream, populating ``accum``.

    The real google-adk events expose:
      - ``event.author`` (agent name)
      - ``event.is_final_response()``
      - ``event.get_function_calls()`` / ``get_function_responses()``
    We use ``getattr``/``hasattr`` so this still works if the API drifts.
    """
    pending_calls_by_name: dict[str, dict[str, Any]] = {}

    async for event in stream:
        for call in _extract_tool_call(event):
            accum.tool_calls.append(call)
            pending_calls_by_name[call["tool_name"]] = call

        for resp in _extract_tool_results(event):
            target = pending_calls_by_name.get(resp["tool_name"])
            if target is not None:
                target["result"] = resp["result"]
            else:
                accum.tool_calls.append(
                    {"tool_name": resp["tool_name"], "args": {}, "result": resp["result"]}
                )

            # Direct path: the inner search_docs_tool returns explicit chunk_ids.
            if resp["tool_name"] == _KNOWLEDGE_TOOL_NAME:
                payload = resp["result"]
                if isinstance(payload, dict):
                    chunk_ids = payload.get("chunk_ids") or []
                    if isinstance(chunk_ids, list):
                        for cid in chunk_ids:
                            if isinstance(cid, str) and cid not in accum.retrieved_chunk_ids:
                                accum.retrieved_chunk_ids.append(cid)

            # Indirect path: when the root agent calls knowledge_agent via
            # AgentTool, the inner search_docs_tool is internalized — but
            # chunk IDs end up cited as ``[chunk_xxx]`` in the AgentTool
            # response text. Pull them out so the trace is still useful.
            if resp["tool_name"] == _KNOWLEDGE_AGENT_NAME:
                for cid in _extract_chunk_ids(resp["result"]):
                    if cid not in accum.retrieved_chunk_ids:
                        accum.retrieved_chunk_ids.append(cid)

        is_final = False
        finalizer = getattr(event, "is_final_response", None)
        if callable(finalizer):
            try:
                is_final = bool(finalizer())
            except Exception:  # noqa: BLE001 - logged
                log.warning("adk_is_final_response_failed")
                is_final = False

        if is_final:
            text = _final_text_from_event(event)
            if text:
                accum.final_text = text
            author = getattr(event, "author", None)
            accum.routed_to = _author_to_route(author)


async def _load_session(session_id: str, db: AsyncSession) -> SessionModel:
    result = await db.execute(
        select(SessionModel).where(SessionModel.session_id == session_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise SessionNotFoundError(f"Session {session_id} does not exist")
    return row


async def _persist_turn(
    *,
    db: AsyncSession,
    session_row: SessionModel,
    state: SessionState,
    user_message: str,
    assistant_text: str,
    trace: _TurnTrace,
    trace_id: str,
    latency_ms: int,
    idempotency_key: str | None,
) -> None:
    """Write Messages + AgentTrace + updated SessionState in one commit."""
    user_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_row.session_id,
        role="user",
        content=user_message,
        trace_id=trace_id,
    )
    asst_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_row.session_id,
        role="assistant",
        content=assistant_text,
        trace_id=trace_id,
        routed_to=trace.routed_to,
        idempotency_key=idempotency_key,
    )
    trace_row = AgentTrace(
        trace_id=trace_id,
        session_id=session_row.session_id,
        routed_to=trace.routed_to,
        tool_calls=trace.tool_calls,
        retrieved_chunk_ids=trace.retrieved_chunk_ids,
        latency_ms=latency_ms,
    )

    state.turn_count += 1
    if trace.routed_to in ("knowledge", "account", "smalltalk"):
        state.last_agent = trace.routed_to  # type: ignore[assignment]

    session_row.state = state.to_db_dict()
    flag_modified(session_row, "state")

    db.add(user_msg)
    db.add(asst_msg)
    db.add(trace_row)
    await db.commit()


async def _run_adk_turn(state: SessionState, user_message: str, accum: _TurnTrace) -> None:
    """Run one turn through google-adk's InMemoryRunner.

    BYOK contract: we hold ``adk_env_lock`` across the swap-in of the
    per-request ``GOOGLE_API_KEY`` and the entire ADK call. ADK reads the
    key from process env, so without the lock two concurrent requests
    with different keys would race. Embeddings + reranker get the key
    via context-var and don't need this lock.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    tracer = get_tracer()
    with tracer.start_as_current_span("adk.run") as span:
        span.set_attribute("adk.app_name", _APP_NAME)
        span.set_attribute("adk.user_id", state.user_id)
        span.set_attribute("adk.message_len", len(user_message))

        active_key = active_google_api_key()
        async with adk_env_lock:
            with adk_env_override(active_key):
                agent = build_root_agent(state)
                runner = InMemoryRunner(agent=agent, app_name=_APP_NAME)

                adk_session = await runner.session_service.create_session(
                    app_name=_APP_NAME, user_id=state.user_id
                )
                new_message = genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text=user_message)],
                )
                stream = runner.run_async(
                    user_id=state.user_id,
                    session_id=adk_session.id,
                    new_message=new_message,
                )
                await _consume_events(stream, accum)

        span.set_attribute("adk.tool_calls", len(accum.tool_calls))
        span.set_attribute("adk.routed_to", accum.routed_to)


async def run(
    session_id: str,
    user_message: str,
    db: AsyncSession,
    *,
    idempotency_key: str | None = None,
) -> PipelineResult:
    """Run one turn of the SROP pipeline."""
    trace_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)
    log.info("pipeline_started", message_len=len(user_message))

    tracer = get_tracer()
    with tracer.start_as_current_span("pipeline.run") as span:
        span.set_attribute("session_id", session_id)
        span.set_attribute("trace_id", trace_id)
        span.set_attribute("idempotency_key", idempotency_key or "")

        session_row = await _load_session(session_id, db)
        state = SessionState.from_db_dict(session_row.state or {})
        span.set_attribute("user_id", state.user_id)
        span.set_attribute("plan_tier", state.plan_tier)

        accum = _TurnTrace()
        start = time.perf_counter()

        if is_out_of_scope(user_message):
            log.info("guardrail_refused")
            accum.final_text = refusal_message(user_message)
            accum.routed_to = "refusal"
        else:
            try:
                await asyncio.wait_for(
                    _run_adk_turn(state, user_message, accum),
                    timeout=settings.llm_timeout_seconds,
                )
            except TimeoutError as exc:
                log.warning("llm_timeout", timeout_s=settings.llm_timeout_seconds)
                raise UpstreamTimeoutError(
                    f"LLM did not respond within {settings.llm_timeout_seconds}s"
                ) from exc
            except HelixError:
                raise
            except Exception as exc:  # noqa: BLE001 - re-classified into a HelixError below
                err_name = type(exc).__name__
                err_text = str(exc)
                if "ResourceExhausted" in err_name or "429" in err_text:
                    log.warning("llm_rate_limited", error=err_text[:200])
                    raise RateLimitedError(
                        "LLM provider rate limit exceeded; please retry shortly"
                    ) from exc
                log.error("llm_call_failed", error_type=err_name, error=err_text[:200])
                raise UpstreamTimeoutError(f"LLM call failed: {err_name}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)

        # AgentTool routing fix: when the root agent delegates via
        # AgentTool, the final event author is the root agent, not the
        # specialist. The authoritative signal is *which* AgentTool was
        # called.
        if accum.routed_to in ("smalltalk",):
            from_tools = _route_from_tool_calls(accum.tool_calls)
            if from_tools:
                accum.routed_to = from_tools

        # Final fallback: pull chunk IDs from the assistant text itself
        # if neither the inner nor outer tool result surfaced them.
        if not accum.retrieved_chunk_ids and accum.final_text:
            for cid in _extract_chunk_ids(accum.final_text):
                if cid not in accum.retrieved_chunk_ids:
                    accum.retrieved_chunk_ids.append(cid)

        if not accum.final_text:
            accum.final_text = (
                "I wasn't able to produce a response. Please try rephrasing."
            )

        await _persist_turn(
            db=db,
            session_row=session_row,
            state=state,
            user_message=user_message,
            assistant_text=accum.final_text,
            trace=accum,
            trace_id=trace_id,
            latency_ms=latency_ms,
            idempotency_key=idempotency_key,
        )

        span.set_attribute("routed_to", accum.routed_to)
        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("tool_calls", len(accum.tool_calls))
        span.set_attribute("retrieved_chunks", len(accum.retrieved_chunk_ids))

        log.info(
            "pipeline_completed",
            routed_to=accum.routed_to,
            latency_ms=latency_ms,
            tool_calls=len(accum.tool_calls),
            chunks=len(accum.retrieved_chunk_ids),
        )

        return PipelineResult(
            content=accum.final_text,
            routed_to=accum.routed_to,
            trace_id=trace_id,
        )


__all__ = ["PipelineResult", "run"]
