"""
Test fixtures.

* ``client`` — async ``httpx.AsyncClient`` mounted on the FastAPI ASGI app
  with the DB dependency overridden to a per-test in-memory SQLite.
* ``mock_adk`` — patches ``app.srop.pipeline.run`` so tests never call a
  real LLM. The mock writes a Message + AgentTrace row exactly as the real
  pipeline would, so route handlers (`GET /v1/traces`) work end-to-end.
* ``seed_vector_store`` — populates a temp Chroma directory with the docs
  corpus using the **local** (deterministic, no-API-key) embedding backend.
  Used by ``tests/test_retriever.py``.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Force the local embedding backend BEFORE app modules import settings/embeddings.
os.environ.setdefault("GOOGLE_API_KEY", "")

from app.db.models import AgentTrace, Base, Message  # noqa: E402
from app.db.session import get_db  # noqa: E402
from app.main import app  # noqa: E402

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest_asyncio.fixture
async def db_sessionmaker(db_engine):
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def db(db_sessionmaker) -> AsyncSession:
    async with db_sessionmaker() as session:
        yield session


@pytest_asyncio.fixture
async def client(db_sessionmaker):
    """Async test client. Each FastAPI dependency invocation gets its own
    short-lived session (matching production ``get_db`` semantics)."""

    async def _override_get_db():
        async with db_sessionmaker() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch):
    """Patch the SROP pipeline so tests don't hit a real LLM.

    The mock inspects the user message, picks a fake routing decision,
    writes the same DB rows the real pipeline would, and returns the same
    ``PipelineResult`` shape. This keeps the integration test honest:
    the route handlers, persistence, and trace endpoint are all exercised
    end-to-end — only the LLM call is replaced.
    """
    from sqlalchemy import select
    from sqlalchemy.orm.attributes import flag_modified

    from app.db.models import Session as SessionModelLocal
    from app.srop import pipeline as pipeline_mod
    from app.srop.state import SessionState

    async def _fake_run(
        session_id: str,
        user_message: str,
        db: AsyncSession,
        *,
        idempotency_key: str | None = None,
    ) -> pipeline_mod.PipelineResult:
        result = await db.execute(
            select(SessionModelLocal).where(SessionModelLocal.session_id == session_id)
        )
        session_row = result.scalar_one_or_none()
        if session_row is None:
            from app.api.errors import SessionNotFoundError
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        state = SessionState.from_db_dict(session_row.state or {})
        msg_lower = user_message.lower()

        tool_calls: list[dict] = []
        retrieved_chunk_ids: list[str] = []

        plan_keywords = ("plan tier", "what plan", "my plan", "remind me what plan")
        if any(w in msg_lower for w in plan_keywords):
            routed_to = "smalltalk"
            reply = (
                f"You're on the {state.plan_tier} plan, {state.user_id}. "
                f"Anything else I can help with?"
            )
        elif any(w in msg_lower for w in ("rotate", "deploy key", "how do i", "how to", "what is")):
            routed_to = "knowledge"
            retrieved_chunk_ids = ["chunk_test_001", "chunk_test_002"]
            tool_calls = [
                {
                    "tool_name": "search_docs_tool",
                    "args": {"query": user_message, "k": 5},
                    "result": {"chunk_ids": retrieved_chunk_ids, "count": 2},
                }
            ]
            reply = (
                "To rotate a deploy key, navigate to Settings → Security "
                "[chunk_test_001]. After rotating, update CI secrets "
                "[chunk_test_002]."
            )
        elif any(w in msg_lower for w in ("build", "account", "usage", "storage")):
            routed_to = "account"
            tool_calls = [
                {
                    "tool_name": "get_recent_builds_tool",
                    "args": {"user_id": state.user_id, "limit": 3},
                    "result": {
                        "builds": [{"build_id": "bld_0001", "status": "failed"}],
                        "count": 1,
                    },
                }
            ]
            reply = (
                f"Your most recent build is bld_0001 (failed). "
                f"Plan tier: {state.plan_tier}."
            )
        else:
            routed_to = "smalltalk"
            reply = "Hi! How can I help with Helix today?"

        trace_id = str(uuid.uuid4())
        db.add(
            Message(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="user",
                content=user_message,
                trace_id=trace_id,
            )
        )
        db.add(
            Message(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content=reply,
                trace_id=trace_id,
                routed_to=routed_to,
                idempotency_key=idempotency_key,
            )
        )
        db.add(
            AgentTrace(
                trace_id=trace_id,
                session_id=session_id,
                routed_to=routed_to,
                tool_calls=tool_calls,
                retrieved_chunk_ids=retrieved_chunk_ids,
                latency_ms=42,
            )
        )

        state.turn_count += 1
        if routed_to in ("knowledge", "account", "smalltalk"):
            state.last_agent = routed_to
        session_row.state = state.to_db_dict()
        flag_modified(session_row, "state")
        await db.commit()

        return pipeline_mod.PipelineResult(
            content=reply, routed_to=routed_to, trace_id=trace_id
        )

    monkeypatch.setattr("app.srop.pipeline.run", _fake_run)
    monkeypatch.setattr("app.api.routes_chat.pipeline.run", _fake_run)
    return _fake_run


@pytest.fixture(scope="session")
def seed_vector_store(tmp_path_factory):
    """Ingest the docs corpus into a per-session temp Chroma directory.

    Uses the **local** embedding backend (deterministic, no API key) so
    this works on CI / clean clones with no secrets.
    """
    import asyncio

    from app.rag import vector_store as vs
    from app.rag.ingest import ingest_directory
    from app.settings import settings as app_settings

    persist_dir = tmp_path_factory.mktemp("chroma_test")
    original = app_settings.chroma_persist_dir
    original_key = app_settings.google_api_key
    app_settings.chroma_persist_dir = str(persist_dir)
    app_settings.google_api_key = ""  # force local embedding backend
    vs.reset_collection_cache()

    docs_path = Path(__file__).resolve().parent.parent / "docs"
    asyncio.run(ingest_directory(docs_path))

    yield persist_dir

    app_settings.chroma_persist_dir = original
    app_settings.google_api_key = original_key
    vs.reset_collection_cache()
