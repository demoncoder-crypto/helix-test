"""
Integration tests — exercise the full SROP pipeline end-to-end.

The LLM is mocked at the ADK boundary via the ``mock_adk`` fixture (which
replaces ``app.srop.pipeline.run`` with a deterministic stand-in that
still writes the same DB rows the real pipeline would).
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post(
        "/v1/sessions", json={"user_id": "u_test_001", "plan_tier": "free"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user_id"] == "u_test_001"
    assert "session_id" in body and len(body["session_id"]) > 0


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client):
    resp = await client.post(
        "/v1/chat/nonexistent-session-id", json={"content": "hello"}
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["title"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_adk):
    """Core integration test: turn 1 routes correctly, turn 2 has prior state."""
    sess = await client.post(
        "/v1/sessions", json={"user_id": "u_test_002", "plan_tier": "pro"}
    )
    assert sess.status_code == 200
    session_id = sess.json()["session_id"]

    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
    )
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["routed_to"] == "knowledge"
    assert "[chunk_" in body1["reply"], body1["reply"]
    trace_id = body1["trace_id"]

    trace = await client.get(f"/v1/traces/{trace_id}")
    assert trace.status_code == 200, trace.text
    tbody = trace.json()
    assert tbody["routed_to"] == "knowledge"
    assert len(tbody["retrieved_chunk_ids"]) > 0
    assert tbody["latency_ms"] >= 0
    assert any(tc["tool_name"] == "search_docs_tool" for tc in tbody["tool_calls"])

    # Turn 2 — state must persist; mock answers from `state.plan_tier`.
    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "What is my plan tier?"},
    )
    assert r2.status_code == 200, r2.text
    assert "pro" in r2.json()["reply"].lower()


@pytest.mark.asyncio
async def test_trace_not_found_returns_404(client):
    resp = await client.get("/v1/traces/no-such-trace")
    assert resp.status_code == 404
    assert resp.json()["title"] == "TRACE_NOT_FOUND"


@pytest.mark.asyncio
async def test_account_query_routes_correctly(client, mock_adk):
    sess = await client.post(
        "/v1/sessions", json={"user_id": "u_test_003", "plan_tier": "enterprise"}
    )
    session_id = sess.json()["session_id"]

    resp = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "Show me my last 3 failed builds"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["routed_to"] == "account"


@pytest.mark.asyncio
async def test_idempotency_returns_cached_reply(client, mock_adk):
    """Same Idempotency-Key returns the same trace_id and the pipeline runs once."""
    sess = await client.post(
        "/v1/sessions", json={"user_id": "u_test_004", "plan_tier": "pro"}
    )
    session_id = sess.json()["session_id"]

    headers = {"Idempotency-Key": "abc-123"}
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
        headers=headers,
    )
    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
        headers=headers,
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["trace_id"] == r2.json()["trace_id"]
    assert r1.json()["reply"] == r2.json()["reply"]


@pytest.mark.asyncio
async def test_guardrail_refuses_out_of_scope(client):
    """Out-of-scope messages are refused without invoking the LLM."""
    sess = await client.post(
        "/v1/sessions", json={"user_id": "u_test_005", "plan_tier": "free"}
    )
    session_id = sess.json()["session_id"]

    resp = await client.post(
        f"/v1/chat/{session_id}", json={"content": "Write me a poem about CI/CD"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["routed_to"] == "refusal"
    assert "helix" in body["reply"].lower()
