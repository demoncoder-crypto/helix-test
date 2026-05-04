"""Unit tests for the guardrails module (Extension E5)."""
from __future__ import annotations

from app.agents.guardrails import is_out_of_scope, redact_event_dict, redact_pii


def test_out_of_scope_detects_creative_writing() -> None:
    assert is_out_of_scope("Write me a poem about debugging")
    assert is_out_of_scope("tell me a joke")
    assert is_out_of_scope("pretend you are a pirate")


def test_in_scope_messages_pass_through() -> None:
    assert not is_out_of_scope("How do I rotate a deploy key?")
    assert not is_out_of_scope("Show me my last 5 builds")
    assert not is_out_of_scope("What plan am I on?")
    assert not is_out_of_scope("")


def test_redact_pii_masks_emails_and_keys() -> None:
    raw = "contact ada@example.com or use sk_live_abcdefghij1234"
    out = redact_pii(raw)
    assert "ada@example.com" not in out
    assert "sk_live_abcdefghij1234" not in out
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_KEY]" in out


def test_redact_event_dict_walks_nested_structures() -> None:
    event = {
        "event": "pipeline_started",
        "user": {"email": "ada@example.com", "id": "u_1"},
        "tags": ["call sk_live_abcdefghij1234 now"],
    }
    out = redact_event_dict(None, "info", event)
    assert "[REDACTED_EMAIL]" in out["user"]["email"]
    assert "[REDACTED_KEY]" in out["tags"][0]
    assert out["event"] == "pipeline_started"
