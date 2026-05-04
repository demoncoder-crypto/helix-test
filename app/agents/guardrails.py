"""
Guardrails — Extension E5.

Two responsibilities:

1. **Out-of-scope detection** — short-circuits before the LLM call when the
   user asks for something a Helix support concierge has no business
   answering (poetry, jokes, unrelated coding help, opinions, role-play).
   The pipeline records ``routed_to="refusal"`` for these turns.

2. **PII redaction** — masks emails, phone numbers, and known API-key
   shapes in any string before it hits the structured logger. Wired in
   as a structlog processor in ``app/obs/logging.py``.

Both checks are deterministic regex — no LLM call, no network, no PII
sent off-host.
"""
from __future__ import annotations

import re
from typing import Any

# Phrases / verbs that strongly indicate creative-writing or off-topic asks.
_OUT_OF_SCOPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwrite (?:me )?(?:a )?(?:poem|haiku|sonnet|song|story|joke)\b", re.I),
    re.compile(r"\btell (?:me )?a (?:joke|story|riddle)\b", re.I),
    re.compile(r"\bplay (?:a )?(?:game|role[- ]play)\b", re.I),
    re.compile(r"\bpretend (?:you are|to be)\b", re.I),
    re.compile(r"\bwhat (?:do you|is your) (?:opinion|favorite|favourite)\b", re.I),
    re.compile(r"\b(?:translate|summarize) (?:this|the following) (?:into|to)\b", re.I),
    re.compile(r"\bwrite (?:python|javascript|java|c\+\+|rust|go) code (?:for|to)\b", re.I),
)

_REFUSAL_TEMPLATE = (
    "I'm the Helix Support Concierge — I can only help with Helix product "
    "questions, your account, and your builds. I can't help with that. "
    "If you have a Helix-related question, ask away."
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Phone matcher requires either a leading ``+`` or at least one separator
# (space, dash, or parenthesis). This avoids eating ISO 8601 timestamps,
# UUIDs, and chunk IDs while still catching real phone numbers like
# "+1 415-555-0123" or "(415) 555-0123".
_PHONE_RE = re.compile(
    r"(?<![\w])(?:"
    r"\+\d[\d \-().]{7,}\d"
    r"|"
    r"\d{1,4}[ \-()][\d \-().]{6,}\d"
    r")(?![\w])"
)
_API_KEY_RE = re.compile(
    r"\b(?:sk_live_|sk_test_|hxk_|AIza|ghp_|gho_|github_pat_)[A-Za-z0-9_\-]{8,}\b"
)


def is_out_of_scope(message: str) -> bool:
    """Return True if the message matches a known off-topic pattern."""
    if not message or not message.strip():
        return False
    return any(p.search(message) for p in _OUT_OF_SCOPE_PATTERNS)


def refusal_message(_message: str) -> str:
    """Polite refusal text used when ``is_out_of_scope`` fires."""
    return _REFUSAL_TEMPLATE


def redact_pii(text: str) -> str:
    """Replace emails, phone numbers and API-key-shaped strings with ``[REDACTED]``."""
    if not isinstance(text, str) or not text:
        return text
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _API_KEY_RE.sub("[REDACTED_KEY]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def redact_event_dict(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor: redact PII recursively in every string value."""

    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            return redact_pii(value)
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_walk(v) for v in value)
        return value

    return {k: _walk(v) for k, v in event_dict.items()}
