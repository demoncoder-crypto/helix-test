"""
Structured logging setup.

All log lines are JSON, augmented with request-scoped contextvars
(``session_id``, ``trace_id``, ``user_id`` when available). PII in any
log payload is redacted by the guardrails processor before serialization
(extension E5).
"""
from __future__ import annotations

import logging
import sys

import structlog

from app.agents.guardrails import redact_event_dict
from app.settings import settings


def _level_from_settings() -> int:
    return getattr(logging, settings.log_level.upper(), logging.INFO)


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_event_dict,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_level_from_settings()),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    )
