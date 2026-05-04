"""
OpenTelemetry wiring (Extension).

A *very* thin layer over ``opentelemetry`` that:

1. ``configure_tracing()`` — call once on FastAPI startup. Sets up a
   ``TracerProvider``, attaches the right span exporter (console for
   demos, OTLP for Jaeger / Tempo), and instruments FastAPI so every
   request becomes a parent span automatically.
2. ``get_tracer()`` — call from anywhere to add child spans (we use it
   in ``pipeline.run``, ``_run_adk_turn``, and ``search_docs``).
3. **Default = no-op**. When ``settings.otel_enabled = false`` we install
   ``opentelemetry``'s own no-op tracer, so ``tracer.start_as_current_span``
   is essentially free at runtime. Toggling OTel never costs perf when off.

This keeps the production hot-path agnostic of OTel: any code path can
``with get_tracer().start_as_current_span("foo"): ...`` without caring
whether tracing is wired up.
"""
from __future__ import annotations

import structlog

from app.settings import settings

log = structlog.get_logger()

_TRACER_NAME = "helix.srop"
_configured = False


def configure_tracing(app: object | None = None) -> None:
    """Set up the OpenTelemetry tracer provider. Idempotent."""
    global _configured
    if _configured:
        return

    if not settings.otel_enabled:
        _configured = True
        log.debug("otel_disabled")
        return

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": "0.1.0",
            "deployment.environment": settings.app_env,
        }
    )
    provider = TracerProvider(resource=resource)

    if settings.otel_console_exporter:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        log.info("otel_console_exporter_enabled")

    if settings.otel_exporter_otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            otlp = OTLPSpanExporter(
                endpoint=settings.otel_exporter_otlp_endpoint.rstrip("/")
                + "/v1/traces"
            )
            provider.add_span_processor(BatchSpanProcessor(otlp))
            log.info(
                "otel_otlp_exporter_enabled",
                endpoint=settings.otel_exporter_otlp_endpoint,
            )
        except Exception as exc:  # noqa: BLE001 - logged + non-fatal
            log.warning(
                "otel_otlp_exporter_setup_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:160],
            )

    trace.set_tracer_provider(provider)

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
            log.info("otel_fastapi_instrumented")
        except Exception as exc:  # noqa: BLE001 - logged + non-fatal
            log.warning(
                "otel_fastapi_instrument_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:160],
            )

    _configured = True
    log.info("otel_configured", service_name=settings.otel_service_name)


def get_tracer() -> object:
    """Return a tracer; safe to call before ``configure_tracing``.

    When OTel isn't configured, this returns the SDK's NoOpTracer, so
    ``start_as_current_span`` is a cheap context manager that produces
    no spans. That means call sites can use the tracer unconditionally.
    """
    from opentelemetry import trace

    return trace.get_tracer(_TRACER_NAME)
