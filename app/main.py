"""
FastAPI application entrypoint.

Wires:
- Structured logging (structlog JSON renderer).
- Async DB schema bootstrap on startup (`create_all`).
- The 3 versioned routers under `/v1`.
- A global exception handler for the typed `HelixError` hierarchy so every
  domain failure renders as RFC 7807 problem-detail JSON.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import routes_chat, routes_sessions, routes_traces
from app.api.errors import HelixError, helix_error_handler
from app.db.session import init_db
from app.obs.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await init_db()
    yield


app = FastAPI(title="Helix SROP", version="0.1.0", lifespan=lifespan)

app.include_router(routes_sessions.router, prefix="/v1")
app.include_router(routes_chat.router, prefix="/v1")
app.include_router(routes_traces.router, prefix="/v1")

app.add_exception_handler(HelixError, helix_error_handler)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
