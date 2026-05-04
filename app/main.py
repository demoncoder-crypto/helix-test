"""
FastAPI application entrypoint.

Wires:
- Structured logging (structlog JSON renderer).
- Async DB schema bootstrap on startup (`create_all`).
- The 3 versioned routers under `/v1`.
- A global exception handler for the typed `HelixError` hierarchy so every
  domain failure renders as RFC 7807 problem-detail JSON.
- A single-page web UI mounted at ``/`` (``app/web/index.html``) so the
  whole demo — backend + frontend — ships from one process and one URL.
- Permissive CORS so the UI works whether it's served same-origin
  (default), from a separate static host, or from a localhost dev port.
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_chat, routes_sessions, routes_traces
from app.api.errors import HelixError, helix_error_handler
from app.db.session import init_db
from app.obs.logging import configure_logging
from app.obs.tracing import configure_tracing

log = structlog.get_logger()
_WEB_DIR = Path(__file__).parent / "web"
_DOCS_DIR = Path(__file__).parent.parent / "docs"


async def _auto_ingest_if_empty() -> None:
    """First-boot RAG bootstrap.

    Hosted demos (Render, Fly, HF Spaces) start with an empty Chroma
    directory. Rather than require reviewers to ssh in and run
    ``python -m app.rag.ingest`` by hand, we check on every startup
    whether the vector store has any rows, and if not we ingest the
    bundled ``docs/`` folder. This is idempotent — a non-empty store
    short-circuits in O(1).
    """
    try:
        from app.rag.vector_store import count

        n = await count()
        if n > 0:
            log.info("auto_ingest_skipped", chunks_already_indexed=n)
            return
        if not _DOCS_DIR.exists():
            log.warning("auto_ingest_no_docs_dir", path=str(_DOCS_DIR))
            return
        log.info("auto_ingest_started", path=str(_DOCS_DIR))
        from app.rag.ingest import ingest_directory

        await ingest_directory(_DOCS_DIR)
        n_after = await count()
        log.info("auto_ingest_completed", chunks_indexed=n_after)
    except Exception as exc:  # noqa: BLE001 - logged + non-fatal
        log.warning(
            "auto_ingest_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    configure_tracing(app)
    await init_db()
    # Run ingest in the background so /healthz is up immediately and
    # the platform's health probe doesn't time out while we embed.
    asyncio.create_task(_auto_ingest_if_empty())
    yield


app = FastAPI(title="Helix SROP", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_sessions.router, prefix="/v1")
app.include_router(routes_chat.router, prefix="/v1")
app.include_router(routes_traces.router, prefix="/v1")

app.add_exception_handler(HelixError, helix_error_handler)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


# ── single-page web UI at ``/`` ──────────────────────────────────────────
# Serve ``app/web/index.html`` so reviewers can interact with the system
# without writing curl. Static-file mounting goes *after* the API routes
# so ``/v1/...`` and ``/healthz`` always win the path match. We expose
# ``/ui`` as a redirect for explicitness in deploy logs / docs links.
if _WEB_DIR.exists():
    @app.get("/", include_in_schema=False)
    async def root_index() -> FileResponse:
        return FileResponse(_WEB_DIR / "index.html")

    @app.get("/ui", include_in_schema=False)
    async def ui_redirect() -> RedirectResponse:
        return RedirectResponse(url="/")

    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")
