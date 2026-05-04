import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change-me-in-prod"

    database_url: str = "sqlite+aiosqlite:///./helix_srop.db"
    chroma_persist_dir: str = "./chroma_db"

    google_api_key: str = ""
    adk_model: str = "gemini-2.0-flash"

    llm_timeout_seconds: int = 30
    tool_timeout_seconds: int = 10

    # E4 — LLM-as-judge reranker. When enabled, search_docs fetches
    # ``reranker_top_n`` chunks from Chroma, asks Gemini to score them, and
    # returns the top-k ranked by judge score. Off by default so demos /
    # tests / cold-clones don't pay the extra LLM call.
    reranker_enabled: bool = False
    reranker_top_n: int = 20
    reranker_model: str = "gemini-flash-latest"

    # OpenTelemetry — when enabled, pipeline.run / agent / tool calls emit
    # spans. ``otel_console_exporter`` ships them to stdout (good for local
    # demos); when ``otel_exporter_otlp_endpoint`` is set, ships to OTLP
    # collectors like Jaeger.
    otel_enabled: bool = False
    otel_console_exporter: bool = False
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "helix-srop"

    # Vector store backend — ``chroma`` (default, local file-based) or
    # ``pgvector`` (Postgres extension). The pgvector path uses the same
    # ``DATABASE_URL`` if it points at Postgres, else its own
    # ``pgvector_database_url``.
    vector_store_backend: str = "chroma"
    pgvector_database_url: str = ""
    pgvector_table: str = "helix_doc_chunks"


settings = Settings()

# google-adk and google.genai both read ``GOOGLE_API_KEY`` from the
# process environment when no explicit ``api_key`` is passed. We load
# the key from .env via pydantic-settings, so re-export it to os.environ
# at import time so ADK's internal client picks it up automatically.
if settings.google_api_key and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
