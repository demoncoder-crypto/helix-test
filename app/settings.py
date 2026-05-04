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


settings = Settings()

# google-adk and google.genai both read ``GOOGLE_API_KEY`` from the
# process environment when no explicit ``api_key`` is passed. We load
# the key from .env via pydantic-settings, so re-export it to os.environ
# at import time so ADK's internal client picks it up automatically.
if settings.google_api_key and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
