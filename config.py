"""
Nutribot Backend — Configuration
Carga variables de entorno con pydantic-settings.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- Database ---
    database_url: str = "postgresql+asyncpg://nutribot:nutribot@localhost:5432/nutribot"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_stt_model: str = "gpt-4o-mini-transcribe"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "coral"

    # --- Evolution API ---
    evolution_api_url: str = "http://localhost:8080"
    evolution_api_key: str = ""
    evolution_instance: str = "nutribot"

    # --- Workers ---
    inbox_poll_interval_seconds: float = 1.0
    outbox_poll_interval_seconds: float = 1.0
    extraction_poll_interval_seconds: float = 5.0
    sweeper_interval_seconds: float = 60.0
    zombie_timeout_minutes: int = 10
    max_retry_count: int = 3

    # --- RAG ---
    openai_embedding_model: str = "text-embedding-3-small"
    rag_threshold: float = 0.65
    rag_limit: int = 3

    # --- App ---
    log_level: str = "INFO"
    debug: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
