"""
Nutribot Backend — Configuration
Carga variables de entorno con pydantic-settings.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- Database ---
    database_url: str = "postgresql+asyncpg://nutribot:nutribot@localhost:5432/nutribot"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

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
    sweeper_interval_seconds: float = 60.0
    zombie_timeout_minutes: int = 10
    max_retry_count: int = 3
    processing_lock_ttl_seconds: int = 60

    # --- RAG ---
    openai_embedding_model: str = "text-embedding-3-small"
    rag_threshold: float = 0.65
    rag_limit: int = 3

    # --- App ---
    log_level: str = "INFO"
    debug: bool = False
    nutribot_mode: str = "monolith"

    # --- Legacy Rate Limiting (compatibilidad) ---
    rate_limit_max_messages: int = 10
    rate_limit_window_seconds: int = 60

    # --- Webhook Security ---
    webhook_secret: str = ""
    webhook_allowed_events: str = "messages.upsert"
    webhook_rate_limit_max_ip: int = 1000
    webhook_rate_limit_max_phone: int = 60
    webhook_rate_limit_window_seconds: int = 60
    webhook_dedup_ttl_seconds: int = 300

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
