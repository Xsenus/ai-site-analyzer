from __future__ import annotations

from pydantic import AliasChoices, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # OpenAI / embeddings
    OPENAI_API_KEY: str | None = None
    CHAT_MODEL: str = "gpt-4o"
    OPENAI_EMBED_MODEL: str = Field(
        default="text-embedding-3-large",
        validation_alias=AliasChoices("OPENAI_EMBED_MODEL", "EMBED_MODEL"),
    )
    VECTOR_DIM: int = 3072

    # Optional internal embedding service
    INTERNAL_EMBED_URL: str | None = None
    INTERNAL_EMBED_TIMEOUT: float = 6.0

    # /api/ai-search runtime guards
    AI_SEARCH_TIMEOUT: float = 12.0
    AI_SEARCH_RATE_LIMIT_PER_MIN: int = 10

    # Logging / CORS
    LOG_LEVEL: str = "INFO"
    CORS_ALLOW_ORIGINS: str = ""
    CORS_ALLOW_METHODS: str = ""
    CORS_ALLOW_HEADERS: str = ""
    CORS_ALLOW_CREDENTIALS: bool = False

    # Embedding debug helpers
    DEBUG_OPENAI_LOG: bool = False
    EMBED_PREVIEW_CHARS: int = 160
    EMBED_BATCH_SIZE: int = 96
    EMBED_MAX_CHARS: int = 200_000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    @computed_field  # type: ignore[misc]
    @property
    def embed_model(self) -> str:
        return self.OPENAI_EMBED_MODEL


settings = Settings()
