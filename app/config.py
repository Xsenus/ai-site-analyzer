# app/config.py
from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, computed_field


def _normalize_psycopg_dsn(raw: Optional[str]) -> Optional[str]:
    """
    Нормализует DSN под psycopg3.
    Поддерживаются входы:
      - postgresql+asyncpg://...  -> postgresql+psycopg://...
      - postgresql://...          -> postgresql+psycopg://...
      - postgresql+psycopg://...  -> без изменений
    """
    if not raw:
        return None
    u = raw.strip()
    if u.startswith("postgresql+asyncpg://"):
        return "postgresql+psycopg://" + u.split("postgresql+asyncpg://", 1)[1]
    if u.startswith("postgresql://"):
        return "postgresql+psycopg://" + u.split("postgresql://", 1)[1]
    return u


class Settings(BaseSettings):
    # =========================
    # Databases (psycopg3 async)
    # =========================
    # Поддерживаем новые и старые имена для DSN
    POSTGRES_URL_RAW: str | None = Field(
        default=None,
        validation_alias=AliasChoices("POSTGRES_URL", "POSTGRES_DATABASE_URL"),
        description="DSN основной БД. Желательно postgresql+psycopg://",
    )
    PARSING_URL_RAW: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PARSING_URL", "PARSING_DATABASE_URL"),
        description="DSN БД parsing_data. Желательно postgresql+psycopg://",
    )

    # Показывать SQL (echo sqlalchemy)
    ECHO_SQL: bool = False

    # =========================
    # OpenAI / Embeddings
    # =========================
    OPENAI_API_KEY: str | None = None
    CHAT_MODEL: str = "gpt-4o"

    # Поддержка старого EMBED_MODEL как алиаса к OPENAI_EMBED_MODEL
    OPENAI_EMBED_MODEL: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("OPENAI_EMBED_MODEL", "EMBED_MODEL"),
    )
    VECTOR_DIM: int = 3072

    # Таймауты/лимиты для /api/ai-search
    AI_SEARCH_TIMEOUT: float = 12.0
    AI_SEARCH_RATE_LIMIT_PER_MIN: int = 10

    # Внутренний сервис эмбеддингов (если есть)
    INTERNAL_EMBED_URL: str | None = None
    INTERNAL_EMBED_TIMEOUT: float = 6.0

    # =========================
    # Режим записи по умолчанию
    # =========================
    # Принимаем оба варианта имён из .env и нормализуем
    DEFAULT_WRITE_MODE_RAW: str = Field(
        default="primary_only",
        validation_alias=AliasChoices("DEFAULT_WRITE_MODE", "DEFAULT_SYNC_MODE"),
    )

    # =========================
    # Логирование / CORS
    # =========================
    LOG_LEVEL: str = "INFO"
    CORS_ALLOW_ORIGINS: str = ""
    CORS_ALLOW_METHODS: str = ""
    CORS_ALLOW_HEADERS: str = ""
    CORS_ALLOW_CREDENTIALS: bool = False

    # =========================
    # Кастомизация справочников (опционально)
    # =========================
    # GOODS TYPES
    IB_GOODS_TYPES_TABLE: str | None = None
    IB_GOODS_TYPES_ID_COLUMN: str | None = None
    IB_GOODS_TYPES_NAME_COLUMN: str | None = None
    IB_GOODS_TYPES_VECTOR_COLUMN: str | None = None
    IB_GOODS_TYPES_COMPANY_ID: int | None = None

    # EQUIPMENT
    IB_EQUIPMENT_TABLE: str | None = None
    IB_EQUIPMENT_ID_COLUMN: str | None = None
    IB_EQUIPMENT_NAME_COLUMN: str | None = None
    IB_EQUIPMENT_VECTOR_COLUMN: str | None = None
    IB_EQUIPMENT_COMPANY_ID: int | None = None

    # =========================
    # Доп. флаги из .env
    # =========================
    DEBUG_OPENAI_LOG: bool = False
    EMBED_PREVIEW_CHARS: int = 160
    EMBED_BATCH_SIZE: int = 96
    EMBED_MAX_CHARS: int = 200_000

    # =========================
    # Pydantic Settings config
    # =========================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # Имена переменных окружения без учёта регистра
        extra="ignore",         # Лишние ключи в .env игнорируем, без падений
        populate_by_name=True,
    )

    # ---------- Вычислимые свойства для БД ----------
    @computed_field  # type: ignore[misc]
    @property
    def postgres_url(self) -> str | None:
        return _normalize_psycopg_dsn(self.POSTGRES_URL_RAW)

    @computed_field  # type: ignore[misc]
    @property
    def parsing_url(self) -> str | None:
        return _normalize_psycopg_dsn(self.PARSING_URL_RAW)

    # ---------- Нормализованный режим записи ----------
    @computed_field  # type: ignore[misc]
    @property
    def default_write_mode(self) -> str:
        """
        Возвращает одно из: primary_only | dual_write | fallback_to_secondary
        """
        val = (self.DEFAULT_WRITE_MODE_RAW or "").strip().lower()
        if val in {"primary_only", "dual_write", "fallback_to_secondary"}:
            return val
        return "primary_only"

    # ---------- Обратная совместимость (CAPS-атрибуты) ----------
    # Некоторые места кода могут обращаться к CAPS-именам.
    @computed_field  # type: ignore[misc]
    @property
    def POSTGRES_URL(self) -> str | None:  # noqa: N802
        return self.postgres_url

    @computed_field  # type: ignore[misc]
    @property
    def PARSING_URL(self) -> str | None:  # noqa: N802
        return self.parsing_url

    @computed_field  # type: ignore[misc]
    @property
    def POSTGRES_DATABASE_URL(self) -> str | None:  # noqa: N802
        return self.postgres_url

    @computed_field  # type: ignore[misc]
    @property
    def PARSING_DATABASE_URL(self) -> str | None:  # noqa: N802
        return self.parsing_url

    # ---------- Удобный alias для эмбед-модели ----------
    @computed_field  # type: ignore[misc]
    @property
    def embed_model(self) -> str:
        return self.OPENAI_EMBED_MODEL


settings = Settings()
