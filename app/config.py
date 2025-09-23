from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # --- DBs ---
    POSTGRES_DATABASE_URL: str = Field(..., description="Основная БД (primary)")
    PARSING_DATABASE_URL: str = Field(..., description="Дублирующая БД (secondary)")

    # --- OpenAI ---
    OPENAI_API_KEY: str
    CHAT_MODEL: str = "gpt-4o"
    EMBED_MODEL: str = "text-embedding-3-small"

    # --- режим записи по умолчанию ---
    # primary_only | dual_write | fallback_to_secondary
    DEFAULT_SYNC_MODE: str = "primary_only"

    # --- логирование / CORS ---
    LOG_LEVEL: str = "INFO"
    CORS_ALLOW_ORIGINS: str = ""
    CORS_ALLOW_METHODS: str = ""
    CORS_ALLOW_HEADERS: str = ""
    CORS_ALLOW_CREDENTIALS: bool = False

    # ======== кастомизация справочников (необязательно) ========
    # GOODS TYPES
    IB_GOODS_TYPES_TABLE: str | None = None                # default: ib_goods_types
    IB_GOODS_TYPES_ID_COLUMN: str | None = None            # default: id
    IB_GOODS_TYPES_NAME_COLUMN: str | None = None          # auto: goods_type_name|name|goods_type|title|label
    IB_GOODS_TYPES_VECTOR_COLUMN: str | None = None        # auto: goods_type_vector|vector|emb|embedding
    IB_GOODS_TYPES_COMPANY_ID: int | None = None           # если нужно фильтровать каталог по company_id

    # EQUIPMENT
    IB_EQUIPMENT_TABLE: str | None = None                  # default: ib_equipment
    IB_EQUIPMENT_ID_COLUMN: str | None = None              # default: id
    IB_EQUIPMENT_NAME_COLUMN: str | None = None            # auto: equipment_name|name|equipment|title|label
    IB_EQUIPMENT_VECTOR_COLUMN: str | None = None          # auto: equipment_vector|vector|emb|embedding
    IB_EQUIPMENT_COMPANY_ID: int | None = None             # если нужно фильтровать каталог по company_id

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
