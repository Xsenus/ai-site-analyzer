# app/db/postgres.py
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import settings

log = logging.getLogger("db.postgres")

_engine_postgres: Optional[AsyncEngine] = None


def _normalize_psycopg_dsn(url: str | None) -> str | None:
    """
    Приводит DSN к драйверу psycopg3 (async).
    Допустимые варианты на входе:
      - postgresql+asyncpg://...
      - postgresql://...
      - postgresql+psycopg://...
    На выходе получаем: postgresql+psycopg://...
    """
    if not url:
        return None
    u = url.strip()
    if u.startswith("postgresql+asyncpg://"):
        u = "postgresql+psycopg://" + u.split("postgresql+asyncpg://", 1)[1]
    elif u.startswith("postgresql://"):
        u = "postgresql+psycopg://" + u.split("postgresql://", 1)[1]
    # если уже psycopg, ничего не делаем
    return u


def get_postgres_engine() -> Optional[AsyncEngine]:
    """
    Ленивая инициализация движка для основной Postgres-БД.
    Настраивается через settings.postgres_url (ENV: POSTGRES_URL).
    Обязательно: DSN должен быть в формате postgresql+psycopg://...
    """
    global _engine_postgres

    raw_url = getattr(settings, "postgres_url", None) or getattr(settings, "POSTGRES_URL", None)
    url = _normalize_psycopg_dsn(raw_url)

    if not url:
        log.warning("POSTGRES_URL не задан — соединение с Postgres отключено")
        return None

    if _engine_postgres is None:
        # ВАЖНО: для async psycopg3 нужно connect_args={"async_": True}
        _engine_postgres = create_async_engine(
            url,
            connect_args={"async_": True},
            pool_pre_ping=True,
            future=True,
            echo=getattr(settings, "ECHO_SQL", False),
        )
    return _engine_postgres


async def ping_postgres() -> bool:
    """
    Лёгкая проверка доступности Postgres. Возвращает True/False.
    """
    eng = get_postgres_engine()
    if eng is None:
        # Если DSN не задан — считаем, что этот коннектор «не требуется»
        return True

    try:
        async with eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as ex:
        log.warning("Postgres ping failed: %s", ex)
        return False
