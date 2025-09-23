# app/db/parsing.py
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import settings

log = logging.getLogger("db.parsing")

_engine_parsing: Optional[AsyncEngine] = None


def _normalize_psycopg_dsn(url: str | None) -> str | None:
    if not url:
        return None
    u = url.strip()
    if u.startswith("postgresql+asyncpg://"):
        u = "postgresql+psycopg://" + u.split("postgresql+asyncpg://", 1)[1]
    elif u.startswith("postgresql://"):
        u = "postgresql+psycopg://" + u.split("postgresql://", 1)[1]
    return u


def get_parsing_engine() -> Optional[AsyncEngine]:
    """
    Ленивая инициализация движка для БД parsing_data.
    Настраивается через settings.parsing_url (ENV: PARSING_DATABASE_URL).
    """
    global _engine_parsing

    raw_url = getattr(settings, "parsing_url", None) or getattr(settings, "PARSING_DATABASE_URL", None)
    url = _normalize_psycopg_dsn(raw_url)

    if not url:
        log.warning("PARSING_DATABASE_URL не задан — соединение с parsing_data отключено")
        return None

    if _engine_parsing is None:
        _engine_parsing = create_async_engine(
            url,
            connect_args={"async_": True},  # psycopg3 async
            pool_pre_ping=True,
            future=True,
            echo=getattr(settings, "ECHO_SQL", False),
        )
    return _engine_parsing


async def ping_parsing() -> bool:
    """
    Лёгкая проверка доступности parsing_data. Возвращает True/False.
    """
    eng = get_parsing_engine()
    if eng is None:
        return True

    try:
        async with eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as ex:
        log.warning("parsing_data ping failed: %s", ex)
        return False
