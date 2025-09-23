from __future__ import annotations
import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text

from app.config import settings

log = logging.getLogger("db.postgres")
_engine: Optional[AsyncEngine] = None


def get_postgres_engine() -> Optional[AsyncEngine]:
    global _engine
    dsn = settings.POSTGRES_DATABASE_URL
    if not dsn:
        log.warning("POSTGRES_DATABASE_URL not set")
        return None
    if _engine is None:
        _engine = create_async_engine(dsn, pool_pre_ping=True)
    return _engine


async def ping_postgres() -> bool:
    eng = get_postgres_engine()
    if not eng:
        return False
    try:
        async with eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.warning("postgres ping failed: %s", e)
        return False
