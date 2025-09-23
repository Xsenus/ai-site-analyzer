from __future__ import annotations
import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text

from app.config import settings

log = logging.getLogger("db.parsing")
_engine: Optional[AsyncEngine] = None


def get_parsing_engine() -> Optional[AsyncEngine]:
    global _engine
    dsn = settings.PARSING_DATABASE_URL
    if not dsn:
        log.warning("PARSING_DATABASE_URL not set")
        return None
    if _engine is None:
        _engine = create_async_engine(dsn, pool_pre_ping=True)
    return _engine


async def ping_parsing() -> bool:
    eng = get_parsing_engine()
    if not eng:
        return False
    try:
        async with eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.warning("parsing ping failed: %s", e)
        return False
