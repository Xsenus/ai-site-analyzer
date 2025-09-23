from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncConnection, create_async_engine
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings

log = logging.getLogger("db")

PRIMARY_URL = settings.POSTGRES_DATABASE_URL
SECONDARY_URL = settings.PARSING_DATABASE_URL

engine_primary: AsyncEngine = create_async_engine(PRIMARY_URL, pool_pre_ping=True)
engine_secondary: AsyncEngine = create_async_engine(SECONDARY_URL, pool_pre_ping=True)

SyncMode = Literal["primary_only", "dual_write", "fallback_to_secondary"]


@asynccontextmanager
async def connect_primary() -> AsyncIterator[AsyncConnection]:
    conn = await engine_primary.connect()
    trans = await conn.begin()
    try:
        yield conn
        await trans.commit()
    except:
        await trans.rollback()
        raise
    finally:
        await conn.close()


@asynccontextmanager
async def connect_secondary() -> AsyncIterator[AsyncConnection]:
    conn = await engine_secondary.connect()
    trans = await conn.begin()
    try:
        yield conn
        await trans.commit()
    except:
        await trans.rollback()
        raise
    finally:
        await conn.close()


@asynccontextmanager
async def dual_write(mode: SyncMode):
    """
    Контекстный менеджер для режимов записи.

    primary_only:  транзакция только в primary.
    dual_write:    параллельно открывает secondary, пишет в обе; если secondary падает — логируем, но primary коммитим.
    fallback_to_secondary: пытаемся primary; при ошибке — откатываем и пробуем secondary.
    """
    if mode == "primary_only":
        async with connect_primary() as p:
            yield {"primary": p, "secondary": None, "used": "primary_only"}
        return

    if mode == "dual_write":
        async with connect_primary() as p:
            try:
                async with connect_secondary() as s:
                    yield {"primary": p, "secondary": s, "used": "dual_write"}
            except SQLAlchemyError as e:
                log.error("Secondary write failed in dual_write: %s", e, exc_info=False)
                # Продолжаем только с primary (primary транзакция всё равно коммитнется)
                yield {"primary": p, "secondary": None, "used": "dual_write(primary_only_effective)"}
        return

    if mode == "fallback_to_secondary":
        try:
            async with connect_primary() as p:
                yield {"primary": p, "secondary": None, "used": "primary"}
        except SQLAlchemyError as e:
            log.error("Primary failed, falling back to secondary: %s", e, exc_info=False)
            async with connect_secondary() as s:
                yield {"primary": None, "secondary": s, "used": "secondary"}
        return

    # неизвестный режим — дефолт в primary
    async with connect_primary() as p:
        yield {"primary": p, "secondary": None, "used": "primary_only(default)"}
