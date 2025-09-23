# app/db/tx.py
from __future__ import annotations

import enum
import logging
from typing import Awaitable, Callable, Optional, Protocol

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import settings

log = logging.getLogger("db.tx")

PRIMARY_URL: Optional[str] = settings.POSTGRES_URL
SECONDARY_URL: Optional[str] = settings.PARSING_URL


class SyncMode(str, enum.Enum):
    PRIMARY_ONLY = "primary_only"
    DUAL_WRITE = "dual_write"
    FALLBACK_TO_SECONDARY = "fallback_to_secondary"


def _resolve_mode(mode: Optional[SyncMode]) -> SyncMode:
    if mode is not None:
        return mode
    # из .env DEFAULT_WRITE_MODE
    try:
        return SyncMode(settings.default_write_mode)
    except Exception:
        return SyncMode.PRIMARY_ONLY


_primary_engine: Optional[AsyncEngine] = None
_secondary_engine: Optional[AsyncEngine] = None


def get_primary_engine() -> Optional[AsyncEngine]:
    global _primary_engine
    if PRIMARY_URL is None:
        log.warning("PRIMARY_URL не задан — primary отключен")
        return None
    if _primary_engine is None:
        _primary_engine = create_async_engine(PRIMARY_URL, pool_pre_ping=True, future=True)
    return _primary_engine


def get_secondary_engine() -> Optional[AsyncEngine]:
    global _secondary_engine
    if SECONDARY_URL is None:
        log.info("SECONDARY_URL не задан — secondary отключен")
        return None
    if _secondary_engine is None:
        _secondary_engine = create_async_engine(SECONDARY_URL, pool_pre_ping=True, future=True)
    return _secondary_engine


class ConnAction(Protocol):
    """
    Протокол для функций, которые принимают connection и что-то делают.
    Пример:
        async def write_user(conn: AsyncConnection) -> None:
            await conn.execute(text("INSERT ..."))
    """
    def __call__(self, conn) -> Awaitable[object]: ...


async def run_on_engine(engine: AsyncEngine, action: ConnAction) -> object:
    async with engine.begin() as conn:
        return await action(conn)


async def dual_write(action: ConnAction, mode: Optional[SyncMode] = None) -> object:
    """
    Выполнить запись согласно режиму синхронизации:
      - primary_only: пишем только в primary (если доступен)
      - dual_write: сначала в primary, затем в secondary (если доступен)
      - fallback_to_secondary: пишем в primary; если ошибка — пробуем secondary

    Возвращает результат выполнения на primary (если был), иначе — на secondary.
    """
    effective_mode = _resolve_mode(mode)

    primary = get_primary_engine()
    secondary = get_secondary_engine()

    # Нет вообще ни одного движка
    if primary is None and secondary is None:
        raise RuntimeError("Нет доступных подключений: PRIMARY_URL и SECONDARY_URL пустые")

    # PRIMARY_ONLY
    if effective_mode == SyncMode.PRIMARY_ONLY:
        if primary is None:
            raise RuntimeError("PRIMARY_ONLY: PRIMARY_URL не задан")
        return await run_on_engine(primary, action)

    # DUAL_WRITE
    if effective_mode == SyncMode.DUAL_WRITE:
        result = None
        if primary is not None:
            result = await run_on_engine(primary, action)
        if secondary is not None:
            try:
                await run_on_engine(secondary, action)
            except Exception as e:
                log.error("DUAL_WRITE: ошибка записи в secondary: %s", e, exc_info=True)
        return result

    # FALLBACK_TO_SECONDARY
    if effective_mode == SyncMode.FALLBACK_TO_SECONDARY:
        try:
            if primary is None:
                raise RuntimeError("PRIMARY недоступен")
            return await run_on_engine(primary, action)
        except Exception as e:
            log.error("FALLBACK: ошибка primary, пробуем secondary: %s", e, exc_info=True)
            if secondary is None:
                raise
            return await run_on_engine(secondary, action)

    # safety
    raise RuntimeError(f"Неизвестный режим синхронизации: {effective_mode}")
