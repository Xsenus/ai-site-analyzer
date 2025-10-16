# app/routers/ai_search.py
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.schemas.ai_search import (
    AiSearchIn,
    AiEmbeddingOut,
    AiIdsOut,
    AiListsOut,
    GoodsRow,
    EquipRow,
    ProdclassRow,
    ErrorOut,
)
from app.services.embeddings import make_embedding_or_none, validate_dim, VECTOR_DIM
from app.utils.rate_limit import SlidingWindowRateLimiter

log = logging.getLogger("routers.ai_search")

# ENV-настройки
AI_SEARCH_TIMEOUT = float(os.getenv("AI_SEARCH_TIMEOUT", "12.0") or "12.0")
RATE_LIMIT_PER_MIN = int(os.getenv("AI_SEARCH_RATE_LIMIT_PER_MIN", "10") or "10")

# Инициализируем простой лимитер
limiter = SlidingWindowRateLimiter(max_per_minute=RATE_LIMIT_PER_MIN, window_seconds=60)

router = APIRouter(prefix="/ai-search", tags=["ai-search"])


def _client_ip(request: Request) -> str:
    # Пытаемся корректно вытащить IP с учётом прокси
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded:
        return forwarded
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    return request.client.host if request.client else "unknown"


def _normalize_query(q: str) -> str:
    q = (q or "").strip()
    # без агрессивной нормализации — только trim;
    # при желании можно добавить lower(), NFC, удаление лишних пробелов и т.д.
    return q


def _dedupe_rows(rows: Optional[List[dict | GoodsRow | EquipRow | ProdclassRow]]) -> Optional[List[dict]]:
    if rows is None:
        return None
    seen: set[int] = set()
    result: List[dict] = []
    for r in rows:
        if isinstance(r, dict):
            rid = r.get("id")
            obj = r
        else:
            # pydantic-модель → dict
            obj = r.model_dump()
            rid = obj.get("id")
        if isinstance(rid, int) and rid not in seen:
            seen.add(rid)
            result.append(obj)
    return result


@router.post(
    "/ai-search",
    responses={
        200: {
            "content": {
                "application/json": {}
            },
            "description": "One of: AiEmbeddingOut | AiIdsOut | AiListsOut",
        },
        400: {"model": ErrorOut},
        429: {"model": ErrorOut},
        500: {"model": ErrorOut},
    },
)
async def ai_search(request: Request, payload: AiSearchIn) -> Union[AiEmbeddingOut, AiIdsOut, AiListsOut, JSONResponse]:
    """
    Основной обработчик:
    1) Rate-limit
    2) Нормализация и валидация q
    3) Попытка получить эмбеддинг (внутренний → OpenAI fallback)
    4) Если удалось — вернуть {"embedding":[...]}
    5) Иначе — альтернативные форматы (ids/lists) по вашей логике (опционально)
    """
    ip = _client_ip(request)

    allowed, remaining = limiter.check_and_hit(ip)
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=ErrorOut(error="Too many requests").model_dump(),
        )

    started = time.time()
    q = _normalize_query(payload.q)
    if not q:
        raise HTTPException(status_code=400, detail="q is required")

    # Логируем вход
    clipped_q = (q[:256] + "…") if len(q) > 256 else q
    log.info("ai-search request ip=%s q='%s' remaining=%d", ip, clipped_q, remaining)

    # Таймаут всей операции
    try:
        async def _work() -> Union[AiEmbeddingOut, AiIdsOut, AiListsOut]:
            # 1) Попытка — эмбеддинг
            vec = await make_embedding_or_none(q, timeout=AI_SEARCH_TIMEOUT * 0.9)
            if vec and validate_dim(vec, VECTOR_DIM):
                elapsed = (time.time() - started) * 1000
                log.info("ai-search response=embedding dim=%d in %.1fms", len(vec), elapsed)
                return AiEmbeddingOut(embedding=vec)

            # 2) Альтернативный сценарий (опционально):
            #    Здесь можно задействовать ваш ранжировщик/БД, чтобы вернуть ID или списки.
            #    Ниже — пустые коллекции как пример «не нашли confident-ответ».
            #    Если хотите вообще возвращать 500 при отсутствии эмбеддинга — замените на исключение.
            ids: Dict[str, List[int]] = {"goods": [], "equipment": [], "prodclasses": []}
            elapsed = (time.time() - started) * 1000
            log.info("ai-search response=ids(empty) in %.1fms", elapsed)
            return AiIdsOut(ids=ids)

        result = await asyncio.wait_for(_work(), timeout=AI_SEARCH_TIMEOUT)
        # FastAPI сам сериализует pydantic-модели
        return result

    except asyncio.TimeoutError:
        log.warning("ai-search timeout after %.1fs", AI_SEARCH_TIMEOUT)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorOut(error="timeout").model_dump(),
        )
    except HTTPException:
        raise
    except ValidationError as ve:
        # На случай ручной сборки Ai*Out где-то выше
        log.exception("ai-search validation error: %s", ve)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorOut(error="validation error").model_dump(),
        )
    except Exception as ex:
        log.exception("ai-search unexpected error: %s", ex)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorOut(error="internal error").model_dump(),
        )
