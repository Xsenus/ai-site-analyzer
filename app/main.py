# app/main.py
from __future__ import annotations

# --- Windows async psycopg fix: must be set before creating event loop / DB engines ---
import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# ------------------------------------------------------------------------------

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import settings
from app.api.routes import router as api_router  # ваши /v1/... маршруты

# БД: две базы с ленивой инициализацией
from app.db.parsing import get_parsing_engine
from app.db.postgres import get_postgres_engine

# <<< ВАЖНО: импортируем схемы и логику на уровне модуля, чтобы Pydantic видел типы >>>
from app.schemas.ai_search import AiSearchIn, AiEmbeddingOut
from app.services.embeddings import embed_many

# Доп. роутер из ТЗ (POST /api/ai-search и GET /api/ai-search/health)
try:
    from app.routers.ai_search import router as ai_search_router  # noqa: F401
    HAS_AI_SEARCH = True
except Exception:
    HAS_AI_SEARCH = False


# ---------------------------
# Логирование
# ---------------------------
LOG_LEVEL = (settings.LOG_LEVEL or "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ai-site-analyzer")


# ---------------------------
# Вспомогательные функции
# ---------------------------
def _parse_csv_env(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _build_app() -> FastAPI:
    return FastAPI(
        title="AI Site Analyzer API",
        version="1.0.0",
        contact={"name": "AI Site Analyzer"},
    )


# ---------------------------
# Middleware для лаконичного access-логирования
# ---------------------------
class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = time.time()
        path = request.url.path
        method = request.method
        client = (
            request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or request.headers.get("x-real-ip")
            or (request.client.host if request.client else "unknown")
        )

        response: Optional[Response] = None
        try:
            response = await call_next(request)
            return response
        except Exception as ex:
            log.exception("Unhandled error for %s %s from %s: %s", method, path, client, ex)
            return JSONResponse(status_code=500, content={"error": "internal error"})
        finally:
            elapsed_ms = (time.time() - started) * 1000.0
            status = getattr(response, "status_code", 500)
            log.info("%s %s → %s (%.1f ms) ip=%s", method, path, status, elapsed_ms, client)


# ---------------------------
# Lifespan: старт/останов
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Инициализация движков БД (если DSN заданы)
    get_parsing_engine()
    get_postgres_engine()
    log.info("Startup complete: engines initialized (where DSN provided).")
    try:
        yield
    finally:
        # Аккуратно закрываем коннекторы (если создавались)
        parsing = get_parsing_engine()
        pg = get_postgres_engine()
        for eng in (parsing, pg):
            if eng is not None:
                await eng.dispose()
        log.info("All database engines disposed.")


# ---------------------------
# Создание приложения
# ---------------------------
def create_app() -> FastAPI:
    app = _build_app()
    app.router.lifespan_context = lifespan

    # CORS из .env
    origins = _parse_csv_env(getattr(settings, "CORS_ALLOW_ORIGINS", ""))
    allow_methods = _parse_csv_env(getattr(settings, "CORS_ALLOW_METHODS", "")) or ["*"]
    allow_headers = _parse_csv_env(getattr(settings, "CORS_ALLOW_HEADERS", "")) or ["*"]
    allow_credentials = bool(getattr(settings, "CORS_ALLOW_CREDENTIALS", False))

    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            allow_credentials=allow_credentials,
        )

    # Access-лог
    app.add_middleware(AccessLogMiddleware)

    # Подключаем ваши /v1/... маршруты
    app.include_router(api_router)

    # Подключаем /api/ai-search, если модуль присутствует
    if HAS_AI_SEARCH:
        from app.routers.ai_search import router as ai_search_router  # локальный импорт для ясности
        app.include_router(ai_search_router, prefix="/api")

    # --- Алиасы без префикса /api, реализованные напрямую (без вызова чужих хэндлеров) ---
    # Контракт: { "embedding": [...] }
    @app.get("/ai-search/health")
    async def _ai_health_alias():
        return {"ok": True}

    @app.post("/ai-search", response_model=AiEmbeddingOut)
    async def _ai_search_alias(body: AiSearchIn):
        vectors = await embed_many([body.q], timeout=12.0)
        if not vectors or not isinstance(vectors[0], list) or not vectors[0]:
            raise HTTPException(status_code=502, detail="embedding provider failed")
        return {"embedding": [float(x) for x in vectors[0]]}

    @app.get("/favicon.ico", include_in_schema=False)
    async def _favicon_alias() -> Response:
        """Return an empty favicon to avoid 404 noise in the logs."""
        return Response(content=b"", media_type="image/x-icon")

    # (опционально) совместимость с Next по умолчанию: POST /
    @app.post("/")
    async def _compat_root(body: AiSearchIn):
        return await _ai_search_alias(body)

    return app


# Экспортируем экземпляр приложения для Uvicorn/Gunicorn
app = create_app()
