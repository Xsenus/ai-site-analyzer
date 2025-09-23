from __future__ import annotations

import logging
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router as api_router

# DB helpers (две БД)
from app.db.parsing import get_parsing_engine, ping_parsing
from app.db.postgres import get_postgres_engine, ping_postgres

# --- Logging ---
LOG_LEVEL = (settings.LOG_LEVEL or "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ai-site-analyzer")

# --- FastAPI app ---
app = FastAPI(title="AI Site Analyzer API", version="1.0.0")

# --- CORS (из .env) ---
origins = [o.strip() for o in (getattr(settings, "CORS_ALLOW_ORIGINS", "") or "").split(",") if o.strip()]
if origins:
    methods = [m.strip() for m in (getattr(settings, "CORS_ALLOW_METHODS", "") or "").split(",") if m.strip()]
    headers = [h.strip() for h in (getattr(settings, "CORS_ALLOW_HEADERS", "") or "").split(",") if h.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=methods or ["*"],
        allow_headers=headers or ["*"],
        allow_credentials=bool(getattr(settings, "CORS_ALLOW_CREDENTIALS", False)),
    )

# Подключаем /v1/... маршруты
app.include_router(api_router)


@app.on_event("startup")
async def on_startup() -> None:
    # Инициализируем коннекторы (если DSN заданы)
    get_parsing_engine()
    get_postgres_engine()
    log.info("Startup complete: engines initialized (where DSN provided).")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    engines: list[AsyncEngine | None] = [
        get_parsing_engine(),
        get_postgres_engine(),
    ]
    for eng in engines:
        if eng is not None:
            await eng.dispose()
    log.info("All database engines disposed.")


@app.get("/health")
async def health():
    """
    Healthcheck пингует обе базы.
    ok=true только если доступны все, для которых заданы DSN.
    """
    results = {
        "parsing_data": await ping_parsing(),
        "postgres": await ping_postgres(),
    }
    ok = all(results.values()) if results else False
    return {"ok": ok, "connections": results}
