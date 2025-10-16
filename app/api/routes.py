from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from app.api.handlers.analyze_json import router as analyze_json_router
from app.services.health import check_health

router = APIRouter()
router.include_router(analyze_json_router)


@router.get("/health")
async def health() -> Dict[str, Any]:
    """Extended health-check that pings both databases."""

    return await check_health()
