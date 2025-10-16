from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from app.api.handlers.analyze_json import router as analyze_json_router
from app.services.health import check_health

router = APIRouter()
router.include_router(analyze_json_router)


@router.get("/health")
async def health() -> Dict[str, Any]:
    """Simple liveness probe used by monitoring and load balancers."""

    return await check_health()
