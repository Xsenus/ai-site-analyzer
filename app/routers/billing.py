from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import BillingSummaryPayload
from app.services.billing import month_to_date_summary

router = APIRouter(prefix="/v1/billing", tags=["billing"])


@router.get("/remaining", response_model=BillingSummaryPayload)
async def billing_remaining(
    project_id: str | None = Query(default=None, description="Опциональный project_id для фильтрации Costs"),
) -> BillingSummaryPayload:
    try:
        summary = await month_to_date_summary(project_id=project_id)
    except Exception as exc:  # pragma: no cover - network/runtime safety
        raise HTTPException(status_code=502, detail=f"billing provider error: {exc}") from exc

    return BillingSummaryPayload.from_summary(summary)
