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
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/runtime safety
        raise HTTPException(status_code=502, detail=f"billing provider error: {exc}") from exc

    return BillingSummaryPayload(
        currency=summary.currency,
        period_start=summary.period_start,
        period_end=summary.period_end,
        spent_usd=summary.spent_usd,
        month_to_date_spend_usd=summary.spent_usd,
        limit_usd=summary.limit_usd,
        budget_monthly_usd=summary.limit_usd,
        prepaid_credits_usd=summary.prepaid_credits_usd,
        remaining_usd=summary.remaining_usd,
    )
