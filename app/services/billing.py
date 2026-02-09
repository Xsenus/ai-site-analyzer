from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import httpx

from app.config import settings


@dataclass
class BillingSummary:
    currency: str
    period_start: int
    period_end: int
    spent_usd: float
    limit_usd: float | None
    prepaid_credits_usd: float | None
    remaining_usd: float | None


def _month_range_utc(now: dt.datetime | None = None) -> tuple[int, int]:
    current = now or dt.datetime.now(dt.timezone.utc)
    month_start = dt.datetime(current.year, current.month, 1, tzinfo=dt.timezone.utc)
    last_day = calendar.monthrange(current.year, current.month)[1]
    month_end = dt.datetime(
        current.year,
        current.month,
        last_day,
        23,
        59,
        59,
        tzinfo=dt.timezone.utc,
    )
    return int(month_start.timestamp()), int(month_end.timestamp())


def _iter_results(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    for bucket in data:
        if not isinstance(bucket, dict):
            continue
        results = bucket.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if isinstance(result, dict):
                yield result


def _extract_amount(result: Dict[str, Any]) -> tuple[float, str | None]:
    amount = result.get("amount")
    if not isinstance(amount, dict):
        return 0.0, None

    value = amount.get("value")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0

    currency = amount.get("currency")
    if not isinstance(currency, str):
        currency = None

    return parsed, currency


async def fetch_costs(
    *,
    start_time: int,
    end_time: int,
    group_by: Optional[List[str]] = None,
    project_id: str | None = None,
) -> Dict[str, Any]:
    admin_key = (settings.OPENAI_ADMIN_KEY or "").strip()
    if not admin_key:
        raise RuntimeError("OPENAI_ADMIN_KEY not configured")

    params: List[tuple[str, str | int]] = [
        ("start_time", start_time),
        ("end_time", end_time),
        ("bucket_width", "1d"),
        ("limit", 90),
    ]

    for group in (group_by or ["project_id"]):
        params.append(("group_by[]", group))

    if project_id:
        params.append(("project_id", project_id))

    base_url = settings.BILLING_COSTS_BASE_URL.rstrip("/")
    url = f"{base_url}/organization/costs"

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            url,
            headers={
                "Authorization": f"Bearer {admin_key}",
                "Content-Type": "application/json",
            },
            params=params,
        )
        response.raise_for_status()
        return response.json()


async def month_to_date_summary(project_id: str | None = None) -> BillingSummary:
    start_ts, end_ts = _month_range_utc()
    payload = await fetch_costs(start_time=start_ts, end_time=end_ts, project_id=project_id)

    spent = 0.0
    currency = "usd"
    for result in _iter_results(payload):
        value, result_currency = _extract_amount(result)
        spent += value
        if result_currency:
            currency = result_currency.lower()

    limit_usd = settings.BILLING_MONTHLY_LIMIT_USD
    prepaid = settings.BILLING_PREPAID_CREDITS_USD

    remaining: float | None = None
    if limit_usd is not None:
        remaining = limit_usd - spent
    elif prepaid is not None:
        remaining = prepaid - spent

    return BillingSummary(
        currency=currency,
        period_start=start_ts,
        period_end=end_ts,
        spent_usd=spent,
        limit_usd=limit_usd,
        prepaid_credits_usd=prepaid,
        remaining_usd=remaining,
    )
