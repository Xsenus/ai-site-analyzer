from __future__ import annotations

import pytest

from app.api.schemas import BillingSummaryPayload
from app.services.billing import _extract_amount, _iter_results
from app.services.pricing import calculate_response_cost_usd


def test_calculate_response_cost_usd_gpt_5_mini_with_cached_tokens():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 250,
        "input_tokens_details": {"cached_tokens": 200},
    }

    cost = calculate_response_cost_usd("gpt-5-mini", usage)

    expected = 800 * (0.45 / 1_000_000) + 200 * (0.045 / 1_000_000) + 250 * (3.60 / 1_000_000)
    assert cost == pytest.approx(expected)


def test_calculate_response_cost_usd_unknown_model_returns_zero():
    assert calculate_response_cost_usd("unknown-model", {"input_tokens": 1, "output_tokens": 1}) == 0.0


def test_iter_results_and_extract_amount():
    payload = {
        "data": [
            {
                "results": [
                    {"amount": {"value": 1.23, "currency": "usd"}},
                    {"amount": {"value": "2.0", "currency": "USD"}},
                ]
            }
        ]
    }

    values = [_extract_amount(result) for result in _iter_results(payload)]

    assert values == [(1.23, "usd"), (2.0, "USD")]


def test_billing_summary_payload_has_legacy_and_new_fields():
    payload = BillingSummaryPayload(
        currency="usd",
        period_start=1,
        period_end=2,
        spent_usd=12.5,
        limit_usd=100.0,
        remaining_usd=87.5,
    )

    assert payload.spent_usd == 12.5
    assert payload.month_to_date_spend_usd == 12.5
    assert payload.budget_monthly_usd == 100.0
    assert payload.remaining_usd == 87.5
