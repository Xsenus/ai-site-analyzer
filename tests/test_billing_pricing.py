from __future__ import annotations

import pytest

from app.api.schemas import BillingSummaryPayload
from app.routers import billing as billing_router
from app.services import billing as billing_service
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



def test_calculate_response_cost_usd_gpt_4o_non_zero():
    usage = {"input_tokens": 1200, "output_tokens": 300, "input_tokens_details": {"cached_tokens": 0}}

    cost = calculate_response_cost_usd("gpt-4o", usage)

    expected = 1200 * (5.0 / 1_000_000) + 300 * (15.0 / 1_000_000)
    assert cost == pytest.approx(expected)
    assert cost > 0


def test_calculate_response_cost_usd_embeddings_non_zero():
    usage = {"total_tokens": 2048}

    cost = calculate_response_cost_usd("text-embedding-3-small", usage)

    expected = 2048 * (0.02 / 1_000_000)
    assert cost == pytest.approx(expected)
    assert cost > 0

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
    assert payload.spend_month_to_date_usd == 12.5
    assert payload.budget_monthly_usd == 100.0
    assert payload.remaining_usd == 87.5
    assert payload.configured is True
    assert payload.error is None


def test_billing_summary_payload_supports_unconfigured_state():
    payload = BillingSummaryPayload(
        currency="usd",
        period_start=1,
        period_end=2,
        spent_usd=None,
        limit_usd=25.0,
        remaining_usd=None,
        configured=False,
        error="OPENAI_ADMIN_KEY not configured",
    )

    assert payload.month_to_date_spend_usd is None
    assert payload.spend_month_to_date_usd is None
    assert payload.budget_monthly_usd == 25.0
    assert payload.configured is False
    assert payload.error == "OPENAI_ADMIN_KEY not configured"


@pytest.mark.anyio
async def test_month_to_date_summary_without_admin_key_returns_null_remaining(monkeypatch):
    billing_service._WARNED_BILLING_REASONS.clear()
    monkeypatch.setattr(billing_service.settings, "OPENAI_ADMIN_KEY", "")

    summary = await billing_service.month_to_date_summary()

    assert summary.remaining_usd is None
    assert summary.limit_usd == billing_service.settings.BILLING_MONTHLY_LIMIT_USD
    assert summary.spent_usd is None
    assert summary.configured is False
    assert summary.error == "OPENAI_ADMIN_KEY not configured"


@pytest.mark.anyio
async def test_month_to_date_summary_without_admin_key_logs_warning_once(monkeypatch, caplog):
    billing_service._WARNED_BILLING_REASONS.clear()
    monkeypatch.setattr(billing_service.settings, "OPENAI_ADMIN_KEY", "")

    caplog.set_level("WARNING", logger="services.billing")
    await billing_service.month_to_date_summary()
    await billing_service.month_to_date_summary()

    assert caplog.messages.count("OPENAI_ADMIN_KEY not configured") == 1


@pytest.mark.anyio
async def test_billing_remaining_endpoint_returns_200_style_payload_without_admin_key(monkeypatch):
    billing_service._WARNED_BILLING_REASONS.clear()
    monkeypatch.setattr(billing_service.settings, "OPENAI_ADMIN_KEY", "")

    payload = await billing_router.billing_remaining(project_id=None)

    assert payload.configured is False
    assert payload.error == "OPENAI_ADMIN_KEY not configured"
    assert payload.spend_month_to_date_usd is None
    assert payload.remaining_usd is None
