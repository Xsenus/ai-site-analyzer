from __future__ import annotations

import json

import pytest

from app.jobs import system_healthcheck as healthcheck_job


def _probe(*, url: str, http_status: int | None, payload: object, reason: str | None) -> healthcheck_job.ProbeResult:
    return healthcheck_job.ProbeResult(
        url=url,
        http_status=http_status,
        payload=payload,
        reason=reason,
    )


def test_build_summary_reports_ok_for_healthy_service_and_billing() -> None:
    summary = healthcheck_job._build_summary(
        base_url="http://127.0.0.1:8123",
        health_probe=_probe(
            url="http://127.0.0.1:8123/health",
            http_status=200,
            payload={"ok": True},
            reason=None,
        ),
        billing_probe=_probe(
            url="http://127.0.0.1:8123/v1/billing/remaining",
            http_status=200,
            payload={
                "currency": "usd",
                "period_start": 1,
                "period_end": 2,
                "spent_usd": 12.5,
                "remaining_usd": 87.5,
                "configured": True,
                "error": None,
            },
            reason=None,
        ),
    )

    assert summary.ok is True
    assert summary.severity == "ok"
    assert summary.reason == "ok"
    assert summary.billing_configured is True
    assert summary.billing_error is None
    assert summary.billing_spent_usd == pytest.approx(12.5)


def test_build_summary_reports_degraded_for_billing_error() -> None:
    summary = healthcheck_job._build_summary(
        base_url="http://127.0.0.1:8123",
        health_probe=_probe(
            url="http://127.0.0.1:8123/health",
            http_status=200,
            payload={"ok": True},
            reason=None,
        ),
        billing_probe=_probe(
            url="http://127.0.0.1:8123/v1/billing/remaining",
            http_status=200,
            payload={
                "currency": "usd",
                "period_start": 1,
                "period_end": 2,
                "spent_usd": None,
                "remaining_usd": None,
                "configured": True,
                "error": "billing provider error: 403 Forbidden",
            },
            reason=None,
        ),
    )

    assert summary.ok is True
    assert summary.severity == "degraded"
    assert summary.reason == "billing_degraded"
    assert summary.billing_error == "billing provider error: 403 Forbidden"


def test_build_summary_reports_unhealthy_when_health_probe_fails() -> None:
    summary = healthcheck_job._build_summary(
        base_url="http://127.0.0.1:8123",
        health_probe=_probe(
            url="http://127.0.0.1:8123/health",
            http_status=503,
            payload=None,
            reason="http_status:503",
        ),
        billing_probe=_probe(
            url="http://127.0.0.1:8123/v1/billing/remaining",
            http_status=None,
            payload=None,
            reason="http_error:ConnectError",
        ),
    )

    assert summary.ok is False
    assert summary.severity == "unhealthy"
    assert summary.reason == "health_http_status:503"


def test_should_send_alert_distinguishes_degradation_and_recovery() -> None:
    assert healthcheck_job._should_send_alert(
        previous_status=None,
        current_status="degraded",
        alert_on_recovery=True,
    )
    assert healthcheck_job._should_send_alert(
        previous_status="ok",
        current_status="unhealthy",
        alert_on_recovery=True,
    )
    assert healthcheck_job._should_send_alert(
        previous_status="degraded",
        current_status="ok",
        alert_on_recovery=True,
    )
    assert not healthcheck_job._should_send_alert(
        previous_status="degraded",
        current_status="ok",
        alert_on_recovery=False,
    )


@pytest.mark.anyio
async def test_run_persists_state_and_artifacts_for_degraded_billing(tmp_path, monkeypatch) -> None:
    state_file = tmp_path / "state.json"
    artifact_dir = tmp_path / "artifacts"

    async def fake_fetch_json(url: str, _timeout: float) -> healthcheck_job.ProbeResult:
        if url.endswith("/health"):
            return _probe(url=url, http_status=200, payload={"ok": True}, reason=None)
        return _probe(
            url=url,
            http_status=200,
            payload={
                "currency": "usd",
                "period_start": 1,
                "period_end": 2,
                "spent_usd": None,
                "remaining_usd": None,
                "configured": False,
                "error": "OPENAI_ADMIN_KEY not configured",
            },
            reason=None,
        )

    monkeypatch.setattr(healthcheck_job, "_fetch_json", fake_fetch_json)

    summary = await healthcheck_job._run(
        base_url="http://127.0.0.1:8123",
        health_url="http://127.0.0.1:8123/health",
        billing_url="http://127.0.0.1:8123/v1/billing/remaining",
        timeout=5.0,
        webhook_url=None,
        state_file=str(state_file),
        artifact_dir=str(artifact_dir),
        alert_on_recovery=True,
    )

    assert summary.ok is True
    assert summary.severity == "degraded"

    stored_state = json.loads(state_file.read_text(encoding="utf-8"))
    assert stored_state["status"] == "degraded"
    assert stored_state["billing_error"] == "OPENAI_ADMIN_KEY not configured"

    latest_artifact = json.loads((artifact_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest_artifact["severity"] == "degraded"
    assert latest_artifact["billing_configured"] is False


@pytest.mark.anyio
async def test_run_marks_webhook_error_as_unhealthy_when_alert_delivery_fails(tmp_path, monkeypatch) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"status": "degraded"}), encoding="utf-8")

    async def fake_fetch_json(url: str, _timeout: float) -> healthcheck_job.ProbeResult:
        if url.endswith("/health"):
            return _probe(url=url, http_status=200, payload={"ok": True}, reason=None)
        return _probe(
            url=url,
            http_status=200,
            payload={
                "currency": "usd",
                "period_start": 1,
                "period_end": 2,
                "spent_usd": 3.0,
                "remaining_usd": 7.0,
                "configured": True,
                "error": None,
            },
            reason=None,
        )

    async def fake_send_webhook(_url: str, _summary: healthcheck_job.ServiceHealthSummary) -> None:
        raise healthcheck_job.httpx.ConnectError("boom")

    monkeypatch.setattr(healthcheck_job, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(healthcheck_job, "_send_webhook", fake_send_webhook)

    summary = await healthcheck_job._run(
        base_url="http://127.0.0.1:8123",
        health_url="http://127.0.0.1:8123/health",
        billing_url="http://127.0.0.1:8123/v1/billing/remaining",
        timeout=5.0,
        webhook_url="https://alerts.example.test/hook",
        state_file=str(state_file),
        artifact_dir=None,
        alert_on_recovery=True,
    )

    assert summary.ok is False
    assert summary.severity == "unhealthy"
    assert summary.reason.startswith("webhook_error:")
