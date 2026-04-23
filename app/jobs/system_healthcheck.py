"""Standalone healthcheck for AI Site Analyzer service and billing chain.

Examples::

    python -m app.jobs.system_healthcheck
    python -m app.jobs.system_healthcheck --json
    python -m app.jobs.system_healthcheck --webhook-url https://example.local/webhook
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8123"
DEFAULT_TIMEOUT = 10.0
DEFAULT_STATE_FILE = "/var/lib/ai-site-analyzer/system-health-state.json"
DEFAULT_ARTIFACT_DIR = "/var/lib/ai-site-analyzer/system-health"


@dataclass(slots=True)
class ProbeResult:
    url: str
    http_status: int | None
    payload: Any
    reason: str | None


@dataclass(slots=True)
class ServiceHealthSummary:
    checked_at: str
    base_url: str
    health_url: str
    billing_url: str
    ok: bool
    severity: str
    reason: str
    health_http_status: int | None
    billing_http_status: int | None
    billing_configured: bool | None
    billing_error: str | None
    billing_currency: str | None
    billing_spent_usd: float | None
    billing_remaining_usd: float | None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _clean_url(value: str) -> str:
    return value.rstrip("/")


def _build_url(base_url: str, suffix: str, override: str | None) -> str:
    raw = (override or "").strip()
    if raw:
        return raw
    return f"{_clean_url(base_url)}{suffix}"


def _load_state(path: str) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: str, state: dict[str, Any]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_to_json(summary: ServiceHealthSummary) -> dict[str, Any]:
    return {
        "checked_at": summary.checked_at,
        "base_url": summary.base_url,
        "health_url": summary.health_url,
        "billing_url": summary.billing_url,
        "ok": summary.ok,
        "severity": summary.severity,
        "reason": summary.reason,
        "health_http_status": summary.health_http_status,
        "billing_http_status": summary.billing_http_status,
        "billing_configured": summary.billing_configured,
        "billing_error": summary.billing_error,
        "billing_currency": summary.billing_currency,
        "billing_spent_usd": summary.billing_spent_usd,
        "billing_remaining_usd": summary.billing_remaining_usd,
    }


def _write_artifacts(artifact_dir: str | None, summary: ServiceHealthSummary) -> None:
    if not artifact_dir:
        return

    target_dir = Path(artifact_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    payload = _summary_to_json(summary)
    timestamp = summary.checked_at.replace(":", "-")
    latest_path = target_dir / "latest.json"
    dated_path = target_dir / f"{timestamp}.json"

    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    dated_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _current_state_label(summary: ServiceHealthSummary) -> str:
    return summary.severity


def _should_send_alert(*, previous_status: str | None, current_status: str, alert_on_recovery: bool) -> bool:
    if current_status in {"degraded", "unhealthy"}:
        return previous_status != current_status
    if current_status == "ok" and alert_on_recovery:
        return previous_status in {"degraded", "unhealthy"}
    return False


def _build_alert_text(summary: ServiceHealthSummary) -> str:
    parts = [
        f"ai-site-analyzer status={summary.severity}",
        f"reason={summary.reason}",
    ]
    if summary.health_http_status is not None:
        parts.append(f"health_http_status={summary.health_http_status}")
    if summary.billing_http_status is not None:
        parts.append(f"billing_http_status={summary.billing_http_status}")
    if summary.billing_configured is not None:
        parts.append(f"billing_configured={summary.billing_configured}")
    if summary.billing_error:
        parts.append(f"billing_error={summary.billing_error}")
    return " | ".join(parts)


async def _fetch_json(url: str, timeout: float) -> ProbeResult:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        return ProbeResult(url=url, http_status=None, payload=None, reason=f"http_error:{exc.__class__.__name__}")

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if response.status_code >= 400:
        return ProbeResult(
            url=url,
            http_status=response.status_code,
            payload=payload,
            reason=f"http_status:{response.status_code}",
        )
    if payload is None:
        return ProbeResult(url=url, http_status=response.status_code, payload=None, reason="invalid_json")

    return ProbeResult(url=url, http_status=response.status_code, payload=payload, reason=None)


def _build_summary(*, base_url: str, health_probe: ProbeResult, billing_probe: ProbeResult) -> ServiceHealthSummary:
    checked_at = _now_iso()

    if health_probe.reason is not None:
        return ServiceHealthSummary(
            checked_at=checked_at,
            base_url=base_url,
            health_url=health_probe.url,
            billing_url=billing_probe.url,
            ok=False,
            severity="unhealthy",
            reason=f"health_{health_probe.reason}",
            health_http_status=health_probe.http_status,
            billing_http_status=billing_probe.http_status,
            billing_configured=None,
            billing_error=None,
            billing_currency=None,
            billing_spent_usd=None,
            billing_remaining_usd=None,
        )

    health_payload = health_probe.payload if isinstance(health_probe.payload, dict) else None
    if not health_payload or health_payload.get("ok") is not True:
        return ServiceHealthSummary(
            checked_at=checked_at,
            base_url=base_url,
            health_url=health_probe.url,
            billing_url=billing_probe.url,
            ok=False,
            severity="unhealthy",
            reason="health_reported_unhealthy",
            health_http_status=health_probe.http_status,
            billing_http_status=billing_probe.http_status,
            billing_configured=None,
            billing_error=None,
            billing_currency=None,
            billing_spent_usd=None,
            billing_remaining_usd=None,
        )

    if billing_probe.reason is not None:
        return ServiceHealthSummary(
            checked_at=checked_at,
            base_url=base_url,
            health_url=health_probe.url,
            billing_url=billing_probe.url,
            ok=False,
            severity="unhealthy",
            reason=f"billing_{billing_probe.reason}",
            health_http_status=health_probe.http_status,
            billing_http_status=billing_probe.http_status,
            billing_configured=None,
            billing_error=None,
            billing_currency=None,
            billing_spent_usd=None,
            billing_remaining_usd=None,
        )

    billing_payload = billing_probe.payload if isinstance(billing_probe.payload, dict) else None
    if billing_payload is None:
        return ServiceHealthSummary(
            checked_at=checked_at,
            base_url=base_url,
            health_url=health_probe.url,
            billing_url=billing_probe.url,
            ok=False,
            severity="unhealthy",
            reason="billing_invalid_payload",
            health_http_status=health_probe.http_status,
            billing_http_status=billing_probe.http_status,
            billing_configured=None,
            billing_error=None,
            billing_currency=None,
            billing_spent_usd=None,
            billing_remaining_usd=None,
        )

    configured = billing_payload.get("configured")
    billing_configured = configured if isinstance(configured, bool) else None
    error = billing_payload.get("error")
    billing_error = error if isinstance(error, str) and error.strip() else None
    billing_currency = billing_payload.get("currency") if isinstance(billing_payload.get("currency"), str) else None
    billing_spent_usd = _float_or_none(billing_payload.get("spent_usd"))
    billing_remaining_usd = _float_or_none(billing_payload.get("remaining_usd"))

    if billing_configured is False or billing_error is not None:
        return ServiceHealthSummary(
            checked_at=checked_at,
            base_url=base_url,
            health_url=health_probe.url,
            billing_url=billing_probe.url,
            ok=True,
            severity="degraded",
            reason="billing_degraded",
            health_http_status=health_probe.http_status,
            billing_http_status=billing_probe.http_status,
            billing_configured=billing_configured,
            billing_error=billing_error,
            billing_currency=billing_currency,
            billing_spent_usd=billing_spent_usd,
            billing_remaining_usd=billing_remaining_usd,
        )

    return ServiceHealthSummary(
        checked_at=checked_at,
        base_url=base_url,
        health_url=health_probe.url,
        billing_url=billing_probe.url,
        ok=True,
        severity="ok",
        reason="ok",
        health_http_status=health_probe.http_status,
        billing_http_status=billing_probe.http_status,
        billing_configured=billing_configured,
        billing_error=billing_error,
        billing_currency=billing_currency,
        billing_spent_usd=billing_spent_usd,
        billing_remaining_usd=billing_remaining_usd,
    )


async def _send_webhook(webhook_url: str, summary: ServiceHealthSummary) -> None:
    payload = {
        "text": _build_alert_text(summary),
        "status": _current_state_label(summary),
        "summary": _summary_to_json(summary),
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(webhook_url, json=payload)
        response.raise_for_status()


async def _run(
    *,
    base_url: str,
    health_url: str,
    billing_url: str,
    timeout: float,
    webhook_url: str | None,
    state_file: str,
    artifact_dir: str | None,
    alert_on_recovery: bool,
) -> ServiceHealthSummary:
    health_probe, billing_probe = await asyncio.gather(
        _fetch_json(health_url, timeout),
        _fetch_json(billing_url, timeout),
    )
    summary = _build_summary(base_url=base_url, health_probe=health_probe, billing_probe=billing_probe)

    previous_state = _load_state(state_file)
    previous_status = previous_state.get("status")
    current_status = _current_state_label(summary)
    webhook_error: str | None = None

    if webhook_url and _should_send_alert(
        previous_status=previous_status,
        current_status=current_status,
        alert_on_recovery=alert_on_recovery,
    ):
        try:
            await _send_webhook(webhook_url, summary)
        except httpx.HTTPError as exc:
            webhook_error = f"{exc.__class__.__name__}: {exc}"

    _save_state(
        state_file,
        {
            "status": current_status,
            "checked_at": summary.checked_at,
            "ok": summary.ok,
            "severity": summary.severity,
            "reason": summary.reason,
            "health_http_status": summary.health_http_status,
            "billing_http_status": summary.billing_http_status,
            "billing_configured": summary.billing_configured,
            "billing_error": summary.billing_error,
            "webhook_error": webhook_error,
        },
    )
    _write_artifacts(artifact_dir, summary)

    if webhook_error is not None and summary.ok:
        return ServiceHealthSummary(
            checked_at=summary.checked_at,
            base_url=summary.base_url,
            health_url=summary.health_url,
            billing_url=summary.billing_url,
            ok=False,
            severity="unhealthy",
            reason=f"webhook_error:{webhook_error}",
            health_http_status=summary.health_http_status,
            billing_http_status=summary.billing_http_status,
            billing_configured=summary.billing_configured,
            billing_error=summary.billing_error,
            billing_currency=summary.billing_currency,
            billing_spent_usd=summary.billing_spent_usd,
            billing_remaining_usd=summary.billing_remaining_usd,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcheck for AI Site Analyzer service chain")
    parser.add_argument(
        "--base-url",
        default=os.getenv("AI_SITE_ANALYZER_HEALTHCHECK_BASE_URL", DEFAULT_BASE_URL),
        help="Base URL for the service, used to derive health and billing URLs",
    )
    parser.add_argument(
        "--health-url",
        default=os.getenv("AI_SITE_ANALYZER_HEALTHCHECK_HEALTH_URL", ""),
        help="Override URL for /health",
    )
    parser.add_argument(
        "--billing-url",
        default=os.getenv("AI_SITE_ANALYZER_HEALTHCHECK_BILLING_URL", ""),
        help="Override URL for /v1/billing/remaining",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AI_SITE_ANALYZER_HEALTHCHECK_TIMEOUT", DEFAULT_TIMEOUT)),
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--state-file",
        default=os.getenv("AI_SITE_ANALYZER_HEALTHCHECK_STATE_FILE", DEFAULT_STATE_FILE),
        help="File used to persist previous health state",
    )
    parser.add_argument(
        "--artifact-dir",
        default=os.getenv("AI_SITE_ANALYZER_HEALTHCHECK_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR),
        help="Optional directory for latest.json and timestamped artifacts",
    )
    parser.add_argument(
        "--webhook-url",
        default=os.getenv("AI_SITE_ANALYZER_HEALTHCHECK_ALERT_WEBHOOK_URL", ""),
        help="Optional webhook URL for degraded/unhealthy transitions",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON",
    )
    parser.add_argument(
        "--alert-on-recovery",
        action="store_true",
        default=_env_bool("AI_SITE_ANALYZER_HEALTHCHECK_ALERT_ON_RECOVERY", True),
        help="Send a webhook when the status returns to ok",
    )

    args = parser.parse_args()
    base_url = _clean_url(args.base_url)
    health_url = _build_url(base_url, "/health", args.health_url)
    billing_url = _build_url(base_url, "/v1/billing/remaining", args.billing_url)

    summary = asyncio.run(
        _run(
            base_url=base_url,
            health_url=health_url,
            billing_url=billing_url,
            timeout=float(args.timeout),
            webhook_url=(args.webhook_url or "").strip() or None,
            state_file=args.state_file,
            artifact_dir=(args.artifact_dir or "").strip() or None,
            alert_on_recovery=bool(args.alert_on_recovery),
        )
    )

    payload = _summary_to_json(summary)
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(
            f"[{summary.severity}] reason={summary.reason} "
            f"health={summary.health_http_status} billing={summary.billing_http_status}\n"
        )

    if not summary.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
