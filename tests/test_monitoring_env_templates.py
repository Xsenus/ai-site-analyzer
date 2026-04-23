from __future__ import annotations

import pathlib
import re

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
MONITORING_SOURCE_PATHS = ("app/jobs/system_healthcheck.py",)
MONITORING_PREFIXES = ("AI_SITE_ANALYZER_HEALTHCHECK_",)


def _read_text(relative_path: str) -> str:
    return (ROOT_DIR / relative_path).read_text(encoding="utf-8")


def _parse_env_keys(relative_path: str) -> set[str]:
    pattern = re.compile(r"^\s*#?\s*([A-Z0-9_]+)\s*=", re.MULTILINE)
    return set(pattern.findall(_read_text(relative_path)))


def _extract_monitoring_env_keys() -> set[str]:
    pattern = re.compile(r'(?:os\.getenv|_env_bool)\("([A-Z0-9_]+)"')
    keys: set[str] = set()

    for relative_path in MONITORING_SOURCE_PATHS:
        keys.update(pattern.findall(_read_text(relative_path)))

    return {key for key in keys if key.startswith(MONITORING_PREFIXES)}


def test_monitoring_env_template_covers_analyzer_monitoring_job() -> None:
    expected = _extract_monitoring_env_keys()
    actual = _parse_env_keys("deploy/systemd/ai-site-analyzer-monitoring.env.example")

    missing = expected - actual
    assert not missing, f"missing analyzer monitoring env keys in template: {sorted(missing)}"


def test_monitoring_keys_are_documented_in_env_example() -> None:
    expected = _extract_monitoring_env_keys()
    actual = _parse_env_keys(".env.example")

    missing = expected - actual
    assert not missing, f"missing analyzer monitoring env keys in .env.example: {sorted(missing)}"


def test_monitoring_service_uses_expected_env_file() -> None:
    expected = "EnvironmentFile=-/etc/default/ai-site-analyzer-monitoring"
    service = _read_text("deploy/systemd/ai-site-analyzer-healthcheck.service")

    assert expected in service


def test_systemd_installer_tracks_env_template_locations() -> None:
    installer = _read_text("deploy/install-ai-site-analyzer-systemd-units.sh")

    assert "ai-site-analyzer-monitoring.env.example" in installer
    assert "/etc/default/ai-site-analyzer-monitoring.example" in installer
    assert "/etc/default/ai-site-analyzer-monitoring" in installer
    assert "AI_SITE_ANALYZER_SYSTEMD_BOOTSTRAP_ENV_FILE" in installer
