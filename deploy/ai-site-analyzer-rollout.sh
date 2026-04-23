#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${APP_DIR:-/opt/ai-site-analyzer}"
ALLOWED_APP_DIR="${AI_SITE_ANALYZER_ROLLOUT_ALLOWED_APP_DIR:-/opt/ai-site-analyzer}"
HEALTH_URL="${AI_SITE_ANALYZER_ROLLOUT_HEALTH_URL:-http://127.0.0.1:8123/health}"
BILLING_URL="${AI_SITE_ANALYZER_ROLLOUT_BILLING_URL:-http://127.0.0.1:8123/v1/billing/remaining}"
HEALTH_TIMEOUT_SECONDS="${AI_SITE_ANALYZER_ROLLOUT_HEALTH_TIMEOUT_SECONDS:-60}"
HEALTH_RETRY_DELAY_SECONDS="${AI_SITE_ANALYZER_ROLLOUT_HEALTH_RETRY_DELAY_SECONDS:-2}"
SERVICES_RAW="${AI_SITE_ANALYZER_ROLLOUT_SERVICES:-ai-site-analyzer.service ai-site-analyzer-healthcheck.timer}"
INSTALL_SYSTEMD_MODE="${AI_SITE_ANALYZER_ROLLOUT_INSTALL_SYSTEMD:-auto}"
MONITORING_ENV_FILE="${AI_SITE_ANALYZER_ROLLOUT_MONITORING_ENV_FILE:-/etc/default/ai-site-analyzer-monitoring}"
PYTHON_BIN_OVERRIDE="${AI_SITE_ANALYZER_ROLLOUT_PYTHON:-}"
PYTEST_ARGS_RAW="${AI_SITE_ANALYZER_ROLLOUT_PYTEST_ARGS:-}"
SKIP_GIT_PULL="${AI_SITE_ANALYZER_ROLLOUT_SKIP_GIT_PULL:-0}"
SKIP_INSTALL="${AI_SITE_ANALYZER_ROLLOUT_SKIP_INSTALL:-0}"
SKIP_TESTS="${AI_SITE_ANALYZER_ROLLOUT_SKIP_TESTS:-0}"
SKIP_COMPILE="${AI_SITE_ANALYZER_ROLLOUT_SKIP_COMPILE:-0}"
SKIP_SMOKE="${AI_SITE_ANALYZER_ROLLOUT_SKIP_SMOKE:-0}"
SKIP_SERVICES="${AI_SITE_ANALYZER_ROLLOUT_SKIP_SERVICES:-0}"

read -r -a SERVICES <<< "$SERVICES_RAW"
read -r -a PYTEST_EXTRA_ARGS <<< "$PYTEST_ARGS_RAW"
MANAGED_SERVICES=()
services_stopped=0
HAS_GIT=0
PYTHON_BIN=""

log() {
  printf '[ai-site-analyzer-rollout] %s\n' "$*" >&2
}

run() {
  log "+ $*"
  "$@"
}

resolve_python_bin() {
  if [[ -n "$PYTHON_BIN_OVERRIDE" ]]; then
    PYTHON_BIN="$PYTHON_BIN_OVERRIDE"
    return
  fi

  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
    return
  fi
  if [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
    return
  fi

  log "python executable not found in .venv; set AI_SITE_ANALYZER_ROLLOUT_PYTHON explicitly"
  exit 2
}

resolve_existing_services() {
  MANAGED_SERVICES=()

  if [[ "$SKIP_SERVICES" == "1" ]]; then
    return
  fi
  if (( ${#SERVICES[@]} == 0 )); then
    return
  fi
  if ! command -v systemctl >/dev/null 2>&1; then
    return
  fi

  local unit
  for unit in "${SERVICES[@]}"; do
    if systemctl cat "$unit" >/dev/null 2>&1; then
      MANAGED_SERVICES+=("$unit")
    else
      log "skipping missing systemd unit: $unit"
    fi
  done
}

start_services_best_effort() {
  if (( ${#MANAGED_SERVICES[@]} == 0 )); then
    return
  fi
  if command -v systemctl >/dev/null 2>&1; then
    systemctl start "${MANAGED_SERVICES[@]}" || true
  fi
}

install_systemd_units_if_needed() {
  local installer_script="deploy/install-ai-site-analyzer-systemd-units.sh"

  case "$INSTALL_SYSTEMD_MODE" in
    never)
      log "skipping systemd unit install because AI_SITE_ANALYZER_ROLLOUT_INSTALL_SYSTEMD=never"
      return 0
      ;;
    auto)
      if ! command -v systemctl >/dev/null 2>&1; then
        log "skipping systemd unit install because systemctl is not available"
        return 0
      fi
      if [[ ! -f "$installer_script" ]]; then
        log "skipping systemd unit install because installer script is missing: $installer_script"
        return 0
      fi
      if [[ "${EUID:-$(id -u)}" != "0" ]]; then
        log "skipping systemd unit install because rollout is not running as root"
        return 0
      fi
      ;;
    always)
      if ! command -v systemctl >/dev/null 2>&1; then
        log "systemd unit install requires systemctl, but it is not available"
        return 2
      fi
      if [[ ! -f "$installer_script" ]]; then
        log "systemd unit installer script is missing: $installer_script"
        return 2
      fi
      if [[ "${EUID:-$(id -u)}" != "0" ]]; then
        log "systemd unit install requires root privileges"
        return 2
      fi
      ;;
    *)
      log "unsupported AI_SITE_ANALYZER_ROLLOUT_INSTALL_SYSTEMD=$INSTALL_SYSTEMD_MODE (expected: auto|always|never)"
      return 2
      ;;
  esac

  run env AI_SITE_ANALYZER_SYSTEMD_ENABLE_TIMERS=0 bash "$installer_script"
}

cleanup() {
  local exit_code=$?
  if (( exit_code != 0 && services_stopped == 1 )); then
    log "rollout failed, starting services back best-effort"
    start_services_best_effort
  fi
  exit "$exit_code"
}
trap cleanup EXIT

wait_for_json_ok() {
  local url="$1"
  local timeout_seconds="$2"
  local retry_delay_seconds="$3"
  local label="$4"
  local attempt=1
  local deadline=$((SECONDS + timeout_seconds))

  while true; do
    log "+ wait_for_json_ok $label $url (attempt $attempt)"
    if "$PYTHON_BIN" - "$url" <<'PY'
from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.load(response)
except (urllib.error.URLError, TimeoutError, ValueError):
    raise SystemExit(1)
if not isinstance(payload, dict) or payload.get("ok") is not True:
    raise SystemExit(1)
PY
    then
      return 0
    fi

    if (( SECONDS >= deadline )); then
      log "$label did not become healthy within ${timeout_seconds}s: $url"
      return 1
    fi

    sleep "$retry_delay_seconds"
    attempt=$((attempt + 1))
  done
}

wait_for_billing_payload() {
  local url="$1"
  local timeout_seconds="$2"
  local retry_delay_seconds="$3"
  local attempt=1
  local deadline=$((SECONDS + timeout_seconds))

  while true; do
    log "+ wait_for_billing_payload $url (attempt $attempt)"
    if "$PYTHON_BIN" - "$url" <<'PY'
from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.load(response)
except (urllib.error.URLError, TimeoutError, ValueError):
    raise SystemExit(1)
if not isinstance(payload, dict):
    raise SystemExit(1)
for required_key in ("currency", "period_start", "period_end", "configured"):
    if required_key not in payload:
        raise SystemExit(1)
PY
    then
      return 0
    fi

    if (( SECONDS >= deadline )); then
      log "billing payload did not become available within ${timeout_seconds}s: $url"
      return 1
    fi

    sleep "$retry_delay_seconds"
    attempt=$((attempt + 1))
  done
}

cd "$APP_DIR"
APP_REALPATH="$(pwd -P)"

if [[ "$APP_REALPATH" != "$ALLOWED_APP_DIR" ]]; then
  log "refusing to run outside $ALLOWED_APP_DIR, got $APP_REALPATH"
  exit 2
fi

if [[ ! -d app || ! -f requirements.txt || ! -f README.md ]]; then
  log "app/, requirements.txt, and README.md are required in $APP_REALPATH"
  exit 2
fi

if [[ -f "$MONITORING_ENV_FILE" ]]; then
  log "loading monitoring env from $MONITORING_ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$MONITORING_ENV_FILE"
  set +a
fi

if [[ -d .git ]]; then
  HAS_GIT=1
fi

resolve_python_bin

log "starting rollout in $APP_REALPATH"
if [[ "$HAS_GIT" == "1" ]]; then
  run git status --short
else
  log "non-git layout detected, git pull/status will be skipped"
fi

if [[ "$HAS_GIT" == "1" && "$SKIP_GIT_PULL" != "1" ]]; then
  run git pull --ff-only origin main
fi

install_systemd_units_if_needed
resolve_existing_services

if (( ${#MANAGED_SERVICES[@]} > 0 )) && command -v systemctl >/dev/null 2>&1; then
  run systemctl stop "${MANAGED_SERVICES[@]}"
  services_stopped=1
fi

run "$PYTHON_BIN" --version
run "$PYTHON_BIN" -m pip --version

if [[ "$SKIP_INSTALL" != "1" ]]; then
  run "$PYTHON_BIN" -m pip install -r requirements.txt
  if [[ "$SKIP_TESTS" != "1" && -f requirements-dev.txt ]]; then
    run "$PYTHON_BIN" -m pip install -r requirements-dev.txt
  fi
fi

if [[ "$SKIP_TESTS" != "1" ]]; then
  run "$PYTHON_BIN" -m pytest "${PYTEST_EXTRA_ARGS[@]}"
fi

if [[ "$SKIP_COMPILE" != "1" ]]; then
  run "$PYTHON_BIN" -m compileall app
fi

if (( ${#MANAGED_SERVICES[@]} > 0 )) && command -v systemctl >/dev/null 2>&1; then
  run systemctl start "${MANAGED_SERVICES[@]}"
  services_stopped=0
  run systemctl is-active "${MANAGED_SERVICES[@]}"
fi

if [[ "$SKIP_SMOKE" != "1" ]]; then
  wait_for_json_ok "$HEALTH_URL" "$HEALTH_TIMEOUT_SECONDS" "$HEALTH_RETRY_DELAY_SECONDS" "service health"
  wait_for_billing_payload "$BILLING_URL" "$HEALTH_TIMEOUT_SECONDS" "$HEALTH_RETRY_DELAY_SECONDS"
  run "$PYTHON_BIN" -m app.jobs.system_healthcheck --json
fi
