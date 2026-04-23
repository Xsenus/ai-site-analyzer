#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_DIR="${AI_SITE_ANALYZER_SYSTEMD_SOURCE_DIR:-$SCRIPT_DIR/systemd}"
TARGET_DIR="${AI_SITE_ANALYZER_SYSTEMD_TARGET_DIR:-/etc/systemd/system}"
UNITS_RAW="${AI_SITE_ANALYZER_SYSTEMD_UNITS:-ai-site-analyzer-healthcheck.service ai-site-analyzer-healthcheck.timer}"
ENV_TEMPLATE_SOURCE="${AI_SITE_ANALYZER_SYSTEMD_ENV_TEMPLATE_SOURCE:-$SCRIPT_DIR/systemd/ai-site-analyzer-monitoring.env.example}"
ENV_TEMPLATE_TARGET="${AI_SITE_ANALYZER_SYSTEMD_ENV_TEMPLATE_TARGET:-/etc/default/ai-site-analyzer-monitoring.example}"
ENV_FILE_TARGET="${AI_SITE_ANALYZER_SYSTEMD_ENV_FILE_TARGET:-/etc/default/ai-site-analyzer-monitoring}"
BOOTSTRAP_ENV_FILE="${AI_SITE_ANALYZER_SYSTEMD_BOOTSTRAP_ENV_FILE:-0}"
ENABLE_TIMERS="${AI_SITE_ANALYZER_SYSTEMD_ENABLE_TIMERS:-1}"
SKIP_SYSTEMCTL="${AI_SITE_ANALYZER_SYSTEMD_SKIP_SYSTEMCTL:-0}"

read -r -a UNITS <<< "$UNITS_RAW"
TIMERS=()
DRY_RUN=0

log() {
  printf '[ai-site-analyzer-systemd-install] %s\n' "$*" >&2
}

run() {
  log "+ $*"
  "$@"
}

is_true() {
  case "${1,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

usage() {
  cat >&2 <<'EOF'
Usage: bash deploy/install-ai-site-analyzer-systemd-units.sh [--dry-run]

Environment overrides:
  AI_SITE_ANALYZER_SYSTEMD_SOURCE_DIR
  AI_SITE_ANALYZER_SYSTEMD_TARGET_DIR
  AI_SITE_ANALYZER_SYSTEMD_UNITS
  AI_SITE_ANALYZER_SYSTEMD_ENV_TEMPLATE_SOURCE
  AI_SITE_ANALYZER_SYSTEMD_ENV_TEMPLATE_TARGET
  AI_SITE_ANALYZER_SYSTEMD_ENV_FILE_TARGET
  AI_SITE_ANALYZER_SYSTEMD_BOOTSTRAP_ENV_FILE=1|0
  AI_SITE_ANALYZER_SYSTEMD_ENABLE_TIMERS=1|0
  AI_SITE_ANALYZER_SYSTEMD_SKIP_SYSTEMCTL=1|0
EOF
}

if (( $# > 1 )); then
  usage
  exit 2
fi
if (( $# == 1 )); then
  if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    usage
    exit 2
  fi
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  log "systemd source directory is missing: $SOURCE_DIR"
  exit 2
fi
if [[ ! -f "$ENV_TEMPLATE_SOURCE" ]]; then
  log "monitoring env template is missing: $ENV_TEMPLATE_SOURCE"
  exit 2
fi
if (( ${#UNITS[@]} == 0 )); then
  log "no systemd units configured via AI_SITE_ANALYZER_SYSTEMD_UNITS"
  exit 2
fi

for unit in "${UNITS[@]}"; do
  if [[ ! -f "$SOURCE_DIR/$unit" ]]; then
    log "systemd unit is missing in source dir: $SOURCE_DIR/$unit"
    exit 2
  fi
  if [[ "$unit" == *.timer ]]; then
    TIMERS+=("$unit")
  fi
done

if (( DRY_RUN == 1 )); then
  log "dry-run mode: files and systemctl actions will not be changed"
fi
if is_true "$SKIP_SYSTEMCTL"; then
  log "systemctl operations are disabled via AI_SITE_ANALYZER_SYSTEMD_SKIP_SYSTEMCTL=1"
fi

if (( DRY_RUN == 0 )); then
  run install -d "$TARGET_DIR"
  for unit in "${UNITS[@]}"; do
    run install -m 0644 "$SOURCE_DIR/$unit" "$TARGET_DIR/$unit"
  done
  run install -d "$(dirname "$ENV_TEMPLATE_TARGET")"
  run install -m 0644 "$ENV_TEMPLATE_SOURCE" "$ENV_TEMPLATE_TARGET"
  if is_true "$BOOTSTRAP_ENV_FILE"; then
    if [[ -f "$ENV_FILE_TARGET" ]]; then
      log "skipping bootstrap of existing env file: $ENV_FILE_TARGET"
    else
      run install -d "$(dirname "$ENV_FILE_TARGET")"
      run install -m 0644 "$ENV_TEMPLATE_SOURCE" "$ENV_FILE_TARGET"
    fi
  fi
else
  for unit in "${UNITS[@]}"; do
    log "would install $SOURCE_DIR/$unit -> $TARGET_DIR/$unit"
  done
  log "would install monitoring env template $ENV_TEMPLATE_SOURCE -> $ENV_TEMPLATE_TARGET"
  if is_true "$BOOTSTRAP_ENV_FILE"; then
    log "would bootstrap monitoring env file $ENV_TEMPLATE_SOURCE -> $ENV_FILE_TARGET if missing"
  fi
fi

if is_true "$SKIP_SYSTEMCTL" || (( DRY_RUN == 1 )); then
  exit 0
fi

if ! command -v systemctl >/dev/null 2>&1; then
  log "systemctl is required unless AI_SITE_ANALYZER_SYSTEMD_SKIP_SYSTEMCTL=1 is set"
  exit 2
fi

run systemctl daemon-reload
if is_true "$ENABLE_TIMERS" && [[ ${#TIMERS[@]} -gt 0 ]]; then
  run systemctl enable --now "${TIMERS[@]}"
else
  log "timer enabling skipped"
fi
