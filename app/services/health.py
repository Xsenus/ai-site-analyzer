from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict

from app.db.parsing import ping_parsing
from app.db.postgres import ping_postgres

log = logging.getLogger("services.health")


async def check_health() -> Dict[str, Any]:
    """Ping primary dependencies and return aggregated health information."""

    try:
        parsing_ok = await ping_parsing()
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("Health ping_parsing failed: %s", exc)
        parsing_ok = False

    try:
        postgres_ok = await ping_postgres()
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("Health ping_postgres failed: %s", exc)
        postgres_ok = False

    connections = {
        "parsing_data": parsing_ok,
        "postgres": postgres_ok,
    }
    ok = all(connections.values()) if connections else False

    return {
        "ok": ok,
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "connections": connections,
    }
