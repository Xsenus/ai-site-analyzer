from __future__ import annotations

import datetime as dt
from typing import Any, Dict


async def check_health() -> Dict[str, Any]:
    """Return a basic uptime payload for liveness probes."""

    return {
        "ok": True,
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
