# app/utils/rate_limit.py
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Tuple


class SlidingWindowRateLimiter:
    """
    Простой in-memory limiter: N запросов в минуту на ключ (обычно IP).
    Не годится для мультипроцесс/мультисервис без шаринга, но быстрый и стартует мгновенно.
    """

    def __init__(self, max_per_minute: int = 10, window_seconds: int = 60) -> None:
        self.max_per_minute = max_per_minute
        self.window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = {}

    def check_and_hit(self, key: str) -> Tuple[bool, int]:
        """
        Возвращает (allowed, remaining).
        При allowed=True происходит запись текущего хита.
        """
        now = time.time()
        wnd_start = now - self.window_seconds
        dq = self._hits.setdefault(key, deque())
        # почистить старые записи
        while dq and dq[0] < wnd_start:
            dq.popleft()
        if len(dq) >= self.max_per_minute:
            remaining = 0
            return False, remaining
        dq.append(now)
        remaining = max(0, self.max_per_minute - len(dq))
        return True, remaining
