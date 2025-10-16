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
        self._last_gc: float = 0.0
        # Чтобы не тратить время на постоянную очистку, выполняем её не чаще половины окна.
        self._gc_interval: float = max(1.0, window_seconds / 2.0)

    def _gc(self, *, now: float) -> None:
        """Удаляет ключи, для которых все события устарели."""

        if now - self._last_gc < self._gc_interval:
            return

        self._last_gc = now
        cutoff = now - self.window_seconds
        stale_keys = [key for key, hits in self._hits.items() if not hits or hits[-1] < cutoff]
        for key in stale_keys:
            self._hits.pop(key, None)

    def check_and_hit(self, key: str) -> Tuple[bool, int]:
        """
        Возвращает (allowed, remaining).
        При allowed=True происходит запись текущего хита.
        """
        now = time.time()
        self._gc(now=now)

        wnd_start = now - self.window_seconds
        dq = self._hits.get(key)
        if dq is None:
            dq = self._hits[key] = deque()
        # почистить старые записи
        while dq and dq[0] < wnd_start:
            dq.popleft()
        if len(dq) >= self.max_per_minute:
            remaining = 0
            return False, remaining
        dq.append(now)
        remaining = max(0, self.max_per_minute - len(dq))
        return True, remaining
