import pytest

from app.utils.rate_limit import SlidingWindowRateLimiter


class _TimeStub:
    def __init__(self, start: float) -> None:
        self.value = start

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def __call__(self) -> float:
        return self.value


@pytest.mark.parametrize("window", [1, 10])
def test_rate_limiter_removes_stale_keys(monkeypatch, window: int) -> None:
    limiter = SlidingWindowRateLimiter(max_per_minute=1, window_seconds=window)

    clock = _TimeStub(1_000.0)
    monkeypatch.setattr("app.utils.rate_limit.time.time", clock)

    assert limiter.check_and_hit("ip-1") == (True, 0)
    assert "ip-1" in limiter._hits

    clock.advance(window * 2)

    # Второй ключ инициирует GC и сам записывается.
    assert limiter.check_and_hit("ip-2") == (True, 0)
    assert "ip-2" in limiter._hits
    assert "ip-1" not in limiter._hits


def test_rate_limiter_reuses_queue_after_cleanup(monkeypatch) -> None:
    limiter = SlidingWindowRateLimiter(max_per_minute=2, window_seconds=5)

    clock = _TimeStub(50.0)
    monkeypatch.setattr("app.utils.rate_limit.time.time", clock)

    assert limiter.check_and_hit("ip-1") == (True, 1)
    clock.advance(10.0)

    # Все события по ip-1 устарели, GC удаляет ключ и создаёт заново
    assert limiter.check_and_hit("ip-1") == (True, 1)
    assert len(limiter._hits["ip-1"]) == 1
