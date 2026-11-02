"""Shared HTTP helpers for MODEL-1's provider clients: a simple sliding-
window rate limiter and a retry-on-429 wrapper. Independent implementation
(no import from the estate)."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > self.window_seconds:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            sleep_for = self.window_seconds - (now - self._calls[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._calls.append(time.monotonic())


class SourceUnavailableError(Exception):
    """Retryable: HTTP 429 (rate limited) or transient network error."""


def retry_with_backoff[T](
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    retryable: tuple[type[BaseException], ...] = (SourceUnavailableError,),
) -> T:
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except retryable as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2**attempt))
    assert last_exc is not None
    raise last_exc
