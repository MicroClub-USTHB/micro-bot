"""In-memory per-IP sliding-window limiting for chat requests."""

from collections import defaultdict, deque
from collections.abc import Callable
from time import monotonic


class SlidingWindowRateLimiter:
    def __init__(
        self,
        requests: int,
        window_s: int,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.requests = requests
        self.window_s = window_s
        self._clock = clock
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_ip: str) -> bool:
        now = self._clock()
        window = self._requests[client_ip]
        cutoff = now - self.window_s
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= self.requests:
            return False
        window.append(now)
        return True
