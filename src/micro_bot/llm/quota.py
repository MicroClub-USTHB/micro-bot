"""Persistent daily quota and local failure circuit breaker for LLM generation."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    import fcntl
except ImportError:  # pragma: no cover - deployed target is Linux
    fcntl = None  # type: ignore[assignment]

Clock = Callable[[], datetime | float]
_ONE_DAY = timedelta(days=1)


def _zone(name: str) -> ZoneInfo | UTC:
    """The provider's reset zone; UTC when tzdata is missing (a container without it)."""
    if name.upper() == "UTC":
        return UTC
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return UTC


class DailyQuota:
    """Atomically reserve non-reserved daily generation capacity.

    ``acquire`` is the operation callers should use before a generation.  It
    increments the UTC-day counter while holding both thread and process locks,
    so concurrent workers cannot spend the same remaining slot.

    ``reset_tz`` is the zone the provider resets its own counter in (Google: Pacific), so a
    stale day is never mistaken for a fresh one.  ``token_limit`` tracks the day's tokens for
    the providers whose binding limit is a token budget (Groq's TPD).
    """

    def __init__(
        self,
        data_dir: Path,
        limit: int,
        reserve: int,
        clock: Clock | None = None,
        *,
        reset_tz: str = "UTC",
        token_limit: int = 0,
    ) -> None:
        self.data_dir = data_dir
        self.limit = limit
        self.reserve = reserve
        self.token_limit = token_limit
        self.clock = clock or (lambda: datetime.now(UTC))
        self.reset_tz = _zone(reset_tz)
        self._thread_lock = Lock()
        self._state_path = self.data_dir / "llm_quota.json"
        self._lock_path = self.data_dir / "llm_quota.lock"

    def allow(self) -> bool:
        """Return whether a non-reserved slot is currently available.

        This is an observation only; callers that intend to make a request must
        use ``acquire`` so the limit check and reservation are atomic.
        """
        return self.used < self._usable_limit and not self.tokens_spent

    def acquire(self, keep: int = 0) -> bool:
        """Atomically reserve one generation slot outside the safety reserve.

        ``keep`` slots must stay free after this one: the query rewrite passes 1, so it never
        takes the last slot from the answer call.
        """
        with self._locked_state() as state:
            if state["count"] + keep >= self._usable_limit:
                return False
            if self._tokens_spent(state):
                return False
            state["count"] += 1
            return True

    def record(self, calls: int = 1) -> int:
        """Record already-authorized calls and return the current UTC-day count."""
        if calls < 0:
            raise ValueError("calls must not be negative")
        with self._locked_state() as state:
            state["count"] += calls
            return state["count"]

    def exhaust(self) -> int:
        """Mark the day's pool as spent after the provider said so (a daily-quota 429).

        Writing the limit instead of the observed count means a restart, or a second worker,
        agrees with the provider without burning another request to find out.
        """
        with self._locked_state() as state:
            state["count"] = max(state["count"], self._usable_limit)
            return state["count"]

    def record_tokens(self, tokens: int) -> int:
        """Add a completed call's tokens to the day's counter."""
        if tokens < 0:
            raise ValueError("tokens must not be negative")
        with self._locked_state() as state:
            state["tokens"] = state.get("tokens", 0) + tokens
            return state["tokens"]

    @property
    def used(self) -> int:
        with self._locked_state() as state:
            return state["count"]

    @property
    def tokens(self) -> int:
        """The day's tokens, 0 when the provider has no token budget."""
        with self._locked_state() as state:
            return state.get("tokens", 0)

    @property
    def tokens_spent(self) -> bool:
        """Whether the day's token budget is used up (always False without one)."""
        return self.token_limit > 0 and self.tokens >= self.token_limit

    @property
    def remaining(self) -> int:
        """Slots left today before the reserve."""
        return max(0, self._usable_limit - self.used)

    def next_reset(self) -> float:
        """Epoch seconds of the provider's next daily reset, in its own zone."""
        moment = self._moment()
        tomorrow = (moment + _ONE_DAY).date()
        return datetime.combine(tomorrow, datetime.min.time(), self.reset_tz).timestamp()

    def _tokens_spent(self, state: dict[str, int | str]) -> bool:
        return self.token_limit > 0 and int(state.get("tokens", 0) or 0) >= self.token_limit

    @property
    def _usable_limit(self) -> int:
        # The reserve can never eat the whole pool: a cap smaller than the reserve would leave
        # no call at all, which silently disables the LLM instead of holding a margin back.
        reserve = min(self.reserve, max(0, self.limit - 1))
        return max(0, self.limit - reserve)

    def _moment(self) -> datetime:
        value = self.clock()
        if isinstance(value, datetime):
            moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        else:
            moment = datetime.fromtimestamp(value, UTC)
        return moment.astimezone(self.reset_tz)

    def _today(self) -> str:
        return self._moment().date().isoformat()

    def _default_state(self) -> dict[str, int | str]:
        state: dict[str, int | str] = {"date": self._today(), "count": 0}
        if self.token_limit:
            state["tokens"] = 0  # only providers with a token budget carry the field
        return state

    def _read_state(self) -> dict[str, int | str]:
        default = self._default_state()
        try:
            raw: Any = json.loads(self._state_path.read_text(encoding="utf-8"))
            if raw.get("date") == default["date"] and isinstance(raw.get("count"), int):
                state: dict[str, int | str] = {
                    "date": default["date"],
                    "count": max(0, raw["count"]),
                }
                if self.token_limit:
                    tokens = raw.get("tokens", 0)
                    state["tokens"] = max(0, tokens if isinstance(tokens, int) else 0)
                return state
        except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
            pass
        return default

    def _write_state(self, state: dict[str, int | str]) -> None:
        fd, temporary_path = tempfile.mkstemp(prefix=".llm_quota-", dir=self.data_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
                json.dump(state, temporary_file, separators=(",", ":"))
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._state_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, int | str]]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            with self._lock_path.open("a+") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                state = self._read_state()
                try:
                    yield state
                finally:
                    self._write_state(state)
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class CircuitBreaker:
    """Open for a cooldown after consecutive failed generation attempts."""

    def __init__(
        self,
        clock: Clock | None = None,
        failure_limit: int = 3,
        cooldown_s: int = 60,
    ) -> None:
        self.clock = clock or (lambda: datetime.now(UTC))
        self.failure_limit = failure_limit
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_until: float | None = None
        self._lock = Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_until is None:
                return True
            if self._now() < self.opened_until:
                return False
            self.failures = 0
            self.opened_until = None
            return True

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_until = None

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failure_limit:
                self.opened_until = self._now() + self.cooldown_s

    def _now(self) -> float:
        value = self.clock()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return value.timestamp()
        return value
