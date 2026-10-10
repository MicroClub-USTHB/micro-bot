"""Persistent daily quota and local failure circuit breaker for LLM generation."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - deployed target is Linux
    fcntl = None  # type: ignore[assignment]

Clock = Callable[[], datetime | float]


class DailyQuota:
    """Atomically reserve non-reserved daily generation capacity.

    ``acquire`` is the operation callers should use before a generation.  It
    increments the UTC-day counter while holding both thread and process locks,
    so concurrent workers cannot spend the same remaining slot.
    """

    def __init__(
        self,
        data_dir: Path,
        limit: int,
        reserve: int,
        clock: Clock | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.limit = limit
        self.reserve = reserve
        self.clock = clock or (lambda: datetime.now(UTC))
        self._thread_lock = Lock()
        self._state_path = self.data_dir / "llm_quota.json"
        self._lock_path = self.data_dir / "llm_quota.lock"

    def allow(self) -> bool:
        """Return whether a non-reserved slot is currently available.

        This is an observation only; callers that intend to make a request must
        use ``acquire`` so the limit check and reservation are atomic.
        """
        return self.used < self._usable_limit

    def acquire(self) -> bool:
        """Atomically reserve one generation slot outside the safety reserve."""
        with self._locked_state() as state:
            if state["count"] >= self._usable_limit:
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

    @property
    def used(self) -> int:
        with self._locked_state() as state:
            return state["count"]

    @property
    def _usable_limit(self) -> int:
        return max(0, self.limit - self.reserve)

    def _today(self) -> str:
        value = self.clock()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return value.astimezone(UTC).date().isoformat()
        return datetime.fromtimestamp(value, UTC).date().isoformat()

    def _default_state(self) -> dict[str, int | str]:
        return {"date": self._today(), "count": 0}

    def _read_state(self) -> dict[str, int | str]:
        default = self._default_state()
        try:
            raw: Any = json.loads(self._state_path.read_text(encoding="utf-8"))
            if raw.get("date") == default["date"] and isinstance(raw.get("count"), int):
                return {"date": default["date"], "count": max(0, raw["count"])}
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

