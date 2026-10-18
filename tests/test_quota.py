from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from micro_bot.llm.quota import CircuitBreaker, DailyQuota


class FakeClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value += delta


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(datetime(2026, 10, 10, 23, 59, tzinfo=UTC))


def test_quota_reserves_the_last_calls_and_persists_utc_day_count(
    tmp_path: Path, clock: FakeClock
) -> None:
    quota = DailyQuota(tmp_path, limit=5, reserve=2, clock=clock)

    assert [quota.acquire() for _ in range(4)] == [True, True, True, False]
    assert quota.used == 3
    assert json.loads((tmp_path / "llm_quota.json").read_text()) == {
        "date": "2026-10-10",
        "count": 3,
    }

    reloaded = DailyQuota(tmp_path, limit=5, reserve=2, clock=clock)
    assert reloaded.allow() is False


def test_quota_resets_when_the_utc_day_changes(tmp_path: Path, clock: FakeClock) -> None:
    quota = DailyQuota(tmp_path, limit=3, reserve=1, clock=clock)
    assert quota.acquire() is True
    assert quota.acquire() is True
    assert quota.allow() is False

    clock.advance(timedelta(minutes=2))

    assert quota.used == 0
    assert quota.acquire() is True


def test_quota_record_rejects_negative_values(tmp_path: Path, clock: FakeClock) -> None:
    quota = DailyQuota(tmp_path, limit=5, reserve=0, clock=clock)

    with pytest.raises(ValueError, match="must not be negative"):
        quota.record(-1)


def test_breaker_opens_after_three_consecutive_failures_and_closes_after_cooldown() -> None:
    now = 1_000.0

    def clock() -> float:
        return now

    breaker = CircuitBreaker(clock=clock)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.allow() is True

    breaker.record_failure()
    assert breaker.allow() is False

    now += 59
    assert breaker.allow() is False
    now += 1
    assert breaker.allow() is True


def test_breaker_success_resets_consecutive_failures() -> None:
    breaker = CircuitBreaker(clock=lambda: 1_000.0)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()

    assert breaker.allow() is True


def test_acquire_keep_leaves_slots_for_the_answer(tmp_path: Path, clock: FakeClock) -> None:
    quota = DailyQuota(tmp_path, limit=3, reserve=0, clock=clock)
    assert quota.acquire(keep=1)
    assert not quota.acquire(keep=2)  # 1 used, 2 left: taking one would leave 1 < 2
    assert quota.acquire(keep=1)
    assert not quota.acquire(keep=1)
    assert quota.acquire()
