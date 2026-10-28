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


def test_the_reserve_never_eats_the_whole_pool(tmp_path: Path, clock: FakeClock) -> None:
    quota = DailyQuota(tmp_path, limit=1, reserve=5, clock=clock)

    assert quota.acquire() is True  # a cap smaller than the reserve would disable the LLM


def test_exhaust_marks_the_day_spent_without_another_request(
    tmp_path: Path, clock: FakeClock
) -> None:
    quota = DailyQuota(tmp_path, limit=5, reserve=1, clock=clock)
    assert quota.acquire() is True

    assert quota.exhaust() == 4  # the usable limit, i.e. what the provider will refuse
    assert quota.allow() is False
    assert quota.remaining == 0
    assert DailyQuota(tmp_path, limit=5, reserve=1, clock=clock).allow() is False  # persisted


def test_a_token_budget_is_part_of_the_daily_quota(tmp_path: Path, clock: FakeClock) -> None:
    quota = DailyQuota(tmp_path, limit=100, reserve=0, clock=clock, token_limit=100)

    assert quota.record_tokens(60) == 60
    assert quota.allow() is True
    assert quota.record_tokens(40) == 100
    assert quota.allow() is False  # Groq's TPD binds before its 1000 RPD
    assert quota.acquire() is False


def test_gemini_resets_at_midnight_pacific_not_midnight_utc(
    tmp_path: Path, clock: FakeClock
) -> None:
    clock.value = datetime(2026, 10, 29, 6, 0, tzinfo=UTC)  # 23:00 in Los Angeles, still the 28th
    quota = DailyQuota(tmp_path, limit=2, reserve=0, clock=clock, reset_tz="America/Los_Angeles")
    assert quota.acquire() is True
    assert quota.used == 1

    clock.advance(timedelta(minutes=30))  # 06:30 UTC is past midnight UTC, not past midnight PT
    assert quota.used == 1

    clock.advance(timedelta(hours=2))  # 08:00 UTC is 01:00 PT: the provider's new day
    assert quota.used == 0


def test_next_reset_is_the_providers_own_midnight(tmp_path: Path, clock: FakeClock) -> None:
    clock.value = datetime(2026, 10, 29, 6, 0, tzinfo=UTC)
    quota = DailyQuota(tmp_path, limit=2, reserve=0, clock=clock, reset_tz="America/Los_Angeles")

    assert quota.next_reset() == datetime(2026, 10, 29, 7, 0, tzinfo=UTC).timestamp()


def test_reading_the_counter_does_not_write_it(tmp_path: Path, clock: FakeClock) -> None:
    quota = DailyQuota(tmp_path, limit=5, reserve=0, clock=clock)

    assert quota.allow() is True  # an observation, not a reservation
    assert quota.used == 0
    assert quota.remaining == 5
    assert not (tmp_path / "llm_quota.json").exists()  # no file, no lock file for a look
    assert not (tmp_path / "llm_quota.lock").exists()

    assert quota.acquire() is True
    assert (tmp_path / "llm_quota.json").exists()


def test_reading_after_a_write_sees_the_reserved_slot(tmp_path: Path, clock: FakeClock) -> None:
    quota = DailyQuota(tmp_path, limit=5, reserve=0, clock=clock)
    quota.acquire()

    assert quota.used == 1  # the atomic replace is what a reader picks up
    assert quota.remaining == 4
