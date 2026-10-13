"""Suites on the fake in-process app: contract, faults, cache, quality, robustness (Req 6, 7)."""

import pytest

from eval.e2e import report, system
from eval.e2e.cases import Case
from eval.e2e.contract import contract
from eval.e2e.drivers import Source
from eval.e2e.generator import Budget, CitingStub, RecordingGenerator
from eval.e2e.suites import Context, quality, robustness
from tests.e2e_helpers import make_driver

PROBE = Case(
    id="p",
    question="Python workshop Saturday room",
    lang="en",
    role="public",
    expect="answer",
    docs=["club-notice"],
    evidence=["every Saturday"],
)


@pytest.fixture
async def driver(tmp_path):
    drv = await make_driver(tmp_path, generator=CitingStub())
    yield drv
    await drv.close()
    await drv.pipe.close()


def _failed(results):
    return {r.id: r.failed() for r in results if not r.passed}


async def test_contract_suite_passes_on_the_app(driver):
    assert _failed(await contract(driver)) == {}


async def test_fault_suite_passes_and_restores_the_state(driver):
    before = driver.state.generator
    assert _failed(await system.faults(driver, PROBE)) == {}
    assert driver.state.generator is before


async def test_cache_suite(driver, tmp_path):
    path = tmp_path / "extra.md"
    path.write_text("# Extra\n\nA new document.")
    source = Source(driver.corpus["club-notice"].model_copy(update={"topic": "x"}), path)
    results = await system.cache(driver, PROBE, lambda: driver.upload(source, "eval-bump"))
    assert _failed(results) == {}


async def test_quality_notes_when_generation_was_not_evaluated(driver):
    driver.state.generator = system.FaultyGenerator("timeout")
    ctx = Context(driver, driver.corpus, driver.topics, Budget(10))
    results = await quality(ctx, [PROBE], "chat")
    assert results[0].path == "llm" and "NOT evaluated" not in " ".join(ctx.notes)
    driver.state.generator = CitingStub()
    retrieval = await quality(ctx, [PROBE], "retrieval")
    assert retrieval[0].passed and retrieval[0].layer == "retrieval"


async def test_budget_spent_marks_not_run(driver, tmp_path):
    budget = Budget(0)
    driver.state.generator = RecordingGenerator(CitingStub(), tmp_path, "live", budget, "m")
    ctx = Context(driver, driver.corpus, driver.topics, budget)
    (result,) = await quality(ctx, [PROBE], "chat")
    assert result.path == "not_run"  # an extractive fallback is not scored as an answer


async def test_robustness_reports_variants_and_repeats(driver):
    ctx = Context(driver, driver.corpus, driver.topics)
    results, variants = await robustness(ctx, [PROBE], repeat=2)
    cases = {c.id: c for c in [PROBE, *variants]}
    summary = report.summarize(results, cases)
    assert summary["metrics"]["instability"] == 0.0
    assert summary["metrics"]["flip_rate"] is not None
    assert {r.id for r in results} == set(cases)


def test_stage_times_from_retrieval_events():
    events = [
        {"event": "retrieval_done", "ms_encode": e, "ms_search": 1, "ms_rerank": 10 * e}
        for e in (1, 2, 3, 4)
    ] + [{"event": "other", "ms_encode": 99}]
    stages = system.stage_times(events)
    assert stages["ms_encode"] == {"n": 4, "p50": 2.5, "p95": 4}
    assert stages["ms_rerank"]["p95"] == 40
    assert system.stage_times([]) == {}


async def test_load_reports_stage_times(driver):
    (result,), stats = await system.load(driver, [PROBE], concurrency=2, duration_s=0.3)
    assert result.passed and stats["requests"] >= 2
    assert set(stats["stages"]) == {"ms_encode", "ms_search", "ms_rerank"}
    assert stats["stages"]["ms_rerank"]["n"] >= 1
