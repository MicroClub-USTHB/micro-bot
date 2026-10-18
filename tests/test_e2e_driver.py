"""In-process and HTTP drivers on fakes: gate text, follow-ups, documents (Req 5, 8)."""

import asyncio
from collections import defaultdict

import httpx
import pytest

from eval.e2e.cases import Case, CorpusDoc
from eval.e2e.drivers import HttpDriver, Source
from eval.e2e.score import GATE_REFUSAL
from micro_bot.messages import UNAVAILABLE
from tests.e2e_helpers import DOCS, make_driver


@pytest.fixture
async def driver(tmp_path):
    drv = await make_driver(tmp_path)
    yield drv
    await drv.close()
    await drv.pipe.close()


def _case(question, **kw):
    return Case(id="t", question=question, lang="en", expect="any", **kw)


async def test_chat_needs_the_api_key(driver):
    body = {"message": "When is the winter retreat in Tikjda?"}
    denied = await driver.request("POST", "/api/v1/chat", json=body)
    assert denied.status_code == 401
    observation = await driver.chat(_case(body["message"]))
    assert observation.status == 200 and observation.body["sources"]


async def test_gate_refusal_text_matches_the_app(driver):
    observation = await driver.chat(_case("Quantum chromodynamics lattice?"))
    assert observation.body["answer"] == GATE_REFUSAL
    assert observation.body["usage"]["llm_calls"] == 0


async def test_generator_failure_falls_back_with_one_call(driver):
    observation = await driver.chat(_case("Python workshop Saturday room"))
    assert observation.body["usage"]["llm_calls"] == 1
    body = observation.body  # fallback: a notice, the passage in the sources, not grounded
    assert body["answer"] == UNAVAILABLE["en"] and body["grounded"] is False
    assert "10:00" in body["sources"][0]["snippet"]


async def test_followup_history_reaches_retrieval(driver):
    history = [{"user_message": "When is the Python workshop?", "assistant_message": "10:00."}]
    await driver.chat(_case("And where?", history=history))
    assert [t.user_message for t in driver.retriever.histories[-1]] == [history[0]["user_message"]]


async def test_documents_api_over_the_pipeline(driver, tmp_path):
    admin = driver.headers(admin=True)
    listed = (await driver.request("GET", "/api/v1/documents", headers=admin)).json()
    assert {d["topic"] for d in listed} == set(DOCS)
    doc_id = next(iter(driver.topics))
    deleted = await driver.request("DELETE", f"/api/v1/documents/{doc_id}", headers=admin)
    assert deleted.json() == {"ok": True}
    assert (await driver.request("GET", "/api/v1/documents")).status_code == 403


async def test_http_driver_paces_and_uploads_with_eval_topics(tmp_path, monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/v1/documents":
            return httpx.Response(200, json={"doc_id": "d1"})
        return httpx.Response(429, json={"error": {"code": "rate_limited", "message": "slow"}})

    monkeypatch.setenv("API_KEY", "m-token")
    drv = HttpDriver("http://app", requests_per_window=1000, window_s=1)
    drv.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://app")
    path = tmp_path / "a.md"
    path.write_text("# A\n\ntext")
    doc = CorpusDoc(file="a.md", topic="rules", title="Rules")
    assert await drv.setup([Source(doc, path)]) == {"d1": "rules"}
    assert b'name="topic"\r\n\r\neval-rules' in seen[0].content
    observation = await drv.chat(_case("q"))
    assert observation.status == 429  # recorded per case, not raised
    assert seen[-1].headers["Authorization"] == "Bearer m-token"
    await drv.close()


def _fast_driver(**kwargs) -> HttpDriver:
    """An HttpDriver on a mock transport that answers instantly, with timestamps kept."""
    driver = HttpDriver("http://app", **kwargs)
    driver.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
        base_url="http://app",
    )
    return driver


async def test_http_driver_default_rate_is_twenty_per_minute():
    drv = _fast_driver()
    assert drv.interval == 3.0  # 60 s / 20, the app's per-IP limit (Req 6.1)
    await drv.close()


async def test_http_driver_applies_a_custom_rate():
    drv = _fast_driver(requests_per_window=240)
    assert drv.interval == pytest.approx(0.25)
    await drv.close()


@pytest.mark.parametrize("value", ["0", "-5"])
def test_requests_per_minute_must_be_positive(value):
    from eval.e2e.run import main

    with pytest.raises(SystemExit):  # argparse usage error, before anything runs
        main(["--target", "http", "--requests-per-minute", value])


async def test_pacing_decisions_are_serialized():
    # The requirement is that two load tasks cannot reserve the same slot, so test the mechanism
    # rather than the clock: while the pacing lock is held, _wait cannot decide its slot.
    drv = _fast_driver(requests_per_window=1, window_s=0.05)
    await drv._pace.acquire()
    task = asyncio.create_task(drv._wait())
    await asyncio.sleep(0)  # one scheduling slice: the task has started and is blocked
    assert not task.done(), "_wait ran without holding the pacing lock"
    drv._pace.release()
    await asyncio.wait_for(task, timeout=1)
    await drv.close()


async def test_pacing_never_has_two_tasks_waiting_for_the_same_slot(monkeypatch):
    # The defect the lock fixes: without it several tasks read the same _next, sleep until it and
    # wake together, so a burst of requests leaves in the same instant. Measured time gaps cannot
    # show this (time.monotonic ticks every ~15.6 ms on Windows), so count the tasks that are
    # sleeping on a slot at the same time: one with the lock, up to seven without it.
    real_sleep = asyncio.sleep
    waiting: dict[str, int] = defaultdict(int)
    peak = 0

    async def counting_sleep(_seconds: float) -> None:
        nonlocal peak
        name = asyncio.current_task().get_name()  # type: ignore[union-attr]
        waiting[name] += 1
        peak = max(peak, sum(waiting.values()))
        await real_sleep(0)
        waiting[name] -= 1

    monkeypatch.setattr(asyncio, "sleep", counting_sleep)
    drv = _fast_driver(requests_per_window=200)
    await asyncio.gather(*(drv._wait() for _ in range(8)))
    assert peak == 1, f"{peak} tasks were waiting for a pacing slot at once"
    await drv.close()


async def test_driver_leaves_the_environment_alone(driver):
    import os

    assert os.environ.get("API_KEY") != driver.api_key  # tokens live in the app settings
    response = await driver.request("GET", "/api/v1/documents", headers=driver.headers(admin=True))
    assert response.status_code == 200
