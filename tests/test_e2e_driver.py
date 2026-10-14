"""In-process and HTTP drivers on fakes: access, gate text, follow-ups, documents (Req 5, 8)."""

import httpx
import pytest

from eval.e2e.cases import Case, CorpusDoc
from eval.e2e.drivers import HttpDriver, Source
from eval.e2e.score import GATE_REFUSAL, score_chat
from mcclub_rag.messages import UNAVAILABLE
from tests.e2e_helpers import DOCS, make_driver


@pytest.fixture
async def driver(tmp_path):
    drv = await make_driver(tmp_path)
    yield drv
    await drv.close()
    await drv.pipe.close()


def _case(question, role="member", **kw):
    return Case(id="t", question=question, lang="en", role=role, expect="any", **kw)


async def test_member_reaches_members_document_public_does_not(driver):
    question = "When is the winter retreat in Tikjda?"
    member = await driver.chat(_case(question))
    assert member.status == 200 and member.body["sources"]
    public_case = _case(question, role="public", must_not=["CANARY-0001", "Tikjda"])
    public = await driver.chat(public_case)
    result = score_chat(public_case, public, driver.corpus, driver.topics)
    assert result.passed, result.failed()


async def test_gate_refusal_text_matches_the_app(driver):
    observation = await driver.chat(_case("Quantum chromodynamics lattice?", role="public"))
    assert observation.body["answer"] == GATE_REFUSAL
    assert observation.body["usage"]["llm_calls"] == 0


async def test_generator_failure_falls_back_with_one_call(driver):
    observation = await driver.chat(_case("Python workshop Saturday room", role="public"))
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

    monkeypatch.setenv("MEMBER_TOKEN", "m-token")
    drv = HttpDriver("http://app", requests_per_window=1000, window_s=1)
    drv.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://app")
    path = tmp_path / "a.md"
    path.write_text("# A\n\ntext")
    doc = CorpusDoc(file="a.md", topic="rules", title="Rules", visibility="public")
    assert await drv.setup([Source(doc, path)]) == {"d1": "rules"}
    assert b'name="topic"\r\n\r\neval-rules' in seen[0].content
    observation = await drv.chat(_case("q"))
    assert observation.status == 429  # recorded per case, not raised
    assert seen[-1].headers["Authorization"] == "Bearer m-token"
    await drv.close()


async def test_driver_leaves_the_environment_alone(driver):
    import os

    assert os.environ.get("MEMBER_TOKEN") != driver.member_token  # tokens live in the app settings
    response = await driver.request("GET", "/api/v1/documents", headers=driver.headers(admin=True))
    assert response.status_code == 200
