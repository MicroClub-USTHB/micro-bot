import asyncio

import httpx
import pytest
from structlog.testing import capture_logs

from mcclub_rag.conversation import Turn
from mcclub_rag.ingest.errors import EmbeddingError, IndexUnavailable, RerankError
from mcclub_rag.schemas import Retrieval
from tests.pipeline_helpers import make_pipeline
from tests.retrieval_helpers import FakeReranker, make_retriever

RULES = """# Cotisation

La cotisation annuelle est de 1500 dinars, payable en septembre.

# Élections

Les élections du bureau ont lieu chaque année en juin, par vote des membres.

# Ateliers

Les ateliers de robotique ont lieu le samedi matin au local du club.
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
async def pipe(tmp_path):
    p = make_pipeline(tmp_path)
    await p.start()
    yield p
    await p.close()


@pytest.fixture
async def rules(pipe, tmp_path):
    return await pipe.ingest_file(_write(tmp_path, "rules.md", RULES), "rules.md", "public")


async def test_chunks_are_ordered_by_rerank_score_and_cut_to_top_n(pipe, rules):
    retriever = make_retriever(pipe, top_n=2)
    result = await retriever.retrieve("élections du bureau en juin", "public", [])
    assert isinstance(result, Retrieval) and len(result.chunks) == 2
    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert result.top_score == scores[0]
    best = result.chunks[0]
    assert "élections du bureau" in best.text.lower()
    assert best.doc_id == rules.doc_id and best.title
    assert not best.text.startswith(best.title)  # stored text, no header


async def test_chunk_id_is_the_point_id(pipe, rules):
    result = await make_retriever(pipe).retrieve("cotisation annuelle", "public", [])
    points, _ = await pipe.client.scroll(pipe.settings.qdrant_collection, limit=100)
    assert {c.chunk_id for c in result.chunks} <= {str(p.id) for p in points}


async def test_reranker_sees_the_header_and_text(pipe, rules):
    reranker = FakeReranker()
    await make_retriever(pipe, reranker).retrieve("cotisation annuelle", "public", [])
    query, passages = reranker.calls[0]
    assert query == "cotisation annuelle"
    assert any(p.startswith("Cotisation\n\n") for p in passages)


async def test_gate_refuses_below_threshold_but_keeps_chunks(pipe, rules):
    result = await make_retriever(pipe, gate_threshold=0.9).retrieve(
        "cotisation trésorier banque virement", "public", []
    )
    assert result.decision == "refuse" and result.chunks
    assert result.top_score < 0.9


async def test_gate_answers_at_or_above_threshold(pipe, rules):
    result = await make_retriever(pipe, gate_threshold=0.5).retrieve(
        "cotisation annuelle", "public", []
    )
    assert result.decision == "answer" and result.top_score >= 0.5


async def test_empty_query_refuses_without_any_call(pipe, rules, monkeypatch):
    async def boom(*args, **kwargs):
        raise AssertionError("Qdrant must not be queried")

    reranker = FakeReranker()
    retriever = make_retriever(pipe, reranker)
    monkeypatch.setattr(pipe.client, "query_points", boom)
    result = await retriever.retrieve("  \n ", "public", [])
    assert result == Retrieval(chunks=[], top_score=0.0, decision="refuse")
    assert reranker.calls == []


async def test_empty_collection_refuses(pipe):
    result = await make_retriever(pipe).retrieve("cotisation", "member", [])
    assert result == Retrieval(chunks=[], top_score=0.0, decision="refuse")


async def test_unknown_role_raises(pipe):
    with pytest.raises(ValueError, match="unknown role"):
        await make_retriever(pipe).retrieve("cotisation", "admin", [])


async def test_concurrent_requests_never_rerank_at_the_same_time(pipe, rules):
    reranker = FakeReranker(delay=0.05)
    retriever = make_retriever(pipe, reranker)
    await asyncio.gather(
        *(retriever.retrieve(q, "public", []) for q in ("cotisation", "élections", "ateliers"))
    )
    assert len(reranker.calls) == 3 and reranker.max_active == 1


async def test_short_follow_up_searches_with_the_previous_question(pipe, rules):
    reranker = FakeReranker()
    history = [
        Turn(user_message="Quand ont lieu les élections du bureau ?", assistant_message="En juin.")
    ]
    result = await make_retriever(pipe, reranker).retrieve("et le vote ?", "public", history)
    assert reranker.calls[0][0] == "Quand ont lieu les élections du bureau ?\net le vote ?"
    assert "élections" in result.chunks[0].text.lower()


async def test_short_new_question_is_searched_alone(pipe, rules):
    reranker = FakeReranker()
    history = [Turn(user_message="Quand ont lieu les élections ?", assistant_message="En juin.")]
    await make_retriever(pipe, reranker).retrieve("cotisation annuelle ?", "public", history)
    await make_retriever(pipe, reranker, followup_cues=False).retrieve(
        "cotisation annuelle ?", "public", history
    )
    assert [c[0] for c in reranker.calls] == [
        "cotisation annuelle ?",
        "Quand ont lieu les élections ?\ncotisation annuelle ?",
    ]


LONG = "# Statuts\n\n" + "\n\n".join(
    f"## Article {i}\n\narticle{i} " + f"mot{i} " * 25 for i in range(4)
)


async def test_neighbours_join_the_surrounding_chunks(pipe, tmp_path):
    await pipe.ingest_file(_write(tmp_path, "long.md", LONG), "long.md", "public")
    plain = await make_retriever(pipe, top_n=1).retrieve("article2", "public", [])
    wide = await make_retriever(pipe, top_n=1, neighbours=True).retrieve("article2", "public", [])
    hit, expanded = plain.chunks[0], wide.chunks[0]
    assert "article2" in hit.text and "article1" not in hit.text
    assert expanded.text.index("article1") < expanded.text.index("article2")
    assert expanded.text.index("article2") < expanded.text.index("article3")
    assert (expanded.chunk_id, expanded.score) == (hit.chunk_id, hit.score)
    assert wide.decision == plain.decision


async def test_one_log_event_without_the_query_text(pipe, rules):
    with capture_logs() as logs:
        await make_retriever(pipe).retrieve("cotisation annuelle", "public", [])
    (event,) = [e for e in logs if e["event"] == "retrieval_done"]
    assert event["role"] == "public" and event["decision"] == "answer"
    assert event["candidates"] >= event["returned"] >= 1
    assert event["history_used"] is False
    assert {"ms_encode", "ms_search", "ms_rerank", "duration_ms", "top_score"} <= set(event)
    assert not any("cotisation" in str(v) for v in event.values())


async def test_open_gate_is_logged_once_at_construction(pipe):
    with capture_logs() as logs:
        make_retriever(pipe)
        make_retriever(pipe, gate_threshold=0.3)
    assert [e["event"] for e in logs].count("retrieval_gate_open") == 1


async def test_encoder_failure_propagates(pipe, rules, monkeypatch):
    def boom(text):
        raise EmbeddingError("embedding model failed: test")

    monkeypatch.setattr(pipe.dense, "embed_query", boom)
    with pytest.raises(EmbeddingError):
        await make_retriever(pipe).retrieve("cotisation", "public", [])


async def test_reranker_failure_propagates(pipe, rules):
    reranker = FakeReranker(error=RerankError("reranker failed: Fail"))
    with pytest.raises(RerankError):
        await make_retriever(pipe, reranker).retrieve("cotisation", "public", [])


async def test_qdrant_failure_becomes_index_unavailable(pipe, rules, monkeypatch):
    async def down(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(pipe.client, "query_points", down)
    with pytest.raises(IndexUnavailable):
        await make_retriever(pipe).retrieve("cotisation", "public", [])
