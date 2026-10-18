from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from micro_bot.ingest.errors import IndexUnavailable
from micro_bot.retrieval.search import Candidate, HybridSearch, dedupe, live_filter
from micro_bot.retrieval.settings import RetrievalSettings
from tests.pipeline_helpers import make_pipeline

T0 = datetime(2026, 10, 10, tzinfo=UTC)


def cand(doc="d", index=0, rank=0, day=0, **kw) -> Candidate:
    fields = dict(
        point_id=f"{doc}-{index}-{rank}",
        doc_id=doc,
        content_hash="h",
        index=index,
        title="T",
        heading_path=(),
        text="x",
        ingested_at=T0 + timedelta(days=day),
        fused_score=1.0 / (rank + 1),
        fused_rank=rank,
    )
    return Candidate(**(fields | kw))


class KeywordDense:
    """Dense fake: cosine grows with the count of KEYWORD, so ranking is controllable."""

    dim = 8
    fingerprint = "keyword-dense"
    KEYWORD = "cotisation"

    def _vector(self, text):
        n = text.lower().count(self.KEYWORD)
        values = [float(n), 1.0] + [0.0] * (self.dim - 2)
        norm = sum(v * v for v in values) ** 0.5
        return [v / norm for v in values]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return [1.0] + [0.0] * (self.dim - 1)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


async def _ingest(pipe, tmp_path, name, text):
    return await pipe.ingest_file(_write(tmp_path, name, text), name)


@pytest.fixture
async def pipe(tmp_path):
    p = make_pipeline(tmp_path, dense=KeywordDense())
    await p.start()
    yield p
    await p.close()


def searcher(pipe, **settings) -> HybridSearch:
    s = RetrievalSettings(qdrant_collection=pipe.settings.qdrant_collection, **settings)
    return HybridSearch(pipe.client, s.qdrant_collection, s)


async def run(pipe, search, query):
    return await search.search(pipe.dense.embed_query(query), pipe.sparse.embed_query(query))


def test_live_filter_keeps_unexpired_documents_only():
    conditions = live_filter(now=100.0).should
    assert conditions[2].range == qm.Range(gt=100.0)
    assert {type(c) for c in conditions[:2]} == {qm.IsEmptyCondition, qm.IsNullCondition}


def test_dedupe_keeps_newest_version_in_fused_order():
    old = cand("a", 0, rank=0, day=0, point_id="old")
    new = cand("a", 0, rank=2, day=1, point_id="new")
    other = cand("b", 0, rank=1)
    assert [c.point_id for c in dedupe([old, other, new])] == ["b-0-1", "new"]


def test_dedupe_tie_goes_to_better_rank():
    first, second = cand("a", 0, rank=0), cand("a", 0, rank=3)
    assert dedupe([second, first]) == [first]


async def test_matching_chunk_comes_back(pipe, tmp_path):
    result = await _ingest(pipe, tmp_path, "rules.md", "# Cotisation\n\nLa cotisation.")
    found = await run(pipe, searcher(pipe), "cotisation annuelle")
    assert found and found[0].doc_id == result.doc_id
    assert found[0].text and found[0].title and found[0].fused_rank == 0


async def test_top_k_is_respected(pipe, tmp_path):
    for i in range(6):
        await _ingest(pipe, tmp_path, f"d{i}.md", f"# D{i}\n\ncotisation {i}")
    assert len(await run(pipe, searcher(pipe, top_k=4), "cotisation")) == 4


async def test_two_versions_give_one_candidate_per_index(pipe, tmp_path):
    await _ingest(pipe, tmp_path, "rules.md", "# Cotisation\n\nLa cotisation.")
    points, _ = await pipe.client.scroll(pipe.settings.qdrant_collection, with_vectors=True)
    stale = [
        qm.PointStruct(
            id=str(i),
            vector=p.vector,
            payload=p.payload | {"content_hash": "old", "ingested_at": "2020-01-01T00:00:00+00:00"},
        )
        for i, p in zip(
            ["00000000-0000-0000-0000-00000000000" + str(n) for n in range(len(points))],
            points,
            strict=True,
        )
    ]
    await pipe.client.upsert(pipe.settings.qdrant_collection, stale, wait=True)
    found = await run(pipe, searcher(pipe), "cotisation")
    assert len(found) == len(points)
    assert {c.content_hash for c in found} != {"old"}


async def test_fusion_settings_reach_qdrant(pipe, tmp_path, monkeypatch):
    seen = {}
    original = pipe.client.query_points

    async def spy(*args, **kwargs):
        seen.update(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(pipe.client, "query_points", spy)
    await run(pipe, searcher(pipe, rrf_k=2, sparse_weight=0.5), "cotisation")
    assert seen["query"] == qm.RrfQuery(rrf=qm.Rrf(k=2, weights=[1.0, 0.5]))
    assert all(p.filter == seen["query_filter"] for p in seen["prefetch"])
    await run(pipe, searcher(pipe, fusion="dbsf"), "cotisation")
    assert seen["query"] == qm.FusionQuery(fusion=qm.Fusion.DBSF)


async def test_qdrant_failure_becomes_index_unavailable():
    client = AsyncQdrantClient(url="http://127.0.0.1:1", timeout=1, check_compatibility=False)
    s = RetrievalSettings()
    with pytest.raises(IndexUnavailable, match="search"):
        await HybridSearch(client, "mcclub", s).search(
            [1.0] * 8, qm.SparseVector(indices=[1], values=[1.0])
        )
    await client.close()


LONG = "# Statuts\n\n" + "\n\n".join(
    f"## Article {i}\n\ncotisation article {i} " + "mot " * 25 for i in range(4)
)


async def test_neighbours_of_middle_and_edge_chunks(pipe, tmp_path):
    await _ingest(pipe, tmp_path, "long.md", LONG)
    found = await run(pipe, searcher(pipe), "cotisation")
    assert len(found) >= 3
    by_index = {c.index: c for c in found}
    middle, first = by_index[1], by_index[0]
    got = await searcher(pipe).neighbours([middle, first])
    key = (middle.doc_id, middle.content_hash)
    assert set(got) == {(*key, 0), (*key, 1), (*key, 2)}
    assert got[(*key, 2)] == by_index[2].text


async def test_neighbours_ignore_other_versions(pipe, tmp_path):
    await _ingest(pipe, tmp_path, "long.md", LONG)
    found = await run(pipe, searcher(pipe), "cotisation")
    hit = next(c for c in found if c.index == 1)
    assert await searcher(pipe).neighbours([replace(hit, content_hash="other")]) == {}
    assert await searcher(pipe).neighbours([]) == {}


async def _set_language(pipe, doc_id, language):
    await pipe.client.set_payload(
        pipe.settings.qdrant_collection,
        {"language": language},
        points=qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]),
        wait=True,
    )


async def _cross_lingual_corpus(pipe, tmp_path):
    for i in range(5):  # same-language block that wins both arms
        await _ingest(pipe, tmp_path, f"fr{i}.md", f"# Doc {i}\n\n" + "cotisation " * (i + 3))
    other = await _ingest(pipe, tmp_path, "en.md", "# Fee\n\nThe annual fee is the cotisation.")
    await _set_language(pipe, other.doc_id, "en")
    return other


async def test_cross_language_prefetch_adds_other_language_candidates(pipe, tmp_path):
    other = await _cross_lingual_corpus(pipe, tmp_path)
    plain = searcher(pipe, top_k=5, prefetch_limit=5)
    assert other.doc_id not in {c.doc_id for c in await run(pipe, plain, "cotisation")}
    cross = searcher(pipe, top_k=5, prefetch_limit=5, cross_lang_prefetch=2)
    found = await cross.search(
        pipe.dense.embed_query("cotisation"),
        pipe.sparse.embed_query("cotisation"),
        query_language="fr",
    )
    assert other.doc_id in {c.doc_id for c in found}


async def test_cross_language_prefetch_keeps_the_live_filter(pipe, tmp_path, monkeypatch):
    await _cross_lingual_corpus(pipe, tmp_path)
    seen = {}
    original = pipe.client.query_points

    async def spy(*args, **kwargs):
        seen.update(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(pipe.client, "query_points", spy)
    cross = searcher(pipe, top_k=5, prefetch_limit=5, cross_lang_prefetch=2)
    dense, sparse = pipe.dense.embed_query("cotisation"), pipe.sparse.embed_query("cotisation")
    await cross.search(dense, sparse, query_language="fr")
    third = seen["prefetch"][2]
    assert third.filter.must[0] == seen["query_filter"]
    assert third.filter.must_not[0].match == qm.MatchValue(value="fr")
    assert seen["query"].rrf.weights is None or len(seen["query"].rrf.weights) == 3
    await cross.search(dense, sparse)  # no language: two arms, as before
    assert len(seen["prefetch"]) == 2


async def test_expired_documents_are_not_retrieved(pipe, tmp_path):
    kept = await _ingest(pipe, tmp_path, "a.md", "# A\n\nLa cotisation.")
    gone = await _ingest(pipe, tmp_path, "b.md", "# B\n\nLa cotisation annuelle.")
    await pipe.store.set_fields(gone.doc_id, {"expires_at": 1.0})  # 1970: long expired
    future = await _ingest(pipe, tmp_path, "c.md", "# C\n\nCotisation future.")
    await pipe.store.set_fields(future.doc_id, {"expires_at": 4_000_000_000.0})
    found = {c.doc_id for c in await run(pipe, searcher(pipe), "cotisation")}
    assert gone.doc_id not in found
    assert kept.doc_id in found and future.doc_id in found


async def test_extra_dense_arms_filter_by_language(pipe, tmp_path, monkeypatch):
    other = await _cross_lingual_corpus(pipe, tmp_path)
    seen = {}
    original = pipe.client.query_points

    async def spy(*args, **kwargs):
        seen.update(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(pipe.client, "query_points", spy)
    search = searcher(pipe, top_k=6, prefetch_limit=6, dense_weight=2.0)
    dense, sparse = pipe.dense.embed_query("cotisation"), pipe.sparse.embed_query("cotisation")
    extra = [(pipe.dense.embed_query("annual fee"), "en"), (dense, None)]
    found = await search.search(dense, sparse, extra_dense=extra)
    assert other.doc_id in {c.doc_id for c in found}
    assert {c.language for c in found if c.doc_id == other.doc_id} == {"en"}
    arms = seen["prefetch"]
    assert len(arms) == 4
    assert arms[2].filter.must[1].match == qm.MatchValue(value="en")
    assert arms[3].filter == arms[0].filter  # None: every language, same live filter
    assert seen["query"].rrf.weights == [2.0, 1.0, 2.0, 2.0]
