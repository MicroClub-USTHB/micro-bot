from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from mcclub_rag.ingest.errors import IndexUnavailable
from mcclub_rag.retrieval.search import Candidate, HybridSearch, dedupe, visibility_filter
from mcclub_rag.retrieval.settings import RetrievalSettings
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
        visibility="public",
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


async def _ingest(pipe, tmp_path, name, text, visibility):
    return await pipe.ingest_file(_write(tmp_path, name, text), name, visibility)


@pytest.fixture
async def pipe(tmp_path):
    p = make_pipeline(tmp_path, dense=KeywordDense())
    await p.start()
    yield p
    await p.close()


def searcher(pipe, **settings) -> HybridSearch:
    s = RetrievalSettings(qdrant_collection=pipe.settings.qdrant_collection, **settings)
    return HybridSearch(pipe.client, s.qdrant_collection, s)


async def run(pipe, search, query, role):
    return await search.search(pipe.dense.embed_query(query), pipe.sparse.embed_query(query), role)


def _match(flt: qm.Filter):
    return flt.must[0].match


def test_visibility_filter_per_role():
    assert _match(visibility_filter("public")) == qm.MatchValue(value="public")
    assert _match(visibility_filter("member")) == qm.MatchAny(any=["public", "members"])
    with pytest.raises(ValueError, match="unknown role"):
        visibility_filter("admin")


def test_dedupe_keeps_newest_version_in_fused_order():
    old = cand("a", 0, rank=0, day=0, point_id="old")
    new = cand("a", 0, rank=2, day=1, point_id="new")
    other = cand("b", 0, rank=1)
    assert [c.point_id for c in dedupe([old, other, new])] == ["b-0-1", "new"]


def test_dedupe_tie_goes_to_better_rank():
    first, second = cand("a", 0, rank=0), cand("a", 0, rank=3)
    assert dedupe([second, first]) == [first]


async def test_matching_chunk_comes_back(pipe, tmp_path):
    result = await _ingest(pipe, tmp_path, "rules.md", "# Cotisation\n\nLa cotisation.", "public")
    found = await run(pipe, searcher(pipe), "cotisation annuelle", "public")
    assert found and found[0].doc_id == result.doc_id
    assert found[0].text and found[0].title and found[0].fused_rank == 0


async def test_members_chunks_never_reach_a_public_role(pipe, tmp_path):
    public = await _ingest(pipe, tmp_path, "pub.md", "# Info\n\nLa cotisation.", "public")
    for i in range(5):  # higher on both arms: more keyword hits for dense and BM25
        text = f"# Secret {i}\n\n" + "cotisation " * (i + 3)
        await _ingest(pipe, tmp_path, f"m{i}.md", text, "members")
    search = searcher(pipe, top_k=3, prefetch_limit=3)
    as_public = await run(pipe, search, "cotisation", "public")
    assert [c.doc_id for c in as_public] == [public.doc_id]
    as_member = await run(pipe, search, "cotisation", "member")
    assert len(as_member) == 3 and all(c.visibility == "members" for c in as_member)


async def test_top_k_is_respected(pipe, tmp_path):
    for i in range(6):
        await _ingest(pipe, tmp_path, f"d{i}.md", f"# D{i}\n\ncotisation {i}", "public")
    assert len(await run(pipe, searcher(pipe, top_k=4), "cotisation", "public")) == 4


async def test_two_versions_give_one_candidate_per_index(pipe, tmp_path):
    await _ingest(pipe, tmp_path, "rules.md", "# Cotisation\n\nLa cotisation.", "public")
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
    found = await run(pipe, searcher(pipe), "cotisation", "public")
    assert len(found) == len(points)
    assert {c.content_hash for c in found} != {"old"}


async def test_fusion_settings_reach_qdrant(pipe, tmp_path, monkeypatch):
    seen = {}
    original = pipe.client.query_points

    async def spy(*args, **kwargs):
        seen.update(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(pipe.client, "query_points", spy)
    await run(pipe, searcher(pipe, rrf_k=2, sparse_weight=0.5), "cotisation", "public")
    assert seen["query"] == qm.RrfQuery(rrf=qm.Rrf(k=2, weights=[1.0, 0.5]))
    assert all(p.filter == seen["query_filter"] for p in seen["prefetch"])
    await run(pipe, searcher(pipe, fusion="dbsf"), "cotisation", "public")
    assert seen["query"] == qm.FusionQuery(fusion=qm.Fusion.DBSF)


async def test_qdrant_failure_becomes_index_unavailable():
    client = AsyncQdrantClient(url="http://127.0.0.1:1", timeout=1, check_compatibility=False)
    s = RetrievalSettings()
    with pytest.raises(IndexUnavailable, match="search"):
        await HybridSearch(client, "mcclub", s).search(
            [1.0] * 8, qm.SparseVector(indices=[1], values=[1.0]), "public"
        )
    await client.close()
