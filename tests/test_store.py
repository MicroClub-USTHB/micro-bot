from datetime import UTC, datetime

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from micro_bot.ingest.errors import IndexStale, IndexUnavailable
from micro_bot.retrieval.store import DENSE, SPARSE, VectorStore

SIG_A = "a" * 64
SIG_B = "b" * 64


@pytest.fixture
async def store():
    client = AsyncQdrantClient(location=":memory:")
    yield VectorStore(client, "mcclub", dim=4)
    await client.close()


def _point(pid: int, **payload) -> qm.PointStruct:
    return qm.PointStruct(
        id=pid,
        vector={DENSE: [1.0, 0, 0, 0], SPARSE: qm.SparseVector(indices=[1], values=[1.0])},
        payload=payload,
    )


async def test_ensure_creates_then_reports_ok_and_stale(store):
    assert await store.ensure(SIG_A) == "created"
    assert await store.ensure(SIG_A) == "ok"
    assert await store.ensure(SIG_B) == "stale"
    info = await store.client.get_collection("mcclub")  # reachable through the alias
    assert info.config.params.vectors[DENSE].size == 4
    assert info.config.params.sparse_vectors[SPARSE].modifier == qm.Modifier.IDF


async def test_real_collection_named_like_the_alias_is_refused(store):
    await store.client.create_collection(
        "mcclub", vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE)
    )
    with pytest.raises(IndexStale, match="not an alias"):
        await store.ensure(SIG_A)


async def test_switch_alias_drops_old_collection_and_keeps_search_working(store):
    await store.ensure(SIG_A)
    new = await store.create_physical(SIG_B)
    await store.client.upsert(new, [_point(1, doc_id="d")], wait=True)
    await store.switch_alias(new)
    assert await store.ensure(SIG_B) == "ok"
    assert not await store.client.collection_exists(store.physical_name(SIG_A))
    hits = await store.client.query_points("mcclub", query=[1.0, 0, 0, 0], using=DENSE)
    assert [p.id for p in hits.points] == [1]


async def test_forced_rebuild_never_targets_the_live_collection(store):
    await store.ensure(SIG_A)
    live = store.physical_name(SIG_A)
    rebuild = await store.next_physical(SIG_A)
    assert rebuild != live
    await store.switch_alias(await store.create_physical(SIG_A, name=rebuild))
    assert await store.ensure(SIG_A) == "ok"
    assert await store.next_physical(SIG_A) == live
    assert not await store.client.collection_exists(live)


async def test_create_physical_replaces_a_half_built_collection(store):
    name = await store.create_physical(SIG_B)
    await store.client.upsert(name, [_point(1)], wait=True)
    await store.create_physical(SIG_B)
    assert (await store.client.count(name)).count == 0


async def test_transport_error_becomes_index_unavailable():
    client = AsyncQdrantClient(url="http://127.0.0.1:1", timeout=1, check_compatibility=False)
    with pytest.raises(IndexUnavailable, match="alias lookup"):
        await VectorStore(client, "mcclub", dim=4).ensure(SIG_A)
    await client.close()


def _chunk_point(doc_id, content_hash, index, *, pid, visibility="public", day=10, count=2):
    payload = {
        "doc_id": doc_id,
        "content_hash": content_hash,
        "index": index,
        "chunk_count": count,
        "visibility": visibility,
        "title": f"title {doc_id}",
        "source": f"{doc_id}.md",
        "source_type": "file",
        "language": "fr",
        "ingested_at": datetime(2026, 10, day, tzinfo=UTC).isoformat(),
        "text": f"chunk {index}",
    }
    return _point(pid, **payload)


@pytest.fixture
async def filled(store):
    await store.ensure(SIG_A)
    await store.upsert(
        [
            _chunk_point("d1", "old", 1, pid=1, day=9),
            _chunk_point("d1", "old", 0, pid=2, day=9),
            _chunk_point("d1", "new", 1, pid=3, day=10),
            _chunk_point("d1", "new", 0, pid=4, day=10),
            _chunk_point("d2", "h2", 1, pid=5, visibility="members"),
            _chunk_point("d2", "h2", 0, pid=6, visibility="members"),
            _chunk_point("d3", "partial", 1, pid=7),  # crashed: no marker
        ]
    )
    return store


async def test_markers_keep_the_newest_version_per_document(filled):
    markers = await filled.markers()
    assert [(m.doc_id, m.content_hash) for m in markers] == [("d1", "new"), ("d2", "h2")]
    assert markers[1].visibility == "members"
    assert markers[0].ingested_at == datetime(2026, 10, 10, tzinfo=UTC)


async def test_versions_and_count(filled):
    assert [m.content_hash for m in await filled.versions("d1")] == ["new", "old"]
    assert await filled.versions("d3") == []
    assert await filled.count("d1", "new") == 2
    assert await filled.count("d3", "partial") == 1


async def test_delete_other_versions_keeps_one_hash(filled):
    await filled.delete_other_versions("d1", "new")
    assert [m.content_hash for m in await filled.versions("d1")] == ["new"]
    assert await filled.count("d1", "old") == 0
    assert await filled.count("d2", "h2") == 2


async def test_delete_version_and_doc(filled):
    await filled.delete_version("d3", "partial")
    assert await filled.count("d3", "partial") == 0
    assert await filled.delete_doc("d2") is True
    assert await filled.delete_doc("d2") is False
    assert [m.doc_id for m in await filled.markers()] == ["d1"]


async def test_set_visibility_touches_every_point(filled):
    await filled.set_visibility("d1", "members")
    points, _ = await filled.client.scroll("mcclub", limit=100, with_payload=True)
    d1 = [p.payload["visibility"] for p in points if p.payload["doc_id"] == "d1"]
    assert d1 == ["members"] * 4


async def test_visibility_filter_inside_the_query(filled):
    public = qm.Filter(
        must=[qm.FieldCondition(key="visibility", match=qm.MatchValue(value="public"))]
    )
    hits = await filled.client.query_points(
        "mcclub", query=[1.0, 0, 0, 0], using=DENSE, query_filter=public, limit=100
    )
    assert {p.payload["doc_id"] for p in hits.points} == {"d1", "d3"}


async def test_upsert_batches(store):
    await store.ensure(SIG_A)
    store.batch = 2
    calls = []
    original = store.client.upsert

    async def spy(collection, points, wait):
        calls.append(len(points))
        return await original(collection, points, wait=wait)

    store.client.upsert = spy
    await store.upsert([_chunk_point("d", "h", i, pid=i + 1) for i in range(5)])
    assert calls == [2, 2, 1]
