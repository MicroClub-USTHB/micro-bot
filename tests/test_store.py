import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from mcclub_rag.ingest.errors import IndexStale, IndexUnavailable
from mcclub_rag.retrieval.store import DENSE, SPARSE, VectorStore

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
