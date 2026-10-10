import pytest

from mcclub_rag.ingest import pipeline as pl
from tests.pipeline_helpers import FakeDenseEncoder, make_pipeline


@pytest.fixture
async def pipe(tmp_path):
    p = make_pipeline(tmp_path)
    await p.start()
    yield p
    await p.close()


@pytest.fixture(autouse=True)
async def reset_singleton():
    yield
    await pl.close_pipeline()


async def test_init_pipeline_creates_collection_and_alias(tmp_path):
    p = await pl.init_pipeline(pipeline=make_pipeline(tmp_path))
    assert await pl.get_pipeline() is p
    assert await p.store.ensure(p.signature) == "ok"
    assert await pl.init_pipeline(pipeline=make_pipeline(tmp_path)) is p  # idempotent


def test_signature_is_stable_and_tracks_every_input(tmp_path):
    base = make_pipeline(tmp_path).signature
    assert make_pipeline(tmp_path).signature == base
    assert make_pipeline(tmp_path, chunk_target_tokens=40).signature != base
    assert make_pipeline(tmp_path, dense=FakeDenseEncoder("other")).signature != base
    assert make_pipeline(tmp_path, bm25_avg_len=50).signature != base
