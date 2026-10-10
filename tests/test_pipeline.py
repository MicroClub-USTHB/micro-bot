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


RULES = "# Règlement\n\nLe club se réunit le mardi.\n\n# Adhésion\n\nLa cotisation est de 500 DA."


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


async def _points(p):
    points, _ = await p.client.scroll(p.store.alias, limit=1000, with_payload=True)
    return points


async def test_ingest_file_creates_points_and_stored_json(pipe, tmp_path):
    result = await pipe.ingest_file(_write(tmp_path, "rules.md", RULES), "rules.md", "members")
    assert result.status == "created"
    assert result.doc_id == pl.doc_id_for("rules")
    assert result.chunk_count >= 1 and result.visibility == "members"
    points = await _points(pipe)
    assert len(points) == result.chunk_count
    payload = next(p.payload for p in points if p.payload["index"] == 0)
    assert set(payload) == {
        "doc_id",
        "content_hash",
        "index",
        "chunk_count",
        "text",
        "heading_path",
        "page_start",
        "page_end",
        "section_index",
        "language",
        "token_count",
        "visibility",
        "title",
        "source",
        "source_type",
        "ingested_at",
    }
    assert payload["visibility"] == "members" and payload["source"] == "rules.md"
    stored = await pipe.docstore.load(result.doc_id)
    assert stored.topic == "rules" and stored.document.content_hash == result.content_hash


async def test_marker_is_written_last(pipe, tmp_path):
    long_text = "\n\n".join(f"# Partie {i}\n\n" + "mot " * 40 for i in range(4))
    order = []
    original = pipe.client.upsert

    async def spy(collection, points, wait):
        order.extend(p.payload["index"] for p in points)
        return await original(collection, points, wait=wait)

    pipe.client.upsert = spy
    result = await pipe.ingest_file(_write(tmp_path, "long.md", long_text), "long.md", "public")
    assert result.chunk_count > 1
    assert order[-1] == 0 and sorted(order) == list(range(result.chunk_count))


async def test_explicit_topic_sets_doc_id(pipe, tmp_path):
    result = await pipe.ingest_file(
        _write(tmp_path, "v2.md", RULES), "v2.md", "public", topic=" Règlement intérieur "
    )
    assert result.doc_id == pl.doc_id_for("Règlement intérieur")


async def test_oversize_file_rejected_before_reading(tmp_path, monkeypatch):
    p = make_pipeline(tmp_path, max_file_mb=1)
    await p.start()
    big = tmp_path / "big.md"
    big.write_bytes(b"x" * (1024 * 1024 + 1))
    monkeypatch.setattr(pl.Path, "read_bytes", lambda self: pytest.fail("file was read"))
    with pytest.raises(pl.FileTooLarge):
        await p.ingest_file(big, "big.md", "public")


async def test_bad_visibility_rejected_before_io(pipe):
    with pytest.raises(pl.InvalidInput):
        await pipe.ingest_file("/nonexistent/x.md", "x.md", "admins")


async def test_missing_file_is_invalid_input(pipe):
    with pytest.raises(pl.InvalidInput, match="not found"):
        await pipe.ingest_file("/nonexistent/x.md", "x.md", "public")


async def test_empty_document_stores_nothing(pipe, tmp_path):
    with pytest.raises(pl.EmptyDocument):
        await pipe.ingest_file(_write(tmp_path, "e.md", "   "), "e.md", "public")
    assert await _points(pipe) == []
    assert await pipe.docstore.load(pl.doc_id_for("e")) is None


async def test_module_level_ingest_file_uses_the_shared_pipeline(tmp_path):
    await pl.init_pipeline(pipeline=make_pipeline(tmp_path))
    result = await pl.ingest_file(_write(tmp_path, "rules.md", RULES), "rules.md", "public")
    assert result.status == "created"
