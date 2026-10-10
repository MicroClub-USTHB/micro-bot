import asyncio

import pytest
from qdrant_client import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException
from structlog.testing import capture_logs

from mcclub_rag.ingest import pipeline as pl
from mcclub_rag.ingest.errors import EmbeddingError, IndexUnavailable
from tests.pipeline_helpers import FakeDenseEncoder, FakePreprocess, make_pipeline


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


async def test_same_file_twice_is_duplicate_without_parsing(tmp_path):
    pre = FakePreprocess()
    p = make_pipeline(tmp_path, preprocess=pre)
    await p.start()
    path = _write(tmp_path, "rules.md", RULES)
    first = await p.ingest_file(path, "rules.md", "public")
    calls = p.dense.calls
    second = await p.ingest_file(path, "rules.md", "public")
    assert second.status == "duplicate"
    assert second.chunk_count == first.chunk_count and second.title == first.title
    assert pre.file_calls == ["rules.md"] and p.dense.calls == calls


async def test_same_file_other_visibility_is_updated(pipe, tmp_path):
    path = _write(tmp_path, "rules.md", RULES)
    first = await pipe.ingest_file(path, "rules.md", "public")
    calls = pipe.dense.calls
    second = await pipe.ingest_file(path, "rules.md", "members")
    assert second.status == "updated" and second.visibility == "members"
    assert pipe.dense.calls == calls
    assert {p.payload["visibility"] for p in await _points(pipe)} == {"members"}
    assert (await pipe.docstore.load(first.doc_id)).document.visibility == "members"


async def test_new_content_same_topic_replaces(pipe, tmp_path):
    first = await pipe.ingest_file(_write(tmp_path, "rules.md", RULES), "rules.md", "public")
    new_text = RULES.replace("500 DA", "800 DA")
    second = await pipe.ingest_file(_write(tmp_path, "rules.md", new_text), "rules.md", "public")
    assert second.status == "replaced" and second.doc_id == first.doc_id
    assert {p.payload["content_hash"] for p in await _points(pipe)} == {second.content_hash}
    stored = await pipe.docstore.load(first.doc_id)
    assert stored.document.content_hash == second.content_hash


async def test_replacement_never_leaves_zero_points(pipe, tmp_path):
    await pipe.ingest_file(_write(tmp_path, "rules.md", RULES), "rules.md", "public")
    seen = []
    for name in ("upsert", "delete"):
        original = getattr(pipe.client, name)

        async def spy(*args, _original=original, **kwargs):
            result = await _original(*args, **kwargs)
            seen.append((await pipe.client.count(pipe.store.alias)).count)
            return result

        setattr(pipe.client, name, spy)
    new_text = RULES + "\n\n# Horaires\n\nDe 9h à 17h."
    await pipe.ingest_file(_write(tmp_path, "rules.md", new_text), "rules.md", "public")
    assert seen and min(seen) > 0


async def test_crashed_half_ingest_is_rewritten_not_duplicate(pipe, tmp_path):
    long_text = "\n\n".join(f"# Partie {i}\n\n" + "mot " * 40 for i in range(4))
    path = _write(tmp_path, "long.md", long_text)
    first = await pipe.ingest_file(path, "long.md", "public")
    # Simulate a crash that lost some non-marker points of this version.
    await pipe.client.delete(
        pipe.store.alias,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(must=[qm.FieldCondition(key="index", match=qm.MatchValue(value=1))])
        ),
        wait=True,
    )
    second = await pipe.ingest_file(path, "long.md", "public")
    assert second.status == "created"
    assert len(await _points(pipe)) == first.chunk_count


async def test_concurrent_ingests_of_one_topic_are_serialized(tmp_path):
    p = make_pipeline(tmp_path, max_concurrency=4)
    await p.start()
    active, peak = 0, 0
    original = p._store_new

    async def tracking(*args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        try:
            return await original(*args)
        finally:
            active -= 1

    p._store_new = tracking
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _write(tmp_path / "a", "rules.md", RULES)
    b = _write(tmp_path / "b", "rules.md", RULES.replace("mardi", "jeudi"))
    results = await asyncio.gather(
        p.ingest_file(a, "rules.md", "public"), p.ingest_file(b, "rules.md", "public")
    )
    assert peak == 1
    assert sorted(r.status for r in results) == ["created", "replaced"]
    last = next(r for r in results if r.status == "replaced")
    assert {pt.payload["content_hash"] for pt in await _points(p)} == {last.content_hash}
    assert (await p.docstore.load(last.doc_id)).document.content_hash == last.content_hash


async def test_encoder_failure_keeps_the_previous_version(pipe, tmp_path):
    first = await pipe.ingest_file(_write(tmp_path, "rules.md", RULES), "rules.md", "public")
    pipe.dense.fail_after = 0
    with pytest.raises(EmbeddingError):
        await pipe.ingest_file(
            _write(tmp_path, "rules.md", RULES + " Nouveau."), "rules.md", "public"
        )
    assert {p.payload["content_hash"] for p in await _points(pipe)} == {first.content_hash}
    assert (await pipe.docstore.load(first.doc_id)).document.content_hash == first.content_hash


async def test_store_failure_mid_upsert_cleans_up(tmp_path):
    p = make_pipeline(tmp_path, upsert_batch=1)
    await p.start()
    first = await p.ingest_file(_write(tmp_path, "rules.md", RULES), "rules.md", "public")
    original, calls = p.client.upsert, 0

    async def flaky(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ResponseHandlingException(ConnectionError("qdrant went away"))
        return await original(*args, **kwargs)

    p.client.upsert = flaky
    text = "\n\n".join(f"# Partie {i}\n\n" + "mot " * 40 for i in range(4))
    with pytest.raises(IndexUnavailable):
        await p.ingest_file(_write(tmp_path, "rules.md", text), "rules.md", "public")
    assert {pt.payload["content_hash"] for pt in await _points(p)} == {first.content_hash}


async def test_logs_one_event_without_document_text(pipe, tmp_path):
    with capture_logs() as logs:
        await pipe.ingest_file(_write(tmp_path, "rules.md", RULES), "rules.md", "public")
        with pytest.raises(pl.InvalidInput):
            await pipe.ingest_file(_write(tmp_path, "x.md", RULES), "x.md", "admins")
    events = [e for e in logs if e["event"].startswith("document_ingest")]
    ok, failed = events
    assert ok["event"] == "document_ingested" and ok["status"] == "created"
    assert {"doc_id", "chunks", "source", "duration_ms", "ms_chunk", "ms_dense", "ms_store"} <= set(
        ok
    )
    assert failed["event"] == "document_ingest_failed"
    assert failed["error_code"] == "invalid_input" and failed["stage"] == "validate"
    assert not any("cotisation" in str(value) for e in logs for value in e.values())


PAGE = "# Programme\n\nAtelier Python le samedi en salle B12."


@pytest.fixture
async def web(tmp_path):
    pre = FakePreprocess()
    p = make_pipeline(tmp_path, preprocess=pre)
    await p.start()
    yield p, pre
    await p.close()


async def test_url_query_and_fragment_do_not_change_doc_id(web):
    p, pre = web
    pre.pages["https://club.example/a"] = ("https://club.example/a?utm=2", PAGE)
    result = await p.ingest_url("https://club.example/a?x=1#top", "public")
    assert result.status == "created" and result.source_type == "web"
    assert result.doc_id == pl.doc_id_for("https://club.example/a")


async def test_redirect_uses_the_final_url_as_topic(web):
    p, pre = web
    pre.pages["https://club.example/old"] = ("https://club.example/new", PAGE)
    result = await p.ingest_url("https://club.example/old", "members")
    assert result.doc_id == pl.doc_id_for("https://club.example/new")
    stored = await p.docstore.load(result.doc_id)
    assert stored.topic == "https://club.example/new"


async def test_same_page_twice_is_duplicate(web):
    p, pre = web
    pre.pages["https://club.example/a"] = ("https://club.example/a", PAGE)
    await p.ingest_url("https://club.example/a", "public")
    assert (await p.ingest_url("https://club.example/a", "public")).status == "duplicate"


async def test_logged_url_has_no_query_string(web):
    p, pre = web
    pre.pages["https://club.example/a"] = ("https://club.example/a", PAGE)
    with capture_logs() as logs:
        await p.ingest_url("https://club.example/a?token=secret", "public")
    event = next(e for e in logs if e["event"] == "document_ingested")
    assert event["source"] == "https://club.example/a"
    assert not any("secret" in str(v) for e in logs for v in e.values())


async def test_module_level_ingest_url(tmp_path):
    pre = FakePreprocess()
    pre.pages["https://club.example/a"] = ("https://club.example/a", PAGE)
    await pl.init_pipeline(pipeline=make_pipeline(tmp_path, preprocess=pre))
    assert (await pl.ingest_url("https://club.example/a", "public")).status == "created"
