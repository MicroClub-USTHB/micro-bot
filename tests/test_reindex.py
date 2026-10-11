import asyncio
import importlib.util
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from mcclub_rag.ingest.errors import IndexStale
from tests.pipeline_helpers import FakePreprocess, make_pipeline

RULES = "# Règlement\n\nLe club se réunit le mardi.\n\n# Adhésion\n\nLa cotisation est de 500 DA."
PAGE = "# Programme\n\nAtelier Python le samedi en salle B12."

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reindex.py"
_spec = importlib.util.spec_from_file_location("reindex_script", _SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


async def _seed(tmp_path, client):
    p = make_pipeline(tmp_path, client=client)
    await p.start()
    rules = await p.ingest_file(_write(tmp_path, "rules.md", RULES), "rules.md", "public")
    await p.ingest_file(_write(tmp_path, "page.md", PAGE), "page.md", "public")
    await p.ingest_file(_write(tmp_path, "rules.md", RULES), "rules.md", "members")  # updated
    return p, rules


async def test_settings_change_rebuilds_from_the_docstore(tmp_path):
    client = AsyncQdrantClient(location=":memory:")
    old, rules = await _seed(tmp_path, client)
    before = await old.corpus_version()
    old_ids = {d.doc_id for d in await old.list_documents()}
    pre = FakePreprocess()
    new = make_pipeline(tmp_path, client=client, preprocess=pre, chunk_target_tokens=40)
    assert await new.start() == "reindexed"
    assert pre.file_calls == []  # no re-parse
    assert not await client.collection_exists(old.store.physical_name(old.signature))
    assert await new.store.ensure(new.signature) == "ok"
    docs = {d.doc_id: d for d in await new.list_documents()}
    assert set(docs) == old_ids and len(docs) == 2
    assert docs[rules.doc_id].visibility == "members"
    assert await new.corpus_version() != before
    hits = await client.query_points(new.store.alias, query=[0.3] * 8, using="dense", limit=10)
    assert hits.points
    await client.close()


async def test_unreadable_json_keeps_the_old_index(tmp_path):
    client = AsyncQdrantClient(location=":memory:")
    old, rules = await _seed(tmp_path, client)
    (old.docstore.root / f"{rules.doc_id}.json").write_text("{broken")
    new = make_pipeline(tmp_path, client=client, chunk_target_tokens=40)
    with pytest.raises(IndexStale, match="previous index is still served"):
        await new.start()
    assert await old.store.ensure(old.signature) == "ok"
    assert not await client.collection_exists(new.store.physical_name(new.signature))
    await client.close()


async def test_indexed_document_without_json_blocks_the_reindex(tmp_path):
    client = AsyncQdrantClient(location=":memory:")
    old, rules = await _seed(tmp_path, client)
    await old.docstore.delete(rules.doc_id)
    with pytest.raises(IndexStale, match=rules.doc_id):
        await old.reindex()
    assert len(await old.list_documents()) == 2
    await client.close()


async def test_stale_index_with_startup_reindex_off_fails_fast(tmp_path):
    client = AsyncQdrantClient(location=":memory:")
    await _seed(tmp_path, client)
    new = make_pipeline(tmp_path, client=client, chunk_target_tokens=40, reindex_on_startup=False)
    with pytest.raises(IndexStale, match="scripts/reindex.py"):
        await new.start()
    await client.close()


async def test_forced_reindex_with_same_settings(tmp_path):
    client = AsyncQdrantClient(location=":memory:")
    p, _ = await _seed(tmp_path, client)
    before = await p.corpus_version()
    await p.reindex()
    assert await p.store.ensure(p.signature) == "ok"
    assert await p.corpus_version() == before
    assert len(await p.list_documents()) == 2
    await client.close()


async def test_ingest_waits_while_a_reindex_runs(tmp_path):
    p = make_pipeline(tmp_path)
    await p.start()
    async with p._exclusive():
        task = asyncio.create_task(
            p.ingest_file(_write(tmp_path, "page.md", PAGE), "page.md", "public")
        )
        await asyncio.sleep(0.05)
        assert not task.done()
    assert (await task).status == "created"
    await p.close()


def test_script_reindexes_and_reports(tmp_path, capsys):
    client = AsyncQdrantClient(location=":memory:")
    asyncio.run(_seed(tmp_path, client))
    rc = script.main([], factory=lambda s: make_pipeline(tmp_path, client=client))
    assert rc == 0
    assert "reindexed into signature" in capsys.readouterr().out


def test_script_reports_failure(tmp_path, capsys):
    client = AsyncQdrantClient(location=":memory:")
    _, rules = asyncio.run(_seed(tmp_path, client))
    (tmp_path / "data" / "documents" / f"{rules.doc_id}.json").write_text("{broken")
    rc = script.main([], factory=lambda s: make_pipeline(tmp_path, client=client))
    assert rc == 1
    assert "index_stale" in capsys.readouterr().err
