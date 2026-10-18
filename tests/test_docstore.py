from datetime import UTC, datetime

from micro_bot.ingest.docstore import DocStore
from micro_bot.ingest.models import ParsedDocument, Section, StoredDocument


def _stored(
    doc_id: str = "d1", visibility: str = "public", text: str = "Bonjour"
) -> StoredDocument:
    document = ParsedDocument(
        content_hash="ab" * 32,
        title="Règlement",
        source="rules.pdf",
        source_type="file",
        mime_type="application/pdf",
        visibility=visibility,
        language="fr",
        text=text,
        sections=[
            Section(text="# Intro\nBonjour", heading="Intro", page=1, language="fr"),
            Section(text="مرحبا", heading=None, page=2, language="ar"),
        ],
        ocr_used=True,
        parser="xberg 1.3.6",
        page_count=2,
        warnings=["1 OCR line removed"],
        created_at=datetime(2026, 10, 9, tzinfo=UTC),
    )
    return StoredDocument(
        doc_id=doc_id,
        topic="rules",
        ingested_at=datetime(2026, 10, 10, tzinfo=UTC),
        document=document,
    )


async def test_round_trip(tmp_path):
    store = DocStore(tmp_path / "documents")
    await store.save(_stored())
    assert await store.load("d1") == _stored()
    assert (tmp_path / "documents" / "d1.json").is_file()


async def test_missing_document_loads_as_none(tmp_path):
    assert await DocStore(tmp_path).load("nope") is None


async def test_overwrite_leaves_no_temp_file(tmp_path):
    store = DocStore(tmp_path)
    await store.save(_stored(text="old"))
    await store.save(_stored(text="new"))
    assert (await store.load("d1")).document.text == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["d1.json"]


async def test_delete_is_idempotent(tmp_path):
    store = DocStore(tmp_path)
    await store.save(_stored())
    await store.delete("d1")
    await store.delete("d1")
    assert await store.load("d1") is None


async def test_load_all_sorted_skips_temp_and_reports_corrupt(tmp_path):
    store = DocStore(tmp_path)
    await store.save(_stored("b"))
    await store.save(_stored("a"))
    (tmp_path / "c.json.tmp").write_text("{half")
    (tmp_path / "z.json").write_text("{not json")
    stored, failed = await store.load_all()
    assert [s.doc_id for s in stored] == ["a", "b"]
    assert failed == ["z"]


async def test_load_all_on_missing_root(tmp_path):
    assert await DocStore(tmp_path / "absent").load_all() == ([], [])


async def test_set_visibility_rewrites_only_that_field(tmp_path):
    store = DocStore(tmp_path)
    await store.save(_stored())
    await store.set_visibility("d1", "members")
    back = await store.load("d1")
    assert back.document.visibility == "members"
    assert back.document.model_copy(update={"visibility": "public"}) == _stored().document
    await store.set_visibility("unknown", "members")  # no file, no error
