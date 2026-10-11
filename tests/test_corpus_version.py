from qdrant_client import AsyncQdrantClient

from tests.pipeline_helpers import make_pipeline

RULES = "# Règlement\n\nLe club se réunit le mardi."


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


async def test_version_changes_on_every_corpus_change(tmp_path):
    p = make_pipeline(tmp_path)
    await p.start()
    path = _write(tmp_path, "rules.md", RULES)
    seen = [await p.corpus_version()]

    async def changed():
        version = await p.corpus_version()
        assert version != seen[-1]
        seen.append(version)

    result = await p.ingest_file(path, "rules.md", "public")
    await changed()
    assert (await p.ingest_file(path, "rules.md", "public")).status == "duplicate"
    assert await p.corpus_version() == seen[-1]
    await p.ingest_file(path, "rules.md", "members")  # updated
    await changed()
    await p.ingest_file(_write(tmp_path, "rules.md", RULES + " Ajout."), "rules.md", "members")
    await changed()  # replaced
    await p.delete_document(result.doc_id)
    await changed()
    assert seen[-1] == seen[0]  # same (empty) corpus, same version
    await p.close()


async def test_version_is_shared_across_processes_and_tracks_settings(tmp_path):
    client = AsyncQdrantClient(location=":memory:")
    api = make_pipeline(tmp_path, client=client)
    cli = make_pipeline(tmp_path, client=client)
    await api.start()
    await cli.ingest_file(_write(tmp_path, "rules.md", RULES), "rules.md", "public")
    assert await api.corpus_version() == await cli.corpus_version()
    other = make_pipeline(tmp_path, client=client, chunk_target_tokens=40)
    assert await other.corpus_version() != await api.corpus_version()
    await client.close()
