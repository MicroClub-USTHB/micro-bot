import json

import pytest

from eval import run_eval
from tests.pipeline_helpers import make_pipeline
from tests.retrieval_helpers import make_retriever

DOCS = {
    "fees": ("# Cotisation\n\nLa cotisation annuelle est de 1500 dinars.", "public"),
    "vote": ("# Élections\n\nLes élections du bureau ont lieu en juin.", "public"),
    "board": ("# Bureau\n\nLe procès-verbal secret du bureau sur les élections.", "members"),
}
QUESTIONS = [
    {"id": "q1", "question": "cotisation annuelle", "lang": "fr", "role": "member",
     "answerable": True, "doc": "fees", "evidence": ["1500 dinars"], "tags": ["french"]},
    {"id": "q2", "question": "et en juin ?", "lang": "fr", "role": "public", "answerable": True,
     "doc": "vote", "evidence": ["ont lieu en juin"], "tags": ["french", "follow_up"],
     "history": [{"user_message": "élections du bureau", "assistant_message": "En juin."}]},
    {"id": "q3", "question": "piscine municipale", "lang": "fr", "role": "member",
     "answerable": False, "tags": ["french", "unanswerable"]},
    {"id": "q4", "question": "procès-verbal secret du bureau", "lang": "fr", "role": "public",
     "answerable": False, "doc": "board", "tags": ["french", "access"]},
]  # fmt: skip


@pytest.fixture
async def setup(tmp_path):
    pipe = make_pipeline(tmp_path)
    await pipe.start()
    doc_ids = {}
    for topic, (text, visibility) in DOCS.items():
        path = tmp_path / f"{topic}.md"
        path.write_text(text, encoding="utf-8")
        doc_ids[topic] = (await pipe.ingest_file(path, path.name, visibility, topic=topic)).doc_id
    yield make_retriever(pipe), doc_ids
    await pipe.close()


DUMP = (
    "# Metadata\n\n```\n{'title': 'x'}\n```\n\n"
    "# Sections (1)\n\n- page=None\n\n"
    "# Text\n\n# Real\n\nBody.\n"
)


def test_document_text_unwraps_parse_dumps():
    assert run_eval.document_text(DUMP) == "# Real\n\nBody.\n"
    assert run_eval.document_text("# Plain\n\nBody.") == "# Plain\n\nBody."


async def test_ingest_corpus_uses_the_text_and_the_manifest_title(tmp_path):
    corpus_dir, work = tmp_path / "corpus", tmp_path / "work"
    corpus_dir.mkdir()
    (corpus_dir / "dump.md").write_text(DUMP, encoding="utf-8")
    manifest = [{"file": "dump.md", "topic": "real", "visibility": "public", "title": "Real doc"}]
    pipe = make_pipeline(tmp_path / "pipe")
    await pipe.start()
    doc_ids = await run_eval.ingest_corpus(pipe, manifest, corpus_dir, work)
    [info] = await pipe.list_documents()
    await pipe.close()
    assert doc_ids == {"real": info.doc_id} and info.title == "Real doc"
    assert info.source == "dump.md" and info.chunk_count == 1


async def test_rows_have_ranks_scores_and_timings(setup):
    retriever, doc_ids = setup
    rows = await run_eval.evaluate(retriever, QUESTIONS, doc_ids)
    by_id = {r["id"]: r for r in rows}
    assert by_id["q1"]["rank_reranked"] == 1 and by_id["q1"]["rank_fused"] is not None
    assert by_id["q2"]["history_used"] and by_id["q2"]["rank_reranked"] == 1
    assert by_id["q3"]["rank_reranked"] is None and by_id["q3"]["top_score"] < 1
    assert by_id["q4"]["visibilities"] == ["public"]  # the members doc never reached public
    assert all(r["ms_total"] >= r["ms_rerank"] >= 0 for r in rows)


async def test_summary_and_result_file(setup, tmp_path):
    retriever, doc_ids = setup
    rows = await run_eval.evaluate(retriever, QUESTIONS, doc_ids)
    summary = run_eval.summarize(rows, max_false_refusal=0.0)
    assert summary["overall"]["n"] == 2 and summary["overall"]["hit@5_reranked"] == 1.0
    assert set(summary["by_tag"]) == {"french", "follow_up"}
    assert summary["access_violations"] == 0
    assert summary["suggested_gate_threshold"] is not None
    assert set(summary["latency_ms"]) == set(run_eval.STAGES)
    path = run_eval.write_result(tmp_path / "results", "unit", {"rows": rows, "summary": summary})
    assert path.name.endswith("-unit.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["summary"]["overall"] == summary["overall"] and len(saved["rows"]) == 4
    run_eval.print_summary(summary, "unit")
