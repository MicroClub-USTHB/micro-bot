"""scripts/build_eval_corpus.py: source list parsing and file names match the manifest."""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_eval_corpus.py"
_spec = importlib.util.spec_from_file_location("build_eval_corpus", _SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

REPO = Path(__file__).resolve().parents[1]
# The first documents of the manifest were parsed by hand before the source list existed.
_PREDATES_LIST = 4


def test_dump_names():
    assert script.dump_name("KENPA SG DOC.pdf") == "KENPA_SG_DOC.md"
    assert (
        script.dump_name("https://finfo.usthb.dz/facult%C3%A9/presentation")
        == "finfo.usthb.dz_faculté_presentation.md"
    )
    assert script.dump_name("https://ar.wikipedia.org/wiki/باب_الزوار") == (
        "ar.wikipedia.org_wiki_باب_الزوار.md"
    )


def test_quote_url_is_idempotent():
    readable = "https://finfo.usthb.dz/faculté/presentation"
    quoted = script.quote_url(readable)
    assert quoted == "https://finfo.usthb.dz/facult%C3%A9/presentation"
    assert script.quote_url(quoted) == quoted


def test_bad_visibility(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("private https://example.com\n")
    with pytest.raises(ValueError, match="bad visibility"):
        script.read_sources(path)


def test_every_listed_source_is_in_the_manifest_with_its_visibility():
    sources = script.read_sources(script.SOURCES)
    manifest = json.loads((REPO / "eval" / "corpus.json").read_text(encoding="utf-8"))
    by_file = {d["file"]: d["visibility"] for d in manifest[_PREDATES_LIST:]}
    built = {script.dump_name(target): visibility for visibility, target in sources}
    assert built == {f: v for f, v in by_file.items() if f in built}
    assert set(by_file) - set(built) <= {f for f in by_file if "scan" in f}
