import importlib.util
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from tests.pipeline_helpers import make_pipeline

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_folder.py"
_spec = importlib.util.spec_from_file_location("ingest_folder", _SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


def _report(capsys) -> list[str]:
    """Script lines only: structlog events (timestamped) go to stdout too."""
    return [line for line in capsys.readouterr().out.splitlines() if not line[:4].isdigit()]


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "corpus"
    (folder / "sub").mkdir(parents=True)
    (folder / "rules.md").write_text("# Règlement\n\nLe club se réunit le mardi.")
    (folder / "page.md").write_text("# Programme\n\nAtelier Python le samedi.")
    (folder / ".hidden.md").write_text("# Caché\n\nIgnoré.")
    (folder / "sub" / "deep.md").write_text("# Profond\n\nSous-dossier.")
    return folder


@pytest.fixture
def factory(tmp_path):
    client = AsyncQdrantClient(location=":memory:")

    def build(settings):
        pipeline = make_pipeline(tmp_path, client=client)

        async def keep_open():  # the in-memory store must survive between runs
            return None

        pipeline.close = keep_open
        return pipeline

    return build


def test_second_run_reports_duplicates(corpus, factory, capsys):
    assert script.main([str(corpus), "--visibility", "public"], factory=factory) == 0
    first = _report(capsys)
    assert [line.split()[0:2] for line in first] == [
        ["created", "page.md"],
        ["created", "rules.md"],
    ]
    assert script.main([str(corpus), "--visibility", "public"], factory=factory) == 0
    second = _report(capsys)
    assert [line.split()[0] for line in second] == ["duplicate", "duplicate"]


def test_recursive_includes_subfolders(corpus, factory, capsys):
    assert (
        script.main([str(corpus), "--visibility", "members", "--recursive"], factory=factory) == 0
    )
    assert any("sub/deep.md" in line for line in _report(capsys))


def test_failed_file_gives_exit_code_1(corpus, factory, capsys):
    (corpus / "blob.bin").write_bytes(b"\x00\x01")
    assert script.main([str(corpus), "--visibility", "public"], factory=factory) == 1
    lines = _report(capsys)
    assert lines[0].startswith("unsupported_file_type") and "blob.bin" in lines[0]
    assert [line.split()[0] for line in lines[1:]] == ["created", "created"]


def test_missing_directory_is_a_usage_error(tmp_path, factory):
    with pytest.raises(SystemExit):
        script.main([str(tmp_path / "absent"), "--visibility", "public"], factory=factory)
