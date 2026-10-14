"""Rebuild the eval corpus dumps in data/parsed from eval/corpus_sources.txt (Req 12.1).

    uv run python scripts/build_eval_corpus.py
    uv run python scripts/build_eval_corpus.py --only finfo.usthb.dz

Each source line is ``<visibility> <URL or file name in data/sources/>``. Every source goes
through the project's own preprocessing (URL guard, fetch, trafilatura; or sniff, xberg, OCR)
and is written in the ``scripts/parse.py`` dump format, under the file name that
``eval/corpus.json`` lists. Web pages are fetched 1 s apart. Prints one line per source and
exits non-zero if any failed. Pages change: rerun the retrieval eval after a rebuild.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from mcclub_rag.ingest.errors import IngestError
from mcclub_rag.ingest.models import ParsedDocument
from mcclub_rag.ingest.preprocess import preprocess_file, preprocess_url

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "eval" / "corpus_sources.txt"
FILES = ROOT / "data" / "sources"
OUT = ROOT / "data" / "parsed"


def read_sources(path: Path) -> list[tuple[str, str]]:
    """``(visibility, target)`` per non-comment line."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            visibility, target = line.split(" ", 1)
            if visibility not in ("public", "members"):
                raise ValueError(f"bad visibility in {line!r}")
            rows.append((visibility, target.strip()))
    return rows


def is_url(target: str) -> bool:
    return target.startswith(("http://", "https://"))


def dump_name(target: str) -> str:
    """File name of the dump: the decoded host + path for URLs, the stem for files."""
    if is_url(target):
        url = urlparse(target)
        name = f"{url.netloc}{unquote(url.path)}"
    else:
        name = Path(target).stem
    return re.sub(r"[^\w.-]+", "_", name).strip("_") + ".md"


def quote_url(target: str) -> str:
    """Percent-encode a readable URL (Arabic, accents) without double-encoding."""
    url = urlparse(target)
    return url._replace(path=quote(unquote(url.path), safe="/-_.~")).geturl()


def render(doc: ParsedDocument) -> str:
    """The ``scripts/parse.py`` dump: metadata, section list, text."""
    lines = ["# Metadata", "", "```", str(doc.model_dump(exclude={"text", "sections"})), "```", ""]
    lines += [f"# Sections ({len(doc.sections)})", ""]
    lines += [
        f"- page={s.page} [{s.language}] {s.heading!r} len={len(s.text)}" for s in doc.sections
    ]
    return "\n".join([*lines, "", "# Text", "", doc.text, ""])


async def build(sources: Sequence[tuple[str, str]], files: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for visibility, target in sources:
        try:
            if is_url(target):
                doc = await preprocess_url(quote_url(target), visibility=visibility)
                time.sleep(1)
            else:
                data = (files / target).read_bytes()
                doc = await preprocess_file(data, target, visibility=visibility)
        except (IngestError, OSError) as exc:
            failures += 1
            print(f"FAIL {target}: {getattr(exc, 'code', type(exc).__name__)}: {exc}")
            continue
        name = dump_name(target)
        (out / name).write_text(render(doc), encoding="utf-8")
        print(f"ok   {doc.language:<5} {len(doc.text.split()):>6}w {name}")
    return failures


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--only", help="rebuild only sources containing this text")
    args = parser.parse_args(argv)
    os.environ.setdefault("TESSDATA_PREFIX", str(ROOT / "models" / "tessdata"))
    sources = [s for s in read_sources(args.sources) if not args.only or args.only in s[1]]
    sys.exit(1 if asyncio.run(build(sources, FILES, OUT)) else 0)


if __name__ == "__main__":
    main()
