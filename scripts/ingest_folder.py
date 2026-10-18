"""Ingest every file of a folder through the pipeline (initial corpus load).

Prints one line per file (status or error code) and exits non-zero if any file failed.
Running it twice is safe: unchanged files come back as ``duplicate``. Topics default to the
filename stem, as for API uploads, so two files with the same stem replace each other.

    uv run python scripts/ingest_folder.py data/corpus
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from micro_bot.ingest.errors import IngestError
from micro_bot.ingest.pipeline import IngestPipeline, build_pipeline
from micro_bot.ingest.settings import IngestSettings, get_ingest_settings


def _files(directory: Path, recursive: bool) -> list[Path]:
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in candidates if p.is_file() and not p.name.startswith("."))


async def run(
    directory: Path,
    recursive: bool,
    factory: Callable[[IngestSettings], IngestPipeline],
) -> int:
    pipeline = factory(get_ingest_settings())
    failures = 0
    try:
        await pipeline.start()
        for path in _files(directory, recursive):
            name = path.relative_to(directory)
            try:
                result = await pipeline.ingest_file(path, path.name)
            except IngestError as exc:
                failures += 1
                print(f"{exc.code:<22} {name}: {exc.message}")
                continue
            print(f"{result.status:<22} {name} ({result.chunk_count} chunks)")
    except IngestError as exc:
        print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    finally:
        await pipeline.close()
    return 1 if failures else 0


def main(
    argv: Sequence[str] | None = None,
    *,
    factory: Callable[[IngestSettings], IngestPipeline] = build_pipeline,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--recursive", action="store_true", help="include subfolders")
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")
    return asyncio.run(run(args.directory, args.recursive, factory))


if __name__ == "__main__":
    raise SystemExit(main())
