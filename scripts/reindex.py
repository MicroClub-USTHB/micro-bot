"""Rebuild the vector index from the stored parsed documents (DATA_DIR/documents).

Use it after changing the embedder or chunking settings when INGEST_REINDEX_ON_STARTUP is
false, or to force a rebuild. Search keeps working on the old collection until the new one
is complete. Do not run it while another process is ingesting.

    uv run python scripts/reindex.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence

from mcclub_rag.ingest.errors import IngestError
from mcclub_rag.ingest.pipeline import IngestPipeline, build_pipeline
from mcclub_rag.ingest.settings import IngestSettings, get_ingest_settings


async def run(factory: Callable[[IngestSettings], IngestPipeline]) -> int:
    settings = get_ingest_settings().model_copy(update={"reindex_on_startup": True})
    pipeline = factory(settings)
    try:
        if await pipeline.start() != "reindexed":
            await pipeline.reindex()
    except IngestError as exc:
        print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    finally:
        await pipeline.close()
    print(f"reindexed into signature {pipeline.signature[:12]}")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    factory: Callable[[IngestSettings], IngestPipeline] = build_pipeline,
) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    return asyncio.run(run(factory))


if __name__ == "__main__":
    raise SystemExit(main())
