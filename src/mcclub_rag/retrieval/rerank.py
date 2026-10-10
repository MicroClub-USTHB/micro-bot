"""Cross-encoder reranking (retrieval Req 4): ONNX int8, one pair per run, sigmoid scores."""

from collections.abc import Sequence


def passage(title: str, heading_path: Sequence[str], text: str, *, header: bool = True) -> str:
    """Reranker passage: the 'title > heading path' header the chunk was embedded with, then text.

    Same rule as ``ingest/chunk.py::_header`` minus its token trim: the reranker truncates the
    passage side only, so a long header just leaves less room for the text.
    """
    entries: list[str] = []
    for entry in (title, *heading_path) if header else ():
        entry = entry.strip()
        if entry and (not entries or entries[-1].casefold() != entry.casefold()):
            entries.append(entry)
    return f"{' > '.join(entries)}\n\n{text}" if entries else text
