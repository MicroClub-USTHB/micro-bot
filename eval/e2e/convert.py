"""Convert the retrieval eval questions into e2e cases (Req 1.3).

    uv run python -m eval.e2e.convert

Reads ``eval/questions.jsonl`` and writes ``eval/e2e/cases_retrieval.jsonl``: ``answerable`` ->
``expect``, ``evidence`` kept, language tags dropped (``lang`` says it), the other tags mapped to
slices. New cases go in ``eval/e2e/cases.jsonl``, never in the generated file.
"""

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "eval" / "questions.jsonl"
TARGET = REPO / "eval" / "e2e" / "cases_retrieval.jsonl"

_LANG_TAGS = {"arabic", "english", "french"}
_TAG_SLICES = {
    "cross_lingual": "cross_lingual",
    "follow_up": "followup",
    "access": "access",
    "darija": "darija",
    "arabizi": "arabizi",
}
# Read by hand: the only unanswerable question that is not about the club or its documents.
_OFF_TOPIC = {"un-08"}
# Facts found in more documents once the corpus grew (2026-10-11): every one is a correct hit.
_MORE_DOCS = {
    "en-07": ["president-candidacy-2026", "mc-code-of-conduct-2025-2026"],
    "x-02": ["president-candidacy-2026", "mc-code-of-conduct-2025-2026"],
}

# The cited snippet may state the answer in other words than the retrieval label: a table row,
# or another document's phrasing. E2E only; the retrieval labels stay as they are.
_MORE_EVIDENCE = {
    "x-12": ["Submissions per minute | 120"],
    "x-13": ["The callback code lives for 120 seconds"],
    "dj-07": ['POST "$MCDI_URL/auth/token"'],
}


def convert_row(row: dict) -> dict:
    tags = set(row["tags"]) - _LANG_TAGS
    slices = [_TAG_SLICES[t] for t in sorted(tags) if t in _TAG_SLICES]
    if "unanswerable" in tags:
        slices.append(
            "unanswerable_off_topic" if row["id"] in _OFF_TOPIC else "unanswerable_in_domain"
        )
    elif row["answerable"] and not slices:
        slices.append("lookup")
    case = {
        "id": row["id"],
        "question": row["question"],
        "lang": row["lang"],
        "role": row["role"],
        "expect": "answer" if row["answerable"] else "refuse",
    }
    if row.get("doc"):
        case["docs"] = [row["doc"], *_MORE_DOCS.get(row["id"], [])]
    if row.get("evidence"):
        case["evidence"] = [*row["evidence"], *_MORE_EVIDENCE.get(row["id"], [])]
    if row.get("history"):
        case["history"] = row["history"]
    case["slices"] = slices
    return case


def render(rows: Iterable[dict]) -> str:
    return "".join(json.dumps(convert_row(r), ensure_ascii=False) + "\n" for r in rows)


def read_rows(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args(argv)
    args.target.write_text(render(read_rows(args.source)), encoding="utf-8")
    print(f"wrote {args.target}")


if __name__ == "__main__":
    main()
