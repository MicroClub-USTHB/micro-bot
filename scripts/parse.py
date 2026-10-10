"""Run the preprocessing pipeline on a file or URL and print the result.

    uv run python scripts/parse.py ~/Downloads/some.pdf
    uv run python scripts/parse.py https://www.bbc.com/arabic
    uv run python scripts/parse.py some.pdf --out result.md

The result is written to a markdown file (default: data/parsed/<name>.md) holding the
metadata, the section list and the full normalized text.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from mcclub_rag.ingest.preprocess import preprocess_file, preprocess_url

ROOT = Path(__file__).resolve().parent.parent
LOCAL_TESSDATA = ROOT / "models" / "tessdata"
DEFAULT_OUT_DIR = ROOT / "data" / "parsed"


def default_out_path(target: str) -> Path:
    if target.startswith(("http://", "https://")):
        url = urlparse(target)
        name = f"{url.netloc}{url.path}"
    else:
        name = Path(target).stem
    slug = re.sub(r"[^\w.-]+", "_", name).strip("_") or "parsed"
    return DEFAULT_OUT_DIR / f"{slug}.md"


async def run(target: str, visibility: str):
    if target.startswith(("http://", "https://")):
        return await preprocess_url(target, visibility=visibility)
    path = Path(target).expanduser()
    return await preprocess_file(path.read_bytes(), path.name, visibility=visibility)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="path to a file, or an http(s) URL")
    parser.add_argument("--visibility", choices=("public", "members"), default="members")
    parser.add_argument("--out", type=Path, help="output file (default: data/parsed/<name>.md)")
    args = parser.parse_args()

    if LOCAL_TESSDATA.is_dir():
        os.environ.setdefault("TESSDATA_PREFIX", str(LOCAL_TESSDATA))

    doc = asyncio.run(run(args.target, args.visibility))
    out = args.out or default_out_path(args.target)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Metadata", "", "```", str(doc.model_dump(exclude={"text", "sections"})), "```", ""]
    lines += [f"# Sections ({len(doc.sections)})", ""]
    lines += [
        f"- page={s.page} [{s.language}] {s.heading!r} len={len(s.text)}" for s in doc.sections
    ]
    lines += ["", "# Text", "", doc.text, ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
