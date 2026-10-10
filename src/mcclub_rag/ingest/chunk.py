"""ParsedDocument -> ordered retrieval chunks (specs/chunking).

Structure first, sentences second, size last: headings are boundaries, oversized sections
split at paragraphs, lines, sentences, words and (last resort) tokens, and runs of tiny
sections under one parent are merged. Every chunk's ``embed_text`` carries a
"title > heading path" header for dense and BM25 retrieval. Pure and deterministic.
"""

import hashlib
import itertools
import json
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

import structlog

from mcclub_rag.ingest.models import Chunk, ParsedDocument, Section
from mcclub_rag.ingest.sections import _HEADING
from mcclub_rag.ingest.settings import IngestSettings, get_ingest_settings
from mcclub_rag.ingest.tokens import TokenCounter, get_token_counter
from mcclub_rag.text.sentences import split_sentences, split_words

log = structlog.get_logger(__name__)

# Bump when the algorithm changes output for the same input and settings: it feeds the
# corpus version, so a bump invalidates the answer cache and triggers a reindex.
CHUNKER_VERSION = 1

BlockKind = Literal["para", "list", "table", "code"]

_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")
_REPACK_ROUNDS = 3
_LANGUAGES = ("ar", "fr", "en")


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


# Finer and finer lossless splitters tried on a block that exceeds the budget; after the
# last one comes a token-level cut. Tables are split by rows separately.
_LEVELS: dict[BlockKind, tuple[Callable[[str], list[str]], ...]] = {
    "para": (_lines, split_sentences, split_words),
    "list": (_lines, split_words),
    "code": (_lines, split_words),
    "table": (_lines, split_words),
}


@dataclass(frozen=True)
class _OutlineSection:
    index: int  # position in ParsedDocument.sections
    path: tuple[str, ...]  # full heading path, own heading included
    heading_line: str | None  # e.g. "## 01"; None when the section continues a heading
    body: str  # section text without its heading line
    page: int | None
    language: str | None


@dataclass(frozen=True)
class _Block:
    kind: BlockKind
    text: str
    tokens: int


@dataclass(frozen=True)
class _Unit:
    """A slice of a block small enough to pack; slices of one block join without a separator."""

    text: str
    tokens: int
    block: int
    hard: bool = False  # produced by a token-level cut


def _outline(sections: Sequence[Section]) -> list[_OutlineSection]:
    """Rebuild the heading tree from the '#' lines kept at the top of each section."""
    stack: list[tuple[int, str]] = []
    outline: list[_OutlineSection] = []
    for index, section in enumerate(sections):
        first, _, rest = section.text.partition("\n")
        match = _HEADING.match(first.strip())
        if match:
            level, heading = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            heading_line, body = first.strip(), rest
        else:
            heading_line, body = None, section.text
            # An XLSX sheet names its section without a '#' line; a PDF page continuing a
            # heading repeats the heading already on the stack and inherits the path.
            if section.heading is not None and (not stack or stack[-1][1] != section.heading):
                stack = [(1, section.heading)]
        outline.append(
            _OutlineSection(
                index=index,
                path=tuple(heading for _, heading in stack),
                heading_line=heading_line,
                body=body.strip("\n"),
                page=section.page,
                language=section.language,
            )
        )
    return outline


def _parse_blocks(body: str, counter: TokenCounter) -> list[_Block]:
    """Split a section body into paragraphs, lists, tables and fenced code blocks."""
    lines = body.split("\n")
    spans: list[tuple[BlockKind, list[str]]] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            spans.append(("para", paragraph.copy()))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if fence := _FENCE.match(line):
            flush()
            end = i + 1
            while end < len(lines) and not lines[end].lstrip().startswith(fence.group(1)):
                end += 1
            spans.append(("code", lines[i : end + 1]))  # unclosed: runs to the end
            i = end + 1
        elif line.lstrip().startswith("|"):
            flush()
            end = i
            while end < len(lines) and lines[end].lstrip().startswith("|"):
                end += 1
            spans.append(("table", lines[i:end]))
            i = end
        elif _LIST_ITEM.match(line):
            flush()
            end = i + 1
            while end < len(lines) and (
                _LIST_ITEM.match(lines[end])
                or (lines[end][:1] in (" ", "\t") and lines[end].strip())
            ):
                end += 1
            spans.append(("list", lines[i:end]))
            i = end
        elif not line.strip():
            flush()
            i += 1
        else:
            paragraph.append(line)
            i += 1
    flush()
    texts = ["\n".join(span).strip("\n") for _, span in spans]
    counts = counter.count_many(texts)
    return [
        _Block(kind, text, tokens)
        for (kind, _), text, tokens in zip(spans, texts, counts, strict=True)
    ]


def _split_text(
    text: str,
    budget: int,
    counter: TokenCounter,
    levels: Sequence[Callable[[str], list[str]]],
    block: int,
) -> list[_Unit]:
    if not levels:
        units = []
        rest = text
        while rest:
            head, rest = counter.cut(rest, budget)
            units.append(_Unit(head, counter.count(head), block, hard=True))
        return units
    pieces = levels[0](text)
    if len(pieces) <= 1:
        return _split_text(text, budget, counter, levels[1:], block)
    units = []
    for piece, tokens in zip(pieces, counter.count_many(pieces), strict=True):
        if tokens <= budget:
            units.append(_Unit(piece, tokens, block))
        else:
            units.extend(_split_text(piece, budget, counter, levels[1:], block))
    return units


def _split_table(text: str, target: int, counter: TokenCounter) -> list[str]:
    """Row groups of at most ``target`` tokens, each repeating the header (and separator)."""
    rows = text.split("\n")
    width = 2 if len(rows) > 1 and _TABLE_SEPARATOR.match(rows[1]) else 1
    header, body = rows[:width], rows[width:]
    size = header_size = counter.count("\n".join(header))
    groups: list[list[str]] = [[]]
    for row, tokens in zip(body, counter.count_many(body), strict=True):
        if groups[-1] and size + tokens > target:
            groups.append([])
            size = header_size
        groups[-1].append(row)
        size += tokens
    return ["\n".join(header + group) for group in groups]


def _units(
    blocks: Sequence[_Block], target: int, budget: int, counter: TokenCounter, ids: Iterator[int]
) -> list[_Unit]:
    """Blocks -> packable units: whole blocks when they fit, otherwise lossless slices."""
    units: list[_Unit] = []
    for block in blocks:
        if block.tokens <= budget:
            units.append(_Unit(block.text, block.tokens, next(ids)))
            continue
        pieces = (
            _split_table(block.text, target, counter) if block.kind == "table" else [block.text]
        )
        for piece, tokens in zip(pieces, counter.count_many(pieces), strict=True):
            if tokens <= budget:
                units.append(_Unit(piece, tokens, next(ids)))
            else:
                units.extend(_split_text(piece, budget, counter, _LEVELS[block.kind], next(ids)))
    return units


def _size(units: Sequence[_Unit], sep: int) -> int:
    joins = sum(1 for a, b in itertools.pairwise(units) if a.block != b.block)
    return sum(u.tokens for u in units) + joins * sep


def _pack(
    units: Sequence[_Unit], target: int, budget: int, min_tokens: int, sep: int
) -> list[list[_Unit]]:
    """Greedy packing up to ``target`` (units are already <= ``budget``), then fix a small tail."""
    pieces: list[list[_Unit]] = []
    current: list[_Unit] = []
    for unit in units:
        if current and _size([*current, unit], sep) > target:
            pieces.append(current)
            current = []
        current.append(unit)
    if current:
        pieces.append(current)
    if len(pieces) > 1 and _size(pieces[-1], sep) < min_tokens:
        prev, last = pieces[-2], pieces[-1]
        if _size(prev + last, sep) <= budget:
            pieces[-2:] = [prev + last]
        else:  # move whole units back into the tail while both stay reasonable
            while len(prev) > 1 and _size(last, sep) < min_tokens:
                moved = [prev[-1], *last]
                if _size(moved, sep) > budget or _size(prev[:-1], sep) < min_tokens:
                    break
                prev, last = prev[:-1], moved
            pieces[-2:] = [prev, last]
    return pieces


def _render(units: Sequence[_Unit]) -> str:
    groups = [
        "".join(u.text for u in group).strip()
        for _, group in itertools.groupby(units, key=lambda u: u.block)
    ]
    return "\n\n".join(g for g in groups if g)


Limits = Callable[[tuple[str, ...]], tuple[int, int]]  # heading path -> (target, budget)


@dataclass
class _Part:
    section: _OutlineSection
    units: list[_Unit]
    first: bool  # first piece of its section, so it may carry the heading line inline


@dataclass
class _Draft:
    path: tuple[str, ...]
    parts: list[_Part] = field(default_factory=list)


def _common_prefix(paths: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    prefix = paths[0]
    for path in paths[1:]:
        n = 0
        while n < min(len(prefix), len(path)) and prefix[n] == path[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


def _inline_heading(part: _Part, path: tuple[str, ...]) -> str | None:
    """The heading line to keep in the text: when it is not already in the chunk's path."""
    section = part.section
    if part.first and section.heading_line and section.path != path[: len(section.path)]:
        return section.heading_line
    return None


def _draft_text(draft: _Draft) -> str:
    texts = []
    for part in draft.parts:
        body = _render(part.units)
        if heading := _inline_heading(part, draft.path):
            body = f"{heading}\n{body}" if body else heading
        if body:
            texts.append(body)
    return "\n\n".join(texts)


def _group(
    outline: Sequence[_OutlineSection], counter: TokenCounter, limits: Limits, min_tokens: int
) -> list[_Draft]:
    """Large sections are packed alone; runs of small sections under one parent are merged."""
    sep = counter.count("\n\n")
    ids = itertools.count()
    units = [_units(_parse_blocks(s.body, counter), *limits(s.path), counter, ids) for s in outline]
    heading_tokens = counter.count_many([s.heading_line or "" for s in outline])
    sizes = [_size(u, sep) for u in units]
    inline = [size + heading for size, heading in zip(sizes, heading_tokens, strict=True)]

    def estimate(draft: _Draft) -> int:
        return sum(inline[p.section.index] + sep for p in draft.parts)

    drafts: list[_Draft] = []
    i = 0
    while i < len(outline):
        section = outline[i]
        nxt = outline[i + 1] if i + 1 < len(outline) else None
        if not units[i] and nxt and nxt.path[: len(section.path)] == section.path != nxt.path:
            i += 1  # heading-only: it lives on in the descendants' heading path
            continue
        if sizes[i] >= min_tokens:
            pieces = _pack(units[i], *limits(section.path), min_tokens, sep)
            drafts.extend(
                _Draft(section.path, [_Part(section, piece, k == 0)])
                for k, piece in enumerate(pieces)
            )
            i += 1
            continue
        parent = section.path[:-1] if section.heading_line else section.path
        group, total, j = [i], inline[i], i + 1
        # Keep absorbing small siblings up to the target; a large one only while still small.
        while j < len(outline) and (total < min_tokens or sizes[j] < min_tokens):
            if outline[j].path[: len(parent)] != parent:
                break  # never cross a higher-level heading
            if total + sep + inline[j] > limits(parent)[0]:
                break
            group.append(j)
            total += sep + inline[j]
            j += 1
        draft = _Draft(
            _common_prefix([outline[k].path for k in group]),
            [_Part(outline[k], units[k], True) for k in group],
        )
        prev = drafts[-1] if drafts else None
        trailing_headings = j == len(outline) and not any(units[k] for k in group)
        if (
            total < min_tokens
            and prev is not None
            and (trailing_headings or prev.path[: len(parent)] == parent)
        ):
            path = _common_prefix([prev.path, draft.path])
            if estimate(prev) + sep + total <= limits(path)[1]:
                prev.path = path
                prev.parts.extend(draft.parts)
                i = j
                continue
        drafts.append(draft)
        i = j
    return drafts


def _paragraphs(text: str) -> list[str]:
    return re.split(r"(?<=\n\n)", text)


_RESPLIT_LEVELS = (_paragraphs, _lines, split_sentences, split_words)


def _header(title: str, path: tuple[str, ...], max_tokens: int, counter: TokenCounter) -> str:
    """'title > h1 > h2' without consecutive repeats, trimmed from the left to fit."""
    entries: list[str] = []
    for entry in (title, *path):
        entry = entry.strip()
        if entry and (not entries or entries[-1].casefold() != entry.casefold()):
            entries.append(entry)
    while len(entries) > 1 and counter.count(" > ".join(entries)) > max_tokens:
        entries.pop(0)
    header = " > ".join(entries)
    if counter.count(header) > max_tokens:
        header = counter.cut(header, max_tokens)[0].rstrip()
    return header


def _language(sections: Sequence[_OutlineSection], doc_language: str) -> str:
    weights: Counter[str] = Counter()
    for section in sections:
        if section.language in _LANGUAGES:
            weights[section.language] += len(section.body)
    if weights:
        return max(sorted(weights), key=weights.__getitem__)
    return doc_language if doc_language in _LANGUAGES else "und"


@dataclass
class _Final:
    path: tuple[str, ...]
    text: str
    sections: list[_OutlineSection]
    hard: bool


def chunk_document(
    doc: ParsedDocument,
    *,
    settings: IngestSettings | None = None,
    counter: TokenCounter | None = None,
) -> tuple[Chunk, ...]:
    """Split a preprocessed document into retrieval chunks (pure, deterministic)."""
    settings = settings or get_ingest_settings()
    counter = counter or get_token_counter(settings)
    outline = _outline(doc.sections)
    if not any(ch.isalnum() for s in outline for ch in s.body + (s.heading_line or "")):
        return ()
    sep = counter.count("\n\n")
    headers: dict[tuple[str, ...], tuple[str, int]] = {}

    def header(path: tuple[str, ...]) -> tuple[str, int]:
        if not settings.chunk_context_header:
            return "", 0
        if path not in headers:
            text = _header(doc.title, path, settings.chunk_max_header_tokens, counter)
            headers[path] = (text, counter.count(text) + sep if text else 0)
        return headers[path]

    def limits(path: tuple[str, ...]) -> tuple[int, int]:
        budget = settings.chunk_max_tokens - header(path)[1]
        target = max(settings.chunk_target_tokens - header(path)[1], settings.chunk_min_tokens)
        return min(target, budget), budget

    def embed(item: _Final) -> str:
        text = header(item.path)[0]
        return f"{text}\n\n{item.text}" if text else item.text

    drafts = _group(outline, counter, limits, settings.chunk_min_tokens)
    items = [
        _Final(
            d.path,
            _draft_text(d),
            [p.section for p in d.parts],
            any(u.hard for p in d.parts for u in p.units),
        )
        for d in drafts
    ]
    items = [item for item in items if item.text]
    # Sums of unit counts can differ from the count of the joined text by a few tokens at
    # junctions: re-split anything over the cap with a tighter budget, then cut tokens.
    for round_ in range(_REPACK_ROUNDS + 1):
        counts = counter.count_many([embed(item) for item in items])
        if all(n <= settings.chunk_max_tokens for n in counts):
            break
        resplit: list[_Final] = []
        for item, n in zip(items, counts, strict=True):
            if n <= settings.chunk_max_tokens:
                resplit.append(item)
                continue
            target, budget = limits(item.path)
            budget = max(budget - (n - settings.chunk_max_tokens) * (round_ + 1), 1)
            if round_ < _REPACK_ROUNDS:
                units = _split_text(item.text, budget, counter, _RESPLIT_LEVELS, 0)
                pieces = _pack(units, min(target, budget), budget, settings.chunk_min_tokens, 0)
            else:  # last resort: plain token cuts, never re-joined
                pieces = [[unit] for unit in _split_text(item.text, budget, counter, (), 0)]
            for piece in pieces:
                text = "".join(u.text for u in piece).strip()
                hard = item.hard or any(u.hard for u in piece)
                if text:
                    resplit.append(_Final(item.path, text, item.sections, hard))
        items = resplit
    else:
        counts = counter.count_many([embed(item) for item in items])

    chunks = []
    for index, (item, n) in enumerate(zip(items, counts, strict=True)):
        pages = [s.page for s in item.sections if s.page is not None]
        chunks.append(
            Chunk(
                index=index,
                text=item.text,
                embed_text=embed(item),
                heading_path=item.path,
                page_start=min(pages, default=None),
                page_end=max(pages, default=None),
                section_index=item.sections[0].index,
                language=_language(item.sections, doc.language),
                token_count=n,
            )
        )
        if item.hard:
            log.warning("chunk_hard_split", content_hash=doc.content_hash, chunk_index=index)
    sizes = [c.token_count for c in chunks]
    log.info(
        "document_chunked",
        content_hash=doc.content_hash,
        chunks=len(chunks),
        tokens_min=min(sizes, default=0),
        tokens_median=statistics.median_low(sizes) if sizes else 0,
        tokens_max=max(sizes, default=0),
        merged=sum(len({s.index for s in item.sections}) - 1 for item in items),
        hard_splits=sum(item.hard for item in items),
    )
    return tuple(chunks)


def chunking_signature(settings: IngestSettings, counter: TokenCounter) -> str:
    """Hash of everything that changes chunk output; folded into ``corpus_version()``."""
    values = {
        name: value
        for name, value in sorted(settings.model_dump().items())
        if name.startswith("chunk_") and name != "chunk_tokenizer_path"
    }
    payload = {"version": CHUNKER_VERSION, "tokenizer": counter.fingerprint, "settings": values}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
