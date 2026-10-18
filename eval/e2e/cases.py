"""E2E eval cases (Req 1) and the corpus manifests they point at (Req 2).

A case says what a correct response is: ``expect`` answer or refuse, ``gold`` strings for the
answer text, ``evidence`` for the cited snippets, ``must_not`` strings that may never appear.
Every public case also gets the canary of each members-only document in ``must_not``, and every
case the injection markers, so a leak or an obeyed injection is a plain string match.
"""

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from micro_bot.schemas import Role, Visibility

# The edge-case catalogue (Req 11), by group: retrieval, language, form, refusal,
# conversation, safety.
SLICES = frozenset(
    {
        *("lookup", "numeric", "date", "list", "yes_no", "multi_doc", "near_duplicate"),
        *("long_doc", "ocr", "table"),
        *("cross_lingual", "code_switch", "darija", "arabizi", "dialect_fr"),
        *("short", "long_input", "noisy", "special_chars", "negation"),
        *("unanswerable_in_domain", "unanswerable_off_topic", "false_premise", "partial"),
        *("future", "ambiguous"),
        *("followup", "topic_switch", "followup_lang_switch", "chitchat"),
        *("access", "injection", "indirect_injection", "history_injection"),
        *("system_prompt_leak", "opinion", "pii", "harmful"),
    }
)


class EvalDataError(ValueError):
    """A case file or corpus manifest is invalid; the message names the case or topic."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TurnIn(_Strict):
    user_message: str
    assistant_message: str


class Case(_Strict):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    lang: Literal["ar", "fr", "en"]
    role: Role
    expect: Literal["answer", "refuse", "any"]  # any: only the safety checks apply
    docs: list[str] = []  # every topic that holds the answer
    gold: list[str] = []
    evidence: list[str] = []
    must_not: list[str] = []
    history: list[TurnIn] = []
    slices: list[str] = []

    @field_validator("slices")
    @classmethod
    def known_slices(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - SLICES)
        if unknown:
            raise ValueError(f"unknown slices {unknown}")
        return value


class CorpusDoc(_Strict):
    file: str
    topic: str
    title: str
    visibility: Visibility
    canary: str | None = None
    injection_marker: str | None = None


def load_corpus(*manifests: Path) -> dict[str, CorpusDoc]:
    """Topic -> document over one or more JSON manifests (a list of documents each)."""
    docs: dict[str, CorpusDoc] = {}
    for manifest in manifests:
        for raw in json.loads(manifest.read_text(encoding="utf-8")):
            try:
                doc = CorpusDoc.model_validate(raw)
            except ValidationError as exc:
                raise EvalDataError(f"{manifest.name}: {raw.get('topic')!r}: {exc}") from None
            if doc.topic in docs:
                raise EvalDataError(f"{manifest.name}: duplicate topic {doc.topic!r}")
            if doc.canary and doc.visibility != "members":
                raise EvalDataError(f"{manifest.name}: canary on public document {doc.topic!r}")
            docs[doc.topic] = doc
    return docs


def load_cases(paths: Path | Iterable[Path], corpus: dict[str, CorpusDoc]) -> list[Case]:
    """Parse and check JSONL case files (ids unique across them), expand ``must_not``."""
    canaries = [d.canary for d in corpus.values() if d.canary]
    markers = [d.injection_marker for d in corpus.values() if d.injection_marker]
    cases: list[Case] = []
    seen: set[str] = set()
    for path, lineno, line in _lines([paths] if isinstance(paths, Path) else paths):
        if not line.strip():
            continue
        raw = json.loads(line)
        where = f"{path.name}:{lineno} {raw.get('id')!r}"
        try:
            case = Case.model_validate(raw)
        except ValidationError as exc:
            raise EvalDataError(f"{where}: {exc}") from None
        if case.id in seen:
            raise EvalDataError(f"{where}: duplicate id {case.id!r}")
        unknown = [topic for topic in case.docs if topic not in corpus]
        if unknown:
            raise EvalDataError(f"{where}: unknown docs {unknown}")
        if case.expect == "answer" and not (case.gold or case.evidence):
            raise EvalDataError(f"{where}: an answer case needs gold or evidence")
        seen.add(case.id)
        extra = (canaries if case.role == "public" else []) + markers
        cases.append(case.model_copy(update={"must_not": _unique([*case.must_not, *extra])}))
    return cases


def split(case_id: str) -> Literal["dev", "test"]:
    """Stable 80/20 split on the base id, so a variant ``<id>~<kind>`` follows its base."""
    base = case_id.split("~", 1)[0]
    return "test" if int(hashlib.sha1(base.encode()).hexdigest()[:8], 16) % 5 == 0 else "dev"


def _lines(paths: Iterable[Path]) -> Iterator[tuple[Path, int, str]]:
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            yield path, lineno, line


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
