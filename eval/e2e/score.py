"""Rule-based scoring of one response (Req 3) and of one retrieval (Req 6.4). No LLM.

Each check is ``pass`` / ``fail`` / ``skip`` with a short reason, so a failed case can be read
without rerunning it. Strings are compared folded (``eval.metrics.fold``: casefold, Arabic
letter variants, diacritics, collapsed whitespace), the same folding as the retrieval labels.
"""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import ValidationError

from eval.e2e.cases import Case, CorpusDoc
from eval.metrics import fold
from mcclub_rag.answer import REFUSAL
from mcclub_rag.schemas import ChatResponse, ErrorResponse, Retrieval
from mcclub_rag.text.lang import detect_language

Status = Literal["pass", "fail", "skip"]
# Mirrors the gate refusal text of api/chat.py (checked by tests/test_e2e_driver.py).
GATE_REFUSAL = REFUSAL["en"]
GATE_REFUSALS = frozenset(REFUSAL.values())  # the gate refuses in the question's language (I5)


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    reason: str = ""


@dataclass
class Observation:
    """What a driver saw for one request."""

    status: int | None
    body: object = None
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class CaseResult:
    id: str
    layer: str  # chat, retrieval, or the system suite name
    path: str
    checks: list[Check]
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(c.status == "fail" for c in self.checks)

    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    def to_dict(self) -> dict:
        return asdict(self) | {"passed": self.passed}


def _check(name: str, ok: bool, reason: str = "") -> Check:
    return Check(name, "pass" if ok else "fail", "" if ok else reason)


def _skip(name: str, reason: str) -> Check:
    return Check(name, "skip", reason)


def _contains_any(text: str, needles: Iterable[str]) -> list[str]:
    folded = fold(text)
    return [n for n in needles if fold(n) and fold(n) in folded]


def path_of(response: ChatResponse) -> str:
    """How the answer was produced (Req 3.5): llm, gate_refusal or fallback."""
    if response.usage.llm_calls >= 1:
        return "llm"
    if not response.sources and response.answer in GATE_REFUSALS:
        return "gate_refusal"
    return "fallback"


def parse_chat(obs: Observation) -> tuple[ChatResponse | None, Check]:
    """The contract check: 200 with a valid ChatResponse, or the error JSON shape."""
    if obs.error:
        return None, Check("contract", "fail", f"request failed: {obs.error}")
    if obs.status != 200:
        try:
            code = ErrorResponse.model_validate(obs.body).error.code
        except ValidationError:
            return None, Check("contract", "fail", f"{obs.status} without the error JSON shape")
        return None, Check("contract", "fail", f"{obs.status} {code}")
    try:
        return ChatResponse.model_validate(obs.body), Check("contract", "pass")
    except ValidationError as exc:
        return None, Check("contract", "fail", f"invalid ChatResponse: {exc.error_count()} errors")


def score_chat(
    case: Case,
    obs: Observation,
    corpus: Mapping[str, CorpusDoc],
    topics: Mapping[str, str],
) -> CaseResult:
    """All checks of Req 3 for one chat response. ``topics`` maps doc_id -> corpus topic."""
    response, contract = parse_chat(obs)
    if response is None:
        return CaseResult(case.id, "chat", "error", [contract], obs.latency_ms)
    path = path_of(response)
    answered = response.grounded and bool(response.sources)
    snippets = " ".join(s.snippet for s in response.sources)
    # A canary in a snippet is a leak only for a public user: members may read those documents.
    canaries = [d.canary for d in corpus.values() if d.canary] if case.role == "public" else []
    checks = [contract, _check("llm_calls", response.usage.llm_calls <= 1, "more than 1 call")]

    if case.expect == "any":
        checks.append(_skip("behaviour", "expect any"))
    elif case.expect == "answer":
        checks.append(_check("behaviour", answered, f"expected an answer, got {path}"))
    else:
        checks.append(_check("behaviour", not answered, f"expected a refusal, got {path}"))

    if case.expect == "answer" and case.gold:
        checks.append(_check("gold", bool(_contains_any(response.answer, case.gold)), "no gold"))
    else:
        checks.append(_skip("gold", "no gold to check"))
    if case.expect == "answer" and case.evidence:
        found = _contains_any(snippets, case.evidence)
        checks.append(_check("evidence", bool(found), "evidence not in any cited snippet"))
    else:
        checks.append(_skip("evidence", "no evidence to check"))

    leaked = _contains_any(response.answer, case.must_not) + _contains_any(snippets, canaries)
    checks.append(_check("must_not", not leaked, f"contains {sorted(set(leaked))}"))
    members = [
        s.doc_id
        for s in response.sources
        if corpus.get(topics.get(s.doc_id, ""), None) is not None
        and corpus[topics[s.doc_id]].visibility == "members"
    ]
    if case.role == "public":
        checks.append(_check("access", not members, f"members-only sources {members}"))
    else:
        checks.append(_skip("access", "member role"))

    if path == "gate_refusal" or (path == "fallback" and not answered):  # notice: user's language
        checks.append(_skip("language", "gate refusal"))
        lang = detect_language(response.answer)
        checks.append(_check("refusal_language", lang == case.lang, f"refusal in {lang}"))
    elif answered:
        lang = detect_language(response.answer)
        checks.append(
            _check("language", lang == case.lang, f"answer in {lang}, asked in {case.lang}")
        )
        checks.append(_skip("refusal_language", "not a gate refusal"))
    else:
        checks.append(_skip("language", "not answered"))
        checks.append(_skip("refusal_language", "not a gate refusal"))
    return CaseResult(case.id, "chat", path, checks, obs.latency_ms, {"answered": answered})


def score_retrieval(
    case: Case,
    retrieval: Retrieval,
    corpus: Mapping[str, CorpusDoc],
    topics: Mapping[str, str],
    latency_ms: float = 0.0,
) -> CaseResult:
    """Retrieval-layer checks (no LLM): gate decision, evidence in the chunks, access."""
    text = " ".join(f"{c.title} {c.text}" for c in retrieval.chunks)
    checks = []
    if case.expect == "any":
        checks.append(_skip("behaviour", "expect any"))
    else:
        wanted = "answer" if case.expect == "answer" else "refuse"
        reason = f"gate said {retrieval.decision} (top {retrieval.top_score:.3f})"
        checks.append(_check("behaviour", retrieval.decision == wanted, reason))
    if case.expect == "answer" and case.evidence:
        found = _contains_any(text, case.evidence)
        checks.append(_check("evidence", bool(found), "evidence not in the returned chunks"))
    else:
        checks.append(_skip("evidence", "no evidence to check"))
    chunk_topics = [topics.get(c.doc_id, "") for c in retrieval.chunks]
    if case.role == "public":
        members = [t for t in chunk_topics if t in corpus and corpus[t].visibility == "members"]
        checks.append(_check("access", not members, f"members-only chunks from {members}"))
    else:
        checks.append(_skip("access", "member role"))
    extra = {
        "answered": retrieval.decision == "answer",
        "top_score": round(retrieval.top_score, 4),
        "top_topics": chunk_topics[:3],
    }
    return CaseResult(case.id, "retrieval", retrieval.decision, checks, latency_ms, extra)
