"""Case suites: quality (chat or retrieval layer) and robustness (Req 3, 5, 6).

A suite takes a driver, the cases and the corpus, and returns one ``CaseResult`` per request.
Chat requests that could reach the LLM stop once the generator's budget is spent: later
answers would be extractive fallbacks, so they are reported ``not_run`` instead of scored.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from eval.e2e.cases import Case, CorpusDoc
from eval.e2e.perturb import variants
from eval.e2e.score import CaseResult, Check, score_chat, score_retrieval


@dataclass
class Context:
    driver: object
    corpus: Mapping[str, CorpusDoc]
    topics: dict[str, str]  # doc_id -> topic
    budget: object | None = None  # eval.e2e.generator.Budget of the in-process generator
    judge: object | None = None  # eval.e2e.judge.Judge, when configured
    notes: list[str] = field(default_factory=list)


def not_run(case: Case, layer: str, reason: str) -> CaseResult:
    return CaseResult(case.id, layer, "not_run", [Check("run", "skip", reason)])


def _spent(ctx: Context) -> bool:
    budget = ctx.budget
    return budget is not None and budget.used >= budget.limit


async def run_case(ctx: Context, case: Case, layer: str) -> CaseResult:
    if case.history and not ctx.driver.supports_history:
        return not_run(case, layer, "history needs the in-process driver")
    if layer == "retrieval":
        retrieval, ms = await ctx.driver.retrieve(case)
        return score_retrieval(case, retrieval, ctx.corpus, ctx.topics, ms)
    spent_before = _spent(ctx)
    observation = await ctx.driver.chat(case)
    result = score_chat(case, observation, ctx.corpus, ctx.topics)
    if spent_before and result.path == "fallback":
        return not_run(case, layer, "LLM budget spent (--max-llm-calls)")
    if ctx.judge is not None and isinstance(observation.body, dict):
        ctx.judge.apply(result, observation.body)
    return result


async def quality(ctx: Context, cases: Sequence[Case], layer: str = "chat") -> list[CaseResult]:
    results = [await run_case(ctx, case, layer) for case in cases]
    if layer == "chat":
        answerable = [r for r, c in zip(results, cases, strict=True) if c.expect == "answer"]
        not_llm = [r for r in answerable if r.path not in ("llm", "not_run")]
        if not_llm:
            paths = sorted({r.path for r in not_llm})
            ctx.notes.append(
                f"generation NOT evaluated: {len(not_llm)}/{len(answerable)} answerable cases "
                f"were not answered by the LLM (paths {paths})"
            )
    return results


async def robustness(
    ctx: Context, cases: Sequence[Case], *, repeat: int = 1, seed: int = 0
) -> tuple[list[CaseResult], list[Case]]:
    """Base cases and their variants on the retrieval layer, then ``repeat`` reruns."""
    variant_cases = [v for case in cases for v in variants(case, seed)]
    results = [await run_case(ctx, c, "retrieval") for c in [*cases, *variant_cases]]
    for i in range(1, repeat):
        for case in cases:
            result = await run_case(ctx, case, "retrieval")
            result.extra["repeat"] = i
            results.append(result)
    if repeat > 1:
        for result in results[: len(cases)]:
            result.extra["repeat"] = 0
    return results, variant_cases
