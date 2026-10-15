"""Summary and markdown of a model matrix run (``eval/bench_models.py``, query-rewrite Req 4).

Retrieval quality is measured on what the LLM sees (``RERANK_TOP_N`` = 5): hit@5 = an answer
chunk holding the evidence is among the 5 reranked chunks. ``recall@20`` is the same before
reranking (the embedder + BM25 candidates). Gate: good = answerable with a hit@5; bad =
unanswerable cases plus answerable misses, so a high AUC means a low top score really flags
bad retrieval.
"""

import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from eval.metrics import auc, best_threshold

GROUPS = ("fr", "en", "ar", "darija", "arabizi", "cross_lingual", "noisy", "followup")
SHORT = {
    "BAAI/bge-m3": "bge-m3 (568 MB)",
    "ibm-granite/granite-embedding-311m-multilingual-r2": "granite-311m (300 MB)",
    "ibm-granite/granite-embedding-97m-multilingual-r2": "granite-97m (98 MB)",
    "intfloat/multilingual-e5-small": "e5-small (118 MB)",
    "BAAI/bge-reranker-v2-m3": "bge-rr (571 MB)",
    "Alibaba-NLP/gte-multilingual-reranker-base": "gte-rr (341 MB)",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": "mmarco-L12 (119 MB)",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "minilm-en (23 MB)",
}


def groups(case: Mapping) -> list[str]:
    """Language group (arabizi and darija slices first, else the case language) + tags."""
    slices = set(case.get("slices", []))
    lang = "arabizi" if "arabizi" in slices else "darija" if "darija" in slices else case["lang"]
    return [lang] + [g for g in ("cross_lingual", "noisy", "followup") if g in slices]


def _pct(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))] if ordered else 0.0


def quality(rows: Sequence[Mapping]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    hit = [r["rank"] for r in rows]
    return {
        "n": n,
        "hit@1": round(sum(r == 1 for r in hit) / n, 3),
        "hit@5": round(sum(r is not None and r <= 5 for r in hit) / n, 3),
        "mrr@10": round(sum(1 / r for r in hit if r) / n, 3),
        "recall@20": round(sum(r["rank_fused"] is not None for r in rows) / n, 3),
    }


def summarize(payload: Mapping) -> list[dict]:
    cases = payload["cases"]
    combos: dict[tuple, list] = defaultdict(list)
    for row in payload["rows"]:
        combos[(row["embedder"], row["reranker"], row["mode"])].append(row)
    out = []
    for (emb, rr, mode), rows in combos.items():
        answer = [r for r in rows if cases[r["case"]]["expect"] == "answer"]
        refuse = [r for r in rows if cases[r["case"]]["expect"] == "refuse"]
        good = [r["top_score"] for r in answer if r["rank"] and r["rank"] <= 5]
        missed = [r["top_score"] for r in answer if not r["rank"] or r["rank"] > 5]
        bad = missed + [r["top_score"] for r in refuse]
        total = [r["ms_encode"] + r["ms_search"] + r["ms_rerank"] for r in rows]
        by_group = defaultdict(list)
        for r in answer:
            for g in groups(cases[r["case"]]):
                by_group[g].append(r)
        out.append(
            {
                "embedder": emb,
                "reranker": rr,
                "mode": mode,
                "overall": quality(answer),
                "groups": {g: quality(by_group[g]) for g in GROUPS if by_group[g]},
                "ms": {
                    "p50": round(statistics.median(total), 1),
                    "p95": round(_pct(total, 0.95), 1),
                    "encode_p50": statistics.median(r["ms_encode"] for r in rows),
                    "rerank_p50": round(statistics.median(r["ms_rerank"] for r in rows), 1),
                },
                "gate": {
                    "auc": auc(good, bad),
                    "auc_unanswerable": auc(good, [r["top_score"] for r in refuse]),
                    "auc_miss": auc(good, missed),
                    "best": best_threshold(good, bad),
                },
            }
        )
    return sorted(out, key=lambda s: -s["overall"].get("hit@5", 0))


def _name(model: str) -> str:
    return SHORT.get(model, model)


def markdown(payload: Mapping) -> str:
    summary = summarize(payload)
    lines = [f"# Model matrix {payload['label']}", "", f"Rewrites by {payload['llm']}.", ""]
    lines += [
        "| Embedder | Reranker | Query | hit@5 | hit@1 | MRR@10 | recall@20 | p50 ms | p95 ms "
        "| gate AUC |",
        "|---" * 10 + "|",
    ]
    for s in summary:
        o, ms = s["overall"], s["ms"]
        lines.append(
            f"| {_name(s['embedder'])} | {_name(s['reranker'])} | {s['mode']} | {o['hit@5']} "
            f"| {o['hit@1']} | {o['mrr@10']} | {o['recall@20']} | {ms['p50']} | {ms['p95']} "
            f"| {s['gate']['auc']} |"
        )
    present = [g for g in GROUPS if any(g in s["groups"] for s in summary)]
    lines += ["", "## hit@5 per group", ""]
    lines += ["| Embedder | Reranker | Query | " + " | ".join(present) + " |"]
    lines += ["|---" * (3 + len(present)) + "|"]
    for s in summary:
        cells = " | ".join(str(s["groups"].get(g, {}).get("hit@5", "")) for g in present)
        lines.append(f"| {_name(s['embedder'])} | {_name(s['reranker'])} | {s['mode']} | {cells} |")
    counts = {g: s["groups"][g]["n"] for s in summary[:1] for g in s["groups"]}
    lines += ["", "Cases per group: " + ", ".join(f"{g} {n}" for g, n in counts.items())]
    lines += ["", "## Gate", "", "| Embedder | Reranker | Query | AUC all | AUC vs unanswerable "]
    lines[-1] += "| AUC vs misses | best t | kept good | refused bad |"
    lines += ["|---" * 9 + "|"]
    for s in summary:
        g, best = s["gate"], s["gate"]["best"] or {}
        lines.append(
            f"| {_name(s['embedder'])} | {_name(s['reranker'])} | {s['mode']} | {g['auc']} "
            f"| {g['auc_unanswerable']} | {g['auc_miss']} | {best.get('threshold')} "
            f"| {best.get('kept_good')} | {best.get('caught_bad')} |"
        )
    rewrites = payload["rewrites"].values()
    langs = Counter((r["rewrite"] or {}).get("lang") or r["error"] or "none" for r in rewrites)
    ms = [r["ms"] for r in rewrites]
    lines += [
        "",
        "## Rewrites",
        "",
        "Languages / errors: " + ", ".join(f"{k} {v}" for k, v in langs.most_common()),
    ]
    lines += [f"Rewrite time p50 {statistics.median(ms)} ms, p95 {_pct(ms, 0.95)} ms (this LLM)."]
    lines += ["", "## Ingest", "", "| Embedder | chunks | seconds | chunks/s |"]
    lines += ["|---|---|---|---|"]
    for name, i in payload["ingest"].items():
        rate = round(i["chunks"] / i["seconds"], 1) if i["seconds"] else ""
        lines.append(f"| {_name(name)} | {i['chunks']} | {i['seconds']} | {rate} |")
    return "\n".join(lines) + "\n"
