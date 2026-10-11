# Eval results

One compact JSON per run (`python -m eval.run_eval --label <name>`): settings, model fingerprints,
corpus version, per-question rows, summary and gate sweep. Corpus: the 4 files of `data/parsed`
(`eval/corpus.json`), 48 questions (36 answerable). Dev machine, 4 threads.

## 2026-10-11: chunk size and context header (clean corpus)

The `data/parsed` files are `scripts/parse.py` dumps (`# Metadata` dict, `# Sections` list, then
`# Text`). Every run up to 2026-10-10 ingested them whole: 9 of the 70 chunks were metadata or
section lists, and every document was titled "Metadata", so every context header started with it.
From here on the eval ingests only the `# Text` part, with the title from `eval/corpus.json`
(61 chunks at the defaults). Runs before this point are not comparable with runs after it.

Defaults otherwise (top_k 20, RRF k 2, gte). Max = target + 100, capped at 400 (the reranker
limit). One question is 0.028 of hit@5 and up to 0.028 of MRR, so read differences under ~0.03
as noise.

| Target / max | Header | Chunks | Fused MRR | Reranked MRR | hit@5 reranked | Cross-lingual reranked | Rerank ms |
|---|---|---|---|---|---|---|---|
| **300 / 400** | **on** | 61 | 0.834 | **0.986** | **1.000** | **0.955** | 1158 |
| 300 / 400 | off | 58 | 0.800 | 0.947 | 1.000 | 0.932 | 1096 |
| 200 / 300 | on | 86 | 0.835 | 0.944 | 0.972 | 0.909 | 852 |
| 200 / 300 | off | 76 | 0.837 | 0.919 | 0.972 | 0.955 | 834 |
| 400 / 400 | on | 54 | **0.847** | 0.958 | 0.972 | 0.955 | 1254 |
| 400 / 400 | off | 49 | 0.820 | 0.921 | 0.972 | 0.939 | 1260 |

Decisions: keep target 300 / max 400 with the header on. The header wins at every size after
reranking (+0.039, +0.025, +0.037 MRR), and it is the only reranked difference above the noise.
200 and 400 each lose a question from the top 5 (en-09). 200 cuts rerank time by a quarter
(shorter pairs), the only argument for it. Gate: answerable questions start at 0.42 (x-02, an
Arabic question on the English VP document, stays at 0.22), unanswerable ones top out at 0.64
(on-topic mcoli-ui questions), so `GATE_THRESHOLD=0.40` stands. No access violations.

## 2026-10-10: fusion, candidate count, reranker

MRR@10 before (fused) and after reranking; hit@5 was 1.0 after reranking in every run (the corpus
is small, so MRR is what separates settings). Rerank = median ms per question.

| Run | Fused MRR | Reranked MRR | Cross-lingual (fused / reranked) | Rerank ms | Gate suggestion |
|---|---|---|---|---|---|
| gte, top_k 30, RRF k 60 | 0.816 | 0.965 | 0.662 / 0.886 | 1891 | 0.41 |
| gte, top_k 20 | 0.816 | 0.968 | 0.662 / 0.894 | 1257 | 0.41 |
| gte, top_k 15 | 0.816 | 0.968 | 0.662 / 0.894 | 971 | 0.41 |
| gte, top_k 10 | 0.816 | 0.972 | 0.662 / 0.909 | 640 | 0.41 |
| gte, RRF k 2 | **0.861** | 0.965 | **0.809** / 0.886 | 1854 | 0.41 |
| gte, DBSF | 0.800 (hit@5 0.972) | 0.965 | 0.632 / 0.886 | 1865 | 0.41 |
| gte, sparse weight 0.5 | 0.744 | 0.965 | 0.509 / 0.886 | 1869 | 0.41 |
| gte, dense weight 0.5 | 0.826 (hit@5 0.972) | 0.965 | 0.700 / 0.886 | 1793 | 0.41 |
| bge, top_k 30 | 0.816 | **0.979** | 0.662 / 0.932 | 4121 | 0.15 |
| **gte, top_k 20, RRF k 2 (defaults)** | 0.861 | 0.968 | 0.809 / 0.894 | 1251 | 0.41 |

Decisions: `RETRIEVE_TOP_K=20` (no MRR loss against 30, a third less rerank time, headroom as the
corpus grows), `RETRIEVAL_RRF_K=2` (better fused order, which matters once the candidate list is
cut), RRF with equal weights, gte kept (bge +0.014 MRR for 2.2x the latency, and worse on
follow-ups). No access violations in any run.

Gate (defaults run): unanswerable questions top out at 0.388, answerable ones start at 0.412
except x-02 (Arabic question on the English VP document, 0.26). `GATE_THRESHOLD=0.40` sits in the
gap: refuses 9 of 12 unanswerable (the 3 kept are on-topic mcoli-ui questions, 0.54 to 0.55) and
1 of 36 answerable. The threshold is per model: recalibrate when `RERANK_MODEL` changes.
