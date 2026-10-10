# Eval results

One compact JSON per run (`python -m eval.run_eval --label <name>`): settings, model fingerprints,
corpus version, per-question rows, summary and gate sweep. Corpus: the 4 files of `data/parsed`
(`eval/corpus.json`, 70 chunks), 48 questions (36 answerable). Dev machine, 4 threads.

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
