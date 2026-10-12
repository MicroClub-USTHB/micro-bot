# Eval results

One compact JSON per run (`python -m eval.run_eval --label <name>`): settings, model fingerprints,
corpus version, per-question rows, summary and gate sweep. Corpus: the 4 files of `data/parsed`
(`eval/corpus.json`), 48 questions (36 answerable), until the sections below say otherwise. Dev
machine, 4 threads.

## 2026-10-12: eval fixes, retrieval side (`fx-*`, `specs/eval-fixes/`)

Granite, gte, defaults unless named, 86 questions (74 answerable). `fx-base` reproduces `xl86-granite`
exactly (0.838 / 0.762), so the runs compare to it.

**Cross-language dense arm (I9, Req 10).** `RETRIEVAL_CROSS_LANG_PREFETCH=n` adds a dense arm over
chunks not in the question's language.

| Run | hit@5 / MRR | Cross-lingual hit@5 / MRR | Darija hit@5 fused / reranked | English hit@5 | Rerank p95 ms |
|---|---|---|---|---|---|
| fx-base | 0.838 / 0.762 | 0.755 / 0.651 | 0.71 / 0.43 | 0.90 | 2086 |
| cross arm 10 | 0.838 / 0.762 | 0.755 / 0.651 | 0.71 / 0.43 | 0.90 | 1901 |
| **cross arm 20** | **0.851 / 0.770** | **0.776 / 0.663** | 0.71 / 0.43 | 0.95 | 1901 |
| cross 20 + prefetch 80, top_k 30 | 0.865 / 0.786 | 0.796 / 0.687 | 0.71 / 0.43 | 1.00 | 2750 |
| bge-m3 + cross 20 | 0.878 / 0.792 | 0.816 / 0.696 | 0.71 / 0.43 | 0.95 | 1879 |

The arm adds candidates (fused MRR 0.582 -> 0.641) and costs nothing measurable at 20; no
language group gets worse. None reaches the 0.85 cross-lingual target (Req 10.2): bge-m3 + arm is
the same 0.816 as bge-m3 alone, and gte still drops darija from 0.71 fused to 0.43. **Default left
at 0** as the plan requires; `RETRIEVAL_CROSS_LANG_PREFETCH=20` is a free +0.02 to turn on. The
remaining levers are the reranker on darija and ingest-time keywords, out of this branch's scope.

**PDF titles (I15, Req 14).** Same corpus with the three PDF titles set three ways:

| Titles | hit@5 / MRR | Cross-lingual MRR |
|---|---|---|
| old (`MICRO CLUB`, `KENPA SG DOC`: what upload gives today) | 0.824 / 0.750 | 0.636 |
| page-1 heuristic (`fx-auto-titles`) | 0.824 / 0.770 | 0.663 |
| curated (`eval/corpus.json`) | 0.838 / 0.762 | 0.651 |

**Snippets (I6, Req 6).** Offline, on the chunk holding each answerable question's evidence: the
first 500 characters show it for 53/74 (72%, ISSUES.md had 70.9%), `best_snippet(chunk, question)`
for 61/74 (82%). 11 of the 13 misses are cross-lingual (no shared words); the LLM path can pass
the answer text instead of the question.

**Injection flag (I3/I11, Req 4.5).** `looks_like_instruction` flags 0 of 2,314 paragraphs of the
real corpus and both adversarial documents.

**Throughput (I19, Req 17).** `python -m eval.bench_load` (`20261012T190242Z-load`): 4 users,
30 s each, retrieval only, 4 reranker threads in total.

| Workers | Rerank top-k | hit@5 / MRR | req/s | p50 / p95 ms | Peak RSS MB |
|---|---|---|---|---|---|
| 1 | 20 (default) | 0.838 / 0.762 | 0.68 | 5552 / 6528 | 2005 |
| 1 | 15 | 0.824 / 0.760 | 0.90 | 4273 / 4885 | 2006 |
| 1 | 10 | 0.757 / 0.693 | 1.31 | 2867 / 3744 | 2006 |
| 2 | 20 | 0.838 / 0.762 | 0.72 | 5251 / 6668 | 4083 (cumulative) |
| 2 | 15 | 0.824 / 0.760 | 0.93 | 4123 / 4930 | 4113 |

A second session buys almost nothing (+6%): the threads are split, so the CPU work is the same,
and memory roughly doubles. The cut is the real lever, but 15 loses 0.014 hit@5 (over the 0.01
limit of Req 1.3) and 10 loses 0.08. p95 is already under the 8 s target at 4 users. **Defaults
kept** (1 worker, all 20); `RETRIEVAL_RERANK_TOP_K=15` is the knob if load grows.

**OCR warning (I13, Req 12).** No signal: xberg's page confidence is 0.95 for the skewed photo
and for the flatbed scans; word counts overlap (photo 150, scans 86–344). Blocked on a new
image dependency.

## 2026-10-12: gate calibration with the noisy variants (eval-fixes Req 3.4, 9.3)

`python -m eval.calibrate_gate e2e/20261012T021600Z-xl-granite.json --split dev` (granite, gte,
1,051 retrieval scores: dev cases and their robustness variants; no new run needed).

| GATE_THRESHOLD | False refusals | False answers | Flip rate |
|---|---|---|---|
| 0.28 | 0.6% | 91.5% | 8.8% |
| 0.31 (suggested: fewest false answers + flips, refusals ≤ 10%) | 1.1% | 83.0% | 11.8% |
| 0.35 | 1.1% | 74.5% | 21.8% |
| **0.40 (current)** | 2.2% | 61.7% | 37.8% |
| 0.45 | 9.4% | 48.9% | 49.2% |

The flip rate at 0.40 matches the e2e robustness result (38.5%, ISSUES.md I7), so the offline
sweep is a fair stand-in. No threshold keeps flips ≤ 10% and false answers low: the scores of
answerable and refuse-expected questions overlap (I8). Below 0.29 the gate is a relevance floor
that almost never flips; refusing unknowns is then the LLM's job. **Kept at 0.40 for now:** with
no LLM wired (I1) a lower gate sends more unrelated chunks to users through the extractive
fallback (I2) and spends more of the 50 calls a day once wired. Lower it (0.28–0.31) together
with the I1/I2 fixes.

## 2026-10-12: cross-lingual questions, granite vs bge-m3 (`xl86-*`)

38 cross-lingual questions added (`x-07`..`x-44`, 86 questions, 74 answerable): Arabic, darija
(Arabic script) and arabizi on English and French documents, English on French, English and
French on Arabic. Each evidence string is in one document only (`x-41`'s date is also in another
notice, but the question names Fulbright). Defaults otherwise (gte, top_k 20, RRF k 2, chunk 300 +
header). bge-m3 is `onnx-community/bge-m3-ONNX` `model_int8.onnx` (568 MB, dense CLS head only),
`EMBED_MODEL=BAAI/bge-m3 EMBED_DIM=1024 INGEST_EMBED_MAX_LENGTH=1024`; chunks stay cut with the
granite tokenizer (`INGEST_CHUNK_TOKENIZER_PATH`), so both runs index the same chunks.

| Run | hit@5 reranked | Reranked MRR | Cross-lingual hit@5 / MRR | Gold doc among candidates (x-lingual) | Darija hit@5 | Encode ms |
|---|---|---|---|---|---|---|
| granite (default) | 0.838 | 0.762 | 0.755 / 0.651 | 78% | 0.43 | 16 |
| **bge-m3 int8** | **0.878** | **0.789** | **0.816 / 0.691** | **86%** | 0.57 | 22 |

By direction (hit@5 reranked, granite -> bge-m3): ar->en 73% -> 82% (n 11), darija->en 20% -> 40%
(n 5), en->ar 80% -> 100% (n 5); ar->fr 50%, en->fr 75%, fr->en 88%, fr->ar 100%, arabizi 100%
unchanged; same-language unchanged (ar->ar 1.000, en->en 0.955 MRR). Groups are small: read
directions, not rates. `x-01`, `x-02` (Arabic on English, missed since `corpus55`) reach rank 3.

Still missed with bge-m3: 7 never reach the reranker (`x-08`, `x-16`, `x-19`, `x-24`, `x-26`,
`x-31`, `x-43`), so the first stage stays the bottleneck. `x-24` and `x-26` lose to the Arabic
faculty notices, whose titles all start with "كلية الإعلام الآلي:". The reranker drops 2 that were
fused in the top 5 (`x-18` darija 1 -> 6, `x-11` 4 -> 6); with granite it took darija hit@5 from
0.71 fused to 0.43, so gte is weak on darija. Rerank scores, hence the gate, are unchanged: same
overlap of answerable and unanswerable scores, suggested threshold 0.42 for both.

E2E check (`e2e/20261012T021600Z-xl-granite`, `e2e/20261012T030628Z-xl-bge-m3`: `--suite quality
robustness --max-llm-calls 0`, 307 cases; chat answers are `not_run`, so only the retrieval layer
and the gate count). Retrieval layer 234 -> 235 of 290 base cases (`cross_lingual` 56 -> 57 of 75,
`darija` 9/15 both), evidence rate 0.830 -> 0.858, false refusals 10.4% -> 9.9%, robustness flip
rate 36.9% -> 33.8%. Fixed `x-01`, `x-17`, `x-36`, `adv-06`; broke `x-11`, `cand-12` (darija),
`usthb-02`. The e2e pass check (evidence in the returned chunks) is coarser than rank, so the gain
shrinks to noise there. Verdict: bge-m3 is a modest, consistent gain, not a fix for cross-lingual
retrieval; the first stage (same-language distractors) and gte on darija are the larger levers.

## 2026-10-12: are the proposed retrieval changes worth it?

One change at a time on granite (86 questions), then bge-m3 with the best one. Translated titles:
each document title translated to ar/fr/en by gemma4:e4b (Ollama) and joined into the title, so
every chunk header carries all three (`INGEST_CHUNK_MAX_HEADER_TOKENS=96`; the original titles
are under 48 tokens, so the baseline chunks do not change); a proxy for keywords at ingest.

| Run | hit@5 / MRR (74) | Cross-lingual hit@5 / MRR | Candidates (all) | Rerank p95 ms | Gate suggestion |
|---|---|---|---|---|---|
| granite (default) | 0.838 / 0.762 | 0.755 / 0.651 | 85% | 1921 | 0.42 |
| prefetch 80, top_k 30 | 0.851 / 0.785 | 0.776 / 0.686 | 89% | 2874 | 0.42 |
| dense weight 2.0 | 0.851 / 0.769 | 0.776 / 0.661 | 86% | 1942 | 0.41 |
| translated titles | 0.824 / 0.769 | 0.755 / 0.685 | 85% | 1889 | 0.41 |
| bge reranker | 0.865 / **0.819** | 0.796 / **0.736** | 86% | 4411 | **0.0** |
| **bge-m3** | **0.878** / 0.789 | **0.816** / 0.691 | **91%** | 1940 | 0.42 |
| bge-m3, prefetch 80, top_k 30 | 0.865 / 0.786 | 0.796 / 0.687 | 91% | ~2900 | 0.42 |

Dense weight: noise. Translated titles: darija->en 20% -> 40% and en->fr 75% -> 100%, but
ar->en 73% -> 55% and en->en 100% -> 91%: not worth an LLM call per document. More candidates
help granite a little for +50% rerank time and add nothing to bge-m3 (91% either way). The bge
reranker ranks best (darija->en 60%) but scores darija and arabizi answers 0.03-0.16 and some
unanswerable questions 0.95: no gate threshold works, and it is 2.3x slower.

Answer level (`e2e/*-llm-granite`, `e2e/*-llm-bge-m3`: `--suite quality --llm ollama`,
gemma4:e4b, 307 cases): a tie. 184 of 237 answerable questions answered by both, 10 gained and
10 lost by bge-m3, chat pass 112 vs 109. What limits answers now is the LLM, not retrieval:
31-35 refusals with the evidence retrieved, 11-12 retrieval misses, 5-6 gate refusals (see
`e2e/ISSUES.md` I22-I24). Verdict: keep granite as the default for now; keep bge-m3 as the A/B
entry and re-run it with the production LLM once I1 and I22 are fixed.

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

## Corpus grown to 55 documents (2026-10-11, `corpus55`)

Defaults (gte, top_k 20, RRF k 2, chunk 300 + header), same 48 questions, 51 documents added:
3 Micro Club internal PDFs (members), 17 MCDI / mcoli-ui docs pages (en), 16 faculty pages
(fr) and 6 faculty notices (ar) from finfo.usthb.dz, 3 usthb.dz pages (en), USTHB on ar/fr
Wikipedia and 4 Arabic Wikipedia articles as distractors. 971 chunks (855 public, 116 members)
instead of 61. Sources in `data/sources/` and `data/parsed/` (gitignored), manifest
`eval/corpus.json`.

| Run | hit@5 reranked | Fused MRR | Reranked MRR | Cross-lingual (fused / reranked) | Rerank ms p50 / p95 | Gate suggestion |
|---|---|---|---|---|---|---|
| defaults, 4 docs (t300 header) | 1.0 | 0.834 | 0.986 | | 1158 / 1267 | 0.42 |
| defaults, 55 docs | **0.944** | 0.709 | 0.912 | 0.455 / 0.758 | 1424 / 1983 | 0.44 |

The eval is no longer saturated. Misses after reranking: x-01 and x-02 (Arabic questions on
English documents, not in the top 10), fr-04 (rank 3), en-07 (rank 2, but "since 1985" is also
in the president candidacy and the Code of Conduct: a label that names one document is too
narrow). At `GATE_THRESHOLD=0.40` the gate now refuses 42% of the unanswerable questions (was
75%) and 3% of the answerable: the bigger corpus has more near-topic text, so unanswerable
questions score higher. Recalibrate on the new questions before changing it.
