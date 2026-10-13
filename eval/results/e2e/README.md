# E2E eval results

Run with `python -m eval.e2e.run` (see [specs/e2e-eval/](../../../specs/e2e-eval/)). Every issue
found, with its status, is in [ISSUES.md](ISSUES.md). The `.md` reports are gitignored and the
`.json` files are committed with `git add -f` when a run is worth keeping; the 2026-10-13 runs
are local only so far. Chat results keep the answer text since `407807e` (I29).

| Run | What | Headline |
|---|---|---|
| `20261011T203423Z-full-1` (+ `-quality-fixed`) | all suites, gte, gate 0.40, no LLM | false answers 62.7%, 9 injection hits, 48 language mismatches (I1–I21) |
| `20261012T021600Z-xl-granite`, `…030628Z-xl-bge-m3` | retrieval, 38 more cross-lingual questions | bge-m3 cross-lingual hit@5 0.816 vs 0.755 (I9) |
| `20261012T043245Z-llm-granite`, `…050640Z-llm-bge-m3` | quality, ollama gemma4:e4b | false answers 0% / 1.7%, citation formats (I22), LLM refusals (I23) |
| `20261012T201158Z-fixes-system`, `…202250Z-fixes-retrieval` | after the eval fixes | contract 22/22, faults 9/9, cache 5/5, ingest 16/16 |
| `20261013T051944Z-bge-12b` | all suites, bge-m3 + bge reranker top 10, gate off, ollama gemma4 12B | refusals came back grounded (I25), prompt leak (I26), example data as facts (I27) |
| `20261013T061449Z-notfound-12b` | the 90 hard cases of `bge-12b` after `NOT_FOUND` (I25) | refuse-expected 41/47 pass; 63/80 answered pass; left: I26–I28, I30–I32 |
| `20261013T174211Z-i30-retrieval` | quality, 349 cases, retrieval layer only, after I30 and the corpus rebuild (I31) | evidence in the chunks 90.9%, p95 2.3 s; false answers 1.0 is the gate being off (I29) |
| `20261013T184620Z-prompt-old-12b` | 349 cases, ollama gemma4 12B, prompt before `8d04277` (A) | 260/349; false answers 9.4%, false refusals 6.6% |
| `20261013T181541Z-prompt-new-12b` | same, first version of the I27/I28 prompt | 260/349; refused yes/no questions (`coc-08`, `fac-22`) |
| `20261013T191937Z-prompt-new2-12b` | same, prompt `8d04277` (C): **latest** | 265/349; false answers 0%, false refusals 11.0%, evidence 80.7%, 29 language mismatches (I4, I6, I9, I23, I32) |
| `20261013T184844Z-replay-lost` | replay of the 10 answerable cases the first new prompt lost against A | 0/10 by design: used to read the refusals (`coc-08`, `fac-22` -> the yes/no rule) and find E6 (`cand-17`) |

## Retrieval-layer preview, 2026-10-11 (`preview-retrieval.json`)

All 269 cases (221 hand-written + 48 converted) through `HybridRetriever.retrieve` only (no
LLM): 63 documents (55 real + 8 adversarial), real models, in-memory Qdrant,
`GATE_THRESHOLD=0.40`. A case fails on: wrong gate decision, evidence not in the returned
chunks, or a members-only chunk for a public role. Not the final scorer (task 3.3).

| | Pass |
|---|---|
| All | 205 / 269 (76%) |
| Arabic / English / French | 76% / 73% / 84% |
| Access leaks (members chunk to a public user) | **0** |
| Gate decision wrong: should refuse, passed | 37 |
| Gate decision wrong: should answer, refused | 5 (cross-lingual and follow-up) |
| Evidence missing from the chunks | 7 |

Weakest slices: `future`, `system_prompt_leak`, `harmful`, `pii`, `injection`,
`unanswerable_in_domain` (the gate passes them: refusing is the LLM's job), `topic_switch`
(1/3), `cross_lingual` (68%). Perfect: `ocr`, `long_doc`, `multi_doc`, `negation`, `yes_no`,
`followup`, `long_input`. Findings recorded in EDGE_CASES.md (gate overlap, short topic switch,
injection text pulled by attack questions, same-language distractors).
