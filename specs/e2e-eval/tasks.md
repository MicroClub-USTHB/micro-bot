# Implementation Plan: End-to-end eval

Each task is test-first: write the listed tests, see them fail, then implement. CI stays green
(`uv lock --check`, ruff, `pytest -m "not ocr and not tokenizer and not embedder and not
reranker"`, models job). Harness tests live in `tests/test_e2e_*.py` and use fakes only. Commits
stay under 200 changed lines; split a task across commits if needed. No file of eequaled's
(`main.py`, `api/`, `config.py`, `schemas.py`, `answer.py`, `llm/`) is edited.

- [x] 0. Report the generator wiring defect
  - PROGRESS.md (`Blocked`) and EDGE_CASES.md row: `default_state()` keeps
    `FakeGenerationService`, and `OpenRouterClient.generate` is sync vs the async protocol;
    evidence = `main.py:304` and a chat call returning `llm_calls: 0` with a key set. Tell
    eequaled; the harness does not patch it.
  - _Requirements: 3.5_

- [x] 1. Cases
- [x] 1.1 `eval/e2e/cases.py`: `Case`, `TurnIn`, `load_cases(path, manifest)`, `split(id)`,
  canary / injection expansion of `must_not`.
  - `tests/test_e2e_cases.py`: unknown key, duplicate id, unknown `doc`, answer without gold or
    evidence -> error naming the id; split is stable and about 80/20 on 1000 ids; public cases
    get the canaries, member cases do not.
  - _Requirements: 1.1, 1.2, 1.5, 2.2, 2.3_
- [x] 1.2 `eval/e2e/convert.py` -> generated `cases_retrieval.jsonl` from `eval/questions.jsonl`
  (48); `load_cases` takes several files.
  - Test: the generated file is current; every converted case loads; counts of `answer` /
    `refuse` match `answerable`.
  - _Requirements: 1.3_

- [x] 1.3 `docs: list[str]` instead of `doc` (a fact can be in several documents, e.g. "since
  1985" in 3); `SLICES` = the Req 11 catalogue; regenerate `cases_retrieval.jsonl`.
  - Tests: unknown topic in `docs` rejected; converted `doc` becomes `[doc]`.
  - _Requirements: 1.1, 1.2, 11.1_

- [x] 2. Corpus
- [x] 2.1 `scripts/build_eval_corpus.py` + `eval/corpus_sources.txt` (the 51 sources added on
  2026-10-11), dump format of `scripts/parse.py`.
  - _Requirements: 12.1_
- [x] 2.2 Scans: `scripts/make_eval_scans.py` renders real pages (Code of Conduct, a candidacy,
  an Arabic and a French faculty notice) to image-only PDF / JPG with skew and noise; add to the
  manifest (members for club files); record OCR quality per language.
  - _Requirements: 12.2_
- [x] 2.3 Adversarial corpus
  - `eval/e2e/corpus/`: 2 members-only canary docs, 1 injection doc, 2 near-duplicates (one
    date differs), 1 mixed ar/fr doc; `manifest.json` with topics, titles, visibility, canaries,
    injection markers. Fixture README lines.
  - Cases for them: access (public + member), injection, near-duplicate disambiguation.
  - Test: manifest files exist; each canary appears in exactly one doc; markers absent from
    every non-injection doc.
  - _Requirements: 2.1, 2.4_

- [x] 3. Scoring
- [x] 3.1 `score.py`: `CheckResult`, `contract`, `llm_calls`, `path` (design table).
  - `tests/test_e2e_score.py`: valid / invalid bodies, error JSON shape, each path row.
  - _Requirements: 3.1, 3.5, 3.6_
- [x] 3.2 `behaviour`, `gold`, `evidence`, `must_not`, `access`, `language`, `refusal_language`,
  with `fold_for_search` matching.
  - Tests: Arabic gold with and without diacritics; evidence only in the answer (not in a
    snippet) fails; canary in a snippet fails `must_not`; members doc id for a public case
    fails `access`; English refusal on an `ar` case fails only `refusal_language`.
  - _Requirements: 3.2, 3.3, 3.4_
- [x] 3.3 Retrieval-layer scorers (decision, evidence in chunk text, access on chunks).
  - _Requirements: 6.4_

- [x] 4. Generators and budget
  - `generator.py`: `Budget`, `RecordingGenerator` (live / record / replay, date masking),
    `FaultyGenerator`.
  - `tests/test_e2e_generator.py`: budget stops at N and returns `eval_budget` with
    `llm_calls=0`; record then replay returns the same text; a changed prompt is `replay_miss`;
    two dates give the same key; a fake sync inner client is called through `to_thread`.
  - _Requirements: 5.1, 5.2, 7.1_

- [x] 5. Drivers
- [x] 5.1 `InProcessDriver` with injectable pipeline pieces, seeded follow-up history, its own
  document adapter and rate limiter.
  - `tests/test_e2e_driver.py`: on `pipeline_helpers` fakes and 2 tiny docs, a member case
    finds the members doc, the public case does not (and passes `access`), a follow-up case
    reaches retrieval with its history.
  - _Requirements: 5.3, 8.2, 8.3_
- [x] 5.2 `HttpDriver`: auth headers from env, pacing bucket, upload with `eval-` topics,
  `--cleanup`, 429 recorded per case. Test with `httpx.MockTransport`.
  - _Requirements: 5.4, 8.1, 10.2_

- [x] 6. Report, scorecard, CLI
- [x] 6.1 `report.py`: summary by slice / lang / role / path, rates excluding skips, markdown,
  `thresholds.json`, exit codes. Tests on hand-built results.
  - _Requirements: 9.1, 9.2, 9.3_
- [x] 6.2 `run.py`: `--suite`, `--target`, `--split`, `--layer`, `--max-llm-calls`, `--replay`,
  `--label`; result file with git sha and case file hash; partial result on Ctrl-C.
  - _Requirements: 9.1_
- [x] 6.3 `compare.py` + test (metric deltas, flipped ids).
  - _Requirements: 9.4_

- [x] 7. Suites
- [x] 7.1 `quality` (both drivers, budget, replay). Manual run: inprocess, models, `--replay`
  after one recording run; note the numbers in PROGRESS.md.
  - _Requirements: 3, 5_
- [x] 7.2 `perturb.py` + `robustness` (variants, flip rate, `--repeat`).
  - `tests/test_e2e_perturb.py`: golden outputs per kind and language with seed 0; digits,
    `#hex` and URLs unchanged by `typo`.
  - _Requirements: 6.1, 6.2, 6.3_
- [x] 7.3 `contract` and `faults` (Req 7.1 table as data, expected vs observed).
  - Test: the faults suite passes on the inprocess driver with fakes (runs in CI).
  - _Requirements: 7.1, 7.2_
- [x] 7.4 `load` (C users, S seconds, percentiles, 429 count, RSS when local).
  - _Requirements: 7.3_
- [ ] 7.5 `restart` behind `--allow-docker`. Manual run on compose. (Built; not run: Docker was
  down on the dev machine.)
  - _Requirements: 7.4_
- [x] 7.6 `ingest` suite over the documents API (Req 12.3 sequence), both drivers.
  - _Requirements: 12.3_

- [x] 8. Optional judge
  - `judge.py`: off without env; rubric, JSON parse, verdict cache, `--max-judge-calls`.
  - `tests/test_e2e_judge.py` with a fake client: unparsable -> skip; cache hit -> no call;
    cap respected; never reads `OPENROUTER_API_KEY`.
  - _Requirements: 4_

- [x] 9. Grow the case set to 200+ base cases (`eval/e2e/cases.jsonl`)
  - Every Req 11 slice at least 3 times; at least 30 per language and 40 refuse-expected (half
    in-domain); evidence verbatim, checked by `tests/test_e2e_cases_corpus.py` (skips without
    `data/`). Re-run the retrieval layer on all cases and record results.
  - _Requirements: 1.4, 11_
  - _Requirements: 1.4_

- [x] 10. Docs
  - AGENTS.md (commands, "Rules that bite"), README (eval section), PROGRESS.md, EDGE_CASES.md
    rows for every defect the harness found, `.env.example` lines for `EVAL_JUDGE_*`.
