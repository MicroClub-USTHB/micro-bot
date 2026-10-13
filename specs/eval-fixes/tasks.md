# Implementation Plan: Eval fixes

Each task is test-first: write the listed tests, see them fail, then implement. Every pushed task
keeps CI green (`uv lock --check`, ruff, `pytest -m "not ocr and not tokenizer and not embedder
and not reranker"`, plus the models job). Commits stay under 200 changed lines; split a task
across commits if needed.

Every task that changes behaviour ends the same way (Req 1): the eval runs named in the task,
before and after, saved under `eval/results/` with a label, compared with
`python -m eval.e2e.compare`; the issue's row in `eval/results/e2e/ISSUES.md` gets its status,
commit and run; EDGE_CASES.md gets a row for any new edge case; each new variable gets a comment
line in `.env.example`. A task is not ticked if a non-regression check of Req 1.2 / 1.3 fails.

**Status 2026-10-12:** all tasks done. The user asked to fix every issue whatever the owner, so
tasks 11–14 were done on this branch too (to review with eequaled). Changes from the plan: 6.2
became a deskew (`ingest/deskew.py`, Pillow now a runtime dependency) because xberg gives no
usable OCR signal; 13.2 keeps the frozen API (blank message -> 200 prompt without retrieval,
delete of an unknown id keeps `{"ok": false}`); 13.3 has no upload `title` field (frozen API);
14 keeps `expires_at`. The full e2e suite was not rerun (the user asked to keep it for the end).

Tasks 1–11 are nisbenz's. Tasks 12–14 are eequaled's files: they are listed so the plan covers
every issue, and are done only with eequaled's agreement (by eequaled, or by nisbenz in a PR
eequaled reviews).

- [x] 1. Eval support for measuring the fixes
- [x] 1.1 Gate-score distribution in the reports
  - `eval/metrics.py`: `score_distribution(scores) -> {min, q1, median, q3, max, n}` and
    `score_overlap(answerable, refuse) -> float` (share of refuse-expected scores ≥ the answerable
    Q1).
  - `eval/run_eval.py` and `eval/e2e/report.py`: a table of both distributions and the overlap.
  - `tests/test_eval_metrics.py`: empty list, one value, known quartiles, overlap 0 and 1.
  - _Requirements: 1.1, 3.5_
- [x] 1.2 Watched cases in the e2e report
  - `eval/e2e/run.py`: `--watch id1,id2,...`; `report.py` prints a pass/fail block per id, at
    retrieval and chat level; unknown id is an error naming it.
  - Tests in `tests/test_e2e_report.py` (or the existing report tests).
  - _Requirements: 1.1, 10.4, 11.4_
- [x] 1.3 Per-stage times in the load suite
  - The in-process system adds a structlog processor collecting `retrieval_done` events during
    the load window; the load report shows p50/p95 of `ms_encode`, `ms_search`, `ms_rerank`.
  - Test with a fake retriever that logs fixed timings.
  - _Requirements: 17.3_
- [x] 1.4 Status column in ISSUES.md and the baseline runs
  - Add `Status` (`open` / `fixed <commit>` / `partly` / `accepted`) to every issue table.
  - Rerun the baselines with the new report fields on this branch: `eval.run_eval --label
    base-fixes` (default questions and `xl86`), `eval.e2e.run --suite all --label base-fixes`.
    These are the "before" files for every later task.
  - _Requirements: 1.1, 1.4_

- [x] 2. Follow-up cues (I10, I12)
- [x] 2.1 `has_followup_cue` in `src/mcclub_rag/retrieval/query.py`
  - Cue table from design §2 (leading conjunctions, pronouns/demonstratives, ellipsis) for en, fr,
    ar and darija, matched on `fold_for_search` tokens from `text/sentences.split_words`.
  - `tests/test_retrieval_query.py`: one parametrized positive per cue, negatives "Who is the
    dean?", "Qui est le doyen ?", "من هو العميد؟", Arabic nouns ending in ه.
  - _Requirements: 11.1, 11.3, 11.5_
- [x] 2.2 Use it in `build_search_query`
  - Merge only when short **and** cued; `RETRIEVAL_FOLLOWUP_CUES` (default `true`) restores the
    old rule when `false`.
  - Tests: cued short message merged, uncued short message alone, long message alone, setting off.
  - Eval: `--suite retrieval,chat --watch conv-05,conv-06,conv-09` plus the `topic_switch`,
    `followup*` slices, before/after.
  - _Requirements: 5.5, 11.1, 11.2, 11.4_

- [x] 3. Query normalization and gate calibration (I7, I8)
- [x] 3.1 `strip_arabic_marks` in `src/mcclub_rag/text/normalize.py`
  - Factor the harakat + tatweel removal out of `fold_for_search` (which then calls it); stored
    text is never passed through it.
  - Tests: `fold_for_search` output unchanged on the existing cases; harakat removed, hamza
    letters kept.
  - _Requirements: 9.4_
- [x] 3.2 `normalize_query` in `retrieval/query.py`, applied before dense encoding and reranking
  - Punctuation/whitespace collapse always; `RETRIEVAL_QUERY_CASEFOLD_SHOUTING` and
    `RETRIEVAL_QUERY_STRIP_HARAKAT` (both default `false` until measured).
  - Tests: shouting lowered, "USTHB" in a normal sentence kept, harakat stripped, settings off =
    today's query.
  - Eval: `--suite robustness` per setting, flips by noise kind; `eval.run_eval` for Req 1.3.
    Turn a setting on by default only if it lowers flips without a quality drop.
  - _Requirements: 9.1, 9.2, 9.4_
- [x] 3.3 `eval/calibrate_gate.py`
  - Sweep `GATE_THRESHOLD` 0.20–0.60 by 0.01 on the `dev` split **with** the robustness
    variants; print false refusals, false answers and flip rate per value; mark the best value
    with false refusals ≤ 10%.
  - Unit test on synthetic scores (the chosen value is the expected one).
  - Set the chosen value in `.env.example` with the run id; update the `retrieval/gate.py`
    docstring and `specs/retrieval/design.md` ("relevance floor").
  - _Requirements: 3.4, 9.2, 9.3, 1.5_

- [x] 4. Share-link pages (I16)
- [x] 4.1 Fixture
  - `tests/fixtures/make_fixtures.py`: `web/share_bar_only.html` in the shape of
    `réintégration-aux-études` (title + share bar, no content); list it in
    `tests/fixtures/README.md`.
  - _Requirements: 15.3_
- [x] 4.2 `_drop_share_blocks` in `src/mcclub_rag/ingest/web.py`
  - Share-link host/path list and class/aria words from design §7; called in `prepare_html`
    before `_inline_links`.
  - Tests: the fixture raises `EmptyDocument`; `web/mcoli_introduction.html` Markdown unchanged
    (golden); a content paragraph with one share link inside is kept.
  - Eval: `--suite ingest`.
  - _Requirements: 15.1, 15.2, 15.3_

- [x] 5. PDF titles (I15)
- [x] 5.1 `page_one_title` and `is_generic_title` in `src/mcclub_rag/ingest/preprocess.py`
  - Rules from design §8 (page-1 headings joined, skip page numbers / dates / "Contents" /
    "Sommaire" / "الفهرس" / < 3 letters; generic = filename stem, contained in the page-1 title,
    producer defaults).
  - Tests with the CoC and SG page-1 section lists as literals; producer defaults; no page-1
    headings -> `None`.
  - _Requirements: 14.2, 14.3_
- [x] 5.2 New title order in `_document`
  - explicit > page-1 title > metadata (unless generic) > stem.
  - Tests: existing preprocess tests still pass; an explicit title still wins.
  - Eval: `eval.run_eval` and `--suite retrieval` **without** the title overrides in
    `eval/corpus.json` vs with them (the title is in the chunk header and the reranker input).
  - _Requirements: 14.2, 14.3, 14.4_

- [x] 6. Low OCR yield warning (I13)
- [x] 6.1 Measure the signals
  - Read xberg's per-page `ocr_confidence` and word counts for the photo and the flatbed scans
    from `scripts/make_eval_scans.py`; record them in EDGE_CASES.md. If neither separates the
    photo from the scans, stop and report (Req 12.3 then goes to the user).
  - _Requirements: 12.2_
- [x] 6.2 `RawPage.ocr_confidence` and `low_ocr_yield`
  - `ingest/models.py`: `ocr_confidence: float | None`; `ingest/parse.py`: fill it, call the pure
    `low_ocr_yield(pages, min_confidence, min_words)` and append `low_ocr_yield: page N`;
    `INGEST_OCR_MIN_CONFIDENCE`, `INGEST_OCR_MIN_WORDS` with the values from 6.1.
  - Unit tests on synthetic pages; `ocr`-marked test: the photo warns, the scans do not.
  - _Requirements: 12.1, 12.2_

- [x] 7. Injection flag and payload schema 2 (I3, I11)
- [x] 7.1 `looks_like_instruction` in `src/mcclub_rag/ingest/injection.py`
  - en/fr/ar patterns from design §6 on `fold_for_search` text.
  - Tests: positives from the adversarial corpus (`eval/e2e/corpus/`), negatives from real
    sentences about "instructions", "règlement", "تعليمات" in a normal sense.
  - _Requirements: 4.3, 4.5_
- [x] 7.2 Payload field and `SCHEMA_VERSION = 2`
  - `ingest/pipeline.py::_payload` adds `instruction_like`; `SCHEMA_VERSION` 1 -> 2;
    `retrieval/store.py` adds `language` to `_KEYWORD_FIELDS` (used by task 8).
  - If eequaled has agreed to keep `expires_at` (task 14) by then, add its payload field in this
    same bump; otherwise leave it to task 14.
  - Tests: ingest writes the flag; reindex from the docstore sets it; a schema-1 index reports
    `index_stale` at startup.
  - _Requirements: 4.3_
- [x] 7.3 Optional ranking penalty
  - `Candidate.instruction_like` (default `False`); `RETRIEVAL_INSTRUCTION_PENALTY` (default
    `1.0` = off) multiplies the rerank score for ranking only; `decide()` gets the unpenalized
    scores.
  - Tests: flagged chunk ranks lower with the penalty; the gate's top score unchanged.
  - Eval: `--suite retrieval,chat` on the `inj-*`, `adv-*` cases; false-positive count on the
    55 real documents printed by `eval/run_eval.py` (target 0).
  - _Requirements: 4.4, 4.5_

- [x] 8. Cross-lingual candidates (I9)
- [x] 8.1 Cross-language dense prefetch in `retrieval/search.py`
  - `RETRIEVAL_CROSS_LANG_PREFETCH` (default `0`); when > 0, a third dense prefetch filtered by
    role AND `language != detect_language(query)`; the role filter stays on every level.
  - Integration tests (in-memory Qdrant): Arabic distractor + English answer, the English chunk is
    a candidate only with the setting on; a public role never gets a `members` chunk through the
    third prefetch; setting 0 = identical request to today.
  - _Requirements: 10.3, 1.2_
- [x] 8.2 Measure and choose
  - Runs on `xl86` and `--suite retrieval --watch coc-23,coc-24,fac-18,note-09,x-01,x-02`:
    cross-language prefetch 10 / 20; prefetch 80 + top_k 30; bge-m3; best combination. Report
    darija hit@5 fused vs reranked for each.
  - Change a default only if cross-lingual hit@5 ≥ 0.85 and same-language hit@5 drops ≤ 0.01;
    otherwise record the best result and mark I9 `partly`.
  - _Requirements: 10.1, 10.2, 10.4, 10.5_

- [x] 9. Rerank throughput (I19)
- [x] 9.1 Reranker session pool
  - `RETRIEVAL_RERANK_WORKERS` (default `1`); `get_reranker` builds that many sessions with
    `max(1, rerank_threads // workers)` threads each; `HybridRetriever` takes sessions from an
    `asyncio.Queue` (returned in `finally`); validator `rerank_workers ≤ rerank_threads`.
  - Tests with fake sessions: two concurrent `retrieve` calls overlap with 2 workers and not with
    1; a failing session is returned and `RerankError` raised.
  - _Requirements: 17.2_
- [x] 9.2 Rerank cut
  - `RETRIEVAL_RERANK_TOP_K` (default = `RETRIEVE_TOP_K`, validator ≤ it): rerank only the first
    k fused candidates.
  - Tests: only k passages scored; default = today.
  - Eval: `--suite load` for workers 1/2 × rerank top-k 20/15/10 with RSS, plus `eval.run_eval`
    for quality; adopt a default only within Req 1.3 and the container memory.
  - _Requirements: 17.1, 17.2, 17.4_

- [x] 10. Snippet helper (I6, I24, nisbenz half)
- [x] 10.1 `best_snippet` in `src/mcclub_rag/text/snippet.py`
  - Sentence split with `text/sentences.split_sentences`, score by `fold_for_search` word
    overlap, window of whole sentences ≤ `max_chars` with "…" at cut ends.
  - Tests: evidence sentence in the middle of a 2,000-character chunk is in the snippet; Arabic
    punctuation; tie -> earlier sentence; short text returned whole.
  - Eval (offline, no API change): rescore the last quality run's snippets with `best_snippet`
    and report the evidence rate it would give.
  - _Requirements: 6.1, 6.2, 6.4_

- [x] 11. Follow-up eval driver cleanup (after task 13.1)
  - `eval/e2e/drivers.py`: stop setting `MEMBER_TOKEN` / `ADMIN_API_KEY` in the environment once
    `get_role` reads the app settings; pass them through `create_app(settings=...)`.
  - Test: the in-process driver works with an empty environment.
  - _Requirements: 16.3_

- [x] 12. Answer layer [eequaled, with agreement]
- [x] 12.1 Wire the LLM (I1)
  - `OpenRouterGenerationService` in `main.py` (`asyncio.to_thread` over
    `OpenRouterClient.generate`); `default_state()` uses it when the key is set, else the fake
    with one `llm_disabled` log.
  - Tests: key set -> real service (fake OpenAI client), one call per question; no key -> fake.
  - Eval: `--suite quality --mode record --max-llm-calls 10`.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.3_
- [x] 12.2 Fallback without pasted chunks (I2, I3, I4)
  - `extractive_fallback` returns the localized notice, top 2 chunks as sources with
    `best_snippet`, `grounded=False`.
  - Tests: no chunk text in `answer`; notice in ar/fr/en; injection marker never in `answer`.
  - _Requirements: 3.1, 3.2, 4.1, 4.2, 5.4_
- [x] 12.3 Refusal language (I5)
  - `REFUSAL[lang]` for the gate refusal in `api/chat.py`.
  - Tests: ar/fr/en questions get the matching refusal; `_REFUSAL_RE` matches all three.
  - _Requirements: 5.3_
- [x] 12.4 Citation formats (I22)
  - `answer.build` accepts `[doc_id: X]`, `[S<n>]` (mapped to `chunks[n-1]`), comma lists.
  - Tests per format; unknown ids still dropped; out-of-range `S<n>` invalid.
  - _Requirements: 7.1, 7.2, 7.3, 7.4_
- [x] 12.5 Snippets in sources (I6, I24)
  - `_source_from_chunk` uses `best_snippet` with the question (fallback) or the citing answer
    sentence (LLM path).
  - Tests: the snippet contains the matching sentence; length ≤ 500.
  - Eval: `--suite quality` evidence rate.
  - _Requirements: 6.1, 6.3_
- [x] 12.6 Prompt and passages for false refusals (I23)
  - First a record-mode run with the production model on the 31 known cases (≤ 10 calls/day);
    then the cross-lingual prompt sentence; nisbenz sweeps `RERANK_TOP_N` 3/5/8 with the local
    Ollama model.
  - Tests: the prompt contains the sentence and still the untrusted-data wording.
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 5.1, 5.2_

- [x] 13. API hygiene [eequaled, with agreement]
- [x] 13.1 Auth reads the app settings (I20)
  - `get_role` / `require_admin` take `request: Request` and read
    `request.app.state.mcclub_state.settings`.
  - Tests: two apps with different tokens in one process each accept only their own token.
  - _Requirements: 16.3, 16.5_
- [x] 13.2 Delete unknown document -> 404 (I17); blank message -> 422 (I18)
  - Tests in `tests/test_api_contract.py`; update the e2e probes `ingest-delete-again`,
    `contract-chat-empty-message`, `contract-chat-whitespace-message` to the strict expectation.
  - _Requirements: 16.1, 16.2, 16.5_
- [x] 13.3 `config.py` defaults (I21) and the upload `title` field (I15)
  - Defaults gte / 20 / 0.40; optional `title` form field passed to `ingest_file` (needs explicit
    approval under the API freeze).
  - Tests: defaults; upload with and without `title`.
  - _Requirements: 16.4, 14.1_

- [x] 14. `expires_at` [eequaled + nisbenz, after agreement] (I14)
  - Agree keep vs remove and write it in AGENTS.md "Open seam items".
  - Keep: `expires_at` keyword in `ingest_file` / `ingest_url`, docstore and payload (schema bump,
    shared with 7.2 if still open), marker and listing, `role_filter(role, now)` on every query
    level; adapter passes it. Remove: reject the form field, update the probe.
  - Tests: expired document absent from retrieval for both roles, present in the listing with its
    date; or the field rejected.
  - Eval: `--suite ingest` 16 / 16.
  - _Requirements: 13.1, 13.2, 13.3, 13.4_
