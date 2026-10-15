# Requirements Document

## Introduction

The e2e eval (`specs/e2e-eval/`, `eval/e2e/`) found 24 issues, listed with their evidence in
`eval/results/e2e/ISSUES.md` (runs `20261011T203423Z-full-1`, `…-full-1-quality-fixed`,
`20261012T*-xl-*` and `20261012T*-llm-*`). This spec says what "fixed" means for each one, so
every fix can be checked by a rerun of the same eval instead of by opinion.

Every issue (I1–I24) maps to at least one requirement below; the table at the end is the index.
Each requirement names its **owner**. Ownership follows AGENTS.md: nisbenz owns ingest,
retrieval, the gate and the eval; eequaled owns `main.py`, `api/`, `config.py`, `schemas.py`,
`answer.py` and `llm/`. Requirements owned by eequaled are written down here so that the whole
list is acknowledged and measurable, but **their code changes need eequaled's approval** and are
out of scope for this branch unless eequaled agrees. Shared requirements are split into a part for
each owner.

Fixed decisions this builds on (AGENTS.md): at most 1 LLM call per question and 50 requests/day;
API routes and parameters are frozen; the response shape is frozen (`schemas.py` changes need the
other owner's PR approval); offline at runtime; no new RAG framework. Several issues (I2, I4, I8,
I23) can only be closed once the LLM is wired (I1); their requirements say what the retrieval side
must do meanwhile.

**Baseline** = the numbers in ISSUES.md. **Scorecard thresholds** = `eval/e2e/thresholds.json`.

## Requirements

### Requirement 1: Measured fixes, no regressions

**User Story:** As the developer, I want every fix proven by a before/after eval run, so that a
fix for one issue cannot quietly break another.

#### Acceptance Criteria

1. WHEN a change claims to fix an issue THEN the change SHALL include a before and an after run
   (`eval.run_eval` and/or `eval.e2e.run` with the suites named in that issue's requirement),
   saved in `eval/results/` and compared with `python -m eval.e2e.compare`.
2. WHEN a change lands THEN access leaks SHALL stay 0, contract failures 0, the fault, ingest and
   load suites all passing, and instability 0.
3. WHEN a retrieval or ingest change lands THEN retrieval hit@5 and MRR on `eval/questions.jsonl`
   (all, Arabic subset, cross-lingual) SHALL NOT drop by more than 0.01, unless the requirement it
   serves accepts the trade and the result file says so.
4. WHEN an issue is fixed, partly fixed or accepted THEN `eval/results/e2e/ISSUES.md` SHALL record
   its status, the commit and the run that shows it, and EDGE_CASES.md SHALL get a row for any new
   edge case.
5. IF a threshold is tuned (gate, fusion, top-k) THEN it SHALL be tuned on the `dev` split only
   and reported on `test` (e2e-eval Req 1.5).

### Requirement 2: The live app calls the LLM (I1)

**Owner:** eequaled.

**User Story:** As a user, I want answers written by the configured LLM, so that I get an answer
to my question and not pasted chunks.

#### Acceptance Criteria

1. WHEN `OPENROUTER_API_KEY` is set THEN `main.default_state()` SHALL use a real generation
   service over `OpenRouterClient`, not `FakeGenerationService`.
2. The generation service SHALL be async-safe: the sync `OpenRouterClient.generate` SHALL run
   outside the event loop (e.g. `asyncio.to_thread`), so one slow LLM call does not block other
   requests.
3. WHEN an answerable question is answered by the LLM THEN the response SHALL report
   `usage.llm_calls == 1`; no path SHALL make more than 1 LLM call per question.
4. IF the key is missing or the daily budget is spent THEN the app SHALL keep working with the
   fallback (Req 3) and say so in the log once.
5. WHEN fixed THEN `--suite quality --mode record --max-llm-calls 10` SHALL report answerable cases
   answered by the LLM (the run must not say "generation NOT evaluated").

### Requirement 3: Questions that must be refused are not answered with unrelated text (I2, I8)

**Owner:** eequaled (fallback and answer layer), nisbenz (gate).

**User Story:** As a user, I want a clear "I don't know" for questions the documents do not
answer (PII, harmful, system prompt, future, unknown in-domain facts), so that I am not misled by
unrelated passages.

#### Acceptance Criteria

1. WHEN the LLM is not used (no key, budget spent, LLM error) THEN the fallback SHALL NOT present
   chunks as the assistant's answer: it SHALL either refuse, or show them labelled as excerpts
   with `grounded=false`.
2. WHEN the fallback refuses or shows excerpts THEN `grounded` SHALL be `false`.
3. WHEN the LLM is wired (Req 2) THEN the false-answer rate on the quality suite SHALL be ≤ 5%
   (baseline 62.7% without LLM; 0.0% / 1.7% measured with gemma4:e4b).
4. (nisbenz) The gate SHALL be documented and tuned as a **relevance floor**: its threshold SHALL
   be chosen to keep false refusals ≤ 10% on the `dev` split, and refusal of in-domain unknowns is
   left to the LLM (I8).
5. (nisbenz) Every retrieval eval report SHALL show the gate-score distributions of answerable and
   refuse-expected questions (min, quartiles, max) so the overlap is tracked per release.

### Requirement 4: Injected instructions never reach the user (I3, I11)

**Owner:** eequaled (answer), nisbenz (ingest flag, retrieval weight).

**User Story:** As a user, I want the assistant to ignore instructions hidden in documents, so that
a malicious upload cannot speak through the bot.

#### Acceptance Criteria

1. WHEN a retrieved chunk contains an injection THEN its text SHALL NOT appear as the assistant's
   reply; excerpts, if shown, SHALL be in `sources[].snippet` only.
2. WHEN the quality suite runs THEN injection markers in the answer SHALL be 0 cases (baseline 9
   without LLM, 2–3 with gemma4:e4b: `inj-02`, `inj-06`, `fac-17`).
3. (nisbenz) WHEN a chunk is ingested THEN the pipeline SHALL flag text that addresses an AI
   assistant with instructions (e.g. "ignore all previous instructions", "reveal your system
   prompt", in ar/fr/en) in the chunk payload, without changing the stored text.
4. (nisbenz) IF a flag exists THEN retrieval MAY down-weight flagged chunks; it SHALL NOT drop them
   from listings, and the change SHALL be measured on `inj-*` and `adv-*` cases (I11).
5. The flag SHALL be deterministic and its false-positive rate on the 55 real documents SHALL be
   reported (target 0 flagged chunks in real documents).

### Requirement 5: Answers and refusals in the question's language (I4, I5, I12)

**Owner:** eequaled (answer text, refusal text), nisbenz (language detection, follow-up retrieval).

**User Story:** As an Arabic or French speaker, I want the reply in my language even when the
document is in another one.

#### Acceptance Criteria

1. WHEN the LLM answers THEN the answer language SHALL match the question's detected language
   (`text/lang.detect_language`); language mismatches on the quality suite SHALL be 0 for
   ar/fr/en questions (baseline 48 without LLM, 8–9 with gemma4:e4b).
2. WHEN the question is darija or arabizi THEN the expected answer language SHALL be defined in
   the cases (Arabic or French) and the mismatch reported as its own slice, not hidden in the total.
3. WHEN the gate refuses THEN the refusal text SHALL be in the question's language (ar/fr/en);
   wrong-language gate refusals SHALL be 0 (baseline 22 / 32).
4. WHEN the fallback cannot translate a passage THEN the message around the excerpts SHALL be in
   the question's language.
5. WHEN a follow-up switches language THEN retrieval SHALL still find the conversation's document
   (`followup_lang_switch` 3/3 at retrieval, baseline 2/3; see Req 12).

### Requirement 6: Cited snippets show the evidence (I6, I24)

**Owner:** eequaled (snippet), nisbenz (ranking).

**User Story:** As a user, I want the source snippet to show the sentence that supports the
answer, so that I can check it.

#### Acceptance Criteria

1. WHEN a source is cited THEN `Source.snippet` SHALL be built around the chunk sentence that
   best matches the question (or the answer, when there is one), not the chunk's first 500
   characters, and SHALL stay within the existing length limit.
2. The snippet choice SHALL be deterministic and need no extra model call (lexical overlap on
   `fold_for_search` text is enough).
3. WHEN the quality suite runs THEN evidence shown in cited snippets SHALL be ≥ 90% of answered
   cases (baseline 70.9% without LLM, 28% with gemma4:e4b).
4. (nisbenz) The sentence splitter SHALL be the existing `text/sentences.py`, so ar/fr/en
   punctuation is handled the same way as in chunking.

### Requirement 7: Citations in other formats are understood (I22)

**Owner:** eequaled.

**User Story:** As a user, I want a correct answer to count as grounded even when the model cites
in a slightly different format.

#### Acceptance Criteria

1. WHEN the answer cites `[doc_id: X]` THEN `answer.build` SHALL treat it as a citation of `X`.
2. WHEN the answer cites a passage label `[S<n>]` THEN it SHALL map to that passage's `doc_id`.
3. IF a citation names a document that was not in the passages THEN it SHALL be ignored, as today.
4. WHEN the quality suite runs with the production model THEN the run info SHALL report the
   citation-rewrite rate, and the eval client SHALL no longer need to rewrite citations.

### Requirement 8: The LLM answers when the evidence was retrieved (I23)

**Owner:** eequaled (prompt), nisbenz (passages sent).

**User Story:** As a user, I want an answer whenever a retrieved passage states it, even in
another language.

#### Acceptance Criteria

1. The false-refusal rate SHALL first be measured with the production model
   (`google/gemma-4-31b-it:free`, record mode within the daily budget) before the prompt changes.
2. WHEN the evidence is in the passages THEN false refusals SHALL be ≤ 10% on the quality suite
   (baseline 22.4% with gemma4:e4b; 31 / 35 cases with evidence retrieved).
3. (nisbenz) The number and order of passages sent (`RERANK_TOP_N`, neighbours) SHALL be chosen by
   a measured run, and the chosen value documented in `.env.example`.
4. Cases that fail because of a retrieval miss SHALL be reported separately from LLM refusals.

### Requirement 9: Robust to small input noise (I7)

**Owner:** nisbenz.

**User Story:** As a user who types fast, I want typos, capitals and extra spaces not to change
whether I get an answer.

#### Acceptance Criteria

1. WHEN a query is encoded or reranked THEN case and whitespace noise SHALL be normalized in the
   query sent to the models (the stored and displayed text unchanged).
2. WHEN the robustness suite runs THEN the flip rate (answer ↔ refuse under any variant) SHALL be
   ≤ 10% of base cases (baseline 38.5%), reported per noise kind (typos, case/spacing, diacritics,
   filler prefix).
3. IF normalization is not enough THEN `GATE_THRESHOLD` SHALL be recalibrated on the `dev` split
   with the noisy variants included, under the false-refusal limit of Req 3.4.
4. Arabic diacritics and tatweel in queries SHALL be handled the same way as at ingest
   (`text/normalize.py`), so a vowelled question finds an unvowelled document.

### Requirement 10: Cross-lingual questions find the right document (I9)

**Owner:** nisbenz.

**User Story:** As an Arabic speaker, I want my question to find the English or French document
that answers it, even when an Arabic document on a nearby topic exists.

#### Acceptance Criteria

1. WHEN a candidate change runs THEN it SHALL be compared on the cross-lingual subset of the
   86-question set (`xl86-*`, incl. darija and arabizi) and the e2e `cross_lingual` slice against
   the baseline (granite cross-lingual hit@5 0.755; bge-m3 0.816; e2e retrieval 234 / 290). The
   single-change grid in `eval/results/README.md` (prefetch 80: 0.776, dense weight 2.0: 0.776,
   bge reranker: 0.796) is the starting point, so it is not re-measured.
2. The change SHALL raise cross-lingual hit@5 to ≥ 0.85 without lowering same-language hit@5 by
   more than 0.01 (Req 1.3).
3. WHEN the query's script differs from a candidate document's language THEN the system MAY use
   more dense candidates or a dense-weighted fusion; any such rule SHALL be a setting, off by
   default until measured.
4. The cases `coc-23`, `coc-24`, `fac-18`, `note-09`, `x-01`, `x-02` SHALL each be reported, pass
   or fail, in the result file.
5. IF the reranker lowers darija hit@5 below the fused order (baseline 0.71 fused → 0.43 reranked)
   THEN the result SHALL say so and the trade SHALL be decided explicitly.

### Requirement 11: A new short question is not glued to the previous turn (I10, I12)

**Owner:** nisbenz.

**User Story:** As a user who changes topic, I want my new question searched on its own.

#### Acceptance Criteria

1. WHEN a message under `RETRIEVAL_FOLLOWUP_MAX_WORDS` words has history AND has no follow-up cue
   (pronoun, demonstrative, ellipsis, leading "and/et/و", question about "it/ça/ذلك", in ar/fr/en)
   THEN retrieval SHALL search the message alone.
2. WHEN it has a follow-up cue THEN retrieval SHALL search the previous user message + the message,
   as today.
3. The cue lists SHALL be a pure, deterministic function with unit tests in ar, fr and en.
4. WHEN the eval runs THEN `conv-06` and the `topic_switch` slice SHALL pass at retrieval (baseline
   1/3) and `conv-05` and the existing follow-up cases SHALL still pass.
5. No LLM call SHALL be used to rewrite the query.

### Requirement 12: Photographed pages are not silently truncated (I13)

**Owner:** nisbenz.

**User Story:** As an admin who uploads a phone photo, I want to know when OCR lost text.

#### Acceptance Criteria

1. WHEN an OCR'd page yields far fewer words than its text area suggests THEN the
   `ParsedDocument.warnings` SHALL include a `low_ocr_yield` warning with the page number.
2. The warning rule SHALL be checked on `sg-candidacy-p2-photo.jpg` (142 of 329 words: warns) and
   on the flatbed scans (93–100% recall: no warning).
3. IF deskewing is added THEN it SHALL run before Tesseract only on images, SHALL NOT add a
   dependency without the user's approval (AGENTS.md dependency rules), and SHALL raise the photo's
   word recall above 90% without lowering the flatbed scans.

### Requirement 13: Document expiry is honoured or removed (I14)

**Owner:** eequaled + nisbenz (agreement first).

**User Story:** As an admin, I want `expires_at` to do what it says, or not be offered.

#### Acceptance Criteria

1. The meaning of `expires_at` SHALL be agreed and written in AGENTS.md ("Open seam items") before
   any code changes.
2. IF it is kept THEN the pipeline SHALL store it in the docstore and in the marker point, the
   listing SHALL return it, and expired documents SHALL be excluded from retrieval (filter inside
   the Qdrant query, like visibility).
3. IF it is removed THEN `POST /documents` SHALL reject it as an unknown field and the e2e probe
   `ingest-expires-at-kept` SHALL be updated accordingly.
4. WHEN fixed THEN the ingest suite SHALL pass 16 / 16.

### Requirement 14: Good document titles (I15)

**Owner:** eequaled (API field), nisbenz (title heuristic).

**User Story:** As a user, I want a source titled "Code of Conduct", not "MICRO CLUB".

#### Acceptance Criteria

1. (eequaled) `POST /documents` SHALL accept an optional `title` form field and pass it to the
   pipeline (the pipeline already takes `title=`).
2. (nisbenz) WHEN no title is given THEN the title SHALL come from the first-page heading (largest
   text or first Markdown heading xberg returns) before the PDF metadata title, and the filename
   stem last.
3. (nisbenz) Without the `eval/corpus.json` overrides, the Code of Conduct's title SHALL contain
   the words of its curated title ("Code of Conduct", "2025–2026"), and the SG candidacy's title
   SHALL come from its page-1 headings, not the filename ("KENPA SG DOC"); neither SHALL be
   generic text alone ("MICRO CLUB", a page number). Page 1 of the SG PDF does not say
   "General Secretary Candidacy", so its curated title needs the upload `title` field (14.1).
4. WHEN titles change THEN the eval SHALL be rerun without the title overrides, since the title is
   in the chunk header and the reranker input.

### Requirement 15: Pages made of share links are rejected (I16)

**Owner:** nisbenz.

**User Story:** As an admin, I want a page with no real content rejected as empty.

#### Acceptance Criteria

1. WHEN an HTML page is extracted THEN list items and blocks made only of share links
   (Facebook, Twitter/X, LinkedIn, WhatsApp, email, print, "share on …" in ar/fr/en) SHALL be
   dropped before the empty-document check.
2. WHEN only such links remain THEN `ingest_url` SHALL fail with `empty_document` (422).
3. The rule SHALL be covered by a committed HTML fixture of the `réintégration-aux-études` shape,
   and the existing web fixture (`web/mcoli_introduction.html`) SHALL keep its output.

### Requirement 16: API hygiene (I17, I18, I20, I21)

**Owner:** eequaled.

**User Story:** As an API client, I want consistent errors and the settings I passed to be used.

#### Acceptance Criteria

1. WHEN `DELETE /api/v1/documents/{doc_id}` names an unknown document THEN the API SHALL return
   404 with the JSON error `not_found` (I17).
2. WHEN a chat `message` is empty or whitespace only THEN the API SHALL return 422 without running
   retrieval (I18).
3. WHEN `create_app(settings=...)` is used THEN `get_role` and `require_admin` SHALL read the
   tokens from the app's settings, not the global ones (I20); the e2e driver SHALL then stop
   setting environment variables.
4. The `config.py` defaults SHALL match the measured retrieval defaults (gte reranker,
   `retrieve_top_k` 20, `gate_threshold` 0.40) or point to `RetrievalSettings` as the single source
   (I21).
5. The routes and parameters SHALL stay as they are (frozen API).

### Requirement 17: Throughput under concurrent users (I19)

**Owner:** nisbenz.

**User Story:** As one of several users at the same time, I want an answer within a few seconds.

#### Acceptance Criteria

1. WHEN 4 users chat concurrently for 60 s (load suite) THEN chat p95 SHALL be ≤ 8 s with 0
   errors (baseline p50 5.9 s, p95 7.0 s, 0.67 req/s), and throughput SHALL be reported next to
   the baseline.
2. The options (lower rerank top-k, a small pool of reranker sessions) SHALL be measured for
   latency, RSS and retrieval quality; a pool SHALL be used only if RSS stays within the container
   budget and per-pair scoring stays one pair per `session.run`.
3. Each retrieval stage SHALL log its time in `retrieval_done` (already the case) and the load
   report SHALL show where the time goes per stage.
4. A change of `RETRIEVE_TOP_K` SHALL respect Req 1.3 (quality drop ≤ 0.01).

## Issue index

| Issue | Severity | Requirement(s) | Owner |
|---|---|---|---|
| I1 LLM never called | critical | 2 | eequaled |
| I2 refuse-expected questions answered | critical | 3 | eequaled + nisbenz |
| I3 fallback repeats injected instructions | critical | 4 | eequaled + nisbenz |
| I4 answer in the document's language | high | 5 | eequaled |
| I5 gate refusal always English | high | 5 | eequaled |
| I6 snippets miss the evidence | medium | 6 | eequaled + nisbenz |
| I7 noise flips the decision | high | 9 | nisbenz |
| I8 gate score overlap | high | 3 | nisbenz |
| I9 cross-lingual distractors | high | 10 | nisbenz |
| I10 short new question glued to history | medium | 11 | nisbenz |
| I11 injection text ranks for attack questions | medium | 4 | nisbenz |
| I12 follow-ups in another language | medium | 5, 11 | nisbenz |
| I13 photographed page truncated | high | 12 | nisbenz |
| I14 `expires_at` dropped | medium | 13 | eequaled + nisbenz |
| I15 no upload title, bad PDF titles | medium | 14 | eequaled + nisbenz |
| I16 share-bar page indexed | medium | 15 | nisbenz |
| I17 delete twice → 200 | low | 16 | eequaled |
| I18 empty message accepted | low | 16 | eequaled |
| I19 one rerank at a time | medium | 17 | nisbenz |
| I20 auth reads global settings | low | 16 | eequaled |
| I21 `config.py` defaults disagree | low | 16 | eequaled |
| I22 citation formats | high | 7 | eequaled |
| I23 LLM refuses with evidence retrieved | high | 8 | eequaled + nisbenz |
| I24 snippets miss evidence with LLM | medium | 6 | eequaled |
