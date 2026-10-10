# Implementation Plan: Chunking

Each task is test-first: write the listed tests, see them fail, then implement. Every pushed task
keeps CI green (`uv lock --check`, ruff, `pytest -m "not ocr and not tokenizer"`, plus the model job).
Unit tests use a `FakeCounter` (1 token per whitespace-separated word), so only task 2's tests and
the golden/perf tests in task 8 need the real `tokenizer.json`.

- [x] 1. Settings, errors and the `Chunk` contract
- [x] 1.1 Chunking settings in `src/mcclub_rag/ingest/settings.py`
  - Add `chunk_target_tokens=300`, `chunk_max_tokens=400`, `chunk_min_tokens=80`,
    `chunk_max_header_tokens=48`, `chunk_context_header=True`, and `chunk_tokenizer_path` (default
    `$MODELS_DIR/embedder/tokenizer.json`, falling back to `/app/models/embedder/tokenizer.json`).
  - Add a `model_validator(mode="after")` enforcing `min < target <= max` and
    `max_header < max / 4`.
  - Append each variable, with one comment line, to `.env.example`.
  - Tests in `tests/test_settings.py`: the defaults; an env override (`INGEST_CHUNK_MAX_TOKENS=500`);
    `min >= target` and `target > max` both raise `ValidationError`.
  - _Requirements: 6.3, 9.1_
- [x] 1.2 `TokenizerConfigError` in `src/mcclub_rag/ingest/errors.py`
  - Subclass of `IngestError` with code `tokenizer_config_error`, not HTTP-mapped (same as
    `ocr_config_error`).
  - Extend `tests/test_errors.py`: the code is stable and the message carries the instruction.
  - _Requirements: 6.2_
- [x] 1.3 `Chunk` model in `src/mcclub_rag/ingest/models.py`
  - Frozen pydantic model with the fields from design "Data Models".
  - Extend `tests/test_models.py`: frozen; `heading_path` is a tuple; round-trips `model_dump()`.
  - _Requirements: 1.1, 1.5, 7.1_

- [x] 2. Token counting
- [x] 2.1 Pinned tokenizer download in `scripts/download_models.py`
  - `download_tokenizer(dest)`: fetch `tokenizer.json` from
    `ibm-granite/granite-embedding-311m-multilingual-r2` at revision
    `44399559930365213510b1ee2eb15ded83374f0e`, verify sha256
    `0087c868b33bad550a78a08d19798cfd7f713cde4f020803b8f51f405503e15f` (reuse `_sha256_file`),
    write atomically, and skip if the file is already present and valid.
  - Add `--only tokenizer` to `main()`; the default destination is `$MODELS_DIR/embedder`.
  - Extend `tests/test_download_models.py` (mocked HTTP, as for tessdata): a sha mismatch raises and
    leaves no file; an existing valid file is not re-downloaded; the CLI flag routes correctly.
  - _Requirements: 6.1, 6.2_
- [x] 2.2 `src/mcclub_rag/ingest/tokens.py`
  - `TokenCounter` protocol (`count`, `count_many`, `cut`, `fingerprint`).
  - `HFTokenCounter(path)`: `Tokenizer.from_file`, then `no_truncation()` and `no_padding()`;
    `add_special_tokens=False`; `cut` splits at `offsets[max_tokens - 1][1]`; `fingerprint` = sha256
    of the file.
  - `verify_tokenizer(settings) -> Path` raises `TokenizerConfigError` with the download command.
  - `get_token_counter(settings)` caches one instance per path.
  - Register the `tokenizer` marker in `pyproject.toml` and auto-apply it to tests using a
    `tokenizer_path` fixture in `tests/conftest.py`. The fixture fails with the instruction when the
    file is missing, mirroring `tessdata_dir`.
  - `tests/test_tokens.py`:
    - A missing file raises `TokenizerConfigError` (no marker needed).
    - `[tokenizer]` a text above 40k tokens counts above 32,768.
    - `[tokenizer]` `count_many` equals `[count(t) ...]`, so padding is off.
    - `[tokenizer]` `cut` is lossless (`head + tail == text`) and `count(head) <= n` on Arabic,
      French and code samples.
    - `[tokenizer]` `fingerprint` is stable.
  - _Requirements: 6.1, 6.2, 6.4_
- [x] 2.3 CI: model job also covers the tokenizer, in `.github/workflows/ci.yml`
  - In the existing OCR job, add a cache step keyed on `scripts/download_models.py` for
    `models/embedder`, run `download_models.py --only tokenizer --dest models/embedder`, then
    `INGEST_CHUNK_TOKENIZER_PATH=$PWD/models/embedder/tokenizer.json uv run pytest -m "ocr or tokenizer"`.
  - Change the fast job to `pytest -m "not ocr and not tokenizer"`.
  - Update the commands in `AGENTS.md` and the `README.md` to match.
  - _Requirements: 6.2_

- [x] 3. Lossless sentence and word splitting in `src/mcclub_rag/text/sentences.py`
  - `split_sentences(text)` and `split_words(text)`, using the rules from design `text/sentences.py`.
    Both are pure, and their pieces keep trailing whitespace.
  - `tests/test_sentences.py`:
    - Arabic `؟`, `؛`, `۔` boundaries.
    - French and English `.`, `!`, `?`, `…`.
    - A terminator followed by `»`, `”` or `)`.
    - Non-boundaries: `3.5`, `e.g. this`, `M. Dupont`, `v1.2`, `example.com`, `p. 12`.
    - Mixed Arabic/French text.
    - Lossless join over parametrized samples and every `tests/fixtures/*.md`/`*.txt`.
  - _Requirements: 3.3, 1.3_

- [x] 4. Outline and blocks in `src/mcclub_rag/ingest/chunk.py` (private helpers)
- [x] 4.1 `_outline(sections) -> list[_OutlineSection]`
  - A heading stack reusing `sections._HEADING`; XLSX sheet reset; PDF continuation inherits; the
    heading line is removed from `body` and kept in `heading_line`.
  - `tests/test_chunk.py::TestOutline`:
    - An H1 > H2 > H3 path, and an H2 after an H3 popping correctly.
    - A page-2 PDF section without a heading line inherits the path.
    - Two XLSX sheets get paths `(sheet,)` each.
    - Text before the first heading has path `()`.
    - A `### x` with no H2 parent gets path `(H1, x)`.
  - _Requirements: 2.2, 2.3, 2.5_
- [x] 4.2 `_parse_blocks(body, counter) -> list[_Block]`
  - Line scanner for code fences (including an unclosed fence), tables, lists with continuation
    lines, and blank-line paragraphs. One `count_many` call per section.
  - `TestBlocks`:
    - Each block kind is detected.
    - An unclosed fence runs to the end of the section.
    - XLSX row lines form one paragraph.
    - Joining the blocks reproduces the body, ignoring whitespace.
  - _Requirements: 3.4_

- [x] 5. Packing and splitting oversized sections
- [x] 5.1 `_split_unit(block, budget, counter)`
  - Levels in order: para → line → sentence → word → token.
  - Tables split at rows, repeating the header and separator rows.
  - Code and lists split at line level.
  - The token level uses `counter.cut` and increments a `hard_splits` counter.
  - `TestSplit`:
    - A 1,000-word paragraph splits at sentences, and every piece ends on a terminator.
    - An oversized table repeats its header in every piece.
    - An oversized code block splits only at newlines.
    - A 600-token single word hard-splits and the pieces rejoin to the original.
  - _Requirements: 3.1, 3.4, 3.5_
- [x] 5.2 `_pack(blocks, target, budget, min_tokens, counter) -> list[list[str]]` with tail rebalance
  - Greedy packing to `target`, never over `budget`; the separator cost is counted. A small tail
    first joins its predecessor, otherwise units are shifted back into it.
  - `TestPack`:
    - A table that fits is never cut.
    - No piece exceeds the budget.
    - A tail under `min` is merged or rebalanced.
    - The long Arabic H2 sample splits at paragraph boundaries.
  - _Requirements: 3.1, 3.2, 3.6_

- [x] 6. Grouping: merging small sections in `chunk.py` (`_group(outline, ...) -> list[_Draft]`)
  - Rules from design "Grouping":
    - A heading-only section whose next section is a descendant is absorbed into the path.
    - Small runs merge under the same parent while `< min` and `≤ target`, with headings kept
      inline.
    - The draft path is the longest common prefix of its members' paths.
    - A leftover small group is appended to the previous draft when the parent matches and the
      budget allows.
    - A trailing heading-only section is appended to the previous draft.
  - Large sections go to `_pack`.
  - `TestGroup`:
    - A slide-style document (`# VP`, `## Contents`, `## 01` … `## 07` with 1 to 8 word bodies)
      gives at most 2 drafts, with `## 01` inline.
    - Merging never crosses an H1 boundary.
    - A heading-only H2 followed by an H3 leaves no `## H2` text and the H3 path includes the H2.
    - A trailing heading-only section ends up in the last draft.
    - Large sections are never merged with each other.
  - _Requirements: 2.1, 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 7. Finalize and the public `chunk_document()`
- [x] 7.1 Header, exact counting, re-pack loop and metadata
  - `_header(title, path, max_header_tokens, counter)`: dedupe consecutive entries, drop from the
    left, then `cut`.
  - `embed_text` is built per `chunk_context_header`.
  - One `count_many` over all `embed_text`s. Over-limit drafts are re-packed with
    `budget - overshoot` (at most 3 rounds), then `cut`.
  - Pages, `section_index`, the language vote and contiguous `index`.
  - `TestFinalize`:
    - A title equal to the H1 appears once in the header.
    - A long path keeps its innermost headings.
    - `context_header=False` gives `embed_text == text`.
    - The text never contains the header.
    - An adversarial counter (whose join cost exceeds the sum of parts) still yields
      `token_count <= max`.
    - A merge across pages 3 and 4 gives `page_start=3`, `page_end=4`.
    - The language vote ignores `und` and falls back to the document language.
  - _Requirements: 1.1, 3.6, 5.1, 5.2, 5.3, 5.4, 7.1, 7.2_
- [x] 7.2 `chunk_document(doc, *, settings=None, counter=None)` and logging
  - Wire outline → blocks → group → finalize.
  - Return `()` for a document with no alphanumeric content.
  - Emit the structlog `document_chunked` event with `title`, `chunks`, `tokens_min`,
    `tokens_median`, `tokens_max`, `merged`, `hard_splits`; emit `chunk_hard_split` per token cut.
  - `TestChunkDocument`, parametrized over synthetic documents (docs-site tree, Arabic article,
    slide PDF, XLSX, plain text, empty), checking these invariants:
    - `index` is contiguous.
    - `token_count <= max`.
    - Lossless coverage (Req 1.3, with repeated table headers exempt).
    - Determinism: two runs give equal tuples.
    - An empty document gives `()`.
  - `capture_logs` checks one `document_chunked` event with the expected keys and no document text
    in any event.
  - _Requirements: 1.2, 1.3, 1.4, 8.1_

- [x] 8. Corpus-version hook and real-tokenizer checks
- [x] 8.1 `chunking_signature(settings, counter) -> str` in `ingest/settings.py` (or `chunk.py`), with
  `CHUNKER_VERSION = 1`
  - sha256 over the sorted `chunk_*` values (excluding the path), `counter.fingerprint` and the
    version.
  - Tests: the signature is stable across calls; it changes when any `chunk_*` field, the counter
    fingerprint, or `CHUNKER_VERSION` changes; it does not change when only the tokenizer path
    changes.
  - _Requirements: 9.1, 9.2_
- [x] 8.2 Golden and performance tests with the real tokenizer (`tokenizer` marker)
  - Extend `tests/fixtures/make_fixtures.py` to write
    `tests/fixtures/chunking/mcoli_introduction.json`: the list of
    `(heading_path, page_start, token_count)` from preprocessing `web/mcoli_introduction.html`
    (mocked fetch, existing helper) and then chunking. Commit it and list it in
    `tests/fixtures/README.md`.
  - `tests/test_chunk_golden.py`:
    - `[tokenizer]` the output equals the golden file.
    - `[tokenizer]` a synthetic 300-section mixed ar/fr/en document chunks in under 2 s.
    - `[tokenizer]` every chunk of the golden document is at most 400 granite tokens.
  - _Requirements: 6.1, 8.2, 2.1_

- [x] 9. Repo docs kept in sync
  - `EDGE_CASES.md`: add these rows, each with its evidence (the probe numbers from the design):
    - The tokenizer's built-in truncation and padding.
    - The reranker/granite token ratio up to 1.28.
    - Tiny sections from slide PDFs.
    - Heading-only sections.
    - Oversized tables.
    - Unclosed code fences.
    - Single words over the budget.
  - `README.md`: a chunking subsection under architecture/key decisions.
  - `PROGRESS.md`: dated entry.
  - `AGENTS.md`: mark `chunk.py` as built and update the "Not yet built" list.
  - _Requirements: all (keeps repo docs in sync per AGENTS.md)_
