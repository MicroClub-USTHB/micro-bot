# Implementation Plan — Data Preprocessing

Assumptions until the design's open questions are answered: `tessdata_fast`, `lang_default="fr"`,
real Arabic scans supplied later by nisbenz, and I own the tessdata part of `scripts/download_models.py`.

Implementation note: Pillow here has libraqm, so a correctly shaped synthetic `arabic_notice.png` is
generated and tested. Real scans (`arabic_scan.pdf`, `arabic_photo.jpg`) are still wanted for
real-world quality; their tests skip until the files exist.
Each task is test-first: write the listed tests, see them fail, then implement. From task 1.3 on, every
pushed task must keep the GitHub Actions CI workflow green.

- [x] 1. Dependencies and settings
- [x] 1.1 Update `pyproject.toml` and lockfile
  - Add `charset-normalizer` to `dependencies` (already transitive via trafilatura; made explicit).
  - Add `pillow`, `openpyxl`, `python-docx`, `pypdf` to the `dev` group (fixture generation only).
  - Run `uv lock` and `uv sync`, then confirm `uv run python -c "import xberg, trafilatura, py3langid"` succeeds.
  - _Requirements: 2.4, 7.2_
- [x] 1.2 Create `src/mcclub_rag/ingest/settings.py`
  - `IngestSettings(BaseSettings)` with every field and default from design "Data Models", env prefix
    `INGEST_`, `tessdata_prefix` read from `TESSDATA_PREFIX`. Add a cached `get_ingest_settings()`.
  - Append the new variables, with one comment line each, to `.env.example`.
  - `tests/test_settings.py`: defaults match the design; an env override works (`INGEST_MAX_PAGES=5`);
    `TESSDATA_PREFIX` is honoured.
  - _Requirements: 7.6_

- [x] 1.3 CI test runner `.github/workflows/ci.yml`
  - Triggers: `push` on any branch, and `pull_request` into `main`.
  - Job `test` on `ubuntu-latest`:
    1. `actions/checkout@v4`
    2. `astral-sh/setup-uv@v6` with `enable-cache: true`
    3. `uv python install 3.12`
    4. `uv sync --locked`, which fails if `uv.lock` is stale; `uv.lock` is committed in 1.1
    5. `uv run ruff check .` then `uv run ruff format --check .`
    6. `uv run pytest -ra`
  - Register the `ocr` marker in `[tool.pytest.ini_options]` (`markers = ["ocr: needs tessdata"]`), so the
    marker from task 7 is declared up front. Add `--strict-markers`.
  - `concurrency: ci-${{ github.ref }}` with `cancel-in-progress: true`, so a new push supersedes the running job.
  - Add a placeholder `tests/test_smoke.py` (`import mcclub_rag`), so the first run collects at least one test.
  - Also covers eequaled's tests as they land; the workflow is not specific to preprocessing.
  - _Requirements: 8.2_

- [x] 2. Output contract and errors
- [x] 2.1 Create `src/mcclub_rag/ingest/models.py`
  - `Visibility`, `SourceType`, frozen `Section` and `ParsedDocument` exactly as in the design.
  - Internal dataclasses `RawPage(number, content, sheet_name, tables)` and `RawExtraction(...)`.
  - `tests/test_models.py`: `visibility="admin"` raises a validation error; models are frozen;
    a `ParsedDocument` JSON round-trip works.
  - _Requirements: 1.1, 1.3_
- [x] 2.2 Create `src/mcclub_rag/ingest/errors.py`
  - `IngestError(code, message)` plus the subclasses and codes from the design's error table
    (`InvalidInput`, `FileTooLarge`, `UnsupportedFileType`, `ParseError`, `ParseTimeout`, `EmptyDocument`,
    `FetchError`, `UrlNotAllowed`, `OcrConfigError`). Add an `ALL_ERRORS` tuple for the coverage check in 12.3.
  - `tests/test_errors.py`: codes are unique; subclass relationships hold (`ParseTimeout` → `ParseError`,
    `UrlNotAllowed` → `FetchError`).
  - _Requirements: 2.5, 2.6, 2.7, 3.4, 3.6, 4.4, 4.5, 4.6, 8.3_

- [x] 3. Text normalization (`src/mcclub_rag/text/normalize.py`)
- [x] 3.1 `normalize_text` and `rejoin_hyphens`
  - `tests/test_normalize.py`, table-driven:
    - CRLF; C0/C1 controls removed
    - U+200B/FEFF/00AD/bidi controls/LRM/RLM removed; ZWNJ/ZWJ kept
    - tatweel removed; NBSP becomes a space
    - space runs collapsed; ≥3 newlines → 2
    - "infor-\nmation" → "information" but "Jean-\nPierre" kept
    - NFC applied
    - idempotence: `f(f(x)) == f(x)`
  - Implement in the order given in the design.
  - _Requirements: 5.1, 5.2, 5.3, 5.5_
- [x] 3.2 `strip_repeated_lines(pages)`
  - Tests:
    - a 5-page input with "Club MC — Page N / 5" footers and a repeated header is stripped;
    - a line repeated on only 2 of 5 pages is kept;
    - inputs with fewer than 3 pages are returned unchanged;
    - body lines that happen to repeat mid-page are untouched.
  - _Requirements: 5.4, 5.5_
- [x] 3.3 `fold_for_search(s)`
  - Tests:
    - أحمد/إحمد/آحمد → احمد; مدرسة → مدرسه; مستشفى → مستشفي
    - harakat and tatweel stripped; ٱ → ا
    - "Événement" → "evenement"; "٣٤" and "۳۴" → "34"
    - output is stable when applied twice
  - _Requirements: 5.3_

- [x] 4. Language detection (`src/mcclub_rag/text/lang.py`)
- [x] 4.1 `detect` and `assign_languages`
  - Lazy singleton `LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)` with
    `set_languages(settings.lang_codes)`. `detect(text) -> LangResult(code, prob)` applying the
    `lang_min_chars` / `lang_min_prob` rules; `assign_languages(sections) -> (sections_with_lang, doc_language)`
    using the char-weighted `mixed` rule.
  - `tests/test_lang.py`:
    - ar, fr and en paragraphs are detected correctly
    - "ok" → `und`
    - an ar section + an fr section of similar length → `mixed`
    - 95% fr + 5% en → `fr`
    - all-empty sections → `und`
  - _Requirements: 6.1, 6.2, 6.3_
- [x] 4.2 `detect_language(text) -> Literal["ar","fr","en"]` (frozen API for the answer layer)
  - Tests:
    - "متى الاجتماع؟" → ar (the Arabic-script shortcut works on short text)
    - "Quand est la réunion du club ?" → fr; "When is the club meeting?" → en
    - "123 ?" → `lang_default`
    - the return value is always in {ar, fr, en}, checked over a small parametrized list incl. "ok"
  - _Requirements: 6.4_

- [x] 5. File type sniffing (`src/mcclub_rag/ingest/sniff.py`)
  - `detect_type(data, filename) -> DetectedType(mime, kind)` per the design's signature table. ZIP
    inspection reads the central directory only (`zipfile.ZipFile(io.BytesIO(data)).namelist()`).
  - `tests/test_sniff.py` (synthetic bytes, no fixtures), one case per table row:
    - each supported signature is recognised;
    - a plain ZIP → `UnsupportedFileType` naming `application/zip`;
    - an OLE2 container named `x.docx` → `ParseError` "encrypted Office file";
    - an OLE2 container named `x.doc` → `UnsupportedFileType`;
    - junk named `a.pdf` → `UnsupportedFileType`;
    - `.txt` with a NUL byte → `UnsupportedFileType`;
    - a `.md` file → `text/markdown`.
  - _Requirements: 2.5, 2.6_

- [x] 6. Test fixtures and tessdata
- [x] 6.1 Fixture generator `tests/fixtures/make_fixtures.py` and committed outputs
  - Deterministic generation (fixed text, no timestamps where the library allows) of:
    - **PDFs:**
      - `text.pdf`: 4 pages of realistic length, with a repeated "Club MC — Page N / 4" footer and a heading per page; minimal hand-written PDF writer as in the design spike
      - `scanned.pdf`: PIL image saved as PDF
      - `mixed.pdf`: page 1 text, page 2 image
      - `encrypted.pdf`: via pypdf
      - `corrupt.pdf`: truncated `text.pdf`
    - **Office files:**
      - `doc.docx`: headings H1/H2 + a table
      - `sheet.xlsx`: "Schedule" sheet with header + 3 rows, a second filled sheet, an empty sheet
    - **Text files:** `notes.md`, `latin1.txt` (French, Latin-1), `arabic_cp1256.txt` (Arabic, Windows-1256)
    - **Images:** `notice.png`, `notice.jpg` (English/French text rendered with DejaVuSans)
  - `tests/fixtures/README.md` lists every file and states that `arabic_notice.jpg` / `arabic_scan.pdf`
    are real scans to be added by nisbenz.
  - _Requirements: 8.1_
- [x] 6.2 Tessdata download step in `scripts/download_models.py`
  - `download_tessdata(dest: Path, variant: Literal["fast","best"]="fast")`:
    - fetches `ara`, `fra`, `eng` from `tesseract-ocr/tessdata_<variant>` at tag `4.1.0` with httpx;
    - verifies pinned SHA-256s (computed once and hard-coded);
    - writes atomically (tmp file + rename);
    - skips files already present with the right hash.
  - CLI: `--only tessdata --dest <dir> --variant fast|best`. Uncomment the Dockerfile `RUN … download_models.py` line.
  - CI: add a step before `pytest` in `.github/workflows/ci.yml`:
    - `actions/cache@v4` on `models/tessdata`, keyed on `hashFiles('scripts/download_models.py')`
    - then `uv run python scripts/download_models.py --only tessdata --dest models/tessdata`
    - `TESSDATA_PREFIX=${{ github.workspace }}/models/tessdata` exported for the job, so `@pytest.mark.ocr`
      tests run in CI instead of failing on missing traineddata.
  - `tests/test_download_models.py` with `httpx.MockTransport`: hash mismatch raises and leaves no file;
    an existing good file isn't re-downloaded.
  - _Requirements: 3.3, 3.4, 7.1_

- [x] 7. OCR configuration (`src/mcclub_rag/ingest/ocr.py`)
  - `verify_tessdata(settings) -> Path`: every `{lang}.traineddata` in `settings.ocr_languages` exists and
    is non-empty, else `OcrConfigError` naming the missing languages and the download command.
  - `timeout_budget(kind, page_count, settings) -> int` and `build_xberg_config(settings, timeout_s)`
    returning the `ExtractionConfig` from the design.
  - `tests/conftest.py` gets a `tessdata_dir` fixture that **fails** (not skips) with the download
    instruction when traineddata is absent, plus a `settings` fixture pointing at it.
  - `tests/test_ocr.py`:
    - a temp dir missing `ara.traineddata` → `OcrConfigError` mentioning "ara"
    - the built config has `use_cache=False` at both levels, `enable_table_detection=False`,
      `tessdata_path` set, `languages == ["ara","fra","eng"]` and `max_pages` from settings
    - budget = 60 + 10×pages for PDFs, 70 for images, 60 otherwise
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 7.1, 7.3_

- [x] 8. File parsing (`src/mcclub_rag/ingest/parse.py`)
- [x] 8.1 `decode_text(data)`
  - Tests: UTF-8 with BOM; `latin1.txt`; `arabic_cp1256.txt` round-trips to the expected Arabic string;
    undecodable random bytes → `ParseError`.
  - _Requirements: 2.4_
- [x] 8.2 `map_xberg_error(exc)`
  - Tests use the exact messages observed in the design spike:
    - "Security violation: Document has too many pages…" → `FileTooLarge`
    - "Security violation: Potential ZIP bomb…" → `ParseError`
    - "Extraction timed out after…" → `ParseTimeout`
    - "Parsing error: … Invalid cross-reference table" → `ParseError` with a ≤ 300-char message
    - the cause is chained
  - _Requirements: 2.6, 2.7, 7.3_
- [x] 8.3 `extract_file(data, detected, settings) -> RawExtraction`
  - MD/TXT go through `decode_text`. Everything else goes through
    `xberg.extract(ExtractInput(kind="bytes", mime_type=detected.mime, ...), build_xberg_config(...))`.
    PDFs are pre-checked with `xberg.pdf_page_count`.
  - Wraps the call in `asyncio.wait_for(budget + 5)` and maps exceptions through `map_xberg_error`. Fills
    `ocr_used` from `extraction_method`/`metadata.ocr_used` and the parser string from `xberg.__version__`.
  - `tests/test_parse.py` (integration, real xberg, `@pytest.mark.ocr` where OCR runs):
    - `text.pdf`: 4 pages, `ocr_used` False, method native
    - `scanned.pdf` and `notice.png`: `ocr_used` True, expected words present
    - `mixed.pdf`: only page 2 was OCR'd
    - `doc.docx`: headings + table in markdown
    - `sheet.xlsx`: pages carry `sheet_name`
    - `encrypted.pdf` → `ParseError` "password-protected"
    - `corrupt.pdf` → `ParseError`
    - `max_pages=1` on `text.pdf` → `FileTooLarge`
    - no writes under `~/.cache/xberg/ocr` (assert via `HOME`/`XDG_CACHE_HOME` pointed at `tmp_path`)
  - _Requirements: 2.1, 2.2, 2.3, 2.6, 2.7, 3.1, 3.2, 3.3, 7.1, 7.3_

- [x] 9. Sectioning (`src/mcclub_rag/ingest/sections.py`)
  - `split_markdown(text, page=None, inherited_heading=None)` and `build_sections(raw: RawExtraction)`
    with the per-format rules from the design, including XLSX rows as `Header: value · Header: value` lines
    built from `RawPage.tables` cells.
  - `tests/test_sections.py` (constructed `RawExtraction`s, no xberg):
    - PDF heading carried across a page break, with correct `page` numbers
    - DOCX split at H1/H2
    - XLSX: one section per non-empty sheet, `heading` = sheet name, row-line format, empty sheet dropped
    - TXT → a single section
    - image → `page=1`
    - whitespace-only sections dropped
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.6_

- [x] 10. SSRF guard (`src/mcclub_rag/ingest/url_guard.py`)
  - `async check_url(url, resolve=default_resolver)`.
  - `tests/test_url_guard.py` with a fake resolver:
    - each of 127.0.0.1, 10.0.0.5, 172.16.0.1, 192.168.1.1, 169.254.169.254, 100.64.0.1, ::1, fc00::1,
      fe80::1, ::ffff:127.0.0.1 → `UrlNotAllowed`
    - a host resolving to one public and one private address → rejected
    - `file://`, `ftp://`, `http://user:pw@host`, a missing host → rejected
    - a public IP → passes
  - _Requirements: 4.5_

- [x] 11. Web fetching and extraction (`src/mcclub_rag/ingest/web.py`)
- [x] 11.1 `fetch(url, settings, client=None, resolve=...) -> FetchedResource`
  - Manual redirect loop with `check_url` on every hop, streamed size cap, UA header, error mapping per the design.
  - `tests/test_web_fetch.py` (`httpx.MockTransport` + fake resolver):
    - 200 HTML
    - 301 → 200 sets `final_url`
    - a redirect to `http://127.0.0.1/` → `UrlNotAllowed`
    - 6 redirects → `FetchError`
    - 404 → `FetchError("HTTP 404")`
    - a timeout → `FetchError("timeout")`
    - `Content-Length` over the cap is rejected before the body is read
    - a chunked body over the cap is aborted
    - the UA header is sent
  - _Requirements: 4.1, 4.4, 4.5, 4.7_
- [x] 11.2 `extract_html(html_bytes, final_url) -> RawExtraction`
  - trafilatura call and metadata title per the design.
  - Tests with an inline HTML article containing nav, cookie banner and footer:
    - boilerplate is absent; headings and table are kept; title is extracted
    - a page with only nav/footer → `EmptyDocument`
  - _Requirements: 4.2, 4.6_

- [x] 12. Orchestration (`src/mcclub_rag/ingest/preprocess.py`)
- [x] 12.1 `preprocess_file(data, filename, *, visibility, title=None) -> ParsedDocument`
  - Steps:
    - validate inputs (`InvalidInput`); size check (`FileTooLarge`)
    - `verify_tessdata` once (cached)
    - semaphore → `detect_type` → `extract_file`
    - for PDFs, `strip_repeated_lines` over page texts
    - `build_sections` → `normalize_text` per section (in `to_thread`) → drop empty → `EmptyDocument` if none
    - `assign_languages`
    - `content_hash` = sha256 of the bytes; title fallback chain
    - one `document_preprocessed` / `document_rejected` structlog event
  - `tests/test_preprocess_file.py`:
    - `content_hash` is identical for the same bytes twice
    - title fallback order (caller > metadata > first heading > filename stem)
    - `sections` and `text` are consistent
    - the PDF footer is gone from `text`
    - a log event is captured with `structlog.testing.capture_logs` and contains no document text
    - oversize file → `FileTooLarge` before parsing (assert xberg not called via monkeypatch)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.7, 5.4, 7.3, 7.4, 7.5_
- [x] 12.2 `preprocess_url(url, *, visibility, title=None) -> ParsedDocument`
  - Routing:
    - HTML → `extract_html`
    - other content types → `detect_type` + `extract_file`, with filename from the URL path and
      `source_type="web"`, `source=final_url`
  - Hashing: `content_hash` over normalized text for HTML, raw bytes otherwise.
  - Logging strips the query string.
  - `tests/test_preprocess_url.py` (MockTransport):
    - an HTML article → `source_type="web"`, `source` = final URL
    - `text.pdf` served as `application/pdf` → parsed via the file path
    - HTML whose ads change but article doesn't → same `content_hash`
    - a URL with `?token=secret` → the token is absent from the captured log
  - _Requirements: 1.1, 1.2, 4.2, 4.3, 7.5_
- [x] 12.3 End-to-end and coverage tests
  - `tests/test_ingest.py` (README-planned name), parametrized over every fixture:
    - `preprocess_file` yields a valid `ParsedDocument` with expected `language` (ar for
      `arabic_cp1256.txt`, fr for `latin1.txt`), `ocr_used` and section count.
    - Arabic scans (`arabic_notice.jpg`, `arabic_scan.pdf`): marked `skipif` the file is absent, with the
      reason "real scan not yet provided". When present, they assert Arabic text in logical order
      (a known word appears unreversed) and `language == "ar"`.
  - Error coverage: a test iterates `ALL_ERRORS` and asserts each class is raised by at least one named
    test case (a mapping from error class to triggering test). `OcrConfigError` is covered by task 7.
  - `uv run ruff check . && uv run pytest` passes locally with networking unavailable (web tests use
    MockTransport only), and the CI workflow from 1.3 is green on the branch.
  - _Requirements: 3.5, 6.1, 8.1, 8.2, 8.3_
