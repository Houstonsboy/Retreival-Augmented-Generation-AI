# Chapter 5 — Testing and Evaluation

*Audit date: 2026-11-09. All values below are sourced from executed commands
(`git`, `pytest`), committed manifests (`requirements.txt`, `package.json`), and
the actual test files in `backend/tests/` and `firestore-tests/`. Nothing is
invented; where evidence is missing or a value could not be measured, it is
stated explicitly rather than estimated.*

---

## 5.4 Testing

### 5.4.1 Test Strategy

The project uses a **layered, mock-first unit-test strategy** with a small set
of integration tests that are gated behind an external service (the Firestore
emulator). The suite is intentionally designed to run offline and without
network access, LLM keys, OCR, or a GPU.

**Framework and environment**

| Item | Value | Evidence |
|---|---|---|
| Test runner | `pytest` 9.2.0 (Python 3.13.7) | executed test header |
| Mocking | `unittest.mock` (`MagicMock`, `patch`) + `pytest-mock` 3.17.0 | `test_history.py`, plugin list |
| Async plugin | `anyio` 4.11.0, `langsmith` 0.4.43 | pytest plugin list |
| Backend test files | `backend/tests/test_history.py`, `backend/tests/test_sccvision_analyst.py` | filesystem |
| Rules test suite | `firestore-tests/firestore.test.ts` (Jest + `@firebase/rules-unit-testing`) | `firestore-tests/package.json` |
| Frontend tests | **None.** `frontend/package.json` has no `test` script (only `dev`, `build`, `start`, `lint`) | `frontend/package.json` |

**Strategy layers**

1. **Pure unit tests** — deterministic functions with no I/O: title cleaning,
   fallback title generation, PII masking, M-Pesa parsing, date checks,
   verdict computation. These run with zero external dependencies.
2. **Mocked persistence tests** — Firestore is replaced with `MagicMock`
   reference chains; `history_store._db`, `_conv_ref`, `_turns_col`, and the
   transaction executors are patched. This exercises save/read/idempotency/
   overwrite logic without a live database.
3. **Flask endpoint tests** — a real `app.test_client()` is used with only the
   Firebase token verification (`auth._verify_and_load`) patched, so routing,
   validation, and status codes are tested against the real blueprints.
4. **Stubbed-LLM pipeline tests** — `SCCVision.analyst` is exercised end-to-end
   with a `_FakeLLM`/`_CaseFakeLLM` whose `complete()` returns a fixed
   `AnalystResult`, so the enrichment, verification, and follow-up layers are
   tested deterministically with no network or OCR.
5. **Emulator-gated integration tests** — `test_history.py::TestEmulatorIntegration`
   performs full Firestore round-trips but is **skipped** unless
   `FIRESTORE_EMULATOR_HOST` is set (it was not set during this audit).
6. **Firestore security rules tests** — `firestore-tests/firestore.test.ts`
   asserts `assertSucceeds`/`assertFails` against `firestore.rules`. These
   require the Firebase emulator and were **not executed** in this environment.

**Execution**

```bash
# Backend unit + endpoint + stubbed-pipeline suite
cd backend && PYTHONPATH=. venv/bin/pytest tests/ -v
# Firestore security rules (requires emulator)
cd firestore-tests && npm test   # or: npm run test:rules
```

### 5.4.2 Acceptance Criteria and Functional Test Matrix

**Executed result**

```
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-9.2.0, pluggy-1.7.0
collected 99 items

tests/test_history.py         66 tests  (62 passed, 4 skipped)
tests/test_sccvision_analyst.py 33 tests (33 passed)

======================== 95 passed, 4 skipped in 15.11s ========================
EXIT_CODE=0
```

The 4 skipped tests are all in `TestEmulatorIntegration` (Firestore emulator
not running). The 33 SCCVision tests and the 62 runnable history tests all pass.

**Table 5.3 — Functional Test Matrix**

| ID | Requirement / Objective | Test case | Type | Level | Result |
|---|---|---|---|---|---|
| H-01 | Fallback conversation titles | `make_fallback_title`: short unchanged, long trimmed at word boundary, 60-char no ellipsis, 61-char ellipsis | Unit | Function | PASS |
| H-02 | Generated title sanitation | `clean_generated_title`: strips `\<think>` blocks, quotes, `[S#]` alias brackets, trailing punctuation, whitespace; caps at 60; empty on think-only | Unit | Function | PASS |
| H-03 | History source payloads | `build_sources_for_history`: snippet ≤300 chars, no full text stored, case metadata populated, constitution/statute type mapping, missing chunk nulls metadata | Unit | Function | PASS |
| H-04 | Turn persistence | `save_turn`: new conversation returns IDs, follow-up not new, idempotent same `client_turn_id`, uses client id as doc id | Unit (mocked) | Function | PASS |
| H-05 | Conversation ownership | `get_or_create_conversation`: existing returns ref, nonexistent raises `LookupError`, foreign conversation raises | Unit (mocked) | Function | PASS |
| H-06 | Title generation | `generate_conversation_title`/`set_generated_title`/`submit_title_generation`: generated title set only over fallback; Groq failure/timeout/empty leave title alone; executor submission | Unit (mocked) | Function | PASS |
| H-07 | Conversation listing | `list_conversations`: ordered list, pagination `next_cursor` when >limit items | Unit (mocked) | Function | PASS |
| H-08 | Conversation reads/manage | `get_conversation` ordering; `rename_conversation` sets `user` source; `set_pinned`; `delete_conversation` calls `recursive_delete` | Unit (mocked) | Function | PASS |
| H-09 | History REST endpoints | `TestHistoryEndpoints`: list 200, bad limit, get 404, PATCH rename, empty/too-long title rejected (400), delete 404/200, regenerate-title 404/202 | Endpoint (mocked auth) | API | PASS |
| H-10 | Document size guard | `_size_guard`: oversized doc truncates (answer + snippet ≤100), normal doc unchanged | Unit | Function | PASS |
| H-11 | Idempotency | `check_idempotency`: none when missing/not-found/error, complete vs failed status, short-circuit on missing args | Unit (mocked) | Function | PASS |
| H-12 | Retry/overwrite | `save_turn(overwrite_turn_index=...)`: calls overwrite transaction not normal, preserves turn index, `is_new=False` | Unit (mocked) | Function | PASS |
| H-13 | Firestore full cycle | save → read → title lifecycle → overwrite-failed-turn (emulator) | Integration | System | **SKIPPED** (no emulator) |
| S-01 | PII masking | `mask_phone`, `mask_id_number`, `mask_pii` (phones masked before IDs) | Unit | Function | PASS |
| S-02 | M-Pesa parsing | full confirmation parse (code/amount/recipient/confidence), requires "Confirmed" | Unit | Function | PASS |
| S-03 | Amount extraction | keyword-quote amount detection | Unit | Function | PASS |
| S-04 | Integrity flags | missing dates; edited + backwards time order | Unit | Function | PASS |
| S-05 | OCR merge | dedupe identical messages at boundary | Unit | Function | PASS |
| S-06 | Analysis input | masks PII, normalizes sender to "them" | Unit | Function | PASS |
| S-07 | Quote grounding | detects unverifiable quote (`quote_not_found`); accepts verbatim | Unit | Function | PASS |
| S-08 | M-Pesa cross-match | flags LLM-only codes, reclassifies invented confirmation to `CLAIMED_ONLY`, keeps verified match | Unit | Function | PASS |
| S-09 | Verdict computation | downgrades `LIKELY VALID` on weak integrity / no dates; no downgrade for non-likely-valid; caps at `POSSIBLY VALID WITH GAPS` | Unit | Function | PASS |
| S-10 | Element status enforcement | no-evidence downgrade to `UNCLEAR`; special statuses restricted | Unit | Function | PASS |
| S-11 | Evidence pipeline | `analyze` returns analysis + verification with stubbed LLM; rejects empty OCR | Unit (stubbed LLM) | Module | PASS |
| S-12 | M-Pesa screenshot | `find_codes` (letter+digit), classify, multi-entry parse, unsupported layout | Unit | Function | PASS |
| S-13 | Date checks | no dates, conversation + due date, overdue | Unit | Function | PASS |
| S-14 | Case-file pipeline | `analyze_case` full response, masks/merges, deterministic follow-ups, dedupe follow-ups | Unit (stubbed LLM) | Module | PASS |
| R-01 | Firestore rules | owner-only reads, denied client writes, role escalation frozen | Rules test (Jest) | System | **NOT RUN** (no emulator) |

**Acceptance criteria met**: all runnable unit, mocked-persistence, endpoint,
and stubbed-LLM tests pass with exit code 0. The emulator-dependent integration
and rules tests are present but could not be executed here because the Firestore
emulator is not running.

**Coverage gaps (honest)**: no frontend test suite exists; the primary RAG
retrieval pipeline (`pixe.py`/`Retriever.py`/`RulesRetriever.py`) and the
`/api/chat` and `/api/generate-legal-answer` routes have **no** automated tests;
retrieval quality (precision/recall) and LLM answer quality are not
unit-tested.

### 5.4.3 Performance, Latency, and Resource Evaluation

**No formal benchmark or load-test harness exists in the repository.** There is
no `pytest-benchmark`, `locust`, or equivalent configuration, and no scripted
latency measurement suite. The following is what can be established from
committed artefacts only.

**Observable performance evidence**

| Signal | Value | Source |
|---|---|---|
| Backend test suite duration | 15.11 s (99 tests) | executed pytest |
| Embedding run 1 | `statutes` → `statutes_test`, 20 docs | `backend/logs/embed_1792955171.log` |
| Embedding run 2 | `statutes` → `statutes`, 1,331 docs | `backend/logs/embed_1792955193.log` |
| Embedding run 3/4 | `cases` → `case_law`, 59 docs | `backend/logs/embed_1792955834.log`, `embed_1792956207.log` |
| Embedding run 5 | `cases` → `case_law`, 78 docs | `backend/logs/embed_1793465232.log` |
| Hallucination/eval log | 51 entries in `hallucinations.jsonl` | `backend/logs/hallucinations.jsonl` |

The embed logs record only `{source, collection, embedded, ts}` counts; they do
**not** capture per-document or per-batch latency, throughput, or memory. The
`hallucinations.jsonl` file records `{timestamp, query, unsupported_claim}` for
a single Amani Logistics test query, all entries with `"unsupported_claim":
"SECTION 1"` — it is an evaluation side-effect log, not a latency benchmark.

**Documented (unmeasured) performance claims** — `README.md` states: CPU-only
inference 3–5 tokens/s for Mistral-7B, ~500 MB one-time model download, GPU
8 GB/16 GB VRAM guidance. These are the original prototype's claims and are
**not** backed by measurements in this repository.

**Resource environment** — the implementation environment report records an
Intel UHD Graphics 620 (no GPU), CPU-only PyTorch, and native host processes
(no Docker). No services were running at audit time, so live memory/CPU could
not be measured.

**Conclusion**: performance evaluation is a documentation gap. Latency
`latency_ms` fields exist in the history data model and the generate endpoint,
but no benchmark run or aggregated latency report is present.

---

## 5.5 Version Control and Reproducibility

**Repository**

| Item | Value |
|---|---|
| Remote | `git@github.com:Houstonsboy/Retreival-Augmented-Generation-AI.git` (fetch & push) |
| Branch | `master` (additional local branches `backup`, `temp-fix`) |
| Total commits | 56 |
| Tags / releases | None (0 tags) |
| Latest commit | `e96979e` — "feat: Round 2 skeleton — viewport-anchored shimmer + unified Skeleton component" |
| Author | Gift Gichuhi <giftgichuhi4138@gmail.com>, 2026-11-01 20:59:06 +0300 |

**Recent commit log (last 15)**

```
e96979e feat: Round 2 skeleton — viewport-anchored shimmer + unified Skeleton component
9f8a335 feat: production skeleton loading for legal research workspace
4608dcb fix: navigate to dashboard after profile setup without requiring refresh
1ffd1be feat: Phase 2 frontend — history, routing, sidebar, idempotency
8c518dc feat: idempotent generate endpoint — cache hit skips LLM, retry overwrites failed turn
fea5b20 docs: add Firestore security rules, composite indexes, and CHAT_HISTORY guide
b1a6a4c feat: add Flask Blueprint for chat history management endpoints
540f0c1 feat: wire chat history into generate-legal-answer endpoint
989fd4b feat: add history persistence layer (history_store.py + tests)
641643b refactor: extract _groq closure to shared groq_client module
57edd66 chore: ignore neo4j runtime data and binaries
211b511 feat: redesign chat workspace UI with gold/cream theme and markdown rendering
0f36d8b Fix F: end-to-end test Amani/M-Pesa query; fix precision tagging for bare [S#]
e807d85 Fix B: compute constitution bboxes via rapidfuzz word-alignment
f284295 Fix A: replace <object> PDF embed with react-pdf + bbox overlay
```

**`.gitignore` coverage**

Correctly ignored: `.env`, `.env.local`, Firebase service-account JSON
(`*-service-account.json`, `*firebase-adminsdk*.json`, `serviceAccount*.json`),
`.firebase/`, `__pycache__/`, `*.py[cod]`, `*.log`, `venv/`/`env/`, IDE files,
`firestore-tests/node_modules/`, `firestore-tests/package-lock.json`, and Neo4j
runtime data/binaries.

Not ignored (reproducibility risk): the ChromaDB data directories
(`CaseLawDB/`, `StatutesDB/`, `LegalSummariesDB/`, `ChunkDB/`,
`MasterRulesDB/`, `BlackLawDictionaryDB/`) and `.bak` directories are untracked
but would be swept in by a future `git add .`; `backend/logs/*.jsonl` is also
untracked and not covered by the `*.log` rule.

**Dependencies (exact, pinned)**

Backend `backend/requirements.txt` (46 lines, all `==` pinned):

- Core: `python-dotenv==1.2.1`, `sentence-transformers==5.1.2`,
  `huggingface-hub==0.36.0`, `langchain==1.0.7`, `langchain-core==1.0.6`,
  `scikit-learn==1.7.2`, `numpy==2.3.5`, `requests==2.32.5`
- PDF/OCR: `pypdf==6.3.0`, `python-docx==1.2.0`, `PyMuPDF==1.24.14`,
  `pytesseract==0.3.13`, `Pillow==11.1.0`, `ocrmypdf==16.5.0`
- Retrieval: `rank-bm25==0.2.2`
- Models: `torch==2.9.1`, `transformers==4.57.1`
- Vector store: `chromadb==1.3.5`, `opentelemetry-*==1.45.1` (api, sdk, otlp)
- API: `Flask==3.1.2`, `flask-cors==6.0.1`
- Auth/persistence: `firebase-admin==7.7.0`
- LLM: `groq==0.35.0`
- OCR (CPU): `rapidocr==3.10.0`, `onnxruntime==1.23.2`,
  `opencv-python-headless==5.0.0.93`

Frontend `frontend/package.json` (semver ranges, no lockfile committed):

- Dependencies: `firebase ^10.12.0`, `lucide-react ^0.562.0`, `next 16.0.3`,
  `react 19.2.0`, `react-dom 19.2.0`, `react-markdown ^10.1.0`,
  `react-pdf ^7.7.3`, `remark-gfm ^4.0.1`
- DevDependencies: `@tailwindcss/postcss ^4`, `@types/node ^20`,
  `@types/react ^19`, `@types/react-dom ^19`, `eslint ^9`,
  `eslint-config-next 16.0.3`, `tailwindcss ^4`, `typescript ^5`
- Scripts: `dev`, `build`, `start`, `lint` (no `test`)

Firestore rules tests `firestore-tests/package.json`:
`@firebase/rules-unit-testing ^3.0.4`, `firebase ^10.12.0`, `jest ^29.7.0`,
`ts-jest ^29.1.3`, `ts-node ^10.9.2`, `typescript ^5.4.5`, `@types/jest ^29.5.12`.

**README / setup documentation**

`README.md` exists but documents the **original prototype** (Hugging Face API
token via `.env`, `pixe.py` as a CLI tool, `.txt`/`.pdf` documents). It does
**not** describe the current Flask API, Firebase authentication, multi-collection
ChromaDB setup, SCCVision, or the Neo4j ingestion subproject. This is a
reproducibility gap: a new developer cannot run the current system from the
README alone. The setup/execution detail lives instead in the chapter-5 audit
documents (`chapter5_environment_report.md`, `chapter5_implementation_report.md`).

---

## 5.6 AI-Assisted Tool Declaration

AI-assisted coding tools were used during the development of this project, in
particular for: scaffolding the React/Next.js frontend components and the
Flask/Firestore history and SCCVision modules; drafting and iterating on
`firestore.rules`; generating the unit-test suites in `backend/tests/` and
`firestore-tests/`; and drafting this chapter-5 audit documentation.

The specific tools used were:

- **Rafiki Code** (CLI coding assistant) — code editing, test authoring, and
  repository exploration.
- **Claude Code** (CLI coding assistant) — code review, refactoring, and
  iteration on implementation logic.
- **Gemini** — drafting and polishing of documentation.

LLM responses were reviewed, verified against the codebase, and revised by the
author before acceptance. All final implementation decisions, testing, and the
factual values in this report were confirmed by the author against executed
commands and committed source. The project makes no claim of fully automated,
un-reviewed code generation.

---

## 5.7 Findings, Objectives Mapping, and Limitations

### 5.7.1 Security Review (OWASP Top 10 2021 alignment)

| OWASP category | Status | Evidence |
|---|---|---|
| A01 Broken Access Control | **Addressed** | Firebase Admin auth + custom `@require_role("advocate"/"admin")` decorator (`auth.py`, `app.py`); Firestore rules enforce owner-only reads and deny all client writes; `type`/`advocateVerified` frozen after creation to block privilege escalation; PDF serving guarded by `secure_filename` + `commonpath` allowlist (`RES_IPSA_LOQUITUR_DIR`, `SCC_CASES_DIR`) |
| A02 Cryptographic Failures | **Addressed (with notes)** | No hard-coded secrets found; `.env`/`.env.local` gitignored and never committed; Firebase service-account JSON gitignored. Residual risk: service-account JSON files exist locally in `backend/`. |
| A03 Injection | **Addressed** | SQLite access uses parameterized queries (`?` placeholders); no string-built SQL; PDF paths resolved under allowlist roots. |
| A05 Security Misconfiguration | **Findings** | Flask runs `debug=True` (interactive debugger) — must be disabled for deployment; Neo4j binary DB index files are tracked in git despite the ignore rule; ChromaDB data dirs not in `.gitignore`. |
| A06 Vulnerable/Outdated Components | **Not assessed** | Dependencies are pinned in `requirements.txt`, but no automated vulnerability scan (e.g. `pip-audit`, `npm audit`) output is present. |
| A07 Identification & Authentication Failures | **Addressed** | Firebase Auth with token verification in `auth.py`; `firestore.rules` require `request.auth != null` and owner equality. |
| A09 Security Logging & Monitoring | **Gap** | No structured security/audit logging; the only eval log is `hallucinations.jsonl`. |
| A10 SSRF | **Not applicable** | No server-side fetch of attacker-controlled URLs identified. |

### 5.7.2 Objectives mapping (summary)

| Objective / functional area | Tested? | Where |
|---|---|---|
| Chat history persistence, idempotency, retry-overwrite | Yes (mocked) + integration present | `test_history.py` |
| History REST API | Yes (endpoint, auth mocked) | `test_history.py` |
| SCCVision evidence analysis, PII masking, M-Pesa, verdicts, quote grounding | Yes (stubbed LLM) | `test_sccvision_analyst.py` |
| Firestore security rules | Tests present, not executed | `firestore-tests/firestore.test.ts` |
| RAG retrieval / reranking quality | **No** | none |
| `/api/chat`, `/api/generate-legal-answer` LLM pipeline | **No** | none |
| Frontend (React/Next.js) | **No** | no test script |
| Performance / latency / load | **No** | no harness |

### 5.7.3 Honest limitations

1. **Emulator-dependent tests were not run.** The 4 `TestEmulatorIntegration`
   tests and the entire `firestore-tests` Jest suite require the Firestore
   emulator, which was not running. Their correctness in this environment is
   unverified.
2. **No frontend tests.** There is no Jest/Vitest/React Testing Library setup
   and no `test` script in `frontend/package.json`.
3. **No tests for the core RAG pipeline.** Retrieval (dense + BM25 + cross-encoder
   reranking), query classification, and the Groq LLM answer generation are
   untested; LLM calls in the existing suites are always stubbed.
4. **No retrieval-quality benchmark.** There is no precision/recall evaluation,
   no golden answer set, and no ground-truth labelled queries. The
   `hallucinations.jsonl` log captures unsupported-claim flags but is not a
   quantitative benchmark.
5. **No performance/load evaluation.** No latency, throughput, or memory
   measurements exist; the README's performance figures are unverified
   prototype claims.
6. **Uncommitted build artefacts.** ChromaDB vector stores, `.bak` directories,
   and eval logs are untracked and not ignored, so a fresh clone cannot
   reproduce the exact indexed corpus without re-running the ingest scripts.
7. **README is stale** and does not reflect the shipped system, so setup
   reproducibility depends on the chapter-5 audit documents rather than the
   repository's own README.
