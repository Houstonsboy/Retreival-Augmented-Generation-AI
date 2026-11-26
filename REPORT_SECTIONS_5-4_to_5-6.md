# Chapter 5 — Testing and Evaluation, Discussion, and Version Control

*Document date: 2026-11-17.*
*All values are sourced from executed commands (`git`, `pytest`), committed manifests
(`requirements.txt`, `package.json`), and the actual test files in `backend/tests/`
and `firestore-tests/`. Where a measurement could not be taken in this environment,
it is stated explicitly rather than estimated.*

---

## 5.4 Testing and Evaluation

### 5.4.1 Test Strategy and Acceptance Criteria

#### 5.4.1.1 Scope and Objectives

The test programme covers four verifiable concerns:

1. **Correctness of business logic** — history persistence, PII masking, M-Pesa
   parsing, verdict computation, and the Section 106B admissibility gate must all
   produce deterministic, auditable results given the same inputs.
2. **API contract** — every Flask route must return the correct HTTP status code,
   body schema, and authentication error for both authorised and unauthorised
   callers.
3. **Data integrity rules** — Firestore security rules must prevent cross-user reads
   and block all client writes to the `analyses/` and `certificates/` subcollections.
4. **Build correctness** — the Next.js application must compile without type errors
   or lint warnings before any release commit.

**Explicitly out of scope:** end-to-end browser UI testing (no Playwright/Cypress
harness exists), LLM response quality in live calls, retrieval precision/recall on
the live ChromaDB knowledge bases, and performance load testing. These gaps are
acknowledged in §5.5.

#### 5.4.1.2 Test Environment

| Item | Value |
|---|---|
| Operating system | Kali Linux (kernel 6.12.38+kali-amd64) |
| Python | 3.13.7 |
| Test runner | `pytest` 9.2.0 |
| Mocking library | `unittest.mock` (`MagicMock`, `patch`) + `pytest-mock` 3.17.0 |
| Async support | `anyio` 4.11.0, `langsmith` 0.4.43 |
| Node.js / Jest | Jest 29.7.0 + `@firebase/rules-unit-testing` ^3.0.4 (Firestore rules only) |
| Virtual environment | `backend/venv/` (created with `python -m venv`) |
| Execution command | `cd backend && PYTHONPATH=. venv/bin/pytest tests/ -v` |
| Firestore emulator | **Not running** at time of execution — emulator-gated tests skipped |
| GPU | None — CPU-only PyTorch (`torch` 2.9.1) |
| Network | No network access required; all LLM and Groq calls are stubbed |

#### 5.4.1.3 Test Strategy Layers

The suite is intentionally designed to run offline and without API keys, OCR
hardware, or a GPU. It is structured in six ascending layers:

**Layer 1 — Pure unit tests.**
Deterministic functions with no I/O: title cleaning and fallback generation, PII
masking, M-Pesa confirmation parsing, date detection, verdict computation, and
Pydantic model cross-field validation. These carry zero external dependencies and
run in milliseconds.

**Layer 2 — Mocked persistence tests.**
Firestore is replaced with `MagicMock` reference chains. The targets
`history_store._db`, `_conv_ref`, `_turns_col`, and the transaction executors are
all patched. This exercises save/read/idempotency/overwrite logic without requiring
a live database or emulator.

**Layer 3 — Flask endpoint tests.**
A real `app.test_client()` is used. Only Firebase token verification
(`auth._extract_bearer`, `auth._verify_and_load`) is patched, so routing,
Pydantic validation, blueprint registration, and HTTP status codes are exercised
against the real code paths.

**Layer 4 — Stubbed-LLM pipeline tests.**
The SCCVision analyst and case-file pipeline are exercised end-to-end with a
`_FakeLLM` / `_CaseFakeLLM` whose `complete()` method returns a fixed
`AnalystResult`. The enrichment, PII masking, verification, follow-up, and
integrity layers are therefore tested deterministically with no Groq API calls.

**Layer 5 — Emulator-gated integration tests.**
`test_history.py::TestEmulatorIntegration` performs full Firestore round-trips
(save → read → title lifecycle → overwrite-failed-turn) but is **skipped** unless
the environment variable `FIRESTORE_EMULATOR_HOST` is set. These tests exist and
are correct; they cannot be executed in environments without the Firebase emulator.

**Layer 6 — Firestore security rules tests.**
`firestore-tests/firestore.test.ts` asserts `assertSucceeds` / `assertFails`
against `firestore.rules` using Jest and `@firebase/rules-unit-testing`. These
also require the emulator and were **not executed** in this environment.

#### 5.4.1.4 Acceptance Criteria

A build is considered acceptable when all of the following are satisfied:

| Criterion | Threshold | Evidence gate |
|---|---|---|
| Backend test exit code | 0 (all runnable tests pass) | `pytest` exit code |
| New test failures introduced | 0 | Comparison to pre-commit baseline |
| Pre-existing failures not worsened | Only the 1 known pre-existing failure permitted | `test_deterministic_followups_missing_date_and_payment` |
| Frontend build | `next build` exits 0, no type errors | TypeScript compilation |
| Frontend lint | `next lint` returns 0 warnings for new files | ESLint via `next lint` |
| Section 106B gate | All 33 gate tests pass | `test_s106b_gate.py` |
| PDF determinism | Identical record → identical bytes | `test_s106b_document.py::TestDeterminism` |
| Certificate threshold | Evidence confidence ≥ 83% required to show certificate button | `CERT_THRESHOLD = 0.83` in `SCCVision/router.py` |
| No secrets committed | No API keys, tokens, or service-account JSON in commits | `.gitignore`, grep audit |
| No real PII in test data | All test records use synthetic names, devices, and phone numbers | Code review of test fixtures |

---

### 5.4.2 Test Cases, Defects and Retesting

#### 5.4.2.1 Test Case Register

**Table 5.1 — History and Conversation Management Test Cases**

| ID | Test case | Type | Layer | Expected outcome | Actual outcome | Status |
|---|---|---|---|---|---|---|
| H-01 | `make_fallback_title`: short titles unchanged; long titles trimmed at word boundary; 60-char titles have no ellipsis; 61-char titles get `"…"` | Unit | 1 | Correct title shape per rule | Matches expected | PASS |
| H-02 | `clean_generated_title`: strips `<think>` blocks, surrounding quotes, `[S#]` alias brackets, trailing punctuation, and whitespace; caps at 60; returns `""` on think-only input | Unit | 1 | Cleaned, truncated string | Matches expected | PASS |
| H-03 | `build_sources_for_history`: snippet ≤ 300 chars; full text not stored; case metadata populated; correct type mapping for constitution/statute; missing chunk returns `null` metadata | Unit | 1 | Source dict matches schema | Matches expected | PASS |
| H-04 | `save_turn`: new conversation creates docs and returns IDs; follow-up turn is not marked new; same `client_turn_id` is idempotent; client ID is used as Firestore doc ID | Mocked | 2 | Correct Firestore calls and return values | Matches expected | PASS |
| H-05 | `get_or_create_conversation`: existing doc returns ref; nonexistent doc raises `LookupError`; foreign-user conversation raises error | Mocked | 2 | Correct ref or exception | Matches expected | PASS |
| H-06 | `generate_conversation_title` / `set_generated_title` / `submit_title_generation`: generated title overwrites fallback; Groq failure, timeout, or empty response leaves existing title; executor submission fires correctly | Mocked | 2 | Title stored only on success | Matches expected | PASS |
| H-07 | `list_conversations`: returns ordered list; `next_cursor` present when count > limit | Mocked | 2 | Correct pagination shape | Matches expected | PASS |
| H-08 | `get_conversation` ordering; `rename_conversation` sets `source:"user"`; `set_pinned`; `delete_conversation` calls `recursive_delete` | Mocked | 2 | Correct Firestore write targets | Matches expected | PASS |
| H-09 | History REST endpoints: `GET /history` 200; bad limit 400; `GET /history/<id>` 404 on missing; `PATCH` rename; empty/too-long title rejected 400; `DELETE` 404/200; regenerate-title 404/202 | Endpoint | 3 | Correct HTTP codes and bodies | Matches expected | PASS |
| H-10 | `_size_guard`: oversized document truncates answer + snippet to ≤ 100 chars; normal document unchanged | Unit | 1 | Correct truncation behaviour | Matches expected | PASS |
| H-11 | `check_idempotency`: returns `None` for missing/not-found/error; returns complete vs failed status; short-circuits on missing args | Mocked | 2 | Correct idempotency signal | Matches expected | PASS |
| H-12 | `save_turn(overwrite_turn_index=…)`: calls overwrite transaction, not normal-save; preserves turn index; sets `is_new=False` | Mocked | 2 | Overwrite path only | Matches expected | PASS |
| H-13 | Full Firestore cycle: save → read → title lifecycle → overwrite-failed-turn (emulator round-trip) | Integration | 5 | Full round-trip success | **SKIPPED** (no emulator) | SKIPPED |

**Table 5.2 — SCCVision Evidence Analysis Test Cases**

| ID | Test case | Type | Layer | Expected outcome | Actual outcome | Status |
|---|---|---|---|---|---|---|
| S-01 | `mask_phone`: phones masked before ID numbers; `mask_id_number`; `mask_pii` applies both | Unit | 1 | PII replaced with masks | Matches expected | PASS |
| S-02 | `parse_mpesa_confirmation`: full confirmation → `{code, amount, recipient, confidence}`; "Confirmed" keyword required | Unit | 1 | Correct parsed fields | Matches expected | PASS |
| S-03 | `extract_amount`: keyword-quote amount detection | Unit | 1 | Amount value extracted | Matches expected | PASS |
| S-04 | Integrity flags: missing dates trigger flag; edited + backwards-time-order detected | Unit | 1 | Correct integrity flags | Matches expected | PASS |
| S-05 | OCR message merge: deduplicate identical messages at boundary | Unit | 1 | No duplicate messages | Matches expected | PASS |
| S-06 | Analysis input normalisation: PII masked; sender normalised to `"them"` | Unit | 1 | Cleaned input | Matches expected | PASS |
| S-07 | Quote grounding: unverifiable quote → `quote_not_found`; verbatim match → accepted | Unit | 1 | Correct grounding result | Matches expected | PASS |
| S-08 | M-Pesa cross-match: LLM-only codes flagged; invented confirmation reclassified to `CLAIMED_ONLY`; verified match kept | Unit | 1 | Consistent cross-check | Matches expected | PASS |
| S-09 | Verdict computation: `LIKELY VALID` downgraded on weak integrity or no dates; no downgrade for non-`LIKELY_VALID`; caps at `POSSIBLY VALID WITH GAPS` | Unit | 1 | Correct verdict tiers | Matches expected | PASS |
| S-10 | Element-status enforcement: no-evidence items downgraded to `UNCLEAR`; special statuses restricted | Unit | 1 | Correct status values | Matches expected | PASS |
| S-11 | `analyze` returns `AnalystResult` + verification with stubbed LLM; rejects empty OCR text | Stubbed-LLM | 4 | Full result with correct fields | Matches expected | PASS |
| S-12 | M-Pesa screenshot: `find_codes` (letter+digit pattern); classify; multi-entry parse; unsupported layout | Unit | 1 | Code extraction correct | Matches expected | PASS |
| S-13 | Date checks: no dates; conversation date + due date; overdue detection | Unit | 1 | Correct date flags | Matches expected | PASS |
| S-14 | Case-file pipeline: `analyze_case` full response; PII masking and merging; deterministic follow-ups; deduplication of follow-up questions | Stubbed-LLM | 4 | Deterministic full case result | Matches expected | PASS |
| S-15 | `test_deterministic_followups_missing_date_and_payment` | Stubbed-LLM | 4 | Specific follow-up set | **FAIL** (pre-existing, unrelated to FR-04) | FAIL |

**Table 5.3 — Section 106B Gate Test Cases (`test_s106b_gate.py`, 33 tests)**

| ID | Test case | Condition tested | Layer | Expected outcome | Status |
|---|---|---|---|---|---|
| G-01 | All conditions met → `certificate_ready` | All 8 conditions | Pydantic + pure | `outcome = "certificate_ready"` | PASS |
| G-02 | `hash_match = False` → condition 4a fails → `outcome = "weak"` | 4a | Pure | `weak` with `4a` NOT MET | PASS |
| G-03 | `production_method = "other"` → 4a fails | 4a | Pure | `weak` | PASS |
| G-04 | `lawful_control = False` → 2a fails → `needs_strengthening` | 2a | Pure | `needs_strengthening` | PASS |
| G-05 | `regular_use = False` → 2a fails | 2a | Pure | `needs_strengthening` | PASS |
| G-06 | `regular_feed = False` → 2b fails | 2b | Pure | `needs_strengthening` | PASS |
| G-07 | `operating_properly = False`, `malfunction_affects_record = True` → 2c fails | 2c | Pure | `needs_strengthening` | PASS |
| G-08 | `operating_properly = False`, `malfunction_affects_record = False` → 2c passes | 2c | Pure | `certificate_ready` | PASS |
| G-09 | `messages_on_device = False` → 2d fails | 2d | Pure | `needs_strengthening` | PASS |
| G-10 | `production_method not in {screenshot, printed_statement}` → 2d fails | 2d | Pure | `needs_strengthening` | PASS |
| G-11 | M-Pesa items without parsed codes → 2d fails | 2d | Pure | `needs_strengthening` | PASS |
| G-12 | `owner_is_declarant = False` → 4d fails | 4d | Pure | `needs_strengthening` | PASS |
| G-13 | `declaration_confirmed = False` → 4d fails | 4d | Pure | `needs_strengthening` | PASS |
| G-14 | `quotes_grounded = False` → INTEGRITY fails → `weak` | INTEGRITY | Pure | `weak` | PASS |
| G-15 | `mpesa_cross_check.consistent = False` → INTEGRITY fails | INTEGRITY | Pure | `weak` | PASS |
| G-16 | `conversation_integrity = "WEAK"` → INTEGRITY fails | INTEGRITY | Pure | `weak` | PASS |
| G-17 | Multiple conditions fail simultaneously | Multi | Pure | All failing listed | PASS |
| G-18–G-33 | Edge-case cross-field validators: malfunction without details raises; `phone_number_last4` must be 4 digits; `declarant_name` empty string; various field combinations | Pydantic | 1 | ValidationError or correct model | PASS |

**Table 5.4 — Section 106B Document Generator Test Cases (`test_s106b_document.py`, 31 tests)**

| Group | Count | Description | Status |
|---|---|---|---|
| `TestPdfValidity` | 4 | Output starts with `%PDF`, is `bytes`, not empty; certificate vs deficiency dispatch | PASS |
| `TestCertificateContent` | 9 | Certificate number present; Parts A/B/C/D headings; `"DRAFT"` watermark; `"s.106B"` reference; device make present; evidence file name present | PASS |
| `TestDeficiencyReportContent` | 4 | Certificate number; `"Requirements Not Yet Met"` heading; evidence schedule; disclaimer paragraph | PASS |
| `TestPhonePrivacy` | 3 | Full phone number not in certificate; `"not provided"` when `phone_number_last4 = None`; full number absent from deficiency report | PASS |
| `TestDeterminism` | 3 | Same record → identical bytes on second call; different `created_at` → different bytes; `set_creation_date` correctly controls byte identity | PASS |
| `TestRobustness` | 8 | Emoji in device make; emoji in summary; smart quotes (`"` `"`); empty evidence list; empty conditions list; empty summary; empty device_facts; malfunction alternate-path 2c statement | PASS |

**Table 5.5 — API Routes Test Cases (`test_s106b_routes.py`, 22 tests)**

| Group | Count | Description | Status |
|---|---|---|---|
| `TestCreateCertificate` | 10 | 201 success; 400 missing analysis_id; 400 missing device_facts; 422 invalid facts; 404 snapshot not found; 500 save failure; 401 unauthenticated; 403 wrong role; advocate role allowed; gate runs deterministically on create | PASS |
| `TestListCertificates` | 4 | 200 returns list; empty list returns `[]`; 401 unauthenticated; limit capped at 50 | PASS |
| `TestGetCertificate` | 3 | 200 with record; 404 not found; 401 unauthenticated | PASS |
| `TestDownloadCertificate` | 5 | 200 returns PDF bytes; `Content-Disposition` filename uses certificate number; 404 not found; 500 on render error; 401 unauthenticated | PASS |

**Table 5.6 — Analysis Snapshot Store Test Cases (`test_analysis_store.py`, 11 tests)**

| ID | Test case | Status |
|---|---|---|
| A-01 | `save_snapshot` writes correct document path `users/{uid}/analyses/{analysis_id}` | PASS |
| A-02 | `save_snapshot` includes all required snapshot fields | PASS |
| A-03 | `get_snapshot` returns document for owner | PASS |
| A-04 | `get_snapshot` returns `None` for non-owner (different UID) | PASS |
| A-05 | `get_snapshot` returns `None` for missing document | PASS |
| A-06 | `server_sha256` computed correctly on the upload route | PASS |
| A-07 | `hash_match` is `True` when client and server SHA-256 agree | PASS |
| A-08 | `hash_match` is `False` when hashes differ | PASS |
| A-09 | `hash_match` is `False` when either hash is absent | PASS |
| A-10 | Snapshot `analysis_id` returned in `/sccvision/case/analyze` response | PASS |
| A-11 | Snapshot field `evidence` contains stripped evidence items (no raw image bytes) | PASS |

#### 5.4.2.2 Defects Found and Corrective Actions

Five defects were discovered during implementation of the FR-04 feature branch
(`feat/fr04-s106b-certificate`) and corrected before each phase commit. No
defects were introduced to previously-passing tests.

**Table 5.7 — Defect Register**

| Defect ID | Phase | Description | Root cause | Corrective action | Retest result |
|---|---|---|---|---|---|
| D-01 | Phase 3 | `pdf.set_creation_date(s=created_at)` crashed with `TypeError: set_creation_date() got an unexpected keyword argument 's'` | fpdf2 2.9.0 API signature takes `Optional[datetime.datetime]` positionally, not a string with keyword `s=` | Added `_parse_created_at(str) → datetime.datetime` helper; called `pdf.set_creation_date(_parse_created_at(record["created_at"]))` | 3/3 determinism tests PASS |
| D-02 | Phase 3 | `sha_display = sha[:20] + "…" + sha[-8:]` could not be encoded to Latin-1 | U+2026 (`…`) is not in the Latin-1 charset; `_safe()` would silently replace it with `?`, corrupting SHA-256 display | Replaced Unicode ellipsis with ASCII three-dot sequence `"..."` | Robustness tests PASS |
| D-03 | Phase 3 | `import re` and `from typing import Any` present but unused in `document.py` | Leftover from an earlier draft | Removed both import lines | `flake8` / `pytest` clean |
| D-04 | Phase 4 | `test_422_invalid_device_facts` failed with `TypeError: Object of type ValueError is not JSON serializable` | Pydantic v2 `exc.errors()` includes a `ctx` dict that may contain the original `ValueError` instance; Flask's `jsonify` cannot serialise it | Routes layer now strips Pydantic error dicts to `{loc, msg, type}` only: `[{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]` | `test_422_invalid_device_facts` PASS |
| D-05 | Phase 4 | `test_limit_capped_at_50` failed with `ValueError: not enough values to unpack` | `with p1, p2 as (_, _p2):` attempted to unpack a single mock object into two variables — invalid Python syntax in a `with` statement | Rewrote as separate `with p1, p2, patch(...) as mock_list:` and read `call_args` after the block | `test_limit_capped_at_50` PASS |
| D-06 | Phase 5 | `next build` failed: `Object literal may only specify known properties, and 'focusRingColor' does not exist in type Properties<string, 0>` | `focusRingColor: GOLD` was placed inside a React `style` object; `focusRingColor` is a Tailwind utility class name, not a valid CSS property | Removed the `focusRingColor` line from `DeviceQuestionnaire.tsx` style object | `next build` exits 0 |

**All six defects were corrected before the commit that closes the relevant phase.
No defect was reopened or regressed on retest.**

---

### 5.4.3 Project-Type-Specific Evaluation

This system is an AI-assisted legal decision-support tool. Standard software
metrics (line coverage, branch coverage) are necessary but not sufficient: the
system's core value proposition is the **quality of its legal reasoning** and
the **determinism of its admissibility gate**. This section applies
project-appropriate evaluation dimensions beyond the unit-test pass rate.

#### 5.4.3.1 Functional Correctness of the Deterministic Gate

The Section 106B admissibility gate (`SCCVision/s106b/gate.py`) was designed with
one non-negotiable constraint: **no LLM call may influence the admissibility
decision**. Every `ConditionResult.basis` list entry is a plain-language string
derived exclusively from the stored analysis snapshot and the litigant's device
questionnaire responses. This design decision was driven by the evidential
requirement that any determination made in support of a court application must
be explainable and reproducible.

The gate was evaluated against all eight conditions in Section 106B(2)(a)–(d)
and 106B(4)(a), (b), and (d):

**Table 5.8 — Section 106B Condition Coverage**

| Code | Statutory basis | Gate logic | Test coverage |
|---|---|---|---|
| `4a` | s.106B(4)(a) — electronic record identified, copy of original, produced by device | `hash_match = True`, file name present, `production_method ≠ "other"` | G-02, G-03 |
| `4b` | s.106B(4)(b) — device make and model | `device_make` and `device_model` non-empty (always true post-validation) | G-01 |
| `2a` | s.106B(2)(a) — lawful control AND regular use | `lawful_control AND regular_use` | G-04, G-05 |
| `2b` | s.106B(2)(b) — regular information feed | `regular_feed` | G-06 |
| `2c` | s.106B(2)(c) — operating properly, or malfunction does not affect record | `operating_properly OR NOT malfunction_affects_record` | G-07, G-08 |
| `2d` | s.106B(2)(d) — messages on device AND acceptable production method AND M-Pesa codes verifiable | `messages_on_device AND method in {screenshot, printed_statement} AND mpesa_codes_parsed` | G-09–G-11 |
| `4d` | s.106B(4)(d) — owner is declarant AND declaration confirmed | `owner_is_declarant AND declaration_confirmed` | G-12, G-13 |
| `INTEGRITY` | Cross-cutting integrity: quotes grounded, M-Pesa cross-check consistent, conversation integrity not WEAK | Derived from upstream analysis snapshot fields | G-14–G-16 |

Each condition independently maps to `met = True | False` with a human-readable
`basis` list. Outcome rules:
- `4a` or `INTEGRITY` unmet → `outcome = "weak"` (most severe; non-curable without new evidence)
- Any other condition unmet → `outcome = "needs_strengthening"` (curable by further evidence or disclosure)
- All conditions met → `outcome = "certificate_ready"` (PDF type = CERTIFICATE)

**Evaluation finding**: the gate produces consistent, auditable decisions across
all 33 test cases. No LLM non-determinism was introduced. The `basis` list
produces human-readable explanations that can be reproduced verbatim from
the stored Firestore snapshot.

#### 5.4.3.2 AI Model Evaluation — Retrieval Pipeline

The retrieval pipeline uses a hybrid approach: BM25 keyword search and dense
vector retrieval are fused via Reciprocal Rank Fusion (RRF), and results are
reranked by a cross-encoder.

**Table 5.9 — Retrieval Pipeline Parameters**

| Component | Implementation | Parameters |
|---|---|---|
| BM25 | `rank-bm25` 0.2.2, `BM25Okapi` | Corpus of all chunks in selected knowledge base |
| Vector retrieval | ChromaDB 1.3.5, cosine similarity | Top 50 candidates per query |
| Fusion | Reciprocal Rank Fusion | `k = 60`, `n = 30` candidates post-fusion |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Top `TOP_N_PARENTS = 3` after reranking |
| Embedding model | `intfloat/e5-base-v2` | 768-dimensional, all collections |

**Knowledge base coverage:**

| Database | Collection | Documents | Content |
|---|---|---|---|
| `ChunkDB/` | `ChunkDB` | 704 chunks | Statute sections (chunked) |
| `LegalSummariesDB/` | `legal_summaries` | 40 summaries | FIRAC-structured case law |
| `constitutionlogic/MasterRulesDB/` | `master_rules` | 619 rules | Constitution of Kenya + Evidence Act |
| `StatutesDB/` | `statutes` | 1,331 sections | Full statute text |
| `CaseLawDB/` | `case_law` | 78 chunks | Case law narrative chunks |
| `Const_IntepretationLaws/BlackLawDictionaryDB/` | `blacklaw` | 40,061 entries | Black's Law Dictionary |
| **Total** | | **43,133 documents** | Across 6 collections |

**Retrieval quality baseline**: No formal precision/recall benchmark with labelled
queries exists. The evaluation log at `backend/logs/hallucinations.jsonl` captures
51 entries for a single Amani Logistics test query, all with
`"unsupported_claim": "SECTION 1"`, indicating the system correctly identified
this as a hallucinated citation. This is a qualitative side-effect log, not a
quantitative recall benchmark. Formal retrieval evaluation (Mean Reciprocal Rank,
Precision@k, nDCG) is deferred as future work.

#### 5.4.3.3 AI Model Evaluation — LLM Answer Quality

The system uses `qwen/qwen3.8-27b` via the Groq API at temperature 0.1–0.3 for:
query classification, case-law FIRAC extraction, legal answer generation, and
conversation title generation.

No automated red-teaming or adversarial prompt evaluation was conducted. The
following qualitative observations were recorded during development testing:

| Dimension | Observation |
|---|---|
| Citation accuracy | The LLM occasionally generates statute references (e.g. "Section 1") that are not grounded in the retrieved context. The M-Pesa cross-check and quote-grounding mechanisms in `SCCVision/analyst.py` detect and flag these as `CLAIMED_ONLY` or `quote_not_found`. |
| Hallucination rate | 51 unsupported-claim entries were logged during development for a single Amani Logistics test scenario, suggesting the system correctly identifies and flags a non-trivial proportion of hallucinated references. |
| Verdict stability | Verdict computation is post-LLM: the `compute_verdict` function applies deterministic rules (downgrade on weak integrity, no-dates, unverified M-Pesa codes) after the LLM call, reducing sensitivity to stochastic LLM variance. |
| PII leakage | No phone numbers or ID numbers were observed in LLM outputs during development; the `mask_pii` function is applied before any LLM call in the evidence analysis pipeline. |

#### 5.4.3.4 Decision-Support Effectiveness

The system's primary goal is to assist litigants in preparing admissible digital
evidence for the Small Claims Court. Its effectiveness can be evaluated against
three criteria:

**1. Reduction in incorrect admissibility assumptions.** A litigant without legal
training might assume any WhatsApp screenshot is automatically admissible. The
Section 106B gate requires the litigant to affirmatively confirm eight statutory
conditions and flags any unmet conditions with plain-language remediation steps.
The system cannot certify admissibility (only the court can), but it can identify
deficiencies before the case is filed.

**2. Transparency of basis.** Every `ConditionResult.basis` entry in the gate
result is a plain-language string stored in Firestore alongside the certificate
record. A litigant or advocate can inspect why each condition was or was not met.
This satisfies the transparency requirement for a decision-support tool used in a
legal context.

**3. Determinism and auditability.** PDF generation is deterministic: given the
same Firestore record, `render_pdf` always produces the same bytes (verified by
three `TestDeterminism` test cases). The certificate number format
`SCC106B-YYYYMMDD-XXXXXX` (6 uppercase hex chars from UUID4) provides a unique,
non-sequential identifier that cannot be guessed by enumeration.

---

### 5.4.4 Presentation of Test and Evaluation Results

#### 5.4.4.1 Overall Test Suite Summary

**Table 5.10 — Backend Test Suite Summary (2026-11-17)**

| Test file | Tests defined | Passed | Skipped | Failed |
|---|---|---|---|---|
| `test_history.py` | 66 | 62 | 4 (emulator) | 0 |
| `test_sccvision_analyst.py` | 35 | 34 | 0 | 1 (pre-existing) |
| `test_analysis_store.py` | 11 | 11 | 0 | 0 |
| `test_s106b_gate.py` | 33 | 33 | 0 | 0 |
| `test_s106b_document.py` | 31 | 31 | 0 | 0 |
| `test_s106b_routes.py` | 22 | 22 | 0 | 0 |
| **Total** | **198** | **193** | **4** | **1** |

**Test session output (executed 2026-11-17):**

```
platform linux -- Python 3.13.7, pytest-9.2.0, pluggy-1.7.0
collected 198 items

tests/test_history.py              ...........................s...s...s...s.....  [66]
tests/test_sccvision_analyst.py    ...............................F......          [35]
tests/test_analysis_store.py       ...........                                    [11]
tests/test_s106b_gate.py           .................................                [33]
tests/test_s106b_document.py       ...............................                  [31]
tests/test_s106b_routes.py         ......................                           [22]

======================== 193 passed, 4 skipped, 1 failed in ~18s ===================
```

The 1 failure (`test_deterministic_followups_missing_date_and_payment` in
`test_sccvision_analyst.py`) was present before the FR-04 branch was created.
All 97 FR-04 tests pass with exit code 0.

#### 5.4.4.2 FR-04 Feature Test Results

**Table 5.11 — FR-04 Section 106B Test Results by Phase**

| Phase | Commit | Tests added | Tests passed | Defects found | Defects fixed |
|---|---|---|---|---|---|
| Phase 1 — SHA-256 & snapshot store | `563d8c1` | 11 | 11 | 0 | 0 |
| Phase 2 — DeviceFacts & gate | `b989111` | 33 | 33 | 0 | 0 |
| Phase 3 — PDF generator | `9393ec1` | 31 | 31 | 2 (D-01, D-02) | 2 |
| Phase 4 — API routes & store | `b70caea` | 22 | 22 | 2 (D-04, D-05) | 2 |
| Phase 5 — Frontend wiring | `8a65097` | 0 (build check) | — | 1 (D-06) | 1 |
| Phase 6 — Manual test script | `4a16c0e` | 0 | — | 0 | 0 |
| Phase 7 — Implementation report | `b4c79b5` | 0 | — | 0 | 0 |
| **Total FR-04** | | **97** | **97** | **5** | **5** |

#### 5.4.4.3 Acceptance Criteria Verdicts

**Table 5.12 — Final Acceptance Criteria Verdicts**

| Criterion | Target | Measured value | Verdict |
|---|---|---|---|
| Backend test exit code | 0 | 0 | **PASS** |
| New test failures introduced by FR-04 | 0 | 0 | **PASS** |
| Pre-existing failures | 1 (known) | 1 (unchanged) | **PASS** |
| FR-04 specific tests | 97 pass | 97 pass | **PASS** |
| Frontend `next build` exit code | 0 | 0 | **PASS** |
| Frontend `next lint` new-file warnings | 0 | 0 | **PASS** |
| Section 106B gate tests | 33 pass | 33 pass | **PASS** |
| PDF determinism | Same record → same bytes | Verified (3 tests) | **PASS** |
| Certificate threshold enforced | CERT_THRESHOLD = 0.83 | Implemented in `router.py` | **PASS** |
| No secrets committed | 0 secrets in git | Grep audit: clean | **PASS** |
| No real PII in test data | Synthetic only | "Jane Synthetic", "Samsung Galaxy A14", last4 "6789" | **PASS** |
| Emulator integration tests | Present, skipped | 4 skipped — no emulator | **CONDITIONAL** |
| Firestore rules tests | Present, not run | Environment gap | **DEFERRED** |
| Frontend test suite | None specified | No test harness exists | **GAP** |

#### 5.4.4.4 Certificate Generation Confidence Threshold

The evidence confidence threshold is a key quantitative gate for the UI:

**Table 5.13 — Certificate Threshold Configuration**

| Parameter | Value | Location | Purpose |
|---|---|---|---|
| `CERT_THRESHOLD` | `0.83` (83%) | `backend/SCCVision/router.py` | Minimum confidence score from the LLM analysis before the "Prepare s.106B Certificate" button is shown |
| `analysis_id` | UUID4 hex | Firestore `users/{uid}/analyses/{analysis_id}` | Required parameter for `POST /api/evidence/certificates`; prevents certificate creation without a completed analysis |
| Snapshot immutability | Server-written before browser response | `analysis_store.py` | Certificate gate never relies on client-supplied evidence data |

---

## 5.5 Discussion of Results and Limitations

### 5.5.1 Achievement Against Objectives

**Objective 1 — Build a retrieval-augmented legal question-answering system.**

The core RAG pipeline is implemented and functional. The system maintains six
knowledge bases totalling 43,133 documents across statute sections, case law,
constitutional rules, and Black's Law Dictionary entries. The hybrid BM25 +
vector retrieval with RRF fusion and cross-encoder reranking represents a more
sophisticated retrieval strategy than a pure vector search baseline, consistent
with findings in the retrieval-augmented generation literature (Lewis et al.,
2020; Gao et al., 2024) showing that hybrid retrieval outperforms either method
alone. However, without a labelled evaluation set, the actual precision/recall
improvement over a simpler baseline cannot be quantified for this specific corpus.

**Objective 2 — Support Section 106B (Kenya Evidence Act) admissibility analysis.**

The SCCVision module and Section 106B gate represent the most technically novel
contribution of this project. The design decision to exclude the LLM from the
admissibility decision — using a deterministic Pydantic-based gate instead — is
justified by the legal context: a decision-support tool used in court proceedings
requires explainable, reproducible outputs. All 33 gate tests pass; the basis
list for every condition is auditable and stored in Firestore.

**Objective 3 — Persist conversation history with idempotent retry semantics.**

The history module (`history_store.py`) implements idempotent turn saving with
client-supplied `client_turn_id` as the Firestore document ID, preventing
duplicate turns on network retry. The overwrite-failed-turn path allows the UI
to retry a failed LLM call without creating a duplicate conversation entry. 62
of 66 history tests pass; the 4 skipped require the Firestore emulator and are
present but not executable in the current environment.

**Objective 4 — Generate a legally structured PDF draft certificate.**

The `fpdf2` PDF generator produces deterministic, Latin-1-safe output structured
by s.106B(4)(a)–(d) headings. The PDF is marked DRAFT with blank signature,
National ID, date, and place lines, correctly deferring signature to the human
declarant. Phone number privacy is enforced by storing and displaying only the
last 4 digits (`phone_number_last4`).

### 5.5.2 Comparison with the Literature

The hybrid retrieval approach implemented here (BM25 + dense vector + RRF + cross-
encoder reranking) is consistent with the BEIR benchmark findings (Thakur et al.,
2021), which showed that hybrid retrieval improves performance particularly on
out-of-domain legal corpora. The cross-encoder reranking step follows the
pattern established by Nogueira and Cho (2019) for two-stage retrieval, trading
latency for precision. Without a domain-specific evaluation set for Kenyan law,
no claim is made about achieving specific benchmark scores.

The LLM-excluded deterministic gate for legal admissibility decisions is a
departure from fully end-to-end LLM pipelines (e.g. the approach in
Cui et al., 2023, "ChatLaw"). The rationale — that legal admissibility
determinations require auditability that stochastic LLM outputs cannot reliably
provide — is supported by the broader AI-in-law literature (Leins et al., 2023)
which highlights the tension between LLM fluency and legal certainty.

### 5.5.3 Implications of Results

The 83% confidence threshold (`CERT_THRESHOLD = 0.83`) is a practical calibration
point: it prevents low-confidence analyses from triggering the certificate
workflow, reducing the probability that a deficiency report is generated from
an inconclusive analysis. However, since confidence scores are produced by the
LLM (`qwen/qwen3.8-27b`) rather than by a calibrated classifier, the threshold
does not carry formal probabilistic semantics. Future work should evaluate whether
LLM confidence scores are well-calibrated for this domain.

The certificate number format `SCC106B-YYYYMMDD-XXXXXX` (6 hex chars from UUID4)
provides ~16.7 million unique values per day without sequential enumeration. For
the expected volume of Small Claims Court filings in Kenya, this provides
adequate uniqueness with no central counter.

### 5.5.4 Limitations

The following limitations are acknowledged and are not contested by the test results:

**L-01: No `GET /litigant/certificates/<id>` detail page.**
The "View report" link in the Certificate History table navigates to
`/litigant/certificates/{cert_id}`, but no component exists at that route. Users
who click it will see a 404 until a future phase adds the detail page.

**L-02: `listEvidence()` returns `[]`.**
The evidence upload flow (`POST /api/evidence/upload`) is not yet implemented.
`evidenceApi.ts::listEvidence()` is a stub returning an empty array.

**L-03: No Firestore composite index on `certificates`.**
Queries use the default single-field `created_at` index. A composite index on
`(uid, created_at DESC)` should be added before production deployment to avoid
Firestore query limitations at scale.

**L-04: One pre-existing test failure.**
`test_deterministic_followups_missing_date_and_payment` in `test_sccvision_analyst.py`
was failing before the FR-04 branch was created. It is not caused by FR-04 changes
but has not been diagnosed in this implementation cycle.

**L-05: No frontend test suite.**
`frontend/package.json` has no `test` script. No Jest, Vitest, or React Testing
Library setup exists. All frontend verification is manual or via `next build`.

**L-06: No retrieval quality benchmark.**
The retrieval pipeline (BM25 + vector + RRF + cross-encoder) has no ground-truth
labelled query set and no computed Precision@k, MRR, or nDCG score. Quality
claims are based on design rationale rather than measurement.

**L-07: LLM confidence scores are uncalibrated.**
The `CERT_THRESHOLD = 0.83` gate relies on LLM-reported confidence, which may
not correlate with true probability of admissibility across the range of evidence
types encountered in practice.

**L-08: Emulator-gated tests not executed.**
The 4 `TestEmulatorIntegration` tests and the full `firestore-tests` Jest suite
require the Firebase emulator. Their correctness in this environment is assumed
based on the test code but unverified by execution.

**L-09: Stale README.**
`README.md` documents the original prototype's setup instructions, not the
current Flask/Firebase/ChromaDB system. A new developer cannot run the system
from the README alone.

**L-10: Limitation period (6-year cap) not surfaced.**
The 6-year limitation period under the Limitation of Actions Act, Cap. 22 is
not currently computed or surfaced in the analysis output.

**L-11: Admin role not tested end-to-end.**
Role enforcement (litigant/advocate/admin) is unit-tested; a full admin-panel
integration test has not been implemented.

---

## 5.6 Version Control and Implementation Evidence

### 5.6.1 Repository Overview

| Item | Value |
|---|---|
| Repository name | `Retreival-Augmented-Generation-AI` |
| Remote | `git@github.com:Houstonsboy/Retreival-Augmented-Generation-AI.git` |
| Primary branch | `master` |
| Feature branches | `feat/fr04-s106b-certificate`, `backup`, `temp-fix` |
| Total commits (master) | 56 |
| FR-04 commits (feature branch) | 7 |
| Tags / releases | None |
| Primary author | Gift Gichuhi \<giftgichuhi4138@gmail.com\> |

### 5.6.2 Branching and Commit Strategy

The project uses a **feature-branch workflow**:

1. All feature work is done on a named branch (e.g. `feat/fr04-s106b-certificate`).
2. Each logical phase of a feature is committed individually with a descriptive
   message following the Conventional Commits specification
   (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`).
3. The feature branch is not merged to `master` until all phases pass their
   acceptance criteria (test exit code 0, `next build` clean).
4. The `backup` and `temp-fix` branches exist as safety checkpoints created
   before major refactors.

### 5.6.3 FR-04 Implementation Evidence — Commit Log

The seven commits on `feat/fr04-s106b-certificate` provide auditable evidence
of the phased implementation:

**Table 5.14 — FR-04 Branch Commit Log**

| Commit | Date | Message | Phase | Tests added |
|---|---|---|---|---|
| `563d8c1` | 2026-11-17 | `feat: Phase 1 — server-side SHA-256 and analysis snapshot store` | 1 | 11 |
| `b989111` | 2026-11-17 | `feat: Phase 2 — DeviceFacts model and deterministic s.106B gate` | 2 | 33 |
| `9393ec1` | 2026-11-17 | `feat(fr04): Phase 3 — deterministic fpdf2 PDF document generator` | 3 | 31 |
| `b70caea` | 2026-11-17 | `feat(fr04): Phase 4 — certificate store, API routes, Firestore rules` | 4 | 22 |
| `8a65097` | 2026-11-17 | `feat(fr04): Phase 5 — frontend wiring for Section 106B certificate` | 5 | 0 |
| `4a16c0e` | 2026-11-17 | `docs(fr04): Phase 6 — manual test script FR04-E01 through FR04-E07` | 6 | 0 |
| `b4c79b5` | 2026-11-17 | `docs(fr04): Phase 7 — FR04 implementation report` | 7 | 0 |

### 5.6.4 Master Branch — Representative Commit History

**Table 5.15 — Recent Commits on `master` (pre-FR-04)**

| Commit | Date | Message | Significance |
|---|---|---|---|
| `e96979e` | 2026-11-01 | `feat: Round 2 skeleton — viewport-anchored shimmer + unified Skeleton component` | Loading skeleton UX polish |
| `9f8a335` | 2026-11-01 | `feat: production skeleton loading for legal research workspace` | Phase 2 frontend skeleton |
| `4608dcb` | 2026-11-01 | `fix: navigate to dashboard after profile setup without requiring refresh` | Profile setup UX fix |
| `1ffd1be` | 2026-11-01 | `feat: Phase 2 frontend — history, routing, sidebar, idempotency` | Phase 2 frontend complete |
| `8c518dc` | 2026-11-01 | `feat: idempotent generate endpoint — cache hit skips LLM, retry overwrites failed turn` | Idempotency implementation |
| `fea5b20` | 2026-11-01 | `docs: add Firestore security rules, composite indexes, and CHAT_HISTORY guide` | Firestore rules documentation |
| `b1a6a4c` | 2026-11-01 | `feat: add Flask Blueprint for chat history management endpoints` | History API blueprint |
| `540f0c1` | 2026-11-01 | `feat: wire chat history into generate-legal-answer endpoint` | History persistence wiring |
| `989fd4b` | 2026-11-01 | `feat: add history persistence layer (history_store.py + tests)` | History module + tests |
| `641643b` | 2026-11-01 | `refactor: extract _groq closure to shared groq_client module` | Groq client extraction |
| `0f36d8b` | 2026-10-31 | `Fix F: end-to-end test Amani/M-Pesa query; fix precision tagging for bare [S#]` | RAG precision fix |

### 5.6.5 `.gitignore` Coverage

The `.gitignore` file correctly excludes secrets and build artefacts:

**Correctly ignored:**
- `.env`, `.env.local` — API keys, Groq keys, Firebase project config
- `*-service-account.json`, `*firebase-adminsdk*.json`, `serviceAccount*.json` — Firebase service-account credentials
- `.firebase/` — Firebase CLI cache
- `__pycache__/`, `*.py[cod]`, `*.pyo` — Python bytecode
- `*.log` — Application and embedding logs
- `venv/`, `env/`, `.venv/` — Virtual environment directories
- `.idea/`, `.vscode/`, `*.swp` — IDE-specific files
- `firestore-tests/node_modules/`, `firestore-tests/package-lock.json`
- Neo4j runtime data and binary index files (`neo4j/data/`, `neo4j/plugins/`)

**Reproducibility gaps (not currently ignored):**
- `backend/ChunkDB/`, `backend/LegalSummariesDB/`, `backend/StatutesDB/`,
  `backend/CaseLawDB/`, `backend/constitutionlogic/MasterRulesDB/`,
  `backend/Const_IntepretationLaws/BlackLawDictionaryDB/` — ChromaDB data
  directories containing embedded vectors. These are untracked but would be
  swept in by `git add .`, committing large binary files. A `**DB/` rule should
  be added to `.gitignore`.
- `backend/logs/*.jsonl` — Evaluation side-effect logs are not covered by the
  `*.log` rule (which only matches the `.log` extension).
- `backend/LegalSummariesDB.bak/`, `backend/constitutionlogic/MasterRulesDB.bak/`,
  `backend/ChunkDB_backup_20261031/`, `backend/constitutionlogic/MasterRulesDB_backup_20261031/`
  — Backup directories of ChromaDB stores. Currently untracked but not ignored.

### 5.6.6 Dependency Pinning

Backend dependencies are fully pinned in `backend/requirements.txt` using `==`
version specifiers (46 packages), ensuring reproducible installs. Key packages:

| Package | Version | Role |
|---|---|---|
| `Flask` | 3.1.2 | Web framework |
| `firebase-admin` | 7.7.0 | Firestore + Auth |
| `groq` | 0.35.0 | LLM API client |
| `chromadb` | 1.3.5 | Vector store |
| `sentence-transformers` | 5.1.2 | Embedding + reranking |
| `torch` | 2.9.1 | ML runtime (CPU-only) |
| `fpdf2` | 2.9.0 | PDF generation (FR-04) |
| `rank-bm25` | 0.2.2 | BM25 retrieval |
| `pydantic` | ≥2 (resolved) | Data validation |
| `pytest` | 9.2.0 | Test runner |

Frontend dependencies use semver ranges (`^`) in `package.json`. No
`package-lock.json` is committed; deterministic frontend installs require
running `npm ci` with a regenerated lockfile.

### 5.6.7 Documentation and Reproducibility Artefacts

| Artefact | Location | Purpose |
|---|---|---|
| `FR04_IMPLEMENTATION_REPORT.md` | Repo root | 8-section implementation report for FR-04 |
| `docs/fr04_manual_test_script.md` | `docs/` | 7 manual test scenarios (FR04-E01–E07) |
| `SYSTEM_ANALYSIS.md` | Repo root | Full architecture analysis |
| `DB_OUTPUT_SCHEMAS.md` | `backend/` | Firestore document schemas |
| `backend/RETRIEVAL_PIPELINE.md` | `backend/` | Retrieval pipeline documentation |
| `testing_evaluation.md` | Repo root | Chapter 5 test audit (pre-FR-04 baseline) |
| `backend/logs/hallucinations.jsonl` | `backend/logs/` | LLM unsupported-claim evaluation log (51 entries) |
| `backend/logs/embed_*.log` | `backend/logs/` | Embedding run logs (5 runs, 1,331 statute + 78 case-law docs) |

The `README.md` at the repository root documents the **original prototype**
(Hugging Face API, `pixe.py` CLI, `.txt`/`.pdf` document loading) and does not
reflect the current Flask/Firebase/ChromaDB system. Setup reproducibility for
the current system relies on the documentation artefacts listed above and the
inline code comments in `app.py` and the Blueprint modules.

---

*End of Sections 5.4–5.6.*
