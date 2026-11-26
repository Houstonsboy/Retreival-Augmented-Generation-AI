# FR-04 Implementation Report — Section 106B Certificate & Deficiency Report

**Branch:** `feat/fr04-s106b-certificate`  
**Date completed:** 2026-11-17  
**Author:** Claude Code (Sonnet 4.6)

---

## 1. Feature Summary

FR-04 adds end-to-end generation of Section 106B (Kenya Evidence Act, Cap. 80) certificate drafts and deficiency reports. After a user completes a contract-evidence analysis in the SCCVision panel, they can fill in a device questionnaire and receive either:

- **CERTIFICATE** — a printable DRAFT PDF with four Parts (A–D) structured by s.106B(4)(a)-(d), ready for hand-signing.
- **DEFICIENCY_REPORT** — a plain-language report listing the unmet conditions and remediation steps.

The admissibility decision is **entirely deterministic** — the LLM is excluded from the gate. All decisions are auditable from the stored analysis snapshot and litigant-supplied device facts.

---

## 2. Architecture Overview

```
Client (browser)
  │
  ├─ POST /api/evidence/certificates   ← DeviceQuestionnaire.tsx
  ├─ GET  /api/evidence/certificates   ← certificates/page.tsx
  ├─ GET  /api/evidence/certificates/<id>
  └─ GET  /api/evidence/certificates/<id>/download

Backend (Flask)
  │
  ├─ SCCVision/s106b/routes.py         ← Blueprint s106b_bp
  ├─ SCCVision/s106b/gate.py           ← Deterministic gate (no LLM)
  ├─ SCCVision/s106b/store.py          ← Firestore persistence
  ├─ SCCVision/s106b/document.py       ← fpdf2 PDF generator
  ├─ SCCVision/s106b/models.py         ← Pydantic: DeviceFacts, GateResult
  └─ SCCVision/analysis_store.py       ← Analysis snapshot (Phase 1 source of truth)

Firestore
  └─ users/{uid}/
        ├─ analyses/{analysis_id}      ← Stripped analysis snapshot
        └─ certificates/{cert_id}      ← Certificate record (no PDF stored)
```

---

## 3. Phase-by-Phase Summary

### Phase 1 — Server-side SHA-256 and analysis snapshot store
**Commit:** `563d8c1`

- `/sccvision/extract`: computes `server_sha256 = hashlib.sha256(raw_bytes).hexdigest()` server-side.
- `/sccvision/evidence/process`: stores `sha256` (client), `server_sha256`, and `hash_match = (client == server and both present)`.
- `/sccvision/case/analyze`: saves a stripped snapshot to `users/{uid}/analyses/{analysis_id}` after each run. The snapshot is the source of truth for certificate generation — never browser-supplied data.
- **Tests:** 11 tests in `test_analysis_store.py`.

### Phase 2 — DeviceFacts model and deterministic s.106B gate
**Commit:** `b989111`

- `DeviceFacts` Pydantic model with cross-field validators (malfunction cross-checks, last4 digits).
- `evaluate(snapshot, facts) → GateResult` checks 8 conditions derived from s.106B(2)(a)-(d) and s.106B(4)(a)-(d):
  - `4a`: evidence identified, hashes match, production method is not "other"
  - `4b`: device make/model present (always met when validation passes)
  - `2a`: lawful_control AND regular_use
  - `2b`: regular_feed
  - `2c`: operating_properly OR malfunction does not affect record
  - `2d`: messages on this device AND method in {screenshot, printed_statement}; M-Pesa items have parsed codes
  - `4d`: owner_is_declarant AND declaration_confirmed
  - `INTEGRITY`: quotes_grounded, mpesa_cross_check.consistent, conversation_integrity ≠ WEAK
- Outcome: `certificate_ready` (all met) → `needs_strengthening` (questionnaire-only unmet) → `weak` (4a or INTEGRITY unmet).
- **Tests:** 33 tests in `test_s106b_gate.py`.

### Phase 3 — Deterministic fpdf2 PDF generator
**Commit:** `9393ec1`

- `render_pdf(record) → bytes` with two layouts (CERTIFICATE / DEFICIENCY_REPORT).
- Deterministic: uses `record["created_at"]` for `pdf.set_creation_date()` so identical records produce identical bytes.
- Latin-1 safe: all text sanitised via `_safe()` — emoji from OCR text becomes `?`, never crashes.
- Phone privacy: last 4 digits displayed as `******XXXX`; full number never written to PDF.
- **Tests:** 31 tests in `test_s106b_document.py` (PDF validity, headings, cert number, privacy, determinism, robustness).

### Phase 4 — Certificate store, API routes, Firestore rules
**Commit:** `b70caea`

- `store.py`: saves records to `users/{uid}/certificates/{cert_id}`; generates `SCC106B-YYYYMMDD-XXXXXX` certificate number.
- `routes.py`: Blueprint registered at `/api/evidence/certificates`; all routes guarded by `@require_role("litigant", "advocate", "admin")`.
- `firestore.rules`: added read-only client rules for `analyses` and `certificates` subcollections.
- Pydantic `ValidationError.errors()` strips non-JSON-serializable `ctx.error` before jsonifying.
- **Tests:** 22 tests in `test_s106b_routes.py`.

### Phase 5 — Frontend wiring
**Commit:** `8a65097`

- `evidenceTypes.ts`: `DeviceFacts`, `ProductionMethod`, updated `CertificateRecord` (snake_case), `CertificateDetail`.
- `evidenceApi.ts`: real `listCertificates()`, `getCertificate()`, `createCertificate()`, `downloadCertificate()`.
- `DeviceQuestionnaire.tsx`: modal form with conditional malfunction section, declaration checkbox.
- `SCCVisionTest.tsx`: `analysis_id` added to `AnalystResponse`; "Prepare s.106B Certificate" button shown after analysis; success banner links to Certificate History.
- `certificates/page.tsx`: enabled Download button (triggers PDF download); updated to snake_case field names.
- `analysis/page.tsx`: removed `setTimeout` simulation stub.

### Phase 6 — Manual test script
**Commit:** `4a16c0e`

- `docs/fr04_manual_test_script.md`: 7 scenarios (FR04-E01 through FR04-E07).

---

## 4. Key Design Decisions

| Decision | Rationale |
|---|---|
| LLM excluded from s.106B gate | Courts require explainable, deterministic admissibility decisions. Every `ConditionResult.basis[]` is auditable. |
| Analysis snapshot as source of truth | The snapshot is written server-side before returning to the browser. Certificate generation never relies on client-supplied evidence data. |
| PDF regenerated on demand | Avoids storing binary blobs in Firestore (1 MB limit). Same record always produces the same bytes (deterministic via `set_creation_date`). |
| `hash_match` from server-side SHA-256 | Proves the file was not modified after upload. Gate condition 4a fails (outcome: weak) if `hash_match = False`. |
| Signature left blank | `s.106B(4)(d)` requires a human signatory. The system prints a DRAFT with blank lines for hand-signing. |
| Certificate number format | `SCC106B-YYYYMMDD-XXXXXX` (6 uppercase hex chars from UUID4) gives uniqueness without sequential enumeration. |

---

## 5. Files Created or Modified

### New files
| File | Purpose |
|---|---|
| `backend/SCCVision/analysis_store.py` | Analysis snapshot persistence |
| `backend/SCCVision/s106b/__init__.py` | GATE_VERSION constant |
| `backend/SCCVision/s106b/models.py` | DeviceFacts, ConditionResult, GateResult |
| `backend/SCCVision/s106b/gate.py` | Deterministic compliance gate |
| `backend/SCCVision/s106b/document.py` | fpdf2 PDF generator |
| `backend/SCCVision/s106b/store.py` | Certificate Firestore store |
| `backend/SCCVision/s106b/routes.py` | Flask blueprint |
| `backend/tests/test_analysis_store.py` | 11 tests |
| `backend/tests/test_s106b_gate.py` | 33 tests |
| `backend/tests/test_s106b_document.py` | 31 tests |
| `backend/tests/test_s106b_routes.py` | 22 tests |
| `frontend/app/components/sccvision/DeviceQuestionnaire.tsx` | Device questionnaire modal |
| `docs/fr04_manual_test_script.md` | 7 manual test scenarios |

### Modified files
| File | Change |
|---|---|
| `backend/SCCVision/router.py` | server_sha256 on /extract; hash_match on /process; analysis_id on /analyze |
| `backend/app.py` | Register s106b_bp blueprint |
| `backend/requirements.txt` | Add fpdf2==2.9.0 |
| `firestore.rules` | Add analyses/{} and certificates/{} read rules |
| `frontend/app/lib/evidenceTypes.ts` | Add DeviceFacts, CertificateRecord, CertificateDetail types |
| `frontend/app/lib/evidenceApi.ts` | Replace stubs with real API calls |
| `frontend/app/components/sccvision/SCCVisionTest.tsx` | Add analysis_id, questionnaire button, cert banner |
| `frontend/app/litigant/certificates/page.tsx` | Enable download, snake_case fields |
| `frontend/app/litigant/analysis/page.tsx` | Remove setTimeout simulation |

---

## 6. Test Results (at time of commit)

| Suite | Tests | Passed | Skipped | Pre-existing failures |
|---|---|---|---|---|
| test_analysis_store.py | 11 | 11 | 0 | 0 |
| test_s106b_gate.py | 33 | 33 | 0 | 0 |
| test_s106b_document.py | 31 | 31 | 0 | 0 |
| test_s106b_routes.py | 22 | 22 | 0 | 0 |
| test_history.py | 133 + 4 skipped | 133 | 4 (emulator) | 0 |
| test_sccvision_analyst.py | — | — | — | 1 (pre-existing, unrelated to FR-04) |
| **Total FR-04** | **97** | **97** | **0** | **0** |

The 1 pre-existing failure (`test_deterministic_followups_missing_date_and_payment`) was present before the branch was created and is unrelated to FR-04.

---

## 7. Security and Privacy Notes

- **No LLM in admissibility gate**: the gate uses only Firestore-stored snapshot data and litigant-supplied facts.
- **No secrets committed**: API keys, service-account credentials referenced by environment variable name only.
- **No real personal data in tests**: all test records use obviously synthetic values ("Jane Synthetic", "Samsung Galaxy A14", last4 "6789").
- **Phone number privacy**: only last 4 digits stored; `phone_number_last4` field. Full number never written to PDF or stored.
- **Firestore write-only via Admin SDK**: client rules are read-only for `analyses/` and `certificates/` — no client can write or delete certificate records.
- **Analysis snapshot stripped**: OCR payloads and raw image data are excluded from the snapshot; only the minimum fields needed for the gate are stored.

---

## 8. Known Limitations and Future Work

1. **No `GET /litigant/certificates/<id>` page**: the "View report" link navigates to that path but no detail page component exists yet. It will show a 404 until a future phase adds it.
2. **`listEvidence()` still returns `[]`**: the evidence upload flow (`POST /api/evidence/upload`) is not yet implemented.
3. **Firestore indexes**: no composite index was added for the `certificates` collection; queries use the default single-field `created_at` index. A composite index on `(uid, created_at DESC)` should be added before production.
4. **Limitation period not implemented**: as noted in PROBE_FR04.md, the 6-year limitation period under the Limitation of Actions Act is not yet surfaced in analysis output.
5. **Admin role not tested end-to-end**: role enforcement is unit-tested; a full admin-panel integration test is deferred.
