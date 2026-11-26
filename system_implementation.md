# FR-04 Implementation Guide — Section 106B Certificate and Deficiency Report

Paste this entire file into your coding agent from the repository root. It is written as instructions to the agent.

---

## 0. Instructions to the agent

You are implementing Chapter 4 requirement **FR-04** in this repository (a legal decision-support system for Kenya's Small Claims Court; SCC module in `backend/SCCVision/`, litigant pages in `frontend/app/litigant/`). A read-only probe has already been done; its findings are summarised in this guide. Follow the phases in order.

**Working rules**
- Create a branch first: `git checkout -b feat/fr04-s106b-certificate`. Commit at the end of each phase with a clear conventional-commit message (`feat:`, `test:`, `fix:`, `docs:`). Do not push unless I ask.
- After each phase, run the full backend test suite (`cd backend && PYTHONPATH=. venv/bin/pytest tests/ -v`) and stop if anything that passed before now fails. Fix it before continuing.
- Do not change the existing behaviour of `/sccvision/case/analyze`, the advocate module, or `AnalystResult` field names. Additive changes only.
- Never print or commit secrets. Never write real personal data into tests — use obviously synthetic values.
- Keep the language model **out** of the Section 106B decision. The gate must be deterministic and explainable.
- Keep a running log in `FR04_IMPLEMENTATION_REPORT.md` (format in Phase 7) as you go — record every decision, every test failure you hit and how you fixed it, and every commit hash. This log is used directly for the project report, so it must be accurate.

**Fixed design decisions (do not change without asking me)**

| Question | Decision |
|---|---|
| When is the questionnaire asked? | After analysis (probe option Q7-b). Analysis runs first; the litigant then answers the device questionnaire; the gate runs; a document is produced. |
| Questionnaire scope | One questionnaire per certificate request, covering the device that holds the evidence. |
| Signature | The system never signs. The PDF is a **draft** with a typed declaration name, a blank signature line, a blank ID-number line and a date line, to be printed and signed by hand. |
| Responsible person's ID number | Not collected or stored. It is written by hand on the printed draft. |
| Document format | Clean A4 PDF generated with fpdf2, headed "DRAFT", structured by the s.106B(4)(a)–(d) headings. No court-form replica. |
| Certificate vs deficiency report | One generator, two document types: `CERTIFICATE` (all conditions met) or `DEFICIENCY_REPORT` (any condition unmet). |
| Contract analysis in the certificate | The certificate contains only the admissibility attestation plus an evidence schedule (files, types, SHA-256 hashes). Contract-element findings appear only in the deficiency report's guidance section. |
| PDF storage | No file storage. The certificate record (data snapshot) is stored in Firestore; the PDF is regenerated deterministically from that snapshot on each download. Firebase Storage is not used. |
| Certificate number | `SCC106B-YYYYMMDD-XXXXXX` where XXXXXX is 6 uppercase hex characters from a UUID4. |
| Who may request a certificate | Roles `litigant`, `advocate`, `admin` (advocates prepare evidence for clients; the declarant must then be the client). |

---

## Phase 1 — Server-side hash and stored analyses

**Goal:** the certificate must be built from data the server produced, not from data the browser sends back.

1. In `SCCVision/router.py`, in the `/sccvision/extract` handler, compute `hashlib.sha256(file_bytes).hexdigest()` on the uploaded bytes and include it in the response as `server_sha256`. (Additive field only.)
2. In `/sccvision/evidence/process`, accept an optional `server_sha256` alongside the existing client `sha256`. Store both on the evidence item and add `hash_match: bool` (true only if both are present and equal).
3. In `/sccvision/case/analyze`, after the existing pipeline returns, save a snapshot to Firestore at `users/{uid}/analyses/{analysis_id}` (new module `SCCVision/analysis_store.py`, modelled on `history_store.py`, reusing its `_size_guard`). Store: `analysis_id` (uuid4), `created_at`, `prompt_version`, `model`, `result` (AnalystResult dump), `verification`, and an `evidence` list with `evidence_id, type, file_name, sha256, server_sha256, hash_match, added_at` (no OCR text, no images). Return `analysis_id` as an additional top-level key in the response.
4. If the Firestore write fails, log it and still return the analysis (with `analysis_id: null`) — saving must not break analysis.
5. Tests (`tests/test_analysis_store.py`): snapshot written with the right path and fields (mocked Firestore); size guard applied; write failure does not raise; response contains `analysis_id`. Update any existing assertion that checks the exact key set of the analyze response so it allows the new key.

## Phase 2 — Device questionnaire model and the deterministic s.106B gate

Create `backend/SCCVision/s106b/` with `__init__.py`, `models.py`, `gate.py`.

**`models.py`** — Pydantic models:

```python
class ProductionMethod(str, Enum):
    SCREENSHOT_ON_DEVICE = "screenshot_on_device"   # taken on the phone that holds the messages
    FORWARDED_OR_COPIED = "forwarded_or_copied"     # received from someone else / copied
    PRINTED_STATEMENT = "printed_statement"
    OTHER = "other"

class DeviceFacts(BaseModel):
    device_make: str                  # e.g. "Samsung"
    device_model: str                 # e.g. "Galaxy A14"
    phone_number_last4: str | None    # last 4 digits only; never store full number
    production_method: ProductionMethod
    owner_is_declarant: bool          # the declarant owned/controlled the device
    lawful_control: bool              # s.106B(2)(a)
    regular_use: bool                 # s.106B(2)(a)
    regular_feed: bool                # s.106B(2)(b)
    operating_properly: bool          # s.106B(2)(c)
    malfunction_details: str | None   # required if operating_properly is False
    malfunction_affects_record: bool | None
    messages_on_this_device: bool     # s.106B(2)(d) — sent/received directly on this device
    declarant_name: str
    declarant_position: str           # e.g. "Owner and sole user of the device"
    declaration_confirmed: bool       # ticked "the above is true to the best of my knowledge"

class ConditionResult(BaseModel):
    code: str            # "4a", "4b", "2a", "2b", "2c", "2d", "4d", "INTEGRITY"
    title: str
    met: bool
    basis: list[str]     # human-readable reasons, citing the answer or check used
    remedy: str | None   # plain-language action if not met

class GateResult(BaseModel):
    document_type: Literal["CERTIFICATE", "DEFICIENCY_REPORT"]
    outcome: Literal["certificate_ready", "needs_strengthening", "weak"]
    conditions: list[ConditionResult]
    unmet: list[str]     # condition codes not met
    gate_version: str    # e.g. "s106b-gate-1.0"
```

Validation: strip whitespace; `device_make`, `device_model`, `declarant_name`, `declarant_position` must be non-empty; `phone_number_last4` must be exactly 4 digits if given; if `operating_properly` is false then `malfunction_details` and `malfunction_affects_record` are required.

**`gate.py`** — `evaluate(analysis_snapshot: dict, facts: DeviceFacts) -> GateResult`. Rules (implement exactly; each produces one `ConditionResult` with `basis` and `remedy`):

| Code | Title | Met when | Remedy if not met |
|---|---|---|---|
| 4a | Electronic record identified and manner of production described | every evidence item has `file_name` and `server_sha256`, and every item has `hash_match` true; `production_method` is not `OTHER` | "Re-upload the original screenshot files directly from your phone." / "Describe how the screenshots were produced." |
| 4b | Particulars of the device | `device_make` and `device_model` provided | "Provide the make and model of the phone that holds the messages." |
| 2a | Device regularly used by a person with lawful control | `lawful_control` and `regular_use` both true | "The certificate must be given by the person who controlled and regularly used the phone at the time." |
| 2b | Information regularly fed into the device | `regular_feed` true | "Confirm that messages/payments of this kind were regularly received or stored on this phone." |
| 2c | Device operating properly | `operating_properly` true, OR malfunction reported with `malfunction_affects_record` false | "A malfunction that affected the record prevents certification; obtain the record from another source (e.g. an M-Pesa statement from Safaricom)." |
| 2d | Record derived from information fed into the device | `messages_on_this_device` true AND `production_method` is `SCREENSHOT_ON_DEVICE` or `PRINTED_STATEMENT`; for M-Pesa items, at least one parsed transaction code exists | "Take the screenshots on the phone that sent or received the messages, not from forwarded copies." |
| 4d | Signed by a person in a responsible position | `owner_is_declarant`, `declaration_confirmed`, non-empty `declarant_name` and `declarant_position` | "The person who owns and controls the phone must make and sign the declaration." |
| INTEGRITY | Evidence integrity checks passed | from the stored `verification`: `quotes_grounded` true, `mpesa_cross_check.consistent` true, `invented_fields` false; and `result.conversation_integrity.rating` is not `WEAK` | "The evidence shows inconsistencies (list them from the checks). Upload the complete, unedited conversation." |

Outcome mapping:
- all conditions met → `document_type = CERTIFICATE`, `outcome = certificate_ready`
- only questionnaire-type conditions unmet (2a, 2b, 2c, 2d, 4b, 4d) and INTEGRITY + 4a met → `DEFICIENCY_REPORT`, `needs_strengthening`
- INTEGRITY or 4a unmet → `DEFICIENCY_REPORT`, `weak`

The contract-analysis verdict (`overall_verdict`) must **not** decide admissibility; record it only for the deficiency report's guidance section.

**Tests (`tests/test_s106b_gate.py`)** — at least: all conditions met → certificate; each of 4a, 4b, 2a, 2b, 2c, 2d, 4d failing individually → correct unmet code and remedy; malfunction reported but not affecting record → 2c met; malfunction affecting record → 2c unmet; hash mismatch → 4a unmet and outcome weak; ungrounded quotes → INTEGRITY unmet and outcome weak; WEAK conversation integrity → weak; overall_verdict "LIKELY NOT VALID" with all s.106B conditions met → still CERTIFICATE (proves admissibility is independent of contract verdict); DeviceFacts validation errors (empty make, bad last4, malfunction without details).

## Phase 3 — Document generator (fpdf2)

1. Add `fpdf2==2.9.0` to `backend/requirements.txt`.
2. Create `backend/SCCVision/s106b/document.py` with `render_pdf(record: dict) -> bytes`. It must be deterministic for the same record (no current-time calls inside; use `record["created_at"]`).
3. Common elements on every page: header "DRAFT — FOR REVIEW AND SIGNATURE", certificate number, generation date, footer with page number and "Generated by the SCC Evidence Assistant (prototype). This draft is not legal advice and has no legal effect until reviewed and signed by the declarant."
4. **CERTIFICATE** layout:
   - Title: "Certificate in respect of Electronic Records — Section 106B(4), Evidence Act (Cap. 80)"
   - Part A, s.106B(4)(a): schedule table of evidence items (No., file name, type, date added, SHA-256 — wrap the hash), plus one sentence describing the production method from the questionnaire.
   - Part B, s.106B(4)(b): device make and model, phone number shown as `******` + last 4.
   - Part C, s.106B(4)(c): one numbered statement per condition (2)(a)–(d), written as a declaration in the first person of the declarant (e.g. "During the period covered by these records, the device was regularly used by me and was under my lawful control.").
   - Part D, s.106B(4)(d): "I, [declarant_name], [declarant_position], certify that the statements above are true to the best of my knowledge and belief." followed by blank lines: Signature ______, National ID No. ______, Date ______, Place ______.
5. **DEFICIENCY_REPORT** layout:
   - Title: "Evidence Admissibility Report — Section 106B, Evidence Act (Cap. 80): Requirements Not Yet Met"
   - Plain-language summary: outcome in one sentence.
   - Table: requirement | status (Met / Not met) | what to do next (remedy).
   - "Other gaps found in your evidence": `missing_information` and `strengthen_checklist` from the analysis snapshot (max 10 items each).
   - Evidence schedule (same as certificate Part A).
   - Disclaimer from the analysis snapshot.
6. Use a built-in core font (Helvetica) and sanitise text to Latin-1 (replace unsupported characters) so rendering never crashes on emoji or smart quotes from OCR.
7. **Tests (`tests/test_s106b_document.py`)**: output starts with `%PDF`; both document types render; text extracted with `pypdf` contains the certificate number, the four Part headings (certificate) or "Requirements Not Yet Met" (report); the phone number never appears in full; rendering the same record twice gives identical bytes after normalising fpdf2's creation-date metadata (set `pdf.set_creation_date(...)` from `created_at` so bytes are identical); OCR text with emoji does not crash.

## Phase 4 — Certificate store and API routes

1. `backend/SCCVision/s106b/store.py` (pattern from `history_store.py`): Firestore path `users/{uid}/certificates/{cert_id}`. Fields: `certificate_number`, `document_type`, `outcome`, `analysis_id`, `evidence` (schedule list from the analysis snapshot), `device_facts` (DeviceFacts dump — contains no full phone number and no ID number), `gate` (GateResult dump), `analysis_summary` (`overall_verdict`, `plain_summary`, `missing_information`, `strengthen_checklist`, `disclaimer`, `prompt_version`, `model`), `created_at`, `schema_version: 1`. Apply `_size_guard`.
2. New Flask blueprint `backend/SCCVision/s106b/routes.py`, registered in `app.py`, all routes `@require_role("litigant", "advocate", "admin")`:

| Method | Path | Behaviour |
|---|---|---|
| POST | `/api/evidence/certificates` | Body `{analysis_id, device_facts}`. Load `users/{g.uid}/analyses/{analysis_id}` (404 if missing — this also enforces ownership). Validate DeviceFacts (400 with field errors). Run gate. Save record. Return 201 with the record summary (`id, certificateNumber, documentType, outcome, unmet, createdAt`). |
| GET | `/api/evidence/certificates` | List the caller's records, newest first, mapped to the frontend `CertificateRecord` shape (`id, certificateNumber, evidenceId` (first evidence id), `fileName` (first file name, plus "+N more" if several), `outcome, createdAt`). |
| GET | `/api/evidence/certificates/<cert_id>` | Full record (gate conditions with basis and remedy, evidence schedule, analysis summary). 404 if not the caller's. |
| GET | `/api/evidence/certificates/<cert_id>/download` | Regenerate the PDF from the stored record; return `application/pdf` with `Content-Disposition: attachment; filename="<certificate_number>.pdf"`. 404 if not the caller's. |

3. `firestore.rules`: add read-own / write-false blocks for `users/{uid}/analyses/{id}` and `users/{uid}/certificates/{id}` (same pattern as conversations). `firestore.indexes.json`: add the index needed to order `certificates` by `created_at` descending if Firestore requires it.
4. **Tests (`tests/test_s106b_routes.py`)** using `app.test_client()` with auth stubbed as in `test_history.py` and Firestore mocked: 201 happy path returns `certificate_ready`; 400 on invalid DeviceFacts; 404 when `analysis_id` belongs to another user / does not exist; 403 for a user without an allowed role; list returns newest first in `CertificateRecord` shape; download returns `application/pdf` and starts with `%PDF`; 404 on another user's certificate.
5. **Firestore rules tests** (`firestore-tests/firestore.test.ts`): owner can read own certificate and analysis; other user cannot; nobody can write from the client. Run with the emulator if available; if not, record "not run — emulator unavailable" in the report.

## Phase 5 — Frontend

1. `frontend/app/lib/evidenceApi.ts`: replace the `listCertificates()` stub with an authenticated call to `GET /api/evidence/certificates` using the existing `api.ts` fetch wrapper. Add `getCertificate(id)`, `createCertificate(analysisId, deviceFacts)` and `downloadCertificate(id, certificateNumber)` (fetch as blob with the auth header, then trigger a download via `URL.createObjectURL`; a plain link will not carry the token). Add matching TypeScript types for `DeviceFacts`, `ConditionResult`, `CertificateDetail`.
2. New component `frontend/app/components/sccvision/DeviceQuestionnaire.tsx`: a single-page form in plain language with the DeviceFacts fields. Yes/No questions as radio buttons, each with a one-line "why we ask" hint referencing the condition (e.g. "The law requires that the phone was working properly — s.106B(2)(c)"). Show malfunction fields only when "operating properly" is No. Client-side validation mirrors the backend. A confirmation checkbox for the declaration. Submit calls `createCertificate`.
3. In `SCCVisionTest.tsx`, after a successful analysis that returned an `analysis_id`, show a "Prepare Section 106B certificate" button that opens the questionnaire. On success, show the outcome: for `certificate_ready`, a "Download draft certificate" button; otherwise the unmet requirements with their remedies and a "Download deficiency report" button. Keep the existing `CertificateModal` working; relabel its print button "Print analysis summary" so it is not confused with the statutory certificate.
4. `frontend/app/litigant/certificates/page.tsx`: enable the Download and View buttons; View opens a detail panel listing each condition with Met/Not met, basis and remedy.
5. `frontend/app/litigant/analysis/page.tsx`: remove the `setTimeout` simulation from `handleSubmit` (route users to the embedded analysis panel instead, or hide the simulated submit). Step 3 "Answer a few questions" now refers to the real questionnaire.
6. Run `npm run build` and `npm run lint` in `frontend/`; fix all errors. Record the build result.

## Phase 6 — End-to-end verification (manual, by me — you prepare the script)

Write `docs/fr04_manual_test_script.md` with these scenarios, each with steps, expected result and a blank "Actual / Evidence (screenshot file) / Pass-Fail" line:

| ID | Scenario | Expected |
|---|---|---|
| FR04-E01 | Upload a genuine M-Pesa screenshot and a matching WhatsApp screenshot taken on the same phone; answer all questions Yes | Outcome certificate_ready; draft certificate PDF downloads with Parts A–D and evidence hashes |
| FR04-E02 | Same evidence, answer "phone not working properly" and "malfunction affected the record" | Deficiency report; 2c unmet with remedy |
| FR04-E03 | Production method "forwarded or copied" | Deficiency report; 2d unmet |
| FR04-E04 | Declarant does not own the device | Deficiency report; 4d unmet |
| FR04-E05 | Submit questionnaire with empty device make | Form blocks submission; backend returns 400 if bypassed |
| FR04-E06 | Second user tries to open the first user's certificate URL | 404 |
| FR04-E07 | Certificates page after E01–E04 | Four records listed newest first with correct outcomes; download works for each |

Use only synthetic screenshots created for testing (state this in the script). List the screenshots I should capture for the report: questionnaire form, certificate-ready result, deficiency result, certificates list, first page of each PDF.

## Phase 7 — Implementation report for the documentation

Finish `FR04_IMPLEMENTATION_REPORT.md` with exactly these sections (facts only, no marketing language, cite files):

1. **Summary** — what was built, in 5–8 sentences.
2. **Requirement traceability** — table: FR-04 clause / Section 106B clause → implementing file(s) → test ID(s).
3. **Design decisions** — the decision table from section 0 of this guide, plus any decision you had to make yourself and why.
4. **Gate rules** — the final rule table as implemented (code, condition, met-when, remedy), and the outcome mapping.
5. **Data model** — Firestore paths and fields for `analyses` and `certificates`; what personal data is and is not stored.
6. **API** — table of new/changed endpoints (method, path, auth, request, response, error codes).
7. **Frontend changes** — files and what changed; which stubs were replaced.
8. **Test results** — command used, totals before and after (passed/failed/skipped), and a table of every new test: test ID (use FR04-U01… for unit, FR04-A01… for API, FR04-R01… for rules), requirement, scenario, expected, actual, verdict.
9. **Defects found and fixed during implementation** — defect ID, symptom, cause, fix, commit hash, retest result. Include every real failure you hit; do not omit them.
10. **Known limitations** — at minimum: the draft has no legal effect until signed; the system cannot verify questionnaire answers; device particulars are self-reported; hash comparison detects mismatched uploads, not tampering before upload; no court-form replica; PDFs regenerated rather than archived.
11. **Commits** — hash and message for every commit on the branch.
12. **Remaining manual steps for me** — the Phase 6 script and the screenshots to capture.

Do not merge the branch. Tell me when all phases are complete and summarise the final test totals.