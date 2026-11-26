# Chapter 5 Probe Results — SYSTEM_CHAPTER5_REVISED

**Branch:** feat/fr04-s106b-certificate  
**Commit:** b4c79b5  
**Date:** 2026-11-17

---

## Part A: Branch and Repository State

### A1. Remote Push Status

The branch `feat/fr04-s106b-certificate` has **not been pushed** to the remote. The only remote branch is `origin/master`.

```
git branch -r
  origin/master
```

Attempting to read `origin/feat/fr04-s106b-certificate` returns a fatal error:

```
fatal: ambiguous argument 'origin/feat/fr04-s106b-certificate': unknown revision or path not in the working tree.
```

Seven commits exist on this branch ahead of `master`, all dated 2026-11-17, all local only:

| Hash (short) | Date | Subject |
|---|---|---|
| b4c79b5 | 2026-11-17 | docs(fr04): Phase 7 — FR04 implementation report |
| 4a16c0e | 2026-11-17 | docs(fr04): Phase 6 — manual test script FR04-E01 through FR04-E07 |
| 8a65097 | 2026-11-17 | feat(fr04): Phase 5 — frontend wiring for Section 106B certificate |
| b70caea | 2026-11-17 | feat(fr04): Phase 4 — certificate store, API routes, Firestore rules |
| 9393ec1 | 2026-11-17 | feat(fr04): Phase 3 — deterministic fpdf2 PDF document generator |
| b98911 | 2026-11-17 | feat: Phase 2 — DeviceFacts model and deterministic s.106B gate |
| 563d8c1 | 2026-11-17 | feat: Phase 1 — server-side SHA-256 and analysis snapshot store |

All seven commits are at risk of loss if the local working tree is destroyed. No remote backup exists as of this audit.

---

### A2. ChromaDB Git Tracking

**22 binary ChromaDB files are tracked by Git.** There is no `.gitignore` rule covering any of them.

Tracked files include:

- `backend/ChunkDB/chroma.sqlite3`
- `backend/ChunkDB/c79b875b-b244-43a5-8154-c671bb89882a/length.bin`
- `backend/LegalSummariesDB/chroma.sqlite3`
- 15 × `backend/LegalSummariesDB/<uuid>/length.bin`
- `backend/constitutionlogic/MasterRulesDB/chroma.sqlite3`
- `backend/constitutionlogic/MasterRulesDB/81b63fd4-aec9-43ea-b12a-74fca93b449e/length.bin`
- `backend/constitutionlogic/chroma.sqlite3`

The root `.gitignore` contains no lines covering `chroma.sqlite3`, `*.bin`, `*.sqlite3`, `ChunkDB/`, `LegalSummariesDB/`, `MasterRulesDB/`, `StatutesDB/`, `CaseLawDB/`, or `*.jsonl`. The only relevant ignores in the repository are Neo4j data directories, `.env`, Python cache, and frontend build artifacts.

**Consequence:** Every `git commit` that follows a re-embedding run will silently bloat the repository history with binary vector-database files. `chroma.sqlite3` files are not human-reviewable diffs and do not belong in source control. The `length.bin` files are ChromaDB segment metadata. Neither type should be committed.

**Recommended fix:**

```
# Add to root .gitignore:
backend/ChunkDB/
backend/LegalSummariesDB/
backend/StatutesDB/
backend/CaseLawDB/
backend/constitutionlogic/MasterRulesDB/
backend/constitutionlogic/BlackLawDictionaryDB/
**/*.sqlite3
**/*.bin
**/*.jsonl
```

After adding the ignore rules, the already-tracked files must be explicitly removed from the index:

```
git rm -r --cached backend/ChunkDB/ backend/LegalSummariesDB/ ...
```

---

### A3. Requirements vs Installed Packages

**Summary:** The virtual environment diverges significantly from `backend/requirements.txt`. Several packages critical to the core pipeline are listed but not installed; several packages used by live code are not listed at all.

#### Packages listed in requirements.txt but NOT present in the venv

| Package | requirements.txt version |
|---|---|
| fpdf2 | 2.9.0 |
| firebase-admin | 7.7.0 |
| flask-cors | 6.0.1 |
| Flask | 3.1.2 |
| rank-bm25 | 0.2.2 |
| rapidocr | 3.10.0 |
| PyMuPDF (fitz) | 1.24.14 |
| pytesseract | 0.3.13 |
| ocrmypdf | 16.5.0 |
| opencv-python-headless | 5.0.0.93 |

Of these, `fpdf2` is the PDF generator introduced on this branch (`backend/SCCVision/s106b/document.py`). Its absence from the venv means PDF generation will fail at runtime in this environment even though all 31 `test_s106b_document.py` tests pass (they apparently ran in a different environment or the package was present during the test run). `firebase-admin`, `Flask`, and `flask-cors` are foundational to the backend; their absence suggests the current venv is not the venv the application runs from.

#### Packages used by live code but NOT listed in requirements.txt

| Package | Where used |
|---|---|
| rapidfuzz | fuzz imports in backend code |
| langchain-text-splitters | `RecursiveCharacterTextSplitter` |
| faiss-cpu | installed in venv (1.12.0), no entry |
| werkzeug | `secure_filename` import |

#### Version mismatches (requirements.txt pinned vs installed)

| Package | requirements.txt | venv installed |
|---|---|---|
| chromadb | 1.3.5 | 1.3.4 |
| groq | 0.35.0 | 0.33.0 |
| langchain | 1.0.7 | 1.0.2 |
| langchain-core | 1.0.6 | 1.0.1 |
| numpy | 2.3.5 | 2.3.4 |
| torch | 2.9.1 | 2.9.0 |
| pypdf | 6.3.0 | 6.1.3 |
| Pillow | 11.1.0 | 12.0.0 (venv is **newer**) |
| python-dotenv | 1.2.1 | 1.1.1 |
| opentelemetry-api | 1.45.1 | 1.38.0 |
| opentelemetry-sdk | 1.45.1 | 1.38.0 |
| opentelemetry-exporter-otlp-proto-common | 1.45.1 | 1.38.0 |
| opentelemetry-exporter-otlp-proto-grpc | 1.45.1 | 1.38.0 |

`openai` (2.6.0) is installed in the venv but absent from `requirements.txt`. No top-level backend `.py` file directly imports `openai` as a module; it appears to be a transitive dependency.

---

### A4. Environment Variable Examples

| File | Status |
|---|---|
| `backend/.env.example` | **DOES NOT EXIST** |
| `.env.example` (root) | Exists; contains one variable: `HF_API_TOKEN=` |
| `frontend/.env.example` | Exists |

Frontend `.env.example` variables:

```
NEXT_PUBLIC_FIREBASE_API_KEY=
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=
NEXT_PUBLIC_FIREBASE_PROJECT_ID=
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=
NEXT_PUBLIC_FIREBASE_APP_ID=
NEXT_PUBLIC_API_URL=
```

**Gap:** `backend/.env.example` is missing. A developer cloning the repository has no documented list of required backend environment variables. At minimum, `GROQ_API_KEY`, `FIREBASE_SERVICE_ACCOUNT_PATH`, and any `GROQ_API_KEY_N` rotation keys should be documented in a `backend/.env.example` file with placeholder values and inline comments.

---

## Part B: SCC Module

### B1. Device Questionnaire

**Status: PRESENT and wired to a real API call.**

The questionnaire is implemented in:

```
frontend/app/components/sccvision/DeviceQuestionnaire.tsx
```

It is not a stub. Every field maps to a real `<input>` or `<button>` control. The widget-to-field mapping is:

| DeviceFacts field | Widget | Lines |
|---|---|---|
| `device_make` | `<input type="text">` | 274–279 |
| `device_model` | `<input type="text">` | 282–287 |
| `lawful_control` | Two `<button>` toggles (Yes/No) | 346–350 |
| `regular_use` | Two `<button>` toggles | 358–362 |
| `regular_feed` | Two `<button>` toggles | 368–372 |
| `operating_properly` | Two `<button>` toggles | 378–382 |
| `messages_on_this_device` | Two `<button>` toggles | 414–418 |
| `declarant_name` | `<input type="text">` | 433–436 |
| `declarant_position` | `<input type="text">` | 440–444 |
| `declaration_confirmed` | `<input type="checkbox">` | 453–457 |

On submit (`handleSubmit`, line 202), the component calls `createCertificate(analysisId, facts)` from `frontend/app/lib/evidenceApi.ts` (lines 22–37). That function issues a real HTTP POST with no `setTimeout` simulation:

- **Endpoint:** `POST /api/evidence/certificates`
- **Body:** `{ "analysis_id": "<string>", "device_facts": <DeviceFacts> }`

**Routing path to reach the form:**

`DeviceQuestionnaire` is imported into `SCCVisionTest` (`SCCVisionTest.tsx` line 11) and mounted at line 1001. `SCCVisionTest` is imported and rendered unconditionally at `frontend/app/litigant/analysis/page.tsx` line 275. The form is reachable at the route `/litigant/analysis`.

**Gate condition:** The questionnaire is only shown when `showQuestionnaire && analysis?.analysis_id` is truthy (`SCCVisionTest.tsx` line 1062). The `analysis_id` is set after a successful call to `POST /sccvision/case/analyze`.

---

### B2. Certificates Page

**Status: WIRED — real network calls, no stubs.**

`listCertificates()` (`frontend/app/lib/evidenceApi.ts` lines 8–13):

```typescript
export async function listCertificates(): Promise<CertificateRecord[]> {
  const res = await apiFetch("/api/evidence/certificates");
  if (!res.ok) return [];
  const body = await res.json().catch(() => ({}));
  return (body.certificates as CertificateRecord[]) ?? [];
}
```

This issues a real HTTP GET to `http://localhost:5000/api/evidence/certificates`. On a non-2xx response it silently returns `[]`, which causes the page to render the empty-state UI rather than surfacing an error — silent failure on network outage.

**Download button:** Enabled whenever a record exists and no download is already in flight for that specific row (`disabled={downloading === rec.cert_id}`). Calls `downloadCertificate(rec.cert_id, rec.certificate_number)` which hits `GET /api/evidence/certificates/{certId}/download`, streams the response as a `Blob`, and triggers a browser anchor click to save as `{certificate_number}.pdf`.

---

### B3. End-to-End Wiring

The main pipeline lives entirely inside `SCCVisionTest` (embedded in `/litigant/analysis`). The dedicated upload page at `/litigant/analysis/page.tsx` and evidence history page at `/litigant/evidence/page.tsx` are separate stubs that do not share state with `SCCVisionTest`.

#### Wired steps

| Step | Frontend file:line | Backend route | Status |
|---|---|---|---|
| Upload + OCR | `SCCVisionTest.tsx:350–389` | `POST /sccvision/extract` | WIRED |
| Evidence normalise | `SCCVisionTest.tsx:366–378` | `POST /sccvision/evidence/process` | WIRED |
| Case analysis | `SCCVisionTest.tsx:407` | `POST /sccvision/case/analyze` | WIRED |
| Device questionnaire | `DeviceQuestionnaire.tsx:211` via `evidenceApi.ts:26` | `POST /api/evidence/certificates` | WIRED |
| Gate (cert vs deficiency) | — server-side only | inside `POST /api/evidence/certificates` | WIRED |
| List certificates | `certificates/page.tsx:77` via `evidenceApi.ts:10` | `GET /api/evidence/certificates` | WIRED |
| Download PDF | `certificates/page.tsx:91` via `evidenceApi.ts:40` | `GET /api/evidence/certificates/<id>/download` | WIRED |

#### Steps that are NOT wired

**Upload via `/litigant/analysis/page.tsx` — STUB**

`handleSubmit` at `analysis/page.tsx` lines 55–74 is entirely commented out. The comment block describes a planned `POST /api/evidence/upload` and states explicitly "not yet implemented." `EvidenceDropzone` collects files but pressing "Analyse evidence" sets `isSubmitting`, then immediately clears it; no request is sent. The backend has no `POST /api/evidence/upload` route. This page is an isolated dead-end — it does not call the working OCR/analysis pipeline in `SCCVisionTest`; it has its own dropzone and its own inert submit handler.

**Evidence history via `/litigant/evidence/page.tsx` — STUB**

`listEvidence()` (`evidenceApi.ts` lines 4–7) is hardcoded:

```typescript
export async function listEvidence(): Promise<never[]> {
  // TODO(backend): GET /api/evidence — returns EvidenceRecord[]
  return [];
}
```

No network request is ever made. The page always renders the empty-state UI. No `GET /api/evidence` route exists in the backend.

**Certificate detail page at `/litigant/certificates/<cert_id>` — PAGE FILE MISSING**

`certificates/page.tsx` links to `/litigant/certificates/${rec.cert_id}` (lines 240, 289). No corresponding `frontend/app/litigant/certificates/[cert_id]/page.tsx` file exists. Clicking "View report" produces a 404.

---

### B4. CERT_THRESHOLD

`CERT_THRESHOLD = 0.83` is defined exclusively in the frontend.

| File:line | Usage |
|---|---|
| `SCCVisionTest.tsx:23` | `const CERT_THRESHOLD = 0.83;` — declaration |
| `SCCVisionTest.tsx:186` | `a.result.confidence >= CERT_THRESHOLD` — gate logic |
| `SCCVisionTest.tsx:1049` | `Meets the {Math.round(CERT_THRESHOLD * 100)}% threshold` — UI label |
| `SCCVisionTest.tsx:1378` | `✓ Meets the {Math.round(CERT_THRESHOLD * 100)}% evidence threshold` — second UI label |

There is **no corresponding definition or enforcement in any backend `.py` file**. The backend computes and returns a confidence value independently; whether that value passes the 83% threshold is evaluated entirely client-side.

**Reachability:** `SCCVisionTest` is imported and rendered unconditionally at `frontend/app/litigant/analysis/page.tsx` line 275 — it is not wrapped in any `isAdvocate`, `isDev`, `isAdmin`, or environment-variable guard. The layout at `/litigant/layout.tsx` provides navigation chrome only and contains no `AuthGuard`, no role redirect, and no conditional that hides child content. Any authenticated litigant who navigates to `/litigant/analysis` sees the full `SCCVisionTest` panel, including the 83% threshold labels.

---

### B5. SHA-256 Flow

The flow proceeds in five stages:

**Stage 1 — Server computes hash on upload**

`backend/SCCVision/router.py` line 70, inside `POST /sccvision/extract`, after reading file bytes and before OCR:

```python
server_sha256 = hashlib.sha256(data).hexdigest()
```

Attached to the OCR response at line 87:

```python
result["server_sha256"] = server_sha256
```

**Stage 2 — Browser independently computes hash**

`SCCVisionTest.tsx` lines 209–213, using the Web Crypto API over the same `ArrayBuffer`:

```typescript
async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
```

**Stage 3 — Backend compares both values**

`backend/SCCVision/router.py` lines 220–223:

```python
client_sha256 = data.get("sha256")
server_sha256 = data.get("server_sha256") or (ocr.get("server_sha256") if isinstance(ocr, dict) else None)
hash_match = (
    bool(client_sha256 and server_sha256 and client_sha256 == server_sha256)
)
```

Both values and the boolean are stored on the evidence item (lines 230–232).

**Stage 4 — Mismatch handling: warning only at upload, hard block at gate**

There is no hard rejection at upload time. A mismatch sets `hash_match = False` but the endpoint still returns HTTP 200. The certificate gate (`backend/SCCVision/s106b/gate.py` lines 53–56) then blocks the certificate:

```python
if hm is False:
    met = False
    basis.append(f"Evidence item '{fn}': client hash does not match server hash — file may have been modified after upload.")
    remedies.append("Re-upload the original screenshot files directly from your phone.")
```

This fails condition `4a` and forces `document_type = "DEFICIENCY_REPORT"` with `outcome = "weak"`.

**Stage 5 — Hash printed on certificate PDF**

`backend/SCCVision/s106b/document.py` lines 130 and 140–149 render an evidence table with column header `"SHA-256 (server)"`. The hash is truncated for display:

```python
sha_display = sha[:20] + "..." + sha[-8:] if len(sha) > 32 else sha
```

The raw hash values are also stored in Firestore via `analysis_store.py` (line 69 explicitly keeps `sha256`, `server_sha256`, and `hash_match`).

---

### B6. Analysis Snapshots

**Collection path:** `users/{uid}/analyses/{analysis_id}` — one document per analysis run, keyed by `uuid4().hex`.

**Fields written** (`backend/SCCVision/analysis_store.py` lines 76–86):

| Field | Value |
|---|---|
| `analysis_id` | `uuid4().hex` |
| `prompt_version` | string from pipeline output |
| `model` | string from pipeline output |
| `result` | full `AnalystResult` dict |
| `verification` | verification block dict |
| `evidence` | list of stripped evidence items (see below) |
| `created_at` | Firestore `SERVER_TIMESTAMP` |
| `schema_version` | integer `1` |
| `truncated` | `bool` — `True` if `_size_guard` fires and trims `plain_summary` |

The `evidence` list is stripped at `analysis_store.py` lines 69–73 to only: `evidence_id`, `type`, `file_name`, `sha256`, `server_sha256`, `hash_match`, `added_at`. The docstring at line 62 states: **"no OCR text, no images."**

**Images:** Never persisted. The router processes image bytes entirely in memory via RapidOCR / `preprocess_image` / `decode`. `preprocess.py` line 38 comment: "Works entirely in memory, never touching disk."

**Raw OCR text:** Not in Firestore. The router at line 265 explicitly drops `parse_layout` and `raw_text` from evidence items before they reach the pipeline that feeds the snapshot. `raw_text` only appears transiently in the in-process M-Pesa parser and is never written to any store.

**Firebase Storage:** Not used. Grep returns zero hits for `firebase_storage`, `getStorage`, `uploadBytes`, `getDownloadURL`, and `storage()` in both backend and frontend. The only storage layer is Firestore.

---

### B7. Gate Conditions

**File:** `backend/SCCVision/s106b/gate.py`

**Module docstring** (line 5): "No LLM calls. Every decision is auditable from the basis[] strings."

Eight conditions are evaluated in sequence (lines 334–343):

#### 4a — Electronic record identified, manner of production described (`_check_4a`, lines 21–72)
- FAIL if `snapshot["evidence"]` is empty or absent
- FAIL per evidence item if: `file_name` missing, `server_sha256` missing, or `hash_match is False`
- FAIL if `facts.production_method == ProductionMethod.OTHER`
- PASS only when all items have file name, server hash, confirmed hash match, and production method is not OTHER

#### 4b — Particulars of the device (`_check_4b`, lines 75–89)
- Always `met = True`. No fail path. Writes `device_make`, `device_model`, and optionally last-4 of phone number into `basis`.

#### 2a — Device regularly used by a person with lawful control (`_check_2a`, lines 92–118)
- `met = facts.lawful_control AND facts.regular_use` — both must be truthy

#### 2b — Information regularly fed into the device (`_check_2b`, lines 121–141)
- `met = facts.regular_feed` — single boolean

#### 2c — Device operating properly (`_check_2c`, lines 144–179)
- `met = True` if `facts.operating_properly` is truthy
- If `operating_properly` is false but `malfunction_affects_record is False` → `met = True` (malfunction did not affect the record)
- FAIL only if `operating_properly` is false AND `malfunction_affects_record` is not explicitly `False`

#### 2d — Record derived from information fed into the device (`_check_2d`, lines 182–236)
- FAIL if `facts.messages_on_this_device` is false
- FAIL if `facts.production_method` not in `{SCREENSHOT_ON_DEVICE, PRINTED_STATEMENT}`
- FAIL if any evidence item has `type == "mpesa_screenshot"` AND no parsed transaction codes exist in `snapshot["result"]["mpesa_payments"]`

#### 4d — Signed by a person in a responsible position (`_check_4d`, lines 239–268)
- FAIL if `facts.owner_is_declarant` is false
- FAIL if `facts.declaration_confirmed` is false
- Both must be true

#### INTEGRITY — Evidence integrity checks passed (`_check_integrity`, lines 271–321)
- FAIL if `snapshot["verification"]["quotes_grounded"]` is explicitly `False`
- FAIL if `snapshot["verification"]["mpesa_cross_check"]["consistent"]` is false OR `invented_fields` is true
- FAIL if `snapshot["result"]["conversation_integrity"]["rating"] == "WEAK"`
- Missing keys default favourably (`quotes_grounded` → `True`, `consistent` → `True`, rating → `"ADEQUATE"`)

**Outcome mapping** (lines 345–359):

| Condition | `document_type` | `outcome` |
|---|---|---|
| All 8 conditions met | `CERTIFICATE` | `certificate_ready` |
| Any unmet AND `INTEGRITY` or `4a` among the unmet | `DEFICIENCY_REPORT` | `weak` |
| Any unmet AND neither `INTEGRITY` nor `4a` unmet | `DEFICIENCY_REPORT` | `needs_strengthening` |

`4a` and `INTEGRITY` are hard blockers. Failure of only the questionnaire-based conditions (2a, 2b, 2c, 2d, 4b, 4d) is recoverable and produces `needs_strengthening`.

**No LLM call or floating-point confidence score is computed inside the gate.** The `conversation_integrity.rating` field (`ADEQUATE`/`WEAK`) consumed by the INTEGRITY check is a categorical string already materialised in the snapshot from an earlier LLM-driven analysis step.

---

### B8. Certificate PDF

**Generator file:** `backend/SCCVision/s106b/document.py`

**Library:** `fpdf2 == 2.9.0` (imported as `from fpdf import FPDF`), pinned in `backend/requirements.txt`.

**Public API:** `render_pdf(record: dict) -> bytes`. Dispatches to `_render_certificate` or `_render_deficiency_report` based on `record["gate"]["document_type"]`.

#### Certificate PDF fields (`_render_certificate`)

- **Header (every page):** "DRAFT - FOR REVIEW AND SIGNATURE", Certificate No., date (YYYY-MM-DD only).
- **Title:** "Certificate in respect of Electronic Records / Section 106B(4), Evidence Act (Cap. 80)".
- **Part A** (s.106B(4)(a)): Evidence schedule table — columns: No., file name (truncated to 40 chars), type, date added (date portion only), SHA-256 (truncated: `first20...last8` if >32 chars). Manner of production label.
- **Part B** (s.106B(4)(b)): `device_make`, `device_model`, masked phone number.
- **Part C** (s.106B(4)(c)): Four numbered statutory statements covering conditions 2(a)–2(d). The 2(c) statement adapts when `malfunction_details` is present.
- **Part D** (s.106B(4)(d)): Declarant name, position, certification statement. Blank signature lines for: Signature, National ID No., Date, Place.

**Phone number masking:** exactly 6 asterisks prepended to the last-4 digits, rendered as `******XXXX`. If last-4 is absent, prints `(not provided)`. The full phone number is never stored or printed.

#### Storage vs regeneration

The PDF is **not stored** anywhere. `GET /api/evidence/certificates/<cert_id>/download` (`backend/SCCVision/s106b/routes.py`) calls `render_pdf(record)` on every download request, regenerating from the Firestore record. The generation is deterministic: `pdf.set_creation_date()` is seeded from `record["created_at"]` (not the current clock), so identical records produce identical bytes across repeated downloads.

#### Deficiency report PDF

Fully implemented in the same file as `_render_deficiency_report`. Emitted when `gate.document_type` is not `"CERTIFICATE"`. Fields include: title "Evidence Admissibility Report / Section 106B … Requirements Not Yet Met", a plain-language summary, a 3-column conditions table (Ref. / [Met|NOT MET] / remedy, capped at 55 and 120 chars), "Other Gaps" bullets from `analysis_summary.missing_information` and `strengthen_checklist` (up to 10 each), the same evidence schedule, and an optional disclaimer.

---

### B9. Test Bundle Results

**File:** `eval/s106b_test_bundles.json`  
**Script:** `eval/run_b9.py`  
**Results:** `eval/logs/s106b_bundles_result.txt`

All 10 bundles pass:

| Bundle | Expected | Actual | Failing condition(s) | Match |
|---|---|---|---|---|
| TB-001 | certificate_ready | certificate_ready | (none) | Y |
| TB-002 | weak | weak | 4a | Y |
| TB-003 | needs_strengthening | needs_strengthening | 2a | Y |
| TB-004 | needs_strengthening | needs_strengthening | 2d | Y |
| TB-005 | needs_strengthening | needs_strengthening | 2c | Y |
| TB-006 | certificate_ready | certificate_ready | (none) | Y |
| TB-007 | weak | weak | INTEGRITY | Y |
| TB-008 | weak | weak | INTEGRITY | Y |
| TB-009 | needs_strengthening | needs_strengthening | 4d | Y |
| TB-010 | needs_strengthening | needs_strengthening | 2d | Y |

**10/10 pass. No failures.**

One import hurdle was encountered during the eval run: `SCCVision/__init__.py` eagerly imports `cv2` (OpenCV), which is not installed in the venv. The script stubs out the top-level `SCCVision` module in `sys.modules` before importing so the self-contained `s106b` subpackage loads without triggering the cv2 dependency. This stubbing pattern would need to be replaced with a proper `cv2` installation or an optional import guard before the eval script can be used in the same process as the full application.

---

### B10. Limitation Period

**No coded rule checks a claim date against the Limitation of Actions Act (Cap. 22) anywhere in the codebase.**

Evidence of the gap:

- `backend/RETRIEVAL_PIPELINE.md:90` — prose annotation: `"temporal": "3-year limitation period under Cap 22"`. This is documentation, not executable logic.
- `backend/validate_knowledge.py:57–69` — a test fixture asserting retrieval quality for the Limitation of Actions Act. It tests whether the vector store surfaces the statute; it does not evaluate whether a claim is time-barred.
- `backend/Qclassifier.py:180,197,276` — prompt-engineering text instructing the LLM to be aware of Cap 22 timelines. This is a hint to the language model, not a deterministic rule.

**`compute_date_checks()` does five things only** (`backend/SCCVision/analyst/dates.py` lines ~182–297):

1. Collects conversation/payment/delivery dates from `user_provided_facts`.
2. Extracts and resolves dates from chat-screenshot OCR chips and message text.
3. Extracts dates from M-Pesa confirmation OCR.
4. Flags dates that fall in the future relative to `analysis_date`.
5. Infers a `performance_due` date and labels it `not_yet_due / due_today / overdue`.

It never computes `analysis_date - occurrence_date`, never compares a delta to any statutory limitation period (6 years for contract under s. 4(1)(a), 3 years for tort under s. 4(2), etc.), and never emits a `time_barred` or `limitation_warning` field. The limitation-period awareness that does exist is entirely LLM-prompt text in `Qclassifier.py` — not executable logic.

---

### B11. Evidence List

**Source: neither backend nor Firestore.**

`listEvidence()` (`frontend/app/lib/evidenceApi.ts` lines 4–7) is a stub that unconditionally returns an empty array:

```typescript
export async function listEvidence(): Promise<never[]> {
  // TODO(backend): GET /api/evidence — returns EvidenceRecord[]
  return [];
}
```

No network request is ever made. No backend route for `GET /api/evidence` exists. The `/litigant/evidence` page always renders the empty-state UI.

---

### B12. Accepted Formats

There is a frontend/backend mismatch in accepted file formats.

**Backend** (`backend/SCCVision/router.py`):

```python
ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
```

Applies to both `POST /sccvision/extract` and `POST /sccvision/analyze`. `.pdf`, `.heic`, and `.eml` return HTTP 400 "Unsupported file type".

**Frontend** (`frontend/app/lib/evidenceConfig.ts` + `EvidenceDropzone`):

```
ACCEPTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".heic", ".eml"]
MAX_FILE_SIZE_BYTES  = 15 * 1024 * 1024  (15 MB)
MAX_FILES_PER_BATCH  = 6
```

**Mismatch table:**

| Format | Frontend dropzone | Backend OCR | End-to-end functional |
|---|---|---|---|
| `.png`, `.jpg`, `.jpeg`, `.webp` | Accepted | Accepted | Yes |
| `.heic` | Accepted | **Rejected (400)** | No |
| `.pdf` | Accepted | **Rejected (400)** | No |
| `.eml` | Accepted | **Rejected (400)** | No |

The frontend promises users that `.pdf`, `.heic`, and `.eml` are valid uploads. The backend rejects all three. The `/api/evidence/upload` route that would handle `.pdf`/`.eml` pre-processing is marked TODO and not implemented. The size limit is also inconsistent: the frontend advertises 15 MB, the backend enforces 10 MB.

---

## Part C: Advocate Module

### C1. Unauthenticated Routes

`require_auth` is defined in `backend/auth.py` and is applied to routes in `backend/SCCVision/router.py`. It is **never imported into `backend/app.py`**. All routes in `app.py` carry no authentication decorator.

The following eight routes in `app.py` are publicly reachable with no token required:

| Route | Line | Notes |
|---|---|---|
| `GET /api/digest/constitution/status` | 852 | — |
| `GET /api/case-pdf` | 957 | — |
| `GET /api/chunks/<chunk_id>/location` | 1073 | — |
| `GET /api/documents/<path:doc_key>/chunks` | 1106 | — |
| `GET /api/cases/<path:case_identifier>/all-chunks` | 1131 | — |
| `GET /api/statutes/<path:source_document>/all-chunks` | 1164 | — |
| `GET /api/rule-pdf` | 1198 | — |
| `GET /api/case-laws/file/<path:filename>` | 1897 | Docstring explicitly acknowledges "no auth — internal tool only" |

The last route serves raw case-law PDF files from `backend/SCCCases/` with no authentication. Any party with network access to the Flask server can download any indexed case PDF without credentials.

---

### C2. History Route Paths

History routes are mounted at `/api/conversations`, **not** `/api/history/conversations`.

**Blueprint registration** (`backend/app.py` line 42):

```python
app.register_blueprint(history_bp)
```

No `url_prefix` is passed.

**Blueprint definition** (`backend/history_routes.py` line 22):

```python
history_bp = Blueprint("history", __name__)
```

No `url_prefix`.

**Route decorators** (`history_routes.py`):

```
@history_bp.route("/api/conversations", methods=["GET"])              # line 43
@history_bp.route("/api/conversations/<conv_id>", methods=["GET"])    # line 62
@history_bp.route("/api/conversations/<conv_id>", methods=["PATCH"])  # line 93
@history_bp.route("/api/conversations/<conv_id>", methods=["DELETE"]) # line 129
@history_bp.route("/api/conversations/<conv_id>/regenerate-title", methods=["POST"])  # line 148
```

There is no `/api/history/` prefix anywhere in the backend.

---

### C3. Size Guard

**File:** `backend/history_store.py` lines 388+

```python
_MAX_DOC_BYTES = 900_000  # guard against the 1 MB Firestore document limit

def _size_guard(doc: dict) -> dict:
    """
    If the document would exceed _MAX_DOC_BYTES when serialised, truncate
    answer_markdown and source snippets and set truncated=True.
    """
    try:
        raw = json.dumps(doc, default=str).encode("utf-8")
    except Exception:
        return doc

    if len(raw) <= _MAX_DOC_BYTES:
        return doc

    overage = len(raw) - _MAX_DOC_BYTES
    doc = dict(doc)  # shallow copy so callers are not mutated
    doc["truncated"] = True

    if "answer_markdown" in doc and doc["answer_markdown"]:
        max_ans = max(100, len(doc["answer_markdown"]) - overage - 1000)
        doc["answer_markdown"] = doc["answer_markdown"][:max_ans]

    for src in doc.get("sources", []):
        if src.get("snippet"):
            src["snippet"] = src["snippet"][:100]

    return doc
```

**Byte limit:** `_MAX_DOC_BYTES = 900,000` bytes (900 KB), set 100 KB below the Firestore 1 MB hard document limit. When exceeded: `answer_markdown` is trimmed by `overage + 1,000` bytes (minimum 100 chars preserved), each source snippet is capped at 100 chars, and `truncated = True` is set on the document.

---

### C4. LOW_CONFIDENCE_THRESHOLD

**Value:** `0.4`  
**File:** `backend/Retriever.py` line 52  
**Comment:** "Cross-encoder score (sigmoid-normalised) below which we warn about low confidence"

The score compared against this threshold is the CE sigmoid output: `ce_sigmoid[i] = _sigmoid_c(float(raw_ce_score))`, a logistic sigmoid applied to the raw cross-encoder logit, mapping to `(0, 1)`. It is set at line 743: `'final_score': ce_sigmoid[i]`.

**Comparison** (`Retriever.py` line 781):

```python
if top_parents and top_parents[0][1]['best_score'] < LOW_CONFIDENCE_THRESHOLD:
    output['warning'] = "Low confidence results — CE score below threshold; consider refining query"
```

**Effect: warning only, no filtering.** The check runs after `format_output` has assembled the full results. When the top-ranked parent's best CE-sigmoid score is below 0.4, a `warning` key is added to the output dict. Results are returned unchanged — nothing is removed, truncated, or re-ranked. Whether the warning is surfaced to the user depends on how the frontend consumes that field.

---

### C5. Retrieval Output

**Statute chunks reaching the LLM prompt** (`app.py` line 1484):

```python
for i, rule in enumerate(statutory_provisions[:5], 1):
    ...
    for j, sch in enumerate(stat_chunks, 1):   # no inner slice — all chunks
        chunk_lines.append(f"  [{chunk_alias}] s.{sec_num} {sec_hdr}\n  {sch.get('text','')[:600]}")
```

- At most **5 statute articles** (outer slice `[:5]`)
- **All chunks** per article (no inner slice) — pool capped at 35 fused chunk IDs before grouping in `RulesRetriever.py` line 620
- Text truncated to **600 characters** per chunk

**Case chunks reaching the LLM prompt** (`app.py` lines 1513/1519):

```python
for i, case in enumerate(case_law_results[:3], 1):
    ...
    case_chunks = case.get("chunks", [])[:3]
    for j, ch in enumerate(case_chunks, 1):
        chunk_lines.append(f"  [{chunk_alias}] {ch.get('text','')[:500]}")
```

- At most **3 cases** (outer slice `[:3]`)
- At most **3 chunks** per case (inner slice `[:3]`)
- Maximum **9 case chunks** total
- Text truncated to **500 characters** per chunk

Inside `Retriever.py`: `MAX_CHUNKS_PER_COMPONENT = 1` (line 50) — only 1 chunk per FIRAC component is kept during case assembly. `app.py` then further caps at 3 via `[:3]`.

| Source | Objects | Chunks per object | Max to LLM | Char limit |
|---|---|---|---|---|
| Statutes | `[:5]` | all (pool capped at 35) | unbounded per article | 600 |
| Cases | `[:3]` | `[:3]` | 9 total | 500 |

---

### C6. Reference Documents

#### UI to attach a document to a query

There is a "probe mode" UI, but it is not a file-upload flow. It works with documents the retriever has already returned in the current session. The user clicks a "Probe" button on a retrieved document card, which toggles that document into a client-side `pinnedDocs` Map. No file is uploaded; only the document's existing database key is captured.

#### Frontend: how `pinned_docs` is built and sent

**Type definition and state** (`ChatView.tsx` lines 76, 199):

```typescript
interface PinnedDocEntry { title: string; type: "case" | "statute"; dbKey: string; }
const [pinnedDocs, setPinnedDocs] = useState<Map<string, PinnedDocEntry>>(new Map());
```

**Toggle handler** (`ChatView.tsx` lines 379–390):

```typescript
const handleToggleDocPin = useCallback((doc: { key: string; title: string; kind: string; dbKey: string }) => {
  setPinnedDocs(prev => {
    const next = new Map(prev);
    if (next.has(doc.key)) next.delete(doc.key);
    else next.set(doc.key, {
      title: doc.title,
      type: doc.kind === "JUDGMENT" ? "case" : "statute",
      dbKey: doc.dbKey,
    });
    return next;
  });
}, []);
```

**Serialised into the query payload** (`ChatView.tsx` lines 482–485):

```typescript
const pinned_docs = [...pinnedDocs.values()].map(d => ({ type: d.type, id: d.dbKey }));
body: JSON.stringify({ query, ...(pinned_docs.length > 0 ? { pinned_docs } : {}) }),
```

Shape: `{ query: "...", pinned_docs: [{ type: "case"|"statute", id: "<dbKey>" }, ...] }`

#### Backend: how `pinned_docs` is consumed

`backend/app.py` around line 1272:

```python
raw_pinned = data.get("pinned_docs") or []
pinned_case_ids    = [pd["id"] for pd in raw_pinned if pd.get("type") == "case"]    or None
pinned_statute_ids = [pd["id"] for pd in raw_pinned if pd.get("type") == "statute"] or None
```

Passed into `Qclassifier` and `retrieve_relevant_rules` as whitelist filters — retrieval is restricted to only those specific Chroma document IDs.

#### What `POST /api/case-laws/upload` does

Requires `advocate` or `admin` role. It does two things:

1. **Saves the file to disk** at `backend/SCCCases/`. Comment: "# Persist locally for RAG pipeline." The file is **not** automatically chunked, embedded, or added to the Chroma vector store — it is staged for the pipeline, not immediately queryable.
2. **Writes metadata to Firestore** (`CaseLaws` collection): `title`, `fileName`, `fileType`, `localPath`, `uploadedBy`, `uploadedAt`, `fileSizeBytes`.

A separate offline step (e.g. `build_case_chunks.py`) is required to push the file into the vector store before it becomes retrievable.

---

### C7. Knowledge Base Contents

**`cases_meta.json` location:** `data/processed/cases/cases_meta.json`

The `case_law` collection in `backend/CaseLawDB/` contains **9 cases, 78 total chunks**:

| case_identifier | Chunks | Case name |
|---|---|---|
| 2001KECA154(KLR) | 7 | Express (Kenya) Limited v Manju Patel |
| 2010KECA346(KLR) | 31 | Safaricom Limited v Ocean View Beach Hotel Limited & 2 others |
| 2014KEHC6468(KLR) | 3 | Nonny Gathoni Njenga & another v Catherine Masitsa & another |
| 2020KEHC8905(KLR) | 5 | Peter Ngethe Ngari t/a PNN Funeral Services v Standard Group Limited PLC & another |
| 2024KEHC15763(KLR) | 10 | Ogembo v Yongo (Civil Appeal E200of2023) |
| 2024KEHC17038(KLR) | 8 | Obado v Anayo (Civil Suit E005of2022) |
| 2026SCC235(KLR) | 4 | New Titanic Jacaranda General Spares v Omil & another |
| 2026SCC241(KLR) | 6 | Ndeti v Kaguchwa (Civil Case E075of2026) |
| 2026SCC242(KLR) | 4 | Umoja Magharibi (K) Ltd v Lubano (Commercial Case E091of2026) |

**Two cases excluded from the collection:**

Both have `"exclude": true` in `cases_meta.json`. `build_case_chunks.py` lines 321–322 skip any entry where `entry.get("exclude", False)` is true.

- **2011KEHC1481(KLR)** — *Republic v Barisa Wayu Mataguda*. A 2011 High Court ruling on admissibility of CCTV footage as electronic evidence under Evidence Act s.106B. The CD was ruled inadmissible for lack of the required s.106B certificate. This case is the foundational authority for the FR04 feature being built on this branch. Embedding it in the general retrieval pool would risk surfacing it out of context or conflating the retrieval path with the dedicated certificate-logic path. `_note`: "Kept excluded per instruction."
- **2021KEELC220(KLR)** — *Republic v Director, Land Administration... Ex Parte Roysa Community Development Society Ltd*. A 2021 Environment and Land Court judicial review case (mandamus for land title processing); application dismissed. Unrelated to the system's primary SCC/digital-evidence domain. `_note`: "Kept excluded per instruction."

Both PDFs are physically present in `backend/SCCCases/`. Exclusion is a deliberate policy decision recorded in `cases_meta.json`.

---

### C8. FIRAC Values

**`firac_component` in `CaseLawDB`:** populated on **0 of 78 chunks**.

The field `firac_component` exists as a key on every chunk's metadata, but every stored value is an empty string `""`. All 78 chunks were built via `build_case_chunks.py`'s page-level splitter (which seeds the field as `""`), not via `ingester.py`'s FIRAC-aware path that would extract and write component values.

**Live-route collection wiring:**

| Collection / DB | Queried by a live route? | Evidence |
|---|---|---|
| `CaseLawDB` / `case_law` | YES | `Retriever.py` imports `CASELAW_DB_DIR`; `app.py` imports and calls `Retriever`; `app.py` opens `CaseLawDB` directly at lines 982, 1140 |
| `StatutesDB` / `statutes` | YES | `RulesRetriever.py` imports `STATUTES_DB_DIR`; `app.py` calls `RulesRetriever`; `app.py` opens `StatutesDB` directly at line 1173 |
| `LegalSummariesDB` / `legal_summaries` | NO | Not referenced in `app.py`, `Retriever.py`, or `RulesRetriever.py`. `config.py` comment states it was replaced by `CaseLawDB`. |
| `ChunkDB` | NO | Not referenced in any of the three files. |
| `BlackLawDictionaryDB` / `black_law_dictionary` | NO | Not referenced in any of the three files. |

`LegalSummariesDB`, `ChunkDB`, and `BlackLawDictionaryDB` are dead/orphaned — their directories exist on disk and are tracked in Git, but no live route queries them.

---

### C9. Chunking Constants

**File:** `backend/constitutionlogic/constitutionbreaker.py` lines 23–25

| Constant | Value |
|---|---|
| `CHUNK_SIZE` | 1000 characters |
| `CHUNK_OVERLAP` | 300 characters |
| `MIN_CHUNK_SIZE` | 150 characters |

`app.py` imports `CHUNK_SIZE` and `CHUNK_OVERLAP` directly from `constitutionbreaker.py` and passes them through to the ingestion stats payload — no override values.

**How the 374 Constitution chunks were produced:**

Confirmed by `backend/constitutionlogic/Constsection/constitution_embeddable_chunks.json`:

```
total_articles:          264
total_embeddable_chunks: 374
chunk_size:              1000
chunk_overlap:           300
source_document:         TheConstitutionOfKenya.pdf
```

264 articles were extracted. Short articles (text ≤ 1,000 chars) produce one chunk each; long articles are split with a sliding window of size 1,000 and stride 700 (1,000 − 300 overlap). The last chunk of a split is merged into its predecessor if it falls below `MIN_CHUNK_SIZE = 150` chars. The 110 extra chunks (374 − 264) come from articles that exceeded 1,000 characters. The `MasterRulesDB` export (which includes the Evidence Act and other rules alongside Constitution chunks) totals 684 chunks across 280 unique articles — a larger combined collection, not the 374 Constitution-only figure.

---

### C10. Answer Schema

**System prompt location:** `backend/app.py` lines 1545–1580

The LLM is instructed to emit exactly this JSON object (all fields required, empty string if not applicable):

```json
{
  "source_review": [
    {"id":"S1","relevant":true,"reason":"one sentence","key_points":["verbatim text"]}
  ],
  "premise_check": "Does the question rely on provisions not in the sources? List them.",
  "short_answer": "2-3 sentences citing [S#]/[C#]. If insufficient: say so.",
  "applicable_law": "Verbatim or near-verbatim text from relevant [S#] sources.",
  "case_law_discussion": "Holdings as stated in [C#] sources — parties, court, outcome, quoted reasoning.",
  "application": "How [S#][C#] provisions apply. No claims beyond source text.",
  "jurisdiction_and_procedure": "From [S#][C#] only. Empty string if not in sources.",
  "gaps": "What the question needed that sources do not cover.",
  "follow_up_questions": ["up to 3 questions the corpus could answer"],
  "citations_used": ["S1.1 — Document s.N chunk", "C1.2 — Case name chunk"]
}
```

This schema has no FACTS/ISSUES/RULES/APPLICATION/CONCLUSION keys. It is not FIRAC-shaped.

**Does the frontend render FIRAC tiles?**

`parseFirac` (`ChatView.tsx` lines 105–126) looks for Markdown headings or bold markers matching the five FIRAC labels. It returns `null` if fewer than 2 such markers are found. Because the backend schema produces none of those headings, `parseFirac` will always return `null` for a normal response from this endpoint.

The ternary at line 1369:

```tsx
{firac ? (
  /* FIRAC tiles with FiracTile letter badges */
) : (
  parseMdFirac(msg.aiResponse || "", activeAlias, onCitationClick, buildCitLabel)
)}
```

**Conclusion:** `FiracTile` components are dead code for this endpoint. The FIRAC tile branch would only activate if the LLM returned free-form Markdown with explicit FACTS/ISSUES/RULES/APPLICATION/CONCLUSION section headings, which the current system prompt explicitly forbids.

---

### C11. Hallucination Log

**File:** `backend/logs/hallucinations.jsonl`  
**Total entries:** 72  
**Distinct queries:** 10  
**Date range:** 2026-10-28 through 2026-11-09 (approximately 12 days)

Distinct `unsupported_claim` values by frequency:

| Claim | Count |
|---|---|
| SECTION 1 | 31 |
| SECTION 106B | 11 |
| SECTION 106B(4) | 7 |
| SECTION 2 | 4 |
| SECTION 5 | 3 |
| SECTION 4 | 3 |
| SECTION 3 | 3 |
| SECTION 106 | 3 |
| KLR 2026SCC242(KLR) | 2 |
| KLR 2020KEHC8905(KLR) | 2 |
| SECTION 2(4) | 1 |
| CASES Kenya) Limited v Manju Patel | 1 |
| CAP 80 | 1 |

**Observations:**

- "SECTION 1" is the most frequent unsupported claim (31/72, 43%). This pattern is consistent with the section-number regex over-matching on the word "section" in boilerplate source text — a false positive in the hallucination detector, not necessarily a true hallucination.
- "SECTION 106B" and "SECTION 106B(4)" together account for 18 entries. The FR04 feature was developed in direct response to these observed hallucinations around the Section 106B certificate requirement.
- Two fabricated case citations appear: `2026SCC242(KLR)` and `2020KEHC8905(KLR)`. Both are cases that exist in the vector store; the hallucination detector may be over-flagging citations that are genuinely grounded but whose citation format does not match the verifier's regex patterns.

---

## Part D: Measurements

### D1. Backend Test Run

**Run date:** 2026-11-17  
**Total collected:** 198  
**Passed:** 193  
**Failed:** 1  
**Skipped:** 4  
**Run time:** 28.78 s  
**JUnit XML:** `eval/logs/pytest_final.xml`

#### Failure

**File:** `tests/test_sccvision_analyst.py`  
**Test:** `test_deterministic_followups_missing_date_and_payment`  
**Assertion:** expected `'upload_mpesa_screenshot:K:payment'` to be in the follow-up IDs list, but only `['provide_date:E:e-chat-1']` was returned.

This test asserts that when both a date and a payment are missing, the system generates follow-up requests for both. The actual output only generates the date follow-up, indicating the payment follow-up rule either does not fire or its condition is not met by the test fixture.

#### FR04 test suites — all pass

**`test_s106b_gate.py` — 33 tests, all pass:**

| # | Test name |
|---|---|
| 1 | test_returns_certificate |
| 2 | test_all_eight_conditions_present |
| 3 | test_contract_verdict_does_not_decide_admissibility |
| 4 | test_no_evidence_items_fails_4a |
| 5 | test_missing_server_sha256_fails_4a |
| 6 | test_hash_mismatch_fails_4a_and_outcome_weak |
| 7 | test_production_method_other_fails_4a |
| 8 | test_4b_always_met_when_make_model_given |
| 9 | test_no_lawful_control_fails_2a |
| 10 | test_no_regular_use_fails_2a |
| 11 | test_remedy_present |
| 12 | test_regular_feed_false_fails_2b |
| 13 | test_malfunction_affecting_record_fails_2c |
| 14 | test_malfunction_not_affecting_record_passes_2c |
| 15 | test_forwarded_copy_fails_2d |
| 16 | test_messages_not_on_device_fails_2d |
| 17 | test_mpesa_no_codes_fails_2d |
| 18 | test_not_owner_fails_4d |
| 19 | test_declaration_not_confirmed_fails_4d |
| 20 | test_ungrounded_quotes_fails_integrity |
| 21 | test_mpesa_inconsistent_fails_integrity |
| 22 | test_weak_conversation_integrity_fails |
| 23 | test_only_questionnaire_unmet_gives_needs_strengthening |
| 24 | test_integrity_unmet_gives_weak |
| 25 | test_4a_unmet_gives_weak |
| 26 | test_empty_device_make_raises |
| 27 | test_whitespace_device_make_raises |
| 28 | test_bad_last4_raises |
| 29 | test_bad_last4_non_digits_raises |
| 30 | test_none_last4_ok |
| 31 | test_malfunction_without_details_raises |
| 32 | test_malfunction_without_affects_raises |
| 33 | test_empty_declarant_name_raises |

**`test_s106b_document.py` — 31 tests, all pass:**

| # | Test name |
|---|---|
| 1 | test_certificate_starts_with_pdf_magic |
| 2 | test_deficiency_report_starts_with_pdf_magic |
| 3 | test_certificate_returns_bytes |
| 4 | test_deficiency_returns_bytes |
| 5 | test_cert_number_in_text |
| 6 | test_part_a_heading_present |
| 7 | test_part_b_heading_present |
| 8 | test_part_c_heading_present |
| 9 | test_part_d_heading_present |
| 10 | test_draft_watermark_present |
| 11 | test_section_106b_reference_present |
| 12 | test_device_make_in_cert |
| 13 | test_evidence_filename_in_cert |
| 14 | test_cert_number_in_deficiency |
| 15 | test_requirements_not_yet_met_heading |
| 16 | test_evidence_schedule_in_deficiency |
| 17 | test_disclaimer_in_deficiency |
| 18 | test_full_phone_not_in_certificate |
| 19 | test_no_phone_number_when_not_provided |
| 20 | test_full_phone_not_in_deficiency |
| 21 | test_certificate_bytes_identical_on_second_render |
| 22 | test_deficiency_bytes_identical_on_second_render |
| 23 | test_different_dates_give_different_bytes |
| 24 | test_emoji_in_device_make_does_not_crash |
| 25 | test_emoji_in_analysis_summary_does_not_crash |
| 26 | test_smart_quotes_in_summary_do_not_crash |
| 27 | test_empty_evidence_list_does_not_crash |
| 28 | test_empty_conditions_list_does_not_crash |
| 29 | test_missing_analysis_summary_does_not_crash |
| 30 | test_missing_device_facts_does_not_crash |
| 31 | test_malfunction_cert_uses_alternate_2c_statement |

**`test_s106b_routes.py` — 20 tests, all pass.**

---

### D2. Emulator Tests

**Firebase CLI:** Installed at `/home/gift/.npm-global/bin/firebase`, version 15.9.0.

**Firestore emulator:** Not running at audit time. Ports 8080 and 4000 unreachable. No emulator process detected.

#### Firestore Security Rules tests (`firestore-tests/firestore.test.ts`)

302 lines, 20 `test()` calls across 10 `describe` blocks covering:

- Own-profile read / cross-user read denial / unauthenticated denial
- Create advocate profile (password + google authProvider)
- Create litigant profile
- Denied create cases: `type=admin`, `advocateVerified=true`, solo write without usernames entry, unauthenticated, writing to another uid
- Username claim in same batch (allowed / duplicate rejected)
- Update own username (batch)
- Update frozen fields (`type`, `advocateVerified`)
- Usernames collection reads (unauthenticated availability check)

Requires Firestore emulator. Command to run:

```bash
cd /home/gift/Retreival-Augmented-Generation-AI
firebase emulators:exec --only firestore --project legalai-rules-test \
  "cd firestore-tests && npm run test:ci"
```

**Status: Not run — emulator offline.**

#### Python history store integration tests (`backend/tests/test_history.py::TestEmulatorIntegration`)

Decorated with:

```python
@pytest.mark.skipif(not EMULATOR, reason="Firestore emulator not running")
```

where `EMULATOR = os.environ.get("FIRESTORE_EMULATOR_HOST")`.

Two tests: `test_full_save_and_read_cycle` and `test_idempotency_with_emulator`.

Command to run:

```bash
FIRESTORE_EMULATOR_HOST=localhost:8080 PYTHONPATH=. pytest backend/tests/test_history.py::TestEmulatorIntegration -v
```

**Status: Skipped — emulator offline.**

---

### D3. Frontend Checks

**Build:** PASS. All 20 static/dynamic routes compiled and generated successfully. No build errors.

**Lint — project-wide:** 41 problems (26 errors, 15 warnings). Errors are dominated by `@typescript-eslint/no-explicit-any` (24 of 26 errors) and two `@typescript-eslint/no-unused-vars`. Errors are entirely in files not touched by this branch (`ChatView.tsx`, `PdfViewer.tsx`, `historyApi.ts`, `firacChecker/page.tsx`, `repository/caselaws/page.tsx`).

**Lint — changed files on this branch only: 3 warnings, 0 errors.**

All three warnings are in `SCCVisionTest.tsx`:

| Line | Issue |
|---|---|
| 430:31 | `_ai` assigned but never used (`@typescript-eslint/no-unused-vars`) |
| 430:50 | `_v` assigned but never used (`@typescript-eslint/no-unused-vars`) |
| 1413:68 | `i` defined but never used (`@typescript-eslint/no-unused-vars`) |

The FR04 branch introduces no new errors — only 3 pre-existing-pattern warnings in `SCCVisionTest.tsx`.

**Changed TypeScript/TSX files on this branch:**

```
frontend/app/components/sccvision/DeviceQuestionnaire.tsx
frontend/app/components/sccvision/SCCVisionTest.tsx
frontend/app/lib/evidenceApi.ts
frontend/app/lib/evidenceTypes.ts
frontend/app/litigant/analysis/page.tsx
frontend/app/litigant/certificates/page.tsx
```

---

### D4. Latency

**Script:** `eval/measure_latency.sh` — present and complete. Starts Flask, waits for the health endpoint, hits four endpoints 10 times each, computes median and p95 latency per endpoint using an inline Python3 heredoc.

**Endpoints measured:**

| Label | Method | Endpoint | Auth required |
|---|---|---|---|
| GET /api/health | GET | `/api/health` | No |
| POST /api/qretrieve | POST | `/api/qretrieve` | Yes — advocate or admin role |
| POST /api/generate-legal-answer | POST | `/api/generate-legal-answer` | Yes — advocate or admin role |
| sccvision health proxy | GET | `/sccvision/health` | No |

**Three required prerequisites:**

1. `GROQ_API_KEY` — in `backend/.env`. Load before running.
2. `FIREBASE_SERVICE_ACCOUNT_PATH` — path to the Firebase service account JSON. Downloaded from Firebase Console > Project Settings > Service Accounts > Generate new private key.
3. `AUTH_TOKEN` — a live Firebase ID token for a user with `type: "advocate"` or `type: "admin"` in Firestore. Firebase ID tokens expire after 1 hour.

**Commands:**

```bash
# Step 1 — Get a fresh ID token
curl -s -X POST \
  "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=<FIREBASE_WEB_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"email":"<user@example.com>","password":"<password>","returnSecureToken":true}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['idToken'])"

# Step 2 — Run the script
cd /home/gift/Retreival-Augmented-Generation-AI
set -a && source backend/.env && set +a
export FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/serviceAccountKey.json
export AUTH_TOKEN="<idToken from step 1>"
bash eval/measure_latency.sh
```

The Firebase Web API key (`NEXT_PUBLIC_FIREBASE_API_KEY`) is in `frontend/.env.example` or `frontend/.env.local`.

**The latency run itself was not executed** at the time of this audit — the backend was not confirmed running and no service account key was available. Actual median/p95 figures are NOT DETERMINABLE from this audit.

---

### D5. Memory

**Host memory at audit time:**

```
              total        used        free      shared  buff/cache   available
Mem:           15Gi        11Gi       424Mi       1.1Gi       5.2Gi       4.1Gi
Swap:          15Gi       7.5Gi       8.4Gi
```

**Python and Node.js processes:**

| Process | RSS | %MEM |
|---|---|---|
| python (Flask/Chroma backend) | ~242 MiB | 1.5% |
| node (×3, Next.js dev server) | ~15 MiB each | 0.0% each |
| python (secondary) | ~13 MiB | 0.0% |

**Assessment:** 7.5 GiB of swap in active use indicates the system is under moderate-to-high memory pressure. No single process is individually runaway. The aggregate load (11 GiB used + 7.5 GiB active swap) is consistent with the Flask backend holding ChromaDB collections and sentence-transformer embeddings in memory simultaneously with other system processes. The ChromaDB embedding/retrieval workload is the most probable contributor to memory pressure. This may cause latency spikes under concurrent load or during re-embedding runs.

---

### D6. Retrieval Baseline

**File:** `eval/retrieval_queries.csv` — PRESENT  
**Queries:** 25 (Q001–Q025)  
**Columns:** `query_id`, `query`, `expected_statute_sections`, `expected_cases`, `verified`  
**Current state:** All 25 rows are marked `UNVERIFIED`. The `expected_cases` column is blank for all 25 queries.

**Statutes indexed in `StatutesDB`** (from `data/processed/statutes/manifest.json`):

Small Claims Court Act, Small Claims Court (Procedure) Rules, Civil Procedure Act, Evidence Act, Limitation of Actions Act, Judicature Act, Employment Act, Consumer Protection Act, Sale of Goods Act, Bills of Exchange Act, Kenya Information and Communications Act, Stamp Duty Act, Landlord and Tenant (Shops, Hotels and Catering Establishments) Act, Law of Contract Act.

The Constitution of Kenya (2010) is indexed separately in `MasterRulesDB` via the constitution fallback path in `RulesRetriever`.

**Notable gaps in the retrieval baseline:**

- Q010 (insurer subrogation) references the Insurance Act, which is **not** in the manifest. That query will produce a statute miss unless the Insurance Act is added to `StatutesDB`.
- The `expected_cases` column being blank for all 25 rows means case-law retrieval accuracy cannot be evaluated until case citation ground-truth is populated.
- The five s.106B-related queries (Q006, Q009, Q015, Q018, Q024) are the highest-priority rows to verify first given the FR04 branch scope.

**Retrieval pipeline summary:**

Both retrievers use `intfloat/e5-base-v2` embeddings and `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking.

- **`RulesRetriever`** (statutes): BM25 + e5-base-v2 vector search → RRF fusion (k=60, top-30) → exact-section injection when Act is named → cross-encoder rerank → schedule penalty → group by article → return top-5.
- **`Retriever`** (case law): BM25 + e5-base-v2 vector search → RRF fusion → cross-encoder rerank → group by `case_identifier` → return top-3.

**Eval stub written to:** `eval/run_retrieval_eval.py`. Run with:

```bash
cd /home/gift/Retreival-Augmented-Generation-AI
backend/venv/bin/python eval/run_retrieval_eval.py --limit 5
```

Actual Hit@3 and Precision@3 figures are NOT DETERMINABLE until the `expected_statute_sections` column is verified against live retrieval output.

---

### D7. Citation Grounding Rate

**What exists:** A two-stage citation grounding pipeline is fully implemented.

**Stage 1 — Prompt-level** (`app.py` lines 1544–1596): The system prompt prohibits the LLM from citing any section number, article, case name, or procedure not present verbatim or near-verbatim in the provided source blocks. Every legal statement must use chunk-level aliases `[S1.1]`, `[C1.2]`, etc. If a provision is not in the sources, the model must say so and must not describe it from memory.

**Stage 2 — Post-generation verification** (`backend/citation_verifier.py`): `verify_and_clean()` runs after every LLM call:

1. Extracts every section/article/rule/Cap/case/KLR reference from the generated answer using six compiled regex patterns.
2. Builds a valid-reference set from the actual source objects in the prompt.
3. Checks each extracted citation against that set.
4. Logs any unsupported citation to `backend/logs/hallucinations.jsonl`.
5. Attempts one automatic regeneration with a follow-up message listing bad citations.
6. If the regen still contains bad citations, `_strip_sentences()` surgically removes every sentence containing an unsupported reference.

**Current hallucination log:** 72 entries spanning 2026-10-28 through 2026-11-09, from live production queries. The dominant pattern (`SECTION 1`, 31 entries) is consistent with the section-number regex over-matching on boilerplate text rather than a true hallucination.

**25-query grounding rate:** NOT DETERMINABLE from this audit. `run_synthesis_tests()` in `validate_knowledge.py` requires a live Groq API key (it explicitly checks `if not real_key.startswith("gsk_"): print("[SKIP]")` and exits early). No offline mock is wired. The grounding rate figure:

```
grounding_rate = (queries where unverified_claims_removed == []) / total_queries
```

is only observable by running `verify_and_clean()` against real LLM outputs and reading the delta from `backend/logs/hallucinations.jsonl`. The commands to execute are documented in D4.

---

## Part E: Defect History

### E1. Specific Bug Fixes

#### (a) fpdf2 TypeError fixed with `_parse_created_at()` helper

**Commit:** `9393ec1` — "feat(fr04): Phase 3 — deterministic fpdf2 PDF document generator"  
**File:line:** `backend/SCCVision/s106b/document.py:159`

The commit introduces `_parse_created_at(created_at: str) -> datetime.datetime` which safely parses a date string or Firestore Timestamp object into a UTC `datetime` before passing it to `pdf.set_creation_date()`. Without this helper, passing a raw string or Timestamp object to `set_creation_date()` would raise a `TypeError`. This was a design-time prevention incorporated at authoring; no prior broken commit exists.

#### (b) Non-Latin-1 ellipsis replaced with ASCII in PDF

**Commit:** `9393ec1` — same Phase 3 commit  
**File:lines:** `backend/SCCVision/s106b/document.py:44–48` and `:143`

`_safe()` at lines 44–48 sanitises all text via `text.encode("latin-1", errors="replace").decode("latin-1")` to prevent any non-Latin-1 character (including Unicode ellipsis U+2026 `…`) from crashing the fpdf2 renderer. SHA-256 truncation at line 143 uses three ASCII full stops `"..."` rather than the Unicode ellipsis. Commit message: "Latin-1 safe: all text sanitised via `_safe()`; emoji and smart quotes replaced with '?' rather than crashing the renderer."

#### (c) Pydantic v2 `ctx` field causing 500 → 422

**Commit:** `b70caea` — "feat(fr04): Phase 4 — certificate store, API routes, Firestore rules"  
**File:line:** `backend/SCCVision/s106b/routes.py:52–58`

Pydantic v2's `ctx` field may contain a live Python exception object that is not JSON-serializable. Without the fix, `ValidationError` during `DeviceFacts` validation causes `jsonify()` to raise `TypeError: Object of type ... is not JSON serializable`, propagating as a 500. The fix strips each error to `{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}` and returns HTTP 422. Documented in `FR04_IMPLEMENTATION_REPORT.md:88`.

#### (d) `focusRingColor` TypeScript build error

**NOT DETERMINABLE.** None of the seven commits on `feat/fr04-s106b-certificate` reference `focusRingColor`. Grep across the entire frontend tree returns no hits. No commit on any branch matches this pattern. This fix cannot be confirmed from available data.

#### (e) Embedding model loaded on every request

**Commit:** `e737d54` (on `master`, not on `feat/fr04-s106b-certificate`) — "Hybrid retrieval: BM25+RRF+cross-encoder, StatutesDB knowledge switch, section injection guard"  
**File:** `backend/Retriever.py`

The commit moves the `CrossEncoder` model to a module-level lazy singleton (via `_ensure_cases_ce()`). However, `SentenceTransformer` for the main retrieval path is **still loaded inside the function body** (`Retriever.py:601`: `embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)` inside `retrieve()`). The fix in `e737d54` only fully resolves the per-request problem for the `CrossEncoder`, not for the primary embedding model. The sentence-transformer re-load on every retrieval call remains.

---

### E2. Other Bugs Fixed on Feature Branch

No commit on this branch carries a `fix:` or `bug:` prefix. All are `feat:` or `docs:`. However, several commits correct defects introduced earlier in the branch or in existing code:

#### `b70caea` — Pydantic `ctx` field, 500 → 422

As described in E1(c). Symptom: `ValidationError` during `DeviceFacts` validation propagated its `.errors()` list including a `ctx` field containing arbitrary Python objects, causing `jsonify()` to throw a `TypeError` and return 500. Fix: strip to `{loc, msg, type}` only; return 422.

#### `9393ec1` — Non-Latin-1 characters crash PDF generation

As described in E1(b). Symptom: emoji or characters outside Latin-1 in device make/model, operator name, or case description passed directly to `fpdf2` core font methods raise `UnicodeEncodeError`. Fix: `_safe(text)` with `errors="replace"` applied to all text before writing to any PDF cell.

#### `563d8c1` — Unhandled Firestore write failure blocking analysis response

**File:** `backend/SCCVision/analysis_store.py`

Symptom: A Firestore write failure (network, quota, missing credentials) would propagate as an unhandled exception and block the analysis response from reaching the client. Fix: `save_analysis_snapshot()` wraps the Firestore write in a broad `except Exception` that logs a warning and returns `None`; `router.py` treats `None` as a valid return value and includes `"analysis_id": null` in the response rather than raising.

#### `8a65097` — Field name mismatch (camelCase vs snake_case) on certificates list

**File:** `frontend/app/litigant/certificates/page.tsx`

Symptom: The page was written using camelCase field names (`certId`, `certificateNumber`, `createdAt`) that did not match the snake_case names the backend API actually returns (`cert_id`, `certificate_number`, `created_at`), causing the certificates list to render empty fields. Fix: updated all field references to snake_case matching the backend.

#### `8a65097` — `setTimeout` simulation stub masking real upload errors

**File:** `frontend/app/litigant/analysis/page.tsx`

Symptom: A `setTimeout` simulation stub fired a no-op after a delay, which would mask real upload errors and leave the UI in a fake "success" state if the actual upload endpoint was absent. Fix: removed the `setTimeout` simulation entirely; the page now only reacts to real API responses. (Note: the submit handler is still commented out, as `POST /api/evidence/upload` is not yet implemented — but the stub simulation is gone.)

---

## Summary Table

| ID | Answer (one line) | Evidence |
|----|-------------------|----------|
| A1 | Branch `feat/fr04-s106b-certificate` is NOT pushed; only `origin/master` exists remotely | `git branch -r`; fatal error on `git log origin/feat/fr04-s106b-certificate` |
| A2 | 22 ChromaDB binary files are tracked by Git; no `.gitignore` rule covers any of them | `git ls-files`; root `.gitignore` audit |
| A3 | requirements.txt lists 10 packages not installed in the venv; 4 packages used by live code not listed; 13 version mismatches | pip freeze vs requirements.txt comparison |
| A4 | `backend/.env.example` does not exist; root and frontend `.env.example` files exist | filesystem check |
| B1 | Device questionnaire is present and wired; calls `POST /api/evidence/certificates` on submit | `DeviceQuestionnaire.tsx:202,211`; `evidenceApi.ts:22–37` |
| B2 | `listCertificates()` makes a real HTTP GET; download button enabled; no stubs | `evidenceApi.ts:8–13`; `certificates/page.tsx:87–96` |
| B3 | 7 steps wired; 3 not wired: upload page stub (`analysis/page.tsx:55–74`), evidence list (`evidenceApi.ts:4–7`), certificate detail page missing | Full trace; `analysis/page.tsx`; `evidenceApi.ts:4–7`; `find` on `[cert_id]/page.tsx` |
| B4 | `CERT_THRESHOLD = 0.83` exists only in the frontend; reachable by any authenticated litigant; no backend enforcement | `SCCVisionTest.tsx:23,186,1049,1378`; `analysis/page.tsx:275` |
| B5 | Server computes SHA-256 at upload; browser computes independently; backend compares at `/evidence/process`; mismatch blocks certificate at gate not at upload; hash printed on PDF | `router.py:70,87,220–223`; `gate.py:53–56`; `document.py:140–149` |
| B6 | Snapshot saved to `users/{uid}/analyses/{analysis_id}` with 8 fields; OCR text and images explicitly excluded; Firebase Storage not used | `analysis_store.py:62–86`; grep for storage calls |
| B7 | 8 conditions evaluated; 4a and INTEGRITY are hard blockers (→ weak); questionnaire-only failures → needs_strengthening; no LLM calls in gate | `gate.py:21–359` |
| B8 | fpdf2 2.9.0; `render_pdf()` dispatches to certificate or deficiency report; PDF regenerated on every download (not stored); deterministic via `set_creation_date()` seeded from `created_at` | `document.py`; `routes.py:103` |
| B9 | 10/10 test bundles pass | `eval/logs/s106b_bundles_result.txt` |
| B10 | No coded rule checks claim date against Limitation of Actions Act; `compute_date_checks()` only extracts, resolves, and flags dates; limitation awareness is LLM prompt text only | `dates.py:182–297`; `Qclassifier.py:180,197,276` |
| B11 | Evidence list page always empty; `listEvidence()` returns `[]` unconditionally; no backend route exists | `evidenceApi.ts:4–7` |
| B12 | Backend accepts only png/jpg/jpeg/webp (10 MB); frontend advertises pdf/heic/eml/png/jpg/jpeg/webp (15 MB); mismatch means heic/pdf/eml are rejected 400 at backend | `router.py:ALLOWED_EXTS`; `evidenceConfig.ts` |
| C1 | 8 routes in `app.py` are unauthenticated; `require_auth` never imported into `app.py`; `/api/case-laws/file/<filename>` explicitly noted "no auth" | `app.py:852,957,1073,1106,1131,1164,1198,1897`; `auth.py` |
| C2 | History routes are at `/api/conversations`, not `/api/history/conversations`; no prefix passed at blueprint registration | `app.py:42`; `history_routes.py:22,43,62,93,129,148` |
| C3 | `_size_guard` truncates at 900,000 bytes; trims `answer_markdown` and caps source snippets at 100 chars; sets `truncated=True` | `history_store.py:388` |
| C4 | `LOW_CONFIDENCE_THRESHOLD = 0.4` compared against CE sigmoid score; adds warning only, does not filter results | `Retriever.py:52,781` |
| C5 | Up to 5 statute articles (all chunks, 600-char limit); up to 3 cases × 3 chunks = 9 max case chunks (500-char limit) reach the LLM | `app.py:1484,1513,1519` |
| C6 | Probe mode pins already-retrieved docs by DB key; `pinned_docs` serialised as `[{type,id}]`; backend uses as whitelist filter; `/api/case-laws/upload` stages to disk + Firestore, does not auto-embed | `ChatView.tsx:379–390,482–485`; `app.py:1272` |
| C7 | 9 cases, 78 chunks in `case_law` collection; 2 cases excluded (`2011KEHC1481`, `2021KEELC220`) with `"exclude": true` in `cases_meta.json` | `cases_meta.json`; `build_case_chunks.py:321–322` |
| C8 | `firac_component` is empty string on all 78 case chunks; `CaseLawDB` and `StatutesDB` are live; `LegalSummariesDB`, `ChunkDB`, `BlackLawDictionaryDB` are dead/orphaned | ChromaDB metadata query; `config.py` comment |
| C9 | `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=300`, `MIN_CHUNK_SIZE=150`; 374 chunks from 264 articles of TheConstitutionOfKenya.pdf | `constitutionbreaker.py:23–25`; `constitution_embeddable_chunks.json` |
| C10 | Backend emits 10-key JSON schema (source_review, premise_check, short_answer, etc.); no FIRAC keys; `FiracTile` components are dead code for this endpoint | `app.py:1545–1580`; `ChatView.tsx:105–126,1369` |
| C11 | 72 hallucination log entries; 10 distinct queries; 13 distinct claims; "SECTION 1" leads (43%); 2 fabricated case citations; covers 2026-10-28 to 2026-11-09 | `backend/logs/hallucinations.jsonl` |
| D1 | 193/198 pass; 1 failure (`test_deterministic_followups_missing_date_and_payment` — expected `upload_mpesa_screenshot:K:payment` follow-up not generated); 4 skipped; 28.78 s | pytest run; `eval/logs/pytest_final.xml` |
| D2 | Firebase CLI v15.9.0 installed; emulator not running; 20 Firestore rules tests and 2 Python history integration tests require emulator; neither suite was run | `firebase --version`; `firestore.test.ts`; `test_history.py:TestEmulatorIntegration` |
| D3 | Build passes; 26 lint errors all in pre-existing files; FR04 branch adds 0 errors, 3 warnings in `SCCVisionTest.tsx` | `next build`; `eslint` |
| D4 | `eval/measure_latency.sh` exists; requires `GROQ_API_KEY`, `FIREBASE_SERVICE_ACCOUNT_PATH`, and a live `AUTH_TOKEN`; actual latency figures NOT DETERMINABLE from this audit | `eval/measure_latency.sh` |
| D5 | 15 GiB RAM; 11 GiB used; 7.5 GiB swap in use (moderate pressure); Flask backend ~242 MiB RSS; Next.js workers ~15 MiB each | `free -h`; `ps` |
| D6 | `eval/retrieval_queries.csv` exists with 25 unverified rows; `expected_cases` blank for all; Q010 references Insurance Act not in manifest; eval stub written to `eval/run_retrieval_eval.py`; actual Hit@3/Precision@3 NOT DETERMINABLE | CSV audit; `manifest.json` |
| D7 | Two-stage citation grounding pipeline implemented (`app.py:1545–1580` + `citation_verifier.py`); 72 hallucination log entries from live queries; 25-query grounding rate NOT DETERMINABLE without live Groq API key | `citation_verifier.py`; `validate_knowledge.py`; `hallucinations.jsonl` |
| E1a | fpdf2 `TypeError` on `set_creation_date()` prevented via `_parse_created_at()` helper (commit `9393ec1`) | `document.py:159`; commit message |
| E1b | Non-Latin-1 / Unicode ellipsis crash in PDF prevented via `_safe()` + ASCII `"..."` (commit `9393ec1`) | `document.py:44–48,143`; commit message |
| E1c | Pydantic v2 `ctx` field causing 500 fixed by stripping to `{loc,msg,type}` → 422 (commit `b70caea`) | `routes.py:52–58`; `FR04_IMPLEMENTATION_REPORT.md:88` |
| E1d | `focusRingColor` TypeScript build error — NOT DETERMINABLE; no matching commit or code found | grep; `git log` |
| E1e | Embedding model per-request fix is PARTIAL — `CrossEncoder` moved to singleton in `e737d54`; `SentenceTransformer` still re-loaded per call in `Retriever.py:601` | `Retriever.py:601`; commit `e737d54` |
| E2 | Four defects corrected on this branch across commits `b70caea`, `9393ec1`, `563d8c1`, `8a65097` — Pydantic 500, Latin-1 crash, Firestore write failure blocking response, camelCase/snake_case field mismatch, setTimeout stub | Commit diffs; source files |

---

## For the Developer

The following questions cannot be answered from the codebase, commit history, or any file in the repository. They require direct input from the developer, supervisor, or stakeholders.

**F1. Why were these 14 Acts chosen (Bills of Exchange, KICA, Stamp Duty included), and why were the Land Acts dropped?**

NOT DETERMINABLE from code or commits. The `data/processed/statutes/manifest.json` records which Acts are indexed but contains no rationale for inclusion or exclusion decisions. The selection (Small Claims Court Act, Evidence Act, Civil Procedure Act, Limitation of Actions Act, etc.) is consistent with a Kenyan Small Claims Court / digital-evidence scope, but the specific decisions to include Bills of Exchange Act, Kenya Information and Communications Act, and Stamp Duty Act — and to exclude the Land Registration Act, Land Act, and related property statutes — are not documented anywhere in the repository. The developer or project brief holds this reasoning.

**F2. Which browser and version did you use for end-to-end checks?**

NOT DETERMINABLE. No browser specification appears in the test scripts (`FR04-E01` through `FR04-E07`), the phase documentation, or any commit message. The manual test scripts describe steps and expected outcomes but do not record the browser or version used.

**F3. Did any person (supervisor, law student, practitioner, or litigant) try the system or review its outputs? If so, when, and what was observed?**

NOT DETERMINABLE. No session notes, feedback records, user test transcripts, or reviewer sign-off documents exist in the repository. The hallucination log (`hallucinations.jsonl`) records 72 entries from live queries between 2026-10-28 and 2026-11-09, suggesting the system was used in some form, but whether these queries were from a developer testing alone or from an external reviewer cannot be determined from the log alone.

**F4. Did the supervisor approve the deviations from the original design — specifically: no Neo4j graph layer, no FIRAC-structured output, screenshots only (no Form SCC 1)?**

NOT DETERMINABLE. No supervisor approval record, change-request log, or design deviation memo exists in the repository. The `.gitignore` excludes Neo4j data directories (suggesting Neo4j was at least considered), and the codebase contains remnants pointing to a FIRAC-structured pipeline (`parseFirac` in ChatView, the `firac_component` metadata field on case chunks), but no document records whether these deviations were formally approved or whether they represent deferred work.

**F5. Was the system ever deployed (for example Firebase Hosting or a VPS), or demonstrated locally only?**

NOT DETERMINABLE. The repository contains no deployment configuration (`firebase.json` with hosting targets, `Dockerfile` with an exposed port, CI/CD pipeline, or hosting provider record). The backend `.env.example` is missing, which is consistent with local-only use. Firebase Hosting is used for the Firestore security rules emulator project (`legalai-rules-test`) but no frontend deployment to Firebase Hosting or any other service can be confirmed from the repository contents alone.

**F6. Who created the 10 synthetic test bundles in `eval/s106b_test_bundles.json`, and who determined the expected outcome for each?**

NOT DETERMINABLE. The file exists and all 10 bundles pass, but the commit history attributes the entire `eval/` directory to the developer's own commits. No external contributor, law-school collaborator, or test-review record is referenced in any commit message or document. The authorship of the expected outcomes (which bundle produces `certificate_ready` vs. `needs_strengthening` vs. `weak`, and under which condition) cannot be attributed to anyone other than the developer from the available evidence.

---

*End of SYSTEM_CHAPTER5_REVISED*
