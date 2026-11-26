# PROBE_FR04.md — Implementation Planning Probe for FR-04

**Date:** 2026-11-17  
**Scope:** Read-only. No modifications, installs, or commits.  
**Purpose:** Identify every asset, gap and integration point needed to implement FR-04 (Section 106B Certificate and Deficiency Report).

---

## 1. Section 106B Requirements (from repo text)

**Source:** `backend/admissibility_notes/s106b_source_text.md` — ChromaDB-verified statutory text.

### 1.1 The four admissibility conditions (s.106B(2)(a)–(d))

A "computer output" (which includes a WhatsApp/SMS screenshot stored or processed by a device) is admissible without further proof only when **all four** of the following are satisfied:

| Condition | Statutory text summary |
|---|---|
| **(a) Regular use** | The computer was used to store or process information for activities regularly carried out during the relevant period by a person having **lawful control** over the use of the computer. |
| **(b) Regular data feed** | Information of the kind contained in the record was **regularly fed** into the computer in the ordinary course of those activities. |
| **(c) Proper operation** | Throughout the material period the computer was **operating properly**, or any malfunction was not such as to affect the record or its accuracy. |
| **(d) Derived data** | The information in the record **reproduces or is derived from** information fed into the computer in the ordinary course of those activities. |

### 1.2 The certificate contents (s.106B(4)(a)–(d))

The certificate must do all of the following:

| Requirement | Content |
|---|---|
| **(4)(a)** | **Identify the electronic record** and describe the manner in which it was produced. |
| **(4)(b)** | Give **particulars of the device** involved in the production of the record sufficient to show it was produced by a computer. |
| **(4)(c)** | Deal with the **s.106B(2)(a)–(d) conditions** (regular use, regular feed, proper operation, derived data). |
| **(4)(d)** | Be **signed by a person occupying a responsible position** in relation to the operation of the relevant device or the management of the relevant activities. |

### 1.3 Mapping conditions to data sources

| s.106B requirement | Automatically evidenced from screenshots? | Requires litigant questionnaire input |
|---|---|---|
| (4)(a) Record identity (file name, SHA-256, screenshots) | **YES** — `sha256` stored per evidence item in `SCCVision/router.py:219`; `file_name` stored at line 213 | No |
| (4)(a) How it was produced (OCR type, source device) | PARTIAL — OCR type inferred (`mpesa_screenshot`/`chat_screenshot`); M-Pesa format identifiable | Litigant must confirm: was it a screenshot, a print, a forwarded file? |
| (4)(b) Device particulars (make, model, IMEI/serial) | **NO** — not extractable from screenshot content | Questionnaire: device make/model, IMEI, phone number associated with device |
| (4)(b) Device type sufficient to show "produced by a computer" | PARTIAL — M-Pesa SMS implies Safaricom system; WhatsApp implies smartphone | Questionnaire: who owns the device; were they the operator |
| (4)(c) Regular use / lawful control (2)(a) | **NO** | Questionnaire: "During the period of these messages, did you use this phone regularly for the activities shown?" |
| (4)(c) Regular data feed (2)(b) | **NO** | Questionnaire: "Were messages/payments like these regularly stored on this phone in the ordinary course of your activities?" |
| (4)(c) Proper operation (2)(c) | **NO** | Questionnaire: "Was the phone working properly during and after the period of these messages? Any known faults?" |
| (4)(c) Derived data (2)(d) | PARTIAL — deterministic M-Pesa parse confirms derivation from Safaricom system | Questionnaire: for chat, "did you receive/send these messages directly on this device?" |
| (4)(d) Responsible person signature | **NO** | Litigant must sign; certificate must name them, their position (owner/operator), and their ID |

**Note from statute:** s.106B(5) clarifies that information is "supplied to a computer" whether supplied directly or via appropriate equipment — this covers WhatsApp/M-Pesa because Safaricom's servers supply the data to the device.

---

## 2. Current Analysis Output Shape

### 2.1 Pydantic models (`backend/SCCVision/analyst/models.py`)

```python
class IntegrityRating(str, Enum):  # STRONG | ADEQUATE | WEAK

class ElementStatus(str, Enum):
    PRESENT | PARTIAL | ABSENT | UNCLEAR | NO_ISSUE_INDICATED | NOT_REQUIRED

class PaymentClassification(str, Enum):
    VERIFIED-LOOKING | PRESENT BUT UNVERIFIED | SUSPICIOUS | CLAIMED ONLY | ABSENT

class OverallVerdict(str, Enum):
    LIKELY VALID | POSSIBLY VALID WITH GAPS | LIKELY NOT VALID | INSUFFICIENT EVIDENCE

class Evidence(BaseModel):
    msg_index: int | None; sender: str | None; time: str | None; quote: str

class Party(BaseModel):
    name: str; role: str | None; notes: str | None

class TimelineEntry(BaseModel):
    msg_index: int | None; evidence_id: str | None; sender: str | None
    time: str | None; event: str

class Element(BaseModel):
    status: ElementStatus
    evidence: list[Evidence]
    reasoning: str
    confidence: float | None

class MpesaPayment(BaseModel):
    code: str | None; amount_kes: float | None; sender: str | None
    recipient: str | None; datetime: str | None; evidence_id: str | None
    basis: list[str]; classification: PaymentClassification; notes: str

class FollowUpRequest(BaseModel):
    id: str                  # "<type>:<element>:<target>"
    element: str; type: str; prompt: str; why_it_matters: str
    blocking: bool; target: str | None; options: list[str]; date_role: str | None
    # types: upload_mpesa_screenshot | upload_chat_screenshot | upload_other_evidence
    #        | provide_date | provide_text | confirm_choice | provide_original_message

class PartyQuestion(BaseModel):
    element: str; question: str

class Integrity(BaseModel):
    rating: IntegrityRating; issues: list[str]

class AnalystResult(BaseModel):
    conversation_integrity: Integrity
    unrelated_messages: list[UnrelatedMessage]
    parties: list[Party]
    timeline: list[TimelineEntry]
    elements: dict[str, Element]          # keys A–K
    mpesa_payments: list[MpesaPayment]
    red_flags: list[str]
    missing_information: list[str]        # ← deficiency list
    overall_verdict: OverallVerdict
    confidence: float | None              # 0.0–1.0
    changes_since_previous: list[str]
    plain_summary: str
    strengthen_checklist: list[str]       # ← actionable deficiency guidance
    follow_up_requests: list[FollowUpRequest]  # ← structured missing-info requests
    questions_for_parties: list[PartyQuestion]
    disclaimer: str
```

**No s.106B-specific field exists.** The model has no `s106b_applicable`, `s106b_verdict`, or `certificate_eligible` field. Eligibility is computed entirely in the frontend at `SCCVisionTest.tsx:177–186`.

### 2.2 Full response shape from `POST /sccvision/case/analyze`

`pipeline.py:analyze_case()` returns a dict with these top-level keys:

```json
{
  "analysis_input":  { ... },   // masked case-file sent to LLM — full object
  "result":          { ... },   // AnalystResult.model_dump() — all fields above
  "verification":    { ... },   // deterministic rules report
  "prompt_version":  "xxxxxxxx",// first 8 hex chars of SHA-256 of system_prompt.md
  "model":           "qwen/qwen3.8-27b"
}
```

The `verification` sub-object (from `verify.py`):

```json
{
  "quotes_grounded":        true,
  "ungrounded_quotes":      [],
  "mpesa_cross_check":      { "consistent": true, "codes_llm_only": [], "codes_parsed_only": [], "invented_fields": false },
  "status_issues":          [],
  "timeline_unreferenced":  [],
  "premature_breach":       false,
  "verdict_before":         "POSSIBLY VALID WITH GAPS",
  "verdict_after":          "POSSIBLY VALID WITH GAPS",
  "verdict_downgraded":     false,
  "downgrade_reasons":      []
}
```

**No s.106B gate verdict** is present in the `verification` object. It checks contract-law elements only.

### 2.3 `PROMPT_VERSION`

Defined in `backend/SCCVision/analyst/__init__.py:17` as the first 8 hex chars of `sha256(system_prompt.md)`. Value changes whenever `system_prompt.md` is edited.

### 2.4 System-prompt references to s.106B

`backend/SCCVision/analyst/system_prompt.md:18` (LEGAL FRAMEWORK section):

> "Kenyan law: Law of Contract Act (Cap 23), Sale of Goods Act, Consumer Protection Act 2012 … Evidence Act s.78A and s.106B … Treat every citation as 'verify before relying'."

The model is told s.106B is in scope but is given **no instructions** to assess the four s.106B(2) conditions, to populate a certificate, or to generate a deficiency report against s.106B. The prompt is focused entirely on contract-validity analysis (elements A–K).

### 2.5 Does the analysis distinguish s.106B admissibility from contract validity?

**NO.** The two are conflated under the single `overall_verdict`. There is no `s106b_verdict` field, no APPLICABLE/INAPPLICABLE/UNCERTAIN enum for the s.106B gate, and no separate "admissibility track". This is the core gap.

### 2.6 Example result object (synthetic test data, safe to print)

The `_FakeLLM` and `_CaseFakeLLM` stubs in `backend/tests/test_sccvision_analyst.py:300–315` produce minimal results. A full `AnalystResult.model_dump()` from a representative synthetic case:

```json
{
  "conversation_integrity": { "rating": "ADEQUATE", "issues": ["One screenshot only; thread may not be complete."] },
  "unrelated_messages": [],
  "parties": [
    { "name": "me", "role": "claimant", "notes": "buyer" },
    { "name": "Neptune Shelters", "role": "respondent", "notes": "service provider" }
  ],
  "timeline": [
    { "msg_index": 1, "evidence_id": null, "sender": "them", "time": "10:00",
      "event": "Provider states service fee of KES 27,750" }
  ],
  "elements": {
    "C": {
      "status": "PARTIAL",
      "evidence": [{ "msg_index": 1, "sender": "them", "time": "10:00", "quote": "I charge 27750 for my services" }],
      "reasoning": "Price quoted; commitment to perform not yet shown.",
      "confidence": 0.75
    },
    "D": { "status": "UNCLEAR", "evidence": [], "reasoning": "No acceptance message in record.", "confidence": 0.3 }
  },
  "mpesa_payments": [],
  "red_flags": ["Single screenshot; prior context unknown."],
  "missing_information": [
    "Screenshot date not provided; timeline cannot be anchored.",
    "No acceptance of offer shown."
  ],
  "overall_verdict": "POSSIBLY VALID WITH GAPS",
  "confidence": 0.55,
  "changes_since_previous": [],
  "plain_summary": "The provider quoted a fee of KES 27,750 but no acceptance or payment confirmation is in the record.",
  "strengthen_checklist": [
    "Upload a screenshot showing your acceptance of the quote.",
    "Upload M-Pesa payment confirmation if payment was made."
  ],
  "follow_up_requests": [
    {
      "id": "provide_date:E:e-chat-1", "element": "E", "type": "provide_date",
      "prompt": "What date was this screenshot taken?",
      "why_it_matters": "Anchors the timeline.", "blocking": true,
      "target": "e-chat-1", "options": [], "date_role": "screenshot_date"
    }
  ],
  "questions_for_parties": [{ "element": "D", "question": "Did you agree to pay KES 27,750?" }],
  "disclaimer": "This is an automated analysis, not legal advice. A qualified advocate should review high-value or contested matters."
}
```

---

## 3. Existing Frontend Pieces

### 3.1 `CertificateModal` (`frontend/app/components/sccvision/SCCVisionTest.tsx`)

**Status: WORKING (developer test panel only)**

**Props** (`SCCVisionTest.tsx:1234–1240`):
```typescript
{
  analysis: AnalystResponse;   // { result: AnalystResult, verification: {...} }
  caseFile:  CaseFile;         // frontend-held case-file object
  asOf:      string;           // ISO date string (todayISO())
  onClose:   () => void;
}
```

**Outcome decision** (`SCCVisionTest.tsx:177–186`):
```typescript
const CERT_THRESHOLD = 0.83;    // SCCVisionTest.tsx:21
function isCertEligible(a: AnalystResponse): boolean {
  const v = a.verification.verdict_after ?? a.result.overall_verdict;
  return (
    a.result.confidence != null &&
    a.result.confidence >= CERT_THRESHOLD &&   // 83%
    v !== "LIKELY NOT VALID" &&
    v !== "INSUFFICIENT EVIDENCE"
  );
}
```

**Sections rendered by `CertificateModal`:**
1. Verdict banner (colour-coded: green/amber/red based on `verdictTone(verdict)`)
2. Confidence bar (percentage, coloured, with "Meets the 83% threshold" text)
3. Parties table (from `result.parties[]`)
4. Contract elements A–K table (status badge + confidence % per element)
5. M-Pesa payments table (filtered to non-ABSENT entries from `result.mpesa_payments[]`)
6. Plain summary (`result.plain_summary`)
7. Strengthen checklist (`result.strengthen_checklist[]`)
8. Red flags (`result.red_flags[]`)
9. Missing information (`result.missing_information[]`)
10. Disclaimer (`result.disclaimer`)
11. Follow-up requests (open requests from `result.follow_up_requests[]`)
12. "Print / Save PDF" button — calls `window.print()` (`handlePrint`)

**SHA-256 flow** (`SCCVisionTest.tsx:206–208, 360–367`):
```typescript
async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, "0")).join("");
}
// Called before POST /sccvision/evidence/process:
const hash = await sha256Hex(await f.arrayBuffer());
// Sent as: { ocr_result, classification, file_name, sha256: hash }
```

### 3.2 `frontend/app/litigant/certificates/page.tsx`

**Status: STUB — UI shell, no data**

- Renders a full filterable table with outcome pills (certificate_ready / needs_strengthening / weak)
- Data source: `listCertificates()` → **returns `[]` unconditionally** (see §3.4)
- Two disabled action buttons with explicit TODO comments:

```tsx
{/* TODO(backend): Download certificate PDF — GET /api/evidence/certificates/{id}/download */}
{/* TODO(backend): View full certificate report — GET /api/evidence/certificates/{id} */}
```

### 3.3 `frontend/app/litigant/analysis/page.tsx`

**Status: STUB upload flow; SCCVisionTest panel WORKING**

4-step visual guide:
1. Upload — Add your files
2. We read it — Extract & check evidence
3. Answer a few questions — About your device or phone (**Step 3 label only; no form exists**)
4. Certificate or guidance — Signed draft or next steps

```typescript
// TODO(backend): POST /api/evidence/upload
//   body: FormData — files[], types[<index>], passwords[<index>], preparedFor
//   Expected response: { records: EvidenceRecord[] }
//   On success, navigate to /litigant/evidence or the individual record page.
await new Promise<void>((r) => setTimeout(r, 2000));  // simulation; navigates nowhere
```

Also has: advocate-only "Preparing this evidence for: Myself / A client" toggle with the note "If it's for a client, the certificate must be signed by your client, because they control the device." — **relevant to s.106B(4)(d) responsible person.**

No questionnaire UI anywhere in `frontend/app/litigant/` (no form for device make/model, operator identity, or working-condition questions).

### 3.4 `frontend/app/lib/evidenceApi.ts` and `evidenceTypes.ts`

**Status: STUB**

```typescript
// evidenceApi.ts:5–12
export async function listEvidence(): Promise<EvidenceRecord[]> {
  // TODO(backend): GET /api/evidence — returns EvidenceRecord[]
  return [];
}
export async function listCertificates(): Promise<CertificateRecord[]> {
  // TODO(backend): GET /api/evidence/certificates — returns CertificateRecord[]
  return [];
}
```

**`EvidenceRecord` type** (`evidenceTypes.ts:43–50`):
```typescript
interface EvidenceRecord {
  id: string;
  fileName: string;
  evidenceType: EvidenceType;   // mpesa_statement|mpesa_sms|whatsapp_sms_chat|email|written_agreement_photo|other
  sizeBytes: number;
  uploadedAt: string;           // ISO-8601
  status: EvidenceStatus;       // uploaded|extracting|analysing|awaiting_answers|complete|failed
}
```

**`CertificateRecord` type** (`evidenceTypes.ts:53–60`):
```typescript
interface CertificateRecord {
  id: string;
  certificateNumber: string;
  evidenceId: string;
  fileName: string;
  outcome: CertificateOutcome;  // certificate_ready|needs_strengthening|weak
  createdAt: string;            // ISO-8601
}
```

The `awaiting_answers` status on `EvidenceRecord` is exactly the hook for the Step 3 questionnaire flow.

### 3.5 `frontend/app/litigant/evidence/page.tsx`

**Status: STUB — UI shell, no data**

Renders a full table of `EvidenceRecord[]` items using `listEvidence()` stub. Status pill includes `awaiting_answers` → "Needs answers" (blue). Evidence is **not persisted in React state or localStorage** — it is fetched from the backend API stub every render. Currently always empty.

---

## 4. Backend Integration Points

### 4.1 Existing SCCVision routes (`backend/SCCVision/router.py`)

| Route | Method | Auth | Key request fields | Key response fields |
|---|---|---|---|---|
| `/sccvision/health` | GET | None | — | `status, module, engine, loaded` |
| `/sccvision/extract` | POST | `@require_auth` | `file` (image, ≤10 MB) | OCR result dict |
| `/sccvision/analyze` | POST | `@require_auth` | `files[]`, `case_context` | Full analysis (deprecated) |
| `/sccvision/analyze-messages` | POST | `@require_auth` | `ocr_results[]`, `case_context` | Full analysis (deprecated) |
| `/sccvision/evidence/process` | POST | `@require_auth` | `ocr_result`, `classification`, `evidence_id`, `file_name`, `sha256` | `{"evidence": {evidence_id, type, file_name, sha256, added_at, ocr, parsed}}` |
| `/sccvision/case/analyze` | POST | `@require_auth` | Full case-file JSON (`case_context`, `evidence[]`, `user_provided_facts[]`, `analyses[]`) | `{analysis_input, result, verification, prompt_version, model}` |

**Per-user storage:** None. No route persists analysis results to Firestore. Every response is ephemeral — the frontend holds the case file entirely in React state.

**Role restriction:** All routes use `@require_auth` only (no `@require_role`). Any authenticated user including `litigant` can call all SCCVision routes.

### 4.2 `auth.py` — authentication decorators

```python
def require_auth(f):
    # Verifies Firebase ID token, loads Firestore profile
    # Sets: g.uid (str), g.profile (dict), g.user_type (str: "advocate"|"litigant"|"admin")

def require_role(*roles):
    # Like require_auth, plus checks g.user_type in roles
    # Returns 403 {"error": "Forbidden: requires role X"} if not in set
```

`auth.py:31`: Firebase Admin SDK is initialized with `storageBucket: 'legalai-scc.firebasestorage.app'`.

### 4.3 `history_store.py` — Firestore helper patterns

Reusable patterns for a new certificate/evidence collection:

```python
# Collection path pattern (line 61–62):
def _conv_ref(uid: str, cid: str):
    return _db().collection("users").document(uid).collection("conversations").document(cid)
# → reuse as: _db().collection("users").document(uid).collection("evidence").document(eid)
#             _db().collection("users").document(uid).collection("certificates").document(cert_id)

# Size guard (line 388–413): _MAX_DOC_BYTES = 900_000
# Truncates oversized docs before writing. Reuse for analysis result storage
# (AnalystResult.model_dump() can be large if many messages/payments).

# Transactional write pattern (line 180–241): @_transactional, Increment(1)
# Reuse for atomic create-evidence + create-certificate-record writes.

# Background executor pattern (line 29): ThreadPoolExecutor(max_workers=2)
# Reuse for async PDF generation after analysis completes.
```

Firestore schema for existing history:
```
users/{uid}/conversations/{cid}         ← metadata doc (title, turn_count, ...)
users/{uid}/conversations/{cid}/turns/{tid}  ← individual turns
```

New schema needed:
```
users/{uid}/evidence/{eid}              ← EvidenceRecord-compatible doc
users/{uid}/certificates/{cert_id}      ← CertificateRecord + full analysis snapshot
```

### 4.4 `firestore.rules` — current rules and what must be added

**Current rules relevant to new subcollections:**
```javascript
// Existing conversation pattern (lines 98–106):
match /users/{uid}/conversations/{convId} {
  allow read: if request.auth != null && request.auth.uid == uid;
  allow write: if false;  // Admin SDK writes only
  match /turns/{turnId} {
    allow read: if request.auth != null && request.auth.uid == uid;
    allow write: if false;
  }
}
// Catch-all (line 108–110):
match /{document=**} { allow read, write: if false; }
```

**Rules to add for FR-04** (follow the same pattern):
```javascript
// Evidence items — written by backend only; litigant reads own
match /users/{uid}/evidence/{evidenceId} {
  allow read: if request.auth != null && request.auth.uid == uid;
  allow write: if false;
}

// Certificates — written by backend only; litigant reads own
match /users/{uid}/certificates/{certId} {
  allow read: if request.auth != null && request.auth.uid == uid;
  allow write: if false;
}
```

**`firestore.indexes.json`:** Currently has no indexes for user subcollections. A `createdAt DESC` composite index on `users/{uid}/certificates` would be needed for the certificate history page's ordered list.

### 4.5 Firebase Storage — `legalai-scc.firebasestorage.app`

**Configuration:** `auth.py:31` passes `storageBucket: 'legalai-scc.firebasestorage.app'` to `firebase_admin.initialize_app()`.

**Import:** `backend/app.py:13` imports `from firebase_admin import firestore as fb_firestore, storage as fb_storage`.

**Usage:** `grep -c "fb_storage" backend/app.py` → **1** (import line only). `fb_storage` is imported but **never called anywhere**. No existing upload helper, download URL helper, or bucket reference code.

The Firebase Admin SDK's `fb_storage.bucket()` → `google.cloud.storage.Bucket` is available but completely unwired.

---

## 5. Available PDF/Document Libraries

| Library | Installed | Version | In `requirements.txt` | Imported in runtime | Notes |
|---|---|---|---|---|---|
| **fpdf2** | YES | 2.9.0 | NO (not in requirements.txt) | **NO** | Available in venv; not wired up anywhere |
| **Jinja2** | YES | 3.1.6 | YES (implicit Flask dep) | **NO** | Available for HTML templating |
| **python-docx** | YES | 1.2.0 | YES (`requirements.txt:13`) | Only in `build_case_chunks.py` (ingestion) | Not used at runtime |
| **reportlab** | NO | — | NO | NO | Not installed |
| **weasyprint** | NO | — | NO | NO | Not installed |

**fpdf2 is NOT in `requirements.txt`**, so it would vanish on a fresh `pip install -r requirements.txt`. It must be added before use.

**Feasible options for certificate PDF generation:**
- **fpdf2 (recommended):** Pure Python, no system deps, A4 layout, Unicode support, already in venv. Add to requirements.txt.
- **Jinja2 + wkhtmltopdf/weasyprint:** HTML→PDF via template; weasyprint must be installed; CSS-driven layout. Good for court-formatted output but adds system binary dep.
- **python-docx:** Already installed; could produce a `.docx` for the litigant to sign and print. Simpler but requires Word/LibreOffice to view.

---

## 6. Tests That Could Be Affected

### 6.1 Test files

| File | Runner | Scope |
|---|---|---|
| `backend/tests/test_sccvision_analyst.py` | `cd backend && PYTHONPATH=. pytest tests/test_sccvision_analyst.py -v` | Enrichment, rules, pipeline, M-Pesa parsing, date checks — **the core module any FR-04 backend change touches** |
| `backend/tests/test_history.py` | `cd backend && PYTHONPATH=. pytest tests/test_history.py -v` | Firestore history helpers — affected if `_size_guard` or collection helpers are changed |
| `backend/test_scc_pipeline.py` | `cd backend && PYTHONPATH=. pytest test_scc_pipeline.py -v` | Full pipeline integration test |
| `firestore-tests/firestore.test.ts` | `cd firestore-tests && npm test` | Security rules — **must add tests for new `users/{uid}/evidence` and `users/{uid}/certificates` rules** |

### 6.2 Test changes required

- **`test_sccvision_analyst.py`:** If `AnalystResult` gains a `s106b_verdict` field or `pipeline.py`'s output dict gains a `s106b_assessment` key, all `_result()` builder calls and the `test_pipeline_returns_analysis_and_verification` / `test_analyze_case_returns_full_response` assertions break.
- **`firestore.test.ts`:** New `certificates` and `evidence` subcollection rules need coverage: (a) litigant reads own, (b) litigant cannot write, (c) cross-user reads denied.

---

## 7. Gaps List — Everything FR-04 Needs That Does Not Exist Yet

### Backend gaps

| # | Gap | Current state | What to build |
|---|---|---|---|
| B1 | **s.106B device questionnaire** data model | ABSENT | A `DeviceFacts` Pydantic model: `device_make`, `device_model`, `imei_or_serial` (optional), `phone_number`, `operator_name`, `lawful_control: bool`, `regular_use: bool`, `regular_feed: bool`, `proper_operation: bool`, `any_malfunction: str | None`, `responsible_person_name`, `responsible_person_id`, `responsible_person_position` |
| B2 | **s.106B compliance gate** | ABSENT | A new module `SCCVision/s106b_gate.py` that takes `AnalystResult` + `DeviceFacts` + evidence items → returns `{passed: bool, conditions: {a: bool, b: bool, c: bool, d: bool}, missing: list[str], certificate_data: dict}` |
| B3 | **s.106B verdict in `AnalystResult`** | ABSENT | Either add `s106b_assessment: S106BAssessment | None` to `AnalystResult`, or compute it in a separate post-analysis step. The gate is deterministic (not LLM) so it belongs outside the LLM result. |
| B4 | **Certificate document generation** | ABSENT (`fpdf2 2.9.0` installed, not wired) | A `certificate_generator.py` that takes `{case_context, evidence[], device_facts, analysis_result, certificate_number}` → returns PDF bytes. Add `fpdf2` to `requirements.txt`. |
| B5 | **Deficiency report generation** | ABSENT | A `deficiency_report.py` or section of the same generator that takes unmet conditions + `missing_information[]` + `strengthen_checklist[]` + `follow_up_requests[]` → PDF/text report. Can reuse existing `AnalystResult` fields. |
| B6 | **Certificate persistence to Firestore** | ABSENT | New `certificate_store.py` modelled on `history_store.py`. Schema: `users/{uid}/certificates/{cert_id}` with fields: `certificate_number`, `evidence_ids[]`, `outcome`, `analysis_snapshot` (stripped), `device_facts`, `pdf_storage_path`, `created_at`, `schema_version`. |
| B7 | **Evidence persistence to Firestore** | ABSENT | `users/{uid}/evidence/{eid}` doc storing `EvidenceRecord`-compatible fields. Currently all evidence lives only in frontend React state. |
| B8 | **Certificate PDF storage** | ABSENT | Use `fb_storage.bucket()` (already configured) to store PDF bytes and generate a signed download URL. No existing helper. |
| B9 | **New API routes** | ABSENT | `POST /api/evidence/upload`, `GET /api/evidence`, `GET /api/evidence/certificates`, `GET /api/evidence/certificates/{id}`, `GET /api/evidence/certificates/{id}/download` |
| B10 | **Certificate number generation** | ABSENT | A sequential or UUID-based numbering scheme (e.g. `SCC-CERT-{uid[:6]}-{YYYYMMDD}-{seq}`) |

### Frontend gaps

| # | Gap | Current state | What to build |
|---|---|---|---|
| F1 | **Step 3 questionnaire page** | ABSENT (label only) | A new page at `frontend/app/litigant/analysis/questionnaire/[evidenceId]/page.tsx` or a modal. Form fields map to `DeviceFacts`: device make/model, phone number, operator name, 4 yes/no condition questions, responsible person name + position + ID. |
| F2 | **`listCertificates()` real API call** | STUB (`return []`) | Replace stub with `apiFetch("/api/evidence/certificates")` |
| F3 | **`listEvidence()` real API call** | STUB (`return []`) | Replace stub with `apiFetch("/api/evidence")` |
| F4 | **Certificate download** | Disabled button | Wire `GET /api/evidence/certificates/{id}/download` → trigger browser download |
| F5 | **Certificate view/report** | Disabled button | Wire `GET /api/evidence/certificates/{id}` → open detail page or modal |
| F6 | **`handleSubmit` in `analysis/page.tsx`** | `setTimeout` simulation | Real `POST /api/evidence/upload` with `FormData`; redirect to questionnaire if `status === "awaiting_answers"` |
| F7 | **`EvidenceStatus: awaiting_answers`** routing | Defined in types, unused | When `evidence.status === "awaiting_answers"`, route to questionnaire page automatically |

### Infrastructure gaps

| # | Gap | Current state | What to add |
|---|---|---|---|
| I1 | **`fpdf2` in `requirements.txt`** | Installed in venv but not in requirements | Add `fpdf2==2.9.0` |
| I2 | **Firestore rules for `evidence` and `certificates`** | No rules (catch-all denies all) | Add two `match` blocks (see §4.4) |
| I3 | **Firestore indexes for certificate listing** | No indexes for user subcollections | Add `createdAt DESC` index on `users/{uid}/certificates` in `firestore.indexes.json` |
| I4 | **Firebase Storage security rules** | Not in repo | Add `service firebase.storage` rules: litigant reads own path; backend writes only |

---

## 8. Questions for You

**Q1 — s.106B scope: screenshot only or multi-modal?**  
The compliance conditions (2)(a)–(d) require statements about the device itself (make, operator, working condition). Will the questionnaire be a simple 5–8 field form the litigant fills in once per device, or does it need to be re-answered per evidence batch? The certificate links to a specific device, not to the case as a whole.

**Q2 — Responsible person: litigant signs, or backend pre-fills + litigant countersigns?**  
s.106B(4)(d) requires a **signature** from the person in a responsible position. The system cannot produce a legally valid signature automatically. Three options: (a) generate a draft PDF the litigant physically prints and signs; (b) integrate a simple e-signature field (typed name as declaration); (c) produce a "ready to sign" PDF with a signature line. Which is in scope?

**Q3 — Certificate format: what does it need to look like?**  
Is a clean A4 PDF with the s.106B(4)(a)–(d) headings sufficient, or does it need to match a specific court form (e.g., Form SCC 1 or similar)? No Form SCC 1 template was found anywhere in the repo.

**Q4 — Deficiency report: separate PDF or same document?**  
When s.106B compliance fails, the design says "generate a clear, plain-language deficiency report". Should this be a separate PDF download, or can it be the same document with a "DEFICIENCY REPORT" header instead of "CERTIFICATE"? The existing `strengthen_checklist[]` and `missing_information[]` from `AnalystResult` already contain the content.

**Q5 — Storage path for PDFs: Firestore (base64) or Firebase Storage?**  
A single-page PDF is ~30–80 KB. The Firestore 1 MB document limit allows storing base64-encoded PDFs directly in the certificate doc (`_size_guard` threshold is 900 KB). Firebase Storage is configured but unwired. Which approach do you prefer? Firestore is simpler (no Storage rules, no signed-URL expiry); Storage is cleaner for large files and allows streaming downloads.

**Q6 — Does the certificate include the full contract analysis (elements A–K), or only the s.106B admissibility attestation?**  
The existing `CertificateModal` renders both. The statutory certificate (s.106B(4)) only requires the four admissibility points. Including the contract analysis makes it more useful but also longer and potentially confusing (a court only needs the admissibility certificate, not the contract analysis).

**Q7 — `EvidenceStatus: awaiting_answers` flow: when does the system ask?**  
Currently the analysis runs immediately on evidence upload. For the questionnaire to gate the certificate, the backend would need to either: (a) block analysis until questionnaire is complete (simpler but slower UX); or (b) run analysis first, then conditionally require the questionnaire before generating the certificate. Option (b) matches the existing `CertificateModal` flow.

**Q8 — Will the litigant-facing certificate history need a certificate number sequence that is court-ready (e.g., includes case number), or is a UUID sufficient for the prototype?**
