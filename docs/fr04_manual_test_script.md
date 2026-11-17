# FR-04 Manual Test Script — Section 106B Certificate & Deficiency Report

**Feature:** FR-04 — Generate a Section 106B (Kenya Evidence Act) certificate or deficiency report from a completed analysis.  
**Branch:** `feat/fr04-s106b-certificate`  
**Prerequisites:** A running backend (`flask run`) and frontend (`npm run dev`), a Firestore-connected Firebase project, and a test account with role `litigant`.

---

## FR04-E01 — Happy path: certificate issued

**Goal:** Confirm that a litigant who answers all device questions correctly receives a CERTIFICATE outcome and can download the PDF.

**Setup:** Upload one clear WhatsApp screenshot and one M-Pesa screenshot. Run analysis until `analysis_id` is returned in the response (check Network tab → `/sccvision/case/analyze` response body).

**Steps:**

1. Navigate to **Evidence Analysis** (`/litigant/analysis` → open SCCVision panel).
2. Upload a WhatsApp chat screenshot and an M-Pesa screenshot. Run analysis.
3. Confirm the verdict banner appears and `analysis_id` is set (confidence ≥ 83%).
4. Click **"Prepare s.106B Certificate"** in the green evidence-threshold banner.
5. In the questionnaire modal:
   - Fill in device make and model (e.g. "Samsung", "Galaxy A14").
   - Enter last 4 digits of phone number (e.g. "6789").
   - Select production method: **Screenshot on this device**.
   - Answer **Yes** to all statutory condition questions (owner, lawful control, regular use, regular feed, operating properly, messages on this device).
   - Enter declarant name and position.
   - Tick the declaration checkbox.
6. Click **"Generate Certificate Draft"**.

**Expected:**
- The modal closes and a green success banner appears: **"SCC106B-YYYYMMDD-XXXXXX created — certificate draft ready for signing."**
- The banner links to **Certificate History**.
- Navigate to Certificate History (`/litigant/certificates`). The new certificate appears at the top with outcome **"Certificate ready"**.
- Click **Download**. A PDF is downloaded as `SCC106B-YYYYMMDD-XXXXXX.pdf`.
- Open the PDF. Verify:
  - Header says "DRAFT - FOR REVIEW AND SIGNATURE".
  - Parts A, B, C, D headings are present.
  - Certificate number matches the one shown in the UI.
  - Evidence schedule lists the uploaded files with SHA-256 hashes.
  - Blank lines for Signature, National ID No., Date, and Place.

---

## FR04-E02 — Deficiency report: missing lawful control

**Goal:** Confirm that answering "No" to lawful control produces a DEFICIENCY_REPORT with outcome "needs_strengthening".

**Setup:** Same as E01 (completed analysis with analysis_id).

**Steps:**

1. Click **"Prepare s.106B Certificate"**.
2. Fill in all fields correctly, but answer **No** to "Did you have lawful control of this phone?"
3. Click **"Generate Certificate Draft"**.

**Expected:**
- Success banner shows certificate number and says **"deficiency report generated."**
- In Certificate History, outcome pill shows **"Needs strengthening"**.
- Downloaded PDF title: **"Evidence Admissibility Report … Requirements Not Yet Met"**.
- Requirements table shows condition 2a (Lawful control) as **NOT MET** with a remedy.

---

## FR04-E03 — Deficiency report: hash mismatch (integrity failure)

**Goal:** Confirm that an evidence item whose client and server SHA-256 hashes differ produces outcome "weak".

**Setup:** Requires a backend test where the evidence item was stored with `hash_match: false`. This can be simulated by:
- Uploading a file, noting the `server_sha256` in the `/sccvision/extract` response.
- Then calling `/sccvision/evidence/process` with a different `sha256` value from the client.
- Running analysis.

**Steps:**

1. With a snapshot that has `hash_match: false` for at least one item, click **"Prepare s.106B Certificate"**.
2. Answer all device questions with "Yes" / correct values.
3. Submit.

**Expected:**
- Certificate outcome is **"Weak"** (red pill in Certificate History).
- Downloaded PDF says "Requirements Not Yet Met".
- Requirements table shows condition 4a (Electronic record identification) as **NOT MET** with note about hash mismatch.

---

## FR04-E04 — Validation: empty device make

**Goal:** Confirm that the form blocks submission when required fields are empty.

**Setup:** Any completed analysis.

**Steps:**

1. Open the questionnaire modal.
2. Leave "Phone make" blank.
3. Answer all other required questions.
4. Click **"Generate Certificate Draft"**.

**Expected:**
- An inline error message appears: "Please complete all required fields before submitting."
- No network request is made to `POST /api/evidence/certificates`.
- The modal remains open.

---

## FR04-E05 — Validation: missing declaration confirmation

**Goal:** Confirm that unchecking the declaration checkbox blocks submission.

**Steps:**

1. Open the questionnaire modal.
2. Fill all fields correctly.
3. Leave the declaration checkbox **unchecked**.
4. Click **"Generate Certificate Draft"**.

**Expected:**
- Inline error appears.
- No certificate is created.

---

## FR04-E06 — Malfunction disclosure flow

**Goal:** Confirm the conditional malfunction section appears and validates correctly.

**Steps:**

1. Open the questionnaire modal, fill basic device fields.
2. Answer **No** to "Was the phone operating properly?"
3. Verify the amber "Describe the malfunction" section appears.
4. Enter malfunction details (e.g. "Screen cracked, battery drains fast").
5. Answer **No** to "Did this malfunction affect the accuracy of the records?"
6. Complete remaining fields and submit.

**Expected:**
- Certificate is created with outcome appropriate to other conditions.
- Downloaded PDF Part C reads: "During the material period the device experienced a malfunction (Screen cracked, battery drains fast), but this malfunction did not affect the electronic record or the accuracy of its contents."

**Negative test — no details:**

1. Answer No to operating properly.
2. Leave malfunction details empty.
3. Submit.

**Expected:** Inline error; no network request.

---

## FR04-E07 — Role enforcement

**Goal:** Confirm that only litigant, advocate, and admin roles can call the certificate endpoints.

**Setup:** A test user with a custom role (or no role set, e.g. a freshly created account where `type` is not in `{litigant, advocate, admin}`).

**Steps (API-level test):**

1. Obtain a Firebase ID token for a user whose Firestore `type` is `"viewer"` or missing.
2. `POST /api/evidence/certificates` with a valid body.

**Expected:** HTTP 403 with `{"error": "Forbidden: requires role litigant or advocate or admin"}`.

**Steps (UI test):**

1. Log in with a standard `litigant` account. Complete an analysis.
2. Confirm the **"Prepare s.106B Certificate"** button is visible.
3. Log out. Log in with an `advocate` account that has access to the same analysis.
4. Confirm advocates can also see and use the button.
