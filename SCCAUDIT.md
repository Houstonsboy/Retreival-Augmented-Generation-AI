# SCC Module Audit — Determinism & Certificate Gate Analysis

> **Purpose**: Map every decision point in the SCC pipeline, identify where non-determinism enters,
> and catalogue concrete strategies to make same-input → same-output without sacrificing independent
> inspection on each run.
>
> **Date**: 2026-11-22  
> **Scope**: `backend/SCCVision/` (analyst pipeline + s.106B gate) and the frontend eligibility check in `SCCVisionTest.tsx`

---

## 1. Pipeline Architecture

```
User uploads screenshots
        │
        ▼
/sccvision/extract          ← preprocess.py + service.py (OCR, structure)
        │  returns: contact, messages[], system_events, mpesa_extracted[]
        ▼
/sccvision/case/analyze     ← router.py
        │
        ├── build_analysis_input_from_case()   ← analyst/enrich.py
        │       date_checks, integrity_flags, masked input JSON
        │
        ├── AnalystLLM.complete()              ← analyst/client.py  ⚠ NON-DETERMINISTIC
        │       model: qwen/qwen3.8-27b via Groq
        │       temperature: 0   (reduces but does NOT eliminate variance)
        │       max_tokens: 8000
        │       returns: AnalystResult (Pydantic)
        │
        ├── verify()                           ← analyst/verify.py  ✓ DETERMINISTIC
        │       check_quotes_grounded()
        │       check_mpesa_cross_match()      (mutates payments → reclassify)
        │       check_element_statuses()       (mutates element statuses)
        │       check_timeline()
        │       check_premature_breach()       (mutates K status)
        │       compute_verdict()              (may downgrade overall_verdict)
        │       merge_followups()
        │
        └── returns: {analysis_input, result, verification, prompt_version, model}

        │  (analysis_id stored in Firestore via analysis_store.py)
        ▼
/sccvision/certificate      ← s106b/routes.py
        │
        ├── load analysis snapshot from Firestore
        ├── validate DeviceFacts (device questionnaire answers)
        │
        ├── gate.evaluate()                   ← s106b/gate.py  ✓ FULLY DETERMINISTIC
        │       8 conditions: 4a, 4b, 2a, 2b, 2c, 2d, 4d, INTEGRITY
        │
        └── document.py → fpdf2 PDF
```

---

## 2. LLM Parameters (AnalystLLM)

| Parameter | Value | Determinism impact |
|---|---|---|
| `model` | `qwen/qwen3.8-27b` (Groq) | Fixed — good |
| `temperature` | `0` | Reduces variance but Groq's qwen at T=0 still shows ~2–5% confidence drift |
| `max_tokens` | `8000` | Fixed |
| `timeout` | `120 s` | Fixed |
| `attempts` | max 2 (retry on parse failure) | Retry uses fresh call — could return different result |
| `api_key` | `GROQ_API_KEY_12` (dedicated key) | Isolates from advocate pipeline rate limits |
| `system_prompt` | `analyst/system_prompt.md` (83 lines) | Fixed per deploy |

---

## 3. The 11 Contract Elements (A–K)

Each element is evaluated by the LLM and then validated/corrected by deterministic rules.

| Element | Description | Allowed statuses | Key rule |
|---|---|---|---|
| A | Parties & identity | PRESENT, PARTIAL, ABSENT, UNCLEAR | Must be at least PARTIAL for LIKELY VALID |
| B | Capacity | PRESENT, PARTIAL, ABSENT, UNCLEAR, **NO_ISSUE_INDICATED** | NO_ISSUE_INDICATED only here |
| C | Offer | PRESENT, PARTIAL, ABSENT, UNCLEAR | Must be PRESENT for LIKELY VALID |
| D | Acceptance | PRESENT, PARTIAL, ABSENT, UNCLEAR | Must be PRESENT for LIKELY VALID |
| E | Consideration | PRESENT, PARTIAL, ABSENT, UNCLEAR | Must be PRESENT for LIKELY VALID |
| F | Intention | PRESENT, PARTIAL, ABSENT, UNCLEAR | Must be PRESENT for LIKELY VALID |
| G | Certainty | PRESENT, PARTIAL, ABSENT, UNCLEAR | Must be PRESENT for LIKELY VALID |
| H | Genuine consent | PRESENT, PARTIAL, ABSENT, UNCLEAR, **NO_ISSUE_INDICATED** | NO_ISSUE_INDICATED only here |
| I | Legality | PRESENT, PARTIAL, ABSENT, UNCLEAR, **NO_ISSUE_INDICATED** | NO_ISSUE_INDICATED only here |
| J | Formalities | PRESENT, PARTIAL, ABSENT, UNCLEAR, **NOT_REQUIRED** | NOT_REQUIRED only here |
| K | Performance & breach | PRESENT, PARTIAL, ABSENT, UNCLEAR | Reset to UNCLEAR if performance not yet due |

**Deterministic status enforcement** (`check_element_statuses`):
- PRESENT or PARTIAL **without** any evidence quotes → forced to UNCLEAR
- NO_ISSUE_INDICATED on A,C,D,E,F,G,H,J,K → forced to UNCLEAR
- NOT_REQUIRED on A–J (excluding J) → forced to UNCLEAR

---

## 4. Verdict Categories and Logic

### 4.1 LLM-Generated (initial)

The LLM picks one of:
```
LIKELY VALID | POSSIBLY VALID WITH GAPS | LIKELY NOT VALID | INSUFFICIENT EVIDENCE
```

### 4.2 Deterministic Downgrade Rules (`compute_verdict` in rules.py)

After the LLM picks a verdict, `compute_verdict` **can only downgrade, never upgrade**:

| Trigger | Downgrade applied |
|---|---|
| `conversation_integrity.rating == "WEAK"` | LIKELY VALID → POSSIBLY VALID WITH GAPS |
| `date_checks.no_dates == True` | LIKELY VALID → POSSIBLY VALID WITH GAPS |
| `"edited_messages"` in integrity_flags | LIKELY VALID → POSSIBLY VALID WITH GAPS |
| C, D, E, F, or G not PRESENT | LIKELY VALID → POSSIBLY VALID WITH GAPS |
| A not PARTIAL or PRESENT | LIKELY VALID → POSSIBLY VALID WITH GAPS |
| Any element is ABSENT | LIKELY VALID → POSSIBLY VALID WITH GAPS |

**Key insight**: The verdict category after verification is reasonably deterministic because the downgrade rules are code. The only source of variance here is whether the LLM initially picks LIKELY VALID vs POSSIBLY VALID WITH GAPS — which then can only get worse.

---

## 5. Frontend Certificate Eligibility (`SCCVisionTest.tsx`)

```typescript
const CERT_THRESHOLD = 0.83;

function isCertEligible(a: AnalystResponse): boolean {
  const v = a.verification.verdict_after ?? a.result.overall_verdict;
  return (
    a.result.confidence != null &&
    a.result.confidence >= CERT_THRESHOLD &&   // ← HARD GATE on LLM confidence
    v !== "LIKELY NOT VALID" &&
    v !== "INSUFFICIENT EVIDENCE"
  );
}
```

**The `confidence` score (0.0–1.0) is entirely LLM-generated.** It is not computed from element statuses. This is the single most dangerous non-determinism point in the system: a confidence of 0.82 vs 0.84 means the difference between being offered the certificate button and not, for identical input.

---

## 6. Section 106B Gate — 8 Conditions (`s106b/gate.py`)

This layer is **fully deterministic** — zero LLM calls. It operates on:
- The stored `analysis_snapshot` (from Firestore)
- `DeviceFacts` (the device questionnaire answers submitted by the litigant)

| Code | Title | Fails when |
|---|---|---|
| `4a` | Electronic record identified, manner of production described | `file_name` missing on any evidence item; no server SHA-256 hash; client hash ≠ server hash; `production_method == OTHER` |
| `4b` | Particulars of the device | Always passes if `device_make` and `device_model` supplied — cannot fail |
| `2a` | Device regularly used by person with lawful control | `lawful_control == False` OR `regular_use == False` |
| `2b` | Information regularly fed into device | `regular_feed == False` |
| `2c` | Device operating properly | `operating_properly == False` AND `malfunction_affects_record != False` |
| `2d` | Record derived from information fed into the device | `messages_on_this_device == False`; production_method not in {SCREENSHOT_ON_DEVICE, PRINTED_STATEMENT}; M-Pesa screenshot present but no parsed transaction codes |
| `4d` | Signed by person in responsible position | `owner_is_declarant == False` OR `declaration_confirmed == False` |
| `INTEGRITY` | Evidence integrity checks passed | `quotes_grounded == False`; M-Pesa cross-check `consistent == False` or `invented_fields == True`; `conversation_integrity.rating == "WEAK"` |

### 6.1 Outcome Mapping

```
All 8 met                           → document_type=CERTIFICATE   outcome=certificate_ready
INTEGRITY or 4a failed              → document_type=DEFICIENCY_REPORT  outcome=weak
Only 2a/2b/2c/2d/4b/4d failed      → document_type=DEFICIENCY_REPORT  outcome=needs_strengthening
```

**INTEGRITY failing is the most common path to refusal** because it ties directly back to what the LLM produced:
- If the LLM cited an ungrounded quote → `quotes_grounded=False` → `INTEGRITY` fails
- If the LLM classified a payment as VERIFIED without a parsed code → reclassified to CLAIMED ONLY → `consistent=False` → `INTEGRITY` fails
- If the LLM rated record integrity WEAK → `INTEGRITY` fails

---

## 7. Deterministic Verification Rules Summary (`analyst/rules.py`)

| Check | What it does | LLM output it can override |
|---|---|---|
| `check_quotes_grounded` | Verifies every element evidence quote exists in the actual messages | Flags hallucinated quotes |
| `check_mpesa_cross_match` | Reclassifies payments LLM claimed as "confirmed" but no parsed code exists | Downgrades VERIFIED-LOOKING → CLAIMED ONLY; nulls out invented codes/datetimes |
| `check_element_statuses` | Enforces status rules (PRESENT requires quotes; B/H/I-only statuses) | Forces PRESENT without quotes → UNCLEAR |
| `check_timeline` | Flags timeline entries with neither msg_index nor evidence_id | Informational only, no mutation |
| `compute_verdict` | Caps LIKELY VALID if elements don't support it | Downgrade-only, never upgrades |
| `check_premature_breach` | Resets K to UNCLEAR if performance not yet due per date_checks | Overrides K status |
| `merge_followups` | Deduplicates LLM + deterministic follow-up requests; caps at 6 | Deterministic requests take priority |

---

## 8. Sources of Non-Determinism — Ranked by Impact

### 8.1 `confidence` score — CRITICAL

**What it is**: The LLM emits a float 0.0–1.0 as a holistic judgment of how confident it is.  
**Why it's a problem**: The 83% frontend gate is a hard threshold on this float. At T=0, runs of identical input against qwen/qwen3.8-27b have been observed to produce confidence variation of ±3–5 percentage points. A case that scores 0.82 on run 1 and 0.85 on run 2 gives different eligibility with zero change in evidence.  
**Downstream effect**: Controls whether the "Prepare s.106B Certificate" button appears at all.

### 8.2 Element statuses (C, D, E, F, G) — HIGH

**What it is**: The LLM assigns PRESENT/PARTIAL/UNCLEAR/ABSENT per element.  
**Why it's a problem**: These feed into `compute_verdict`. If G goes PARTIAL on run 2 instead of PRESENT on run 1, the verdict downgrades from LIKELY VALID to POSSIBLY VALID WITH GAPS.  
**Most volatile elements**: G (Certainty) — requires a judgment about whether terms are "definite or ascertainable"; K (Performance) — depends on date reasoning; A (Parties) — depends on how well the LLM resolves informal identities.

### 8.3 `conversation_integrity.rating` — HIGH

**What it is**: The LLM rates the record STRONG, ADEQUATE, or WEAK.  
**Why it's a problem**: WEAK triggers verdict downgrade AND blocks the s106b gate via INTEGRITY condition.  
**Borderline case**: A conversation with one edited message and partial timestamps might be ADEQUATE on run 1 and WEAK on run 2.

### 8.4 M-Pesa classification (pre-reclassification) — MEDIUM

**What it is**: The LLM classifies each payment as VERIFIED-LOOKING / PRESENT BUT UNVERIFIED / SUSPICIOUS / CLAIMED ONLY.  
**Why it's a problem**: The deterministic cross-match layer catches hallucinated confirmations, but VERIFIED-LOOKING vs PRESENT BUT UNVERIFIED for a real payment is still LLM judgment.  
**Mitigation already in place**: `check_mpesa_cross_match` will reclassify any payment the LLM claims is confirmed if no parsed code exists — this is effective.

### 8.5 Quote selection — MEDIUM

**What it is**: Which exact verbatim quotes the LLM selects for each element.  
**Why it's a problem**: The `check_quotes_grounded` check is strict (substring match against normalized messages). Different quote phrasings between runs could cause grounding failures, triggering the INTEGRITY block.

### 8.6 Retry behaviour — LOW

**What it is**: On parse failure, the system retries once (fresh call if >5000 chars, continuation otherwise).  
**Why it's a problem**: A retry always produces a different result. Rare but possible, especially on complex cases that push token limits.

---

## 9. Strategies for Determinism Without Sacrificing Independent Inspection

### Strategy 1 — Replace LLM confidence with a deterministic formula (highest impact)

Compute `confidence` from element statuses **after** verification, not from the LLM.

**Proposed formula** (weights reflect legal importance for contract validity):

```python
ELEMENT_WEIGHTS = {
    "C": 0.18,  # Offer — essential
    "D": 0.18,  # Acceptance — essential
    "E": 0.14,  # Consideration — essential
    "F": 0.10,  # Intention
    "G": 0.12,  # Certainty
    "A": 0.10,  # Parties
    "K": 0.10,  # Performance/breach
    "B": 0.02, "H": 0.02, "I": 0.02, "J": 0.02,
}
STATUS_SCORES = {
    "PRESENT": 1.0, "PARTIAL": 0.5,
    "UNCLEAR": 0.25, "NO_ISSUE_INDICATED": 1.0,
    "NOT_REQUIRED": 1.0, "ABSENT": 0.0,
}

def compute_deterministic_confidence(elements: dict) -> float:
    total_weight = sum(ELEMENT_WEIGHTS.values())
    score = sum(
        ELEMENT_WEIGHTS.get(k, 0) * STATUS_SCORES.get(v.status, 0)
        for k, v in elements.items()
    )
    return round(score / total_weight, 4)
```

Applied in `verify()` after `check_element_statuses()`. The LLM's emitted `confidence` is ignored and replaced. This makes the confidence score **100% reproducible** from the same set of element statuses.

**Trade-off**: The LLM's holistic judgment (e.g. "this is unusually weak for a PRESENT") is lost. Mitigation: expose the formula to the user so they know exactly what drove the number.

---

### Strategy 2 — Replace LLM overall_verdict with a deterministic decision tree

The current approach only downgrades the LLM verdict. The LLM is never upgraded and downgrades are deterministic — but the LLM's starting point is still probabilistic.

**Proposed fully deterministic verdict**:

```python
def deterministic_verdict(elements, date_checks, integrity_flags) -> str:
    statuses = {k: e.status for k, e in elements.items()}
    required_present = {"C", "D", "E", "F", "G"}
    
    if any(statuses.get(k) == "ABSENT" for k in required_present):
        return "LIKELY NOT VALID"
    
    all_present = all(statuses.get(k) == "PRESENT" for k in required_present)
    a_ok = statuses.get("A") in ("PRESENT", "PARTIAL")
    no_gaps = not date_checks.get("no_dates") and "WEAK" not in integrity_flags
    
    if all_present and a_ok and no_gaps:
        return "LIKELY VALID"
    
    any_absent = any(s == "ABSENT" for s in statuses.values())
    if any_absent:
        return "LIKELY NOT VALID"
    
    insufficient = all(
        statuses.get(k) in (None, "UNCLEAR", "ABSENT")
        for k in required_present
    )
    if insufficient:
        return "INSUFFICIENT EVIDENCE"
    
    return "POSSIBLY VALID WITH GAPS"
```

This runs **after** element status correction. The LLM's emitted verdict is used only as a cross-check; discrepancies are logged.

**Trade-off**: The LLM's nuanced reading (e.g. "this is PARTIAL but legally sufficient in context") is overridden. Keep the LLM verdict in the payload as `result.llm_verdict` for transparency.

---

### Strategy 3 — Confidence band / hysteresis around the 83% threshold

If using Strategy 1 (deterministic confidence), this is not needed. If keeping LLM-emitted confidence, add a "borderline" zone:

```python
CERT_THRESHOLD_HARD  = 0.86   # Clear pass
CERT_THRESHOLD_LOW   = 0.78   # Clear fail below this
# Between 0.78 and 0.86 → "Borderline" banner, still offer button but with warning
```

This eliminates the cliff where 0.82 vs 0.84 gives completely different UX.

---

### Strategy 4 — Anchor confidence ranges in the system prompt

Add an explicit rubric to `system_prompt.md` that maps confidence to verdict:

```
CONFIDENCE SCALE:
0.90–1.00  → all core elements PRESENT, strong record, payment confirmed
0.80–0.89  → all core elements PRESENT but some gaps (dates, integrity)
0.65–0.79  → some core elements PARTIAL; use POSSIBLY VALID WITH GAPS
0.40–0.64  → significant gaps; use POSSIBLY VALID WITH GAPS or LIKELY NOT VALID
0.00–0.39  → core elements ABSENT or insufficient evidence
You must stay within these ranges per your chosen overall_verdict.
```

This constrains the LLM's float without replacing it, maintaining its holistic judgment within tighter bands. At T=0, the model tends to follow explicit numeric rubrics consistently.

---

### Strategy 5 — Promote record integrity rating to deterministic code

Currently the LLM decides STRONG / ADEQUATE / WEAK. This should be computed deterministically from `integrity_flags` and `date_checks`:

```python
def compute_integrity_rating(integrity_flags, date_checks, messages) -> str:
    weak_triggers = {
        "edited_messages", "deleted_messages", "low_ocr_confidence",
        "screenshot_boundary", "gap_detected",
    }
    present_flags = set(integrity_flags)
    
    if present_flags & {"low_ocr_confidence", "deleted_messages"}:
        return "WEAK"
    if date_checks.get("no_dates"):
        return "WEAK" if len(messages) < 5 else "ADEQUATE"
    if present_flags & weak_triggers:
        return "ADEQUATE"
    return "STRONG"
```

This removes the LLM's rating from the INTEGRITY gate condition (`_check_integrity` in gate.py), making that check reproducible.

---

### Strategy 6 — Structured output enforcement via JSON schema

Groq supports `response_format={"type": "json_object"}`. Add `json_schema` mode if the endpoint supports it. Combined with Pydantic validation on `AnalystResult`, this enforces that confidence is a float, statuses are enum values, etc. It doesn't eliminate variance but eliminates malformed outputs that trigger a retry (which produces a different result).

Already partially in place via Pydantic retry. Formal schema enforcement would reduce retries further.

---

### Strategy 7 — Pin analysis snapshots for the s106b gate

The s106b gate (`gate.py`) already operates on the **stored** analysis snapshot, not a live re-run. This is the right design — the gate is deterministic on a fixed snapshot.

**What to verify**: Confirm that the `analysis_id` stored in Firestore (`analysis_store.save_analysis_snapshot`) is always the snapshot from the **same** run that the user reviewed. If a user re-runs the analysis, a new `analysis_id` is created. The certificate questionnaire should always reference the snapshot the user confirmed, not the latest run.

Currently `DeviceQuestionnaire` sends `analysis_id` and the backend loads that specific snapshot. This is correct.

---

### Strategy 8 — Consistency check across element statuses (guard rail)

Add a cross-element consistency check in `rules.py` after all status corrections:

```python
def check_cross_element_consistency(result: AnalystResult) -> list[str]:
    """Flag logical contradictions between elements."""
    warnings = []
    statuses = {k: e.status for k, e in (result.elements or {}).items()}
    
    # If E (consideration) is ABSENT but K (performance) is PRESENT, contradiction
    if statuses.get("E") == ElementStatus.ABSENT and statuses.get("K") == ElementStatus.PRESENT:
        warnings.append("E_ABSENT_but_K_PRESENT: payment confirmed with no agreement on price")
    
    # If D (acceptance) is ABSENT but E is PRESENT, cannot have considered without accepting
    if statuses.get("D") == ElementStatus.ABSENT and statuses.get("E") == ElementStatus.PRESENT:
        warnings.append("D_ABSENT_but_E_PRESENT: payment evidenced but acceptance not found")
    
    return warnings
```

Include in the verification output. These don't mutate anything but flag when the LLM produced a logically inconsistent set of statuses — a signal that the run should be treated as lower confidence or the user prompted to re-run.

---

## 10. Priority Implementation Order

| Priority | Strategy | Effort | Impact |
|---|---|---|---|
| 1 | **Strategy 1** — Deterministic confidence formula | Low (add to `rules.py` + `verify()`) | CRITICAL — eliminates the main gate variance |
| 2 | **Strategy 5** — Deterministic integrity rating | Low (pure function in `enrich.py` or `rules.py`) | HIGH — removes INTEGRITY gate variability |
| 3 | **Strategy 4** — Anchor confidence ranges in prompt | Minimal (edit `system_prompt.md`) | MEDIUM — constrains LLM float at source |
| 4 | **Strategy 2** — Deterministic verdict tree | Medium (replace `compute_verdict`) | HIGH — but only matters after confidence is fixed |
| 5 | **Strategy 8** — Cross-element consistency check | Low (add to `rules.py`) | MEDIUM — catches logical contradictions |
| 6 | **Strategy 3** — Confidence hysteresis band (frontend) | Low (edit `SCCVisionTest.tsx`) | MEDIUM — safety net if LLM confidence is kept |
| 7 | **Strategy 6** — JSON schema mode on Groq | Low (edit `client.py`) | LOW — reduces retry variance |

---

## 11. What Must Stay LLM-Driven

The following **should not** be replaced with deterministic code because they require reading and understanding the conversation text:

- **Verbatim quote selection** — only the LLM can read message text and pick the right quotes
- **Element reasoning** — the natural language explanation of why an element is PRESENT/ABSENT
- **`plain_summary`** — the user-facing 200-word narrative
- **`red_flags`** — requires reading for implicit issues (threats, inconsistent stories, etc.)
- **`questions_for_parties`** — requires understanding who the parties are and what's contested
- **M-Pesa pattern matching** (pre-reclassification) — recognising non-standard confirmation formats
- **`strengthen_checklist`** — advice specific to the case

The goal is to have the LLM do what it's good at (reading + reasoning over text) and have deterministic code do what should be reproducible (scoring, thresholds, structural validation).

---

## 12. Files to Modify per Strategy

| File | Change needed |
|---|---|
| `backend/SCCVision/analyst/rules.py` | Add `compute_deterministic_confidence()`, `compute_integrity_rating()`, `check_cross_element_consistency()` |
| `backend/SCCVision/analyst/verify.py` | Call new functions; replace `result.confidence` after verification |
| `backend/SCCVision/analyst/system_prompt.md` | Add confidence scale rubric (Strategy 4) |
| `backend/SCCVision/analyst/client.py` | Add `response_format` JSON mode (Strategy 6) |
| `frontend/app/components/sccvision/SCCVisionTest.tsx` | Adjust `CERT_THRESHOLD` bands if keeping LLM confidence (Strategy 3) |
| `backend/SCCVision/s106b/gate.py` | No changes needed — already fully deterministic |
| `backend/SCCVision/analyst/enrich.py` | Move integrity rating computation here (deterministic, feeds into analysis_input) |
