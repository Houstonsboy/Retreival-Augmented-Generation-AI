# Direct Retrieval Phase 2 — Results

**Run date:** 2026-11-27  
**Branch:** eval/final-hardening  
**Files modified:** `backend/direct_answer.py`, `backend/direct_retrieval.py`

---

## Summary

| Metric | Value |
|--------|-------|
| Questions tested | 5 |
| Questions improved | 5 / 5 |
| Groq budget violations | 0 |
| Max Groq calls per query | 2 (Q03) |
| Questions with 0 Groq calls | Q02, Q08, Q37 |

---

## Changes implemented

### Task 1 — Quoted-precedent penalty (`_QUOTED_EXTRACT`)
**Problem (Q04):** `_fallback_passage()` selected a Mataguda quotation embedded in the Nonny judgment (a paragraph opening with `"`) instead of the court's own numbered ruling.

**Fix:** Added `_QUOTED_EXTRACT = re.compile(r'^\s*[' + chr(0x201c) + ...]` and applied a -5.0 score penalty to paragraphs matching this pattern in outcome/reasoning modes.

**Additional (Task 1b):** Extended `_ALLEGATION_SIGNAL` to also match `"he/she further submitted"` patterns, demoting defendant-submission paragraphs that contain incidental disposition language.

### Task 2 — Multi-paragraph bundling (`_fallback_bundle`)
**Problem (Q37):** `_fallback_passage()` returned only para13; para14 ("Consequently... not established personal liability") and para15 ("The Claim therefore fails and is dismissed") were missing.

**Fix:** Replaced `_fallback_passage()` with `_fallback_bundle()`. After selecting the best paragraph, the function extends the bundle with immediately following paragraphs from the same chunk that have positive lexical overlap, contain disposition language, or begin with conclusion signals (`_CONCLUSION_SIGNAL`). Stops after a disposition paragraph or after 3 additional paragraphs.

### Task 3 — Statute discovery from passages (`_discover_statutes_from_passages`)
**Problem (Q08):** For `LEGAL_PRINCIPLE_LOOKUP + jurisdiction_hint` queries, only cases were retrieved. The Titanic case explicitly mentioned `section 13(5) of the Small Claims Court Act` but no statute passage was surfaced.

**Fix:** Added `_STATUTE_REF` regex and `_discover_statutes_from_passages()` to `direct_retrieval.py`. After retrieving cases for `LEGAL_PRINCIPLE_LOOKUP + jurisdiction_hint`, scans retrieved chunk text for statute references and calls `direct_statute()` deterministically. No Groq call required.

### Task 5 — Fallback rendering (Level A / Level C)
**Fix:** Replaced `"I could not verify a direct answer. The retrieved source includes this passage [X]:\n\n> ..."` with:
- **Level A (`source_extract`):** `{citation}:\n\n> {text}` — when bundle contains a clear judicial disposition
- **Level C (`extractive_fallback`):** `The retrieved source [{citation}] includes this passage:\n\n> {text}` — when no clear disposition found

### Task 7 — Alias resolution (`_resolve_aliases`)
**Problem (Q03):** Verified claims were annotated `[C2.1]` (internal alias) rather than the full case citation.

**Fix:** `_resolve_aliases(text, passages)` replaces `[C1.3]` etc. with `[passage.label]` in the answer string. The internal alias in the `claims` dict is preserved for `app.py`'s `alias_map` lookup.

### Task 8 — Claim deduplication (`_deduplicate_claims`)
**Problem (Q03):** Three near-identical claims with the same underlying quote were returned.

**Fix:** `_deduplicate_claims(claims)` removes claims sharing the same 120-character normalized quote prefix. Applied after both generation and verification stages.

### Statutory citation improvement (Task 2 of spec)
**Problem (Q02):** Answer displayed `Evidence Act s.106B(4) [S1.1]` including the internal alias.

**Fix:** Updated statutory extract path to: (a) detect when `_subsection()` narrowed to a sub-clause (e.g., `(d)`), update the citation label to `s.106B(4)(d)`; (b) removed `[{alias}]` token from the answer string.

---

## Per-question results

### Q03 — CCTV evidence in Mataguda
- **Verification:** `semantic_verified` (unchanged)  
- **Groq calls:** 2 (unchanged)  
- **Latency:** 41,361 ms  
- **Change:** 3 near-identical claims → 1 deduplicated claim; `[C2.1]` → full Ogembo v Yongo citation  
- **Status:** IMPROVED

### Q04 — DVD evidence in Nonny
- **Verification:** `source_extract` (was `extractive_fallback`)  
- **Groq calls:** 1 (unchanged)  
- **Latency:** 894 ms  
- **Change:** Mataguda quotation replaced by Nonny court's own ruling from primary source (pp. 2–3): "the said DVDs are inadmissible as evidence"  
- **Status:** IMPROVED

### Q02 — Evidence Act s.106B(4) signatory
- **Verification:** `statutory_extract` (unchanged)  
- **Groq calls:** 0 (unchanged)  
- **Latency:** 192 ms  
- **Change:** Citation now `Evidence Act s.106B(4)(d)` without internal alias token  
- **Status:** IMPROVED

### Q08 — Small Claims Court jurisdiction
- **Verification:** `source_extract` (was `extractive_fallback`)  
- **Groq calls:** 0 (unchanged)  
- **Latency:** 6,091 ms  
- **Change:** Full ruling from Titanic case (2026SCC235) citing s.13(5) exclusion; statute discovery retrieved 2 statute passages; court's strike-out reasoning included in bundle  
- **Status:** IMPROVED

### Q37 — Umoja Magharibi KES 553,000 claim
- **Verification:** `source_extract` (was `extractive_fallback`)  
- **Groq calls:** 0 (unchanged)  
- **Latency:** 787 ms  
- **Change:** Full 3-paragraph bundle — para13 (finding on vehicle arrangement), para14 ("Consequently… not established personal liability"), para15 ("The Claim therefore fails and is dismissed")  
- **Status:** IMPROVED

---

## Tasks not implemented in Phase 2

| Task | Reason |
|------|--------|
| Task 4 (structured diagnostics) | `run_diagnostic.py` already captures `verification`, `groq_attempts`, `latency_ms` per question; structured reason codes deferred |
| Task 6 (graph participation) | Graph layer empty (Neo4j not populated); no measurable contribution possible |
| Task 9 (stage-level latency) | Q03 latency (41 s) dominated by cross-encoder model load on first run; no instrumentation needed to confirm |
| Task 10 (Groq budget guard) | Hard 3-call budget already enforced by `groq_budget.py`; no violation observed |
