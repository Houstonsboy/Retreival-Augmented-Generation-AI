# Prompt — Re-index Case Law & Build the Case-Law GraphRAG (v1)

> For the coding agent. Prerequisite: `GRAPHRAG_SETUP_PROMPT.md` is done (`backend/graph/`, migration 0001, Docker Neo4j).
> References in the repo or project: `GRAPHRAG_PART1_CASELAW.md` (analysis), `scc_caselaw_seed_graph.json` (hand-checked golden graph: 149 nodes, 269 edges).
> Source judgments: `SCCCases/` (11 files: 10 PDF, 1 DOCX).

## 0. Verify before building (mandatory)

Before writing code, inspect the codebase and write `docs/graph/PREFLIGHT.md` answering:
1. How `Retriever.py`, `RulesRetriever.py` and `Qclassifier.py` read the `case_law` collection: metadata keys used, filters, BM25 build, ID format. Which changes below would break them?
2. The current ingestion script for case law (path, chunking, embedding prefix `passage:`, model `intfloat/e5-base-v2`).
3. Whether `backend/graph/` (setup prompt) exists and its tests pass, and whether Neo4j is reachable (`python -m graph.admin status`).
4. Whether the Black's Law pipeline is fully removed (`grep -ri "black_law\|blacklaw\|BlackLaw" --exclude-dir=.git` → empty). If not, finish Phase A step 3 of the setup prompt first.
5. For each instruction below: **is it logical and optimal given the actual code?** If something conflicts (e.g. a retriever depends on a metadata key being removed, or a better hook exists), propose the adjustment in PREFLIGHT.md and follow the adjusted plan. Don't silently deviate, and don't blindly follow an instruction that breaks working code.

Stop and report if Neo4j isn't reachable or the setup tests fail.

---

## Part A — Fix and re-index the vector DB (Chroma `case_law`)

Goal: all 11 judgments indexed, clean text, **one record per numbered paragraph**, IDs shared with the graph.

### A1. Text extraction and repair (`backend/ingest/caselaw/extract.py`)
- PDFs: PyMuPDF (already in the stack). DOCX: python-docx. Keep page numbers per line.
- **Ligature repair:** the Kenya Law PDFs (Ogembo, Obado) lost "fi/fl/ff". Apply a repair dictionary before anything else. At minimum:
  `certicate→certificate, led→filed (only where followed by a document/date context; review list), nding→finding, satised→satisfied, oending→offending, justication→justification, adavit→affidavit, plainti→plaintiff, dierent→different, aected→affected, suered→suffered, ecient→efficient, denes→defines, conrm→confirm, specic→specific, signicant→significant, sucient→sufficient, eect→effect, oce→office, ofce→office, rst→first, nal→final`.
  Use word-boundary regexes. Add a check: count tokens not in a wordlist (e.g. `/usr/share/dict/words` plus legal terms). If more than 0.5% are unknown after repair, log it and list the top 30 unknown tokens in the ingest report.
  Note: "led" is also a real word. Only replace it in patterns like "led a", "led on", "led in court", "led submissions", "led by way". Log each replacement.
- Strip page furniture: Kenya Law URLs, "Judgment SCCCOMM No… pg. N", "Hon A.Z Ogange- R.M pg. N", lone page numbers, "I certify that this is a true copy".

### A2. Header parsing (deterministic)
Extract into a `CaseMeta` pydantic model:
- `case_key`: use `graph.keys.case_key`; neutral citation from the filename (e.g. `2024KEHC15763(KLR)` → `KEHC-2024-15763`)
- `name`, `aliases[]`, `citation`
- `court_code` (SCC, KEHC, KECA, KEELC…), `station`
- `case_number`
- `decision_date`, `decision_type` (Judgment/Ruling)
- `judges[]`
- `parties` (roles only + names in text; no party nodes)
- `appeal_from` (if stated)

Where regex fails, fill from a small hand-written overrides file `ingest/caselaw/overrides.yaml` (e.g. Ndeti's .docx has no neutral-citation header). **Don't use the LLM for header fields.**

Expected (check against the seed graph): Barisa KEHC-2011-1481 HC Mombasa 2011-07-22 Ruling; Nonny KEHC-2014-6468; PNN KEHC-2020-8905; Obado KEHC-2024-17038; Ogembo KEHC-2024-15763; Express KECA-2001-154; Safaricom KECA-2010-346; Roysa KEELC-2021-220; Ndeti SCC-2026-241; New Titanic SCC-2026-235; Umoja SCC-2026-242.

### A3. Paragraph segmentation
- Split on judgment paragraph numbers (`^\s*\d{1,3}\.\s`), keeping sub-lists (a)(b)(c) and block quotes with their parent paragraph.
- Text before paragraph 1 (title, parties, coram) → `para_no = 0` ("header").
- **Unnumbered judgments** (Barisa, Express, Safaricom): split on blank-line paragraph boundaries and assign sequential numbers with `para_numbering = "synthetic"`.
- **Separate opinions:** Safaricom has `RULING OF OMOLO, J.A`, and separate Waki and Nyamu sections; Express has `JUDGMENT OF SHAH, J.A.` Detect `(JUDGMENT|RULING) OF <NAME>` headings. Each paragraph gets `opinion_author` and `opinion_kind` (lead | concurring | dissent | sole). For Safaricom, record the opinions; mark `kind` as `unknown` if it can't be determined, never guess.
- **Long paragraphs** over 2,500 characters: split into `p44a`, `p44b`… with the same `para_no`. Never split mid-sentence.
- **Quote spans:** detect quoted blocks (indented, or within “ ” after "stated:", "held:", "observed that:"). Record `quote_spans: [{start, end, attributed_to_raw}]`, with attribution taken from the nearest preceding citation in the same paragraph.

### A4. Re-index Chroma
- Build into a **new** collection `case_law_v2`. Don't overwrite `case_law` until A5 passes, then switch the retriever config and keep `case_law` for one release as a rollback.
- Record ID = `para_key` (`KEHC-2024-15763#p44`, `…#p44a`).
- Document text = repaired paragraph text, prefixed for embedding as today (`passage: `).
- Metadata, flat only (Chroma doesn't allow lists; join lists with ` | `):
  `case_key, para_no, para_sub, para_numbering, case_name, citation, court_code, court_level, station, decision_date, year, decision_type, judges, opinion_author, opinion_kind, domains, has_quote (bool), quote_sources, page_start, page_end, source_file, doc_hash, ingest_run_id, text_repaired (bool)`.
  Keep any metadata key the PREFLIGHT found the retrievers depend on (e.g. `file_name`, `parties`, `legal_domain`, `chunk_index`). Map them from the new fields; don't break the retrievers.
- `legal_domain` stays for compatibility, but holds the primary domain from the corrected list. `domains` holds all of them. Use these corrections: Ogembo = Evidence Law | Defamation; Nonny = Evidence Law | IP; Ndeti = Civil Procedure | Insurance; Safaricom = Arbitration | Civil Procedure.
- Drop `firac_component`, `label`, `winning_party` (empty or unreliable), **unless** PREFLIGHT shows code reads them. Outcomes now live in the graph.
- Remove the two orphaned HNSW folders from the old store (back them up first).

### A5. Retrieval regression check (gate before switching)
Create `eval/caselaw_questions.yaml` with 20 questions and the expected case and paragraph for each. Start with these and add 10 more:
1. Is a s.106B certificate required for WhatsApp evidence? → Ogembo p38–46, Barisa
2. Who must sign a s.106B certificate? → Barisa, Obado p29
3. Can a certificate be filed late? → Nonny p13, Ogembo p44
4. Is an expert affidavit a valid s.106B certificate? → Obado p28–29
5. Are emails admissible if the maker testifies? → PNN p13
6. Does the SCC have jurisdiction over employee theft claims? → New Titanic p10–12
7. When will a regular default judgment be set aside? → Ndeti p32–38, Express
8. Is service via WhatsApp valid? → Ndeti p25–26
9. Adverse inference for failing to call a witness? → Umoja p9
10. Is the SCC bound by strict rules of evidence? → Umoja p12, Ogembo p38

Run the existing retriever (BM25 + vector + RRF + rerank) against `case_law` and against `case_law_v2`. Report recall@5 and MRR per question. **Switch only if v2 is at least as good overall and no question drops from a hit to a miss.** Otherwise report and stop.

---

## Part B — Build the case-law graph

### B1. Migration `0002_caselaw_v1.cypher` (new file; never edit 0001)
Add uniqueness constraints (`IF NOT EXISTS`) for any labels missing from 0001:
- `Opinion(opinion_key)`, `Article(article_key)`, `Rule(rule_key)`, `Definition(definition_key)`
- `ClaimType(code)`, `Outcome(code)`, `Award(award_key)`
- `ContractElement(code)`, `GateCondition(code)`
- `Station(station_key)`, `PartyType(code)`

Add indexes: `Paragraph(case_key, para_no)` if missing; `Provision(statute_key)`; `Award(type)`.

Seed fixed reference nodes, idempotent with `MERGE`:
- **Courts and BINDS:** SC→CoA→{HC, ELC, ELRC}→{SCC, MC}; also EACA (historic CoA) and FOREIGN rank 9.
- **ContractElement A–K** (names as in the SCC analyser).
- **GateCondition:** 4a, 4b, 2a, 2b, 2c, 2d, 4d, INTEGRITY. Each with `-[:GROUNDED_IN]->` the matching `Provision` stub (s.106B(4)(a), (4)(b), (2)(a)…(2)(d), (4)(d)); INTEGRITY → s.106B(2).
- **EvidenceType:** whatsapp, sms, mpesa, email, photo, cctv, dvd, social_post, document.
- **Outcome codes:** ALLOWED, DISMISSED, STRUCK_OUT, SET_ASIDE_REFUSED, SET_ASIDE_GRANTED, OBJECTION_UPHELD, OBJECTION_DISMISSED, PARTLY_ALLOWED, GRANTED, REFUSED.
- **ClaimType:** money_lent, sale_goods_services, property_damage_tort, personal_injury, subrogation_recovery, employment_founded, defamation, bailment, other.
- **PartyType:** individual, company, insurer, employer, government.

`ClaimType -[:WITHIN|EXCLUDED_BY]-> Provision`: create `employment_founded -[:EXCLUDED_BY]-> SCC Act s.13(5)` now. Create the `WITHIN` edges only after the SCC Act is ingested in the statutes phase; leave a TODO.

### B2. Final edge set (v1). Implement exactly these; add them to the `RelType` enum
Direction rule: from the dependent thing to the authority.
- **Documents:**
  - `SourceDocument-CONTAINS->Case`
  - `Case-HAS_PARAGRAPH->Paragraph`
  - `Paragraph-NEXT->Paragraph`
  - `Case-HAS_OPINION->Opinion-AUTHORED_BY->Judge`
  - `Paragraph-IN_OPINION->Opinion`
- **Citations:**
  - `Paragraph-CITES{treatment, via, confidence}->Case`, where treatment ∈ FOLLOWED, APPLIED, CONSIDERED, DISTINGUISHED, DEPARTED_FROM, OVERRULED, QUOTED, CITED_BY_PARTY
  - `Paragraph-QUOTES->Case`
  - Cited cases not in the corpus become stub `Case{in_corpus:false}`
  - **Don't store Case→Case CITES.** Provide a Cypher view or helper that derives it.
- **Courts:**
  - `Case-DECIDED_IN->Court`, `Court-BINDS->Court`, `Case-APPEAL_FROM->Case|LowerCourtDecision`
  - **Don't store "binding" on edges.** Add `graph/authority.py: is_binding(cited_court, forum_court)` computed from BINDS paths.
- **Meaning:**
  - `Paragraph-STATES{role: ratio|obiter}->Principle`
  - `Principle-ANSWERS->Issue`
  - `Case-DECIDES->Issue`
  - `Principle-QUALIFIES|CONFLICTS_WITH|SUPPORTS->Principle`
- **Law** (provisions as stubs now; full text in the statutes phase):
  - `Paragraph-INTERPRETS|APPLIES|MENTIONS->Provision|Article|Rule`
  - `Principle-DERIVED_FROM->Provision|Article`
  - `Provision-PART_OF->Provision|Statute`
  - `Rule-MADE_UNDER->Statute`
- **SCC:**
  - `Case-CAUSE_OF_ACTION->ClaimType`
  - `Case-HAS_OUTCOME->Outcome`
  - `Case-AWARDED->Award{type, amount_kes, payer_role}`
  - `Principle-BEARS_ON->ContractElement`
  - `GateCondition-GROUNDED_IN->Provision`
- **Evidence and doctrine:**
  - `Paragraph-RULED_ON_EVIDENCE{result}->EvidenceType`
  - `Case-INVOLVES->Concept`
- **Cheap extras (from headers, no LLM):**
  - `Case-DECIDED_BY->Judge`
  - `Case-AT_STATION->Station`
  - `Case-HAS_PARTY_TYPE->PartyType`

Reserved for the statutes phase, **not built now:**
- `Provision-REFERS_TO->Provision`, `Statute-AMENDS|REPEALS->Statute`
- `Provision VERSION {valid_from, valid_to}`
- `Statute-GIVES_EFFECT_TO->Article`
- `Provision-DEFINES->Definition`, `Concept-DEFINED_AS->Definition`, `Definition-SAME_TERM_AS->Definition`

Out of scope: party or advocate names as nodes, client facts, user activity.

### B3. Deterministic extraction (no LLM) — `backend/ingest/caselaw/deterministic.py`
- **Citations:** regexes for
  - `[YYYY] eKLR`
  - `[YYYY] (KEHC|KECA|KESC|KEELC|KEELRC|SCC) n (KLR)`
  - `(YYYY) n? EA n` and `[YYYY] EA n`
  - `(YYYY) KLR n`
  - English reports (AC, QB, WLR, All ER), Ugandan `UGHC`
  - unreported `Civil Appeal No. n of YYYY`

  Each citation is captured with the preceding case name (`X v/vs/versus Y`).
- **Provisions:**
  - `[Ss]ection n[A-Z]?(\(n\))*(\([a-z]\))? of the <Act>`, `S\.\s?106B(4)(d)` shorthand
  - `Order [IVXL\d]+ [Rr]ule n(\(\w\))?`
  - `Rule n(\(n\))? of the <Rules>`
  - `Article n(\(n\))?(\([a-z]\))? of the Constitution`

  Resolve the Act from context when it's omitted ("the Act" → the last named Act in the paragraph or the previous one). Build `provision_key` with `graph.keys`.
- **Entity resolution:**
  - Normalise names with `keys.normalise_name`.
  - Merge stubs by citation key first. Otherwise merge by fuzzy name match (`rapidfuzz` token_set_ratio ≥ 90 **and** same year) into a single node with `aliases[]`.
  - Must merge: Mataguda / Matuguda / Matriguda / Berisa → `KEHC-2011-1481`; "Giulla vs Cassman Brown" → Giella v Cassman Brown; Nonny cited as "Civil Case 490 of 2013" → `KEHC-2014-6468`.
  - Log every merge with its score.
- **From headers:** DECIDED_IN, DECIDED_BY, AT_STATION, APPEAL_FROM, HAS_OPINION.
- **Outcomes and awards:** regex for the final orders (dismissed/struck out/allowed…, `costs assessed at Kshs. n`, `each party to bear own costs`). Cases where the regex can't decide go to the LLM pass (B4) as `outcome_unresolved`.
- **QUOTES:** from A3 quote spans plus the resolved citation.

### B4. LLM extraction — 2 calls per judgment, at build time only (Groq `qwen/qwen3.8-27b`, T=0, fixed seed, strict JSON)
11 judgments × 2 calls = 22 calls in total. Use the existing rate limiter. Cache responses by `doc_hash + prompt_version` in `ingest/caselaw/cache/`, so reruns cost 0 calls.

**Input:** numbered paragraphs (compact `p<n>: text`), plus the deterministic findings (citations and provisions found per paragraph). The LLM only **labels and links what's already found**, and may add items only with paragraph references.

**Call 1 — paragraph roles and citation treatment:**
```json
{"paragraphs":[{"para":"p44","firac":"facts|issue|rule|analysis|conclusion|orders|header|submissions_claimant|submissions_respondent",
  "citations":[{"citation_id":"<from deterministic list>","treatment":"FOLLOWED|APPLIED|CONSIDERED|DISTINGUISHED|DEPARTED_FROM|OVERRULED|QUOTED|CITED_BY_PARTY","reason_span":"<=15 words copied from the paragraph"}],
  "provisions":[{"provision_id":"<from list>","mode":"INTERPRETS|APPLIES|MENTIONS"}],
  "evidence_rulings":[{"type":"whatsapp|sms|mpesa|email|photo|cctv|dvd|social_post|document","result":"ADMITTED|EXCLUDED|EXCLUDED_WITH_LEAVE_TO_CURE|ACCEPTED_FOR_SERVICE|ACCEPTED_TO_LINK_IDENTITY|RELIED_ON"}]}],
 "outcome":{"code":"<Outcome code>","para":"p51"}, "claim_type":"<ClaimType code>", "party_types":["individual","insurer"]}
```
Rules for the prompt:
- Citations made in a paragraph summarising a party's submissions are `CITED_BY_PARTY`, unless the court later adopts them.
- `DEPARTED_FROM` and `OVERRULED` need explicit language.
- If unsure, use `CONSIDERED`.

**Call 2 — Issues and Principles:**
- Input: the curated **Issue list** `ingest/caselaw/issues.yaml` (seed it with the 18 Issues in `scc_caselaw_seed_graph.json`), the existing **Principle list** (from previously processed cases), and the judgment's `analysis`/`conclusion` paragraphs.
- Output:
```json
{"issues":[{"issue_id":"<existing id>"|null,"proposed_question":"<only if no existing id fits>"}],
 "principles":[{"principle_id":"<existing id>"|null,"proposed_text":"<neutral one-sentence rule if new>","role":"ratio|obiter",
   "paras":["p41","p42"],"answers_issue":"<issue id or proposed>","derived_from":["<provision_id>"],
   "bears_on":["A".."K"],"relations":[{"type":"QUALIFIES|CONFLICTS_WITH|SUPPORTS","target_principle_id":"<existing id>","para":"p38"}]}],
 "concepts":["adverse_inference","subrogation"]}
```
- **New Issues and Principles are written with `status:"proposed"`** and listed in `ingest/caselaw/review_queue.yaml` for human approval (supervisor or advocate). Proposed nodes are visible to query code only when `GRAPH_INCLUDE_PROPOSED=true` (default false). Ship `scripts/approve_proposals.py` to approve, merge into an existing id, or reject.
- **Process cases oldest first** (Express 2001 → … → 2026 SCC cases). Later cases then link to principles that already exist, and QUALIFIES/CONFLICTS relations resolve.

### B5. Grounding and validation (code, after each LLM call)
Reject an item, log it, and move on if:
- a `para` doesn't exist;
- a `citation_id` or `provision_id` isn't in the deterministic list;
- a `reason_span` isn't found in the paragraph (fuzzy ≥ 90);
- a principle has no supporting `paras`;
- a relation target doesn't exist.

The LLM can never create a citation that the regex didn't find.

### B6. Writing to Neo4j
- Use `graph.repository` (`MERGE` on keys, `UNWIND` batches). Every node and edge carries `source_doc_hash, ingest_run_id, extractor (regex|header|llm:<model>@<prompt_version>|manual|seed)`.
- **One `IngestRun` node per run.** Re-ingesting a document deletes only that document's derived edges (matched by `source_doc_hash`) before rewriting. Stub nodes and seeded reference nodes are never deleted.
- **Paragraph nodes store** `text`, `para_no`, `firac`, `quote_spans` (JSON string), `page_start/end`. The **embedding stays in Chroma** (Neo4j `embedding` property left empty; vector index unused for now).
- CLI: `python -m ingest.caselaw run [--case KEY] [--dry-run] [--no-llm]`. `--no-llm` builds only the deterministic layers (useful when the Groq quota is low).

### B7. Validation against the golden seed graph
`scripts/eval_graph_vs_seed.py` compares the built graph with `scc_caselaw_seed_graph.json` and reports precision and recall per edge type:
- CITES (src case, dst case, treatment)
- INTERPRETS/APPLIES (case, provision)
- HOLDS/STATES (case, principle, matched by semantic similarity ≥ 0.85 or explicit id)
- RULED_ON_EVIDENCE
- QUALIFIES/CONFLICTS_WITH
- Outcome

Targets:

| Edge type | Target |
|---|---|
| CITES target resolution | recall ≥ 0.95 |
| treatment | accuracy ≥ 0.85 |
| provisions | recall ≥ 0.90 |
| principles | recall ≥ 0.80 |
| QUALIFIES/CONFLICTS | find all 4 seed relations |

Include specific checks:
- Barisa has in-degree ≥ 3 (Nonny, PNN via MNN, Ogembo);
- Ogembo→Nonny is DISTINGUISHED at p44;
- New Titanic→Maisha Mapya is DEPARTED_FROM;
- Umoja's "SCC not bound by strict rules of evidence" QUALIFIES "certificate mandatory";
- Ndeti p17 RULED_ON_EVIDENCE mpesa ACCEPTED_TO_LINK_IDENTITY, BEARS_ON A.

Where the seed is right and the extraction is wrong, fix the prompts or regexes. **Don't edit the seed to match.** If you believe the seed itself is wrong, list it in the report with the paragraph text.

### B8. Query helpers (no pipeline wiring yet)
In `backend/graph/queries.py`, write parameterised templates. Each returns `para_key`s plus a path explanation:
- `doctrinal(issue_ids|principle_ids, forum="SCC")`: Principles → supporting paragraphs → QUALIFIES/CONFLICTS (always included) → binding flag.
- `case_lookup(case_key)`: holdings, outgoing citations with treatment, incoming citations.
- `provision(provision_key)`: walks the PART_OF tree; cases ranked by mode (INTERPRETS > APPLIES > MENTIONS), then authority.
- `from_paragraphs(para_keys)`: anchor Chroma hits, then expand ≤ 2 hops, capped at 12 paragraphs.

Unit-test each against the built graph, e.g. `doctrinal("issue:e-evidence-in-scc")` must return paragraphs from Umoja and Ogembo p38 **and** the qualified rule. **Don't hook into `Retriever.py`.** Query-time integration is the next prompt.

## Acceptance criteria
1. PREFLIGHT.md written. Any deviations from this prompt are justified there.
2. `case_law_v2` holds all 11 judgments, paragraph records with `para_key` IDs and no ligature artefacts (`certicate` → 0 hits). It's at least as good as `case_law` on the 20 eval questions, with no hit turned into a miss. The retriever switch happens only after that, with `case_law` kept for rollback.
3. Migration 0002 applied; seed reference nodes present.
4. `python -m ingest.caselaw run` builds the graph in ≤ 22 Groq calls, and a rerun uses 0 calls (cache).
5. The seed-graph evaluation meets the targets in B7. Report attached.
6. Query helper tests pass. `GRAPH_ENABLED=false` still leaves the live pipeline unchanged.

## Report back (keep it short)
- PREFLIGHT deviations.
- Eval table (v1 vs v2 retrieval).
- Graph counts by label and edge type.
- B7 precision/recall table.
- The review queue (proposed Issues/Principles).
- Groq calls used.
- Anything blocked.