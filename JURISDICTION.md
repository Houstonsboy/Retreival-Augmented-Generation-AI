# SCC RAG Pipeline Audit Report

> **Scope:** Read-only forensic audit of the Kenyan Small Claims Court legal research and evidence inspection RAG system.
> **Trigger:** Dual failure on employer-storekeeper cash-recovery prompt — holding inversion + single-intent tunnel vision.
> **Date:** 2026-11-22

---

## 1. Pipeline Architecture Map (Files & Line Numbers)

- **Entrypoint / Chat Layer:** [`backend/app.py:62–78`](backend/app.py#L62) — Raw query extracted at line 62 (`query = data.get('message', '')`), passed **directly** to `run_rag_pipeline(query)` at line 78 (defined in `pixe.py`). **No intent classification, no query rewriting, no multi-query splitting** at this entry point. A second entry point at `app.py:1253–1335` (`/api/qretrieve`) does invoke `Qclassifier` (line 1281) but that route is only used by the advanced research panel, not the main chat.

- **Intent Classifier:** [`backend/Qclassifier.py:423–741`](backend/Qclassifier.py#L423) — `classify_legal_query()` calls `qwen/qwen3.8-27b` via Groq (temp 0.1, max 1500 tokens) to produce a single `vector_query` string (comma-separated keywords) plus `retrieval_mode`, `intents[]`, and `legal_domains[]`. **Critical gap: a multi-intent prompt is collapsed into one `vector_query` string — there is no multi-query fan-out.**

- **Ingestion & Chunking Script:** [`backend/build_case_chunks.py:35–417`](backend/build_case_chunks.py#L35) — `MAX_CHARS_PER_CHUNK = 3000`, `OVERLAP_PARAGRAPHS = 1` (one paragraph of overlap). PyMuPDF (`fitz`) for PDF coordinate-aware extraction; `python-docx` for DOCX (paragraph-split, no coordinates). No FIRAC annotation — `firac_component` is stored as `""` for all chunks in the current DB. Embedding model: `intfloat/e5-base-v2` (768-dim). Chunk prefix: `"passage: "` for documents, `"query: "` for queries.

- **Vector Store & Retrieval Layer:** [`backend/Retriever.py:563–790`](backend/Retriever.py#L563) — ChromaDB collection `case_law` (`backend/CaseLawDB/`). Hybrid retrieval: BM25 top-50 + vector top-50 → RRF fusion (`k=60, n=30`) → cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2`). **`TOP_N_PARENTS = 3`** (line 47): returns at most 3 distinct parent cases. **`MAX_CHUNKS_PER_COMPONENT = 1`** (line 50): 1 chunk per FIRAC component per case. **No MMR / diversity mechanism.** Final `n_results` to ChromaDB: `min(50, collection.count())` = 50 (line 682).

- **Prompt & Generation Layer:** [`backend/Retriever.py:123–183`](backend/Retriever.py#L123) — System prompt: `"You are a legal research assistant analyzing a court case for relevance to a user's query."` LLM: `qwen/qwen3.8-27b`, temperature 0.3, max_tokens 1000, Groq API, timeout 30 s. The prompt injects `winning_party` as a metadata label but does **not** instruct the model to use it as the authoritative holding — the model is free to draw conclusions from chunk text alone.

---

## 2. Database Forensics: `2026SCC235(KLR)` in ChromaDB

- **Total chunks stored for this case:** `4`

- **Total vectors in DB:** `78` across 9 cases:
  ```json
  {
    "2026SCC242(KLR)":  4,
    "2026SCC235(KLR)":  4,
    "2026SCC241(KLR)":  6,
    "2001KECA154(KLR)": 7,
    "2014KEHC6468(KLR)":3,
    "2024KEHC17038(KLR)":8,
    "2024KEHC15763(KLR)":10,
    "2020KEHC8905(KLR)": 5,
    "2010KECA346(KLR)": 31
  }
  ```

- **Stored metadata keys & sample values (all 4 chunks share the same schema):**
  ```json
  {
    "case_identifier":  "2026SCC235(KLR)",
    "firac_component":  "",
    "file_name":        "New Titanic Jacaranda General Spares v Omil  another (Commercial Case E095of2026) 2026SCC235(KLR) (31August2026) (Ruling).pdf",
    "parties":          "New Titanic Jacaranda General Spares v Omil  another",
    "court_level":      "Small Claims Court",
    "judge":            "A.Z. Ogange",
    "year":             "2026",
    "legal_domain":     "Employment Law",
    "winning_party":    "Respondents",
    "document_hash":    "6737faf2be45672d",
    "chunk_index":      0,
    "chunk_total":      4,
    "content_length":   2688,
    "page_start":       0,
    "page_end":         1,
    "ocr":              false,
    "prev_chunk_id":    "",
    "next_chunk_id":    "2026SCC235KLR_1"
  }
  ```
  > **Note:** `firac_component` is `""` for every chunk — the ingestion pipeline stores it but the classifier that was intended to populate it was never run. This means the retriever cannot distinguish submission paragraphs from holding paragraphs.

---

### Chunk-by-Chunk Breakdown

**Chunk 1 (ID: `2026SCC235KLR_0` — pages 0–1) — Respondents' PO + Claimant's submission**

> Content: Case header (SCCOMM E095/2026). Respondents' Preliminary Objection: s.13(5) SCC Act excludes employment-related claims; reliance on *Mukisa Biscuit* and *Lillian S* (jurisdiction must be present before the court can act). **Claimant's counter-submission:** although the relationship was employer–employee, the claim is for civil recovery of concealed cash proceeds; *Maisha Mapya Kenya Foundation Ltd v Shikuku & Another [2025] KEELRC 761 (KLR)* cited in support; objection should be dismissed.
>
> **Speaker attribution:** Respondents' submissions (paragraphs 3) + Claimant's submissions (paragraph 4). **Not the Court's holding.**

---

**Chunk 2 (ID: `2026SCC235KLR_1` — page 1) — Claimant's submission tail + Court begins analysis**

> Content: Overlap paragraph repeating the Claimant's *Maisha Mapya* argument. Court's opening analysis: cites *Flodida Construction Machinery Limited v Peter Wepukhulu Wanyonyi [2026] KEELRC 992* — court acknowledged divergent decisions but held that cash-diversion claim arises from employment relationship → ELRC jurisdiction. Court notes it had previously relied on *Maisha Mapya* itself, but signals reconsideration given "subsequent and binding jurisprudence of the Court of Appeal."
>
> **Speaker attribution:** Paragraph 5 is the Court citing an ELRC authority; paragraph 6 is the Court hedging away from its own earlier *Maisha Mapya* reliance. The language is transitional — the Court has not yet issued its final holding. **Easily misread as indeterminate on the jurisdiction question.**

---

**Chunk 3 (ID: `2026SCC235KLR_2` — pages 1–2) — Court's binding authority analysis**

> Content: Court identifies binding Court of Appeal authorities: *KMRI v Davy Kiprotich Koech [2018] KECA 128* (employment disputes → ELRC); *Hantex Garments EPZ v Kazungu [2023] KECA 1287* (constitutional demarcation, Article 162(2)). Court holds: "jurisdiction of the specialised court flows from the Constitution and legislation and cannot be assumed or enlarged by the nature of the relief sought." Para 9: applies to present facts — 1st Respondent was an employee when he concealed KSh 232,295.45; cause of action is founded on the employment relationship; subsequent termination does not alter character of the claim.
>
> **Speaker attribution:** Court's reasoning — binding authority analysis and factual application. Contains the logical chain leading to the final order, but the dispositive words have not yet appeared.

---

**Chunk 4 (ID: `2026SCC235KLR_3` — pages 2–3) — FINAL DISPOSITION (verbatim)**

> This is the unretrieved chunk containing the operative ruling. Full verbatim text:
>
> ```text
> 9. In the present matter, the Claimant's own pleadings disclose that the
> 1st Respondent was working for the Claimant when, on 21st March 2026, he
> allegedly received and concealed cash proceeds amounting to Kshs.232,295.45.
> The claim is therefore not founded upon an independent transaction between
> strangers but upon acts allegedly committed by the 1st Respondent in the
> course of his employment. The fact that the employment relationship
> subsequently ceased does not, in my view, alter the character of the cause
> of action.
>
> 10. More importantly, section 13(5) of the Small Claims Court Act expressly
> provides that a claim shall not be brought before the Small Claims Court
> where the cause of action is founded upon employment and labour relations.
> The statutory exclusion is categorical. It is therefore immaterial that the
> relief sought is framed as recovery of money or special damages if, upon
> examination of the pleaded cause of action, the claim is founded upon the
> employment relationship.
>
> 11. I therefore find that, although this Court has previously adopted the
> reasoning in Maisha Mapya, the subsequent consideration of the Court of
> Appeal authorities persuades me that the earlier approach ought not to be
> followed. This is not a departure undertaken lightly; it is necessitated by
> the hierarchical doctrine of precedent.
>
> 12. Accordingly, I find that the present claim is founded upon an
> employer–employee relationship and falls within the exclusion contemplated
> under section 13(5) of the Small Claims Court Act. This Court consequently
> lacks jurisdiction to entertain the claim.
>
> 13. Counsel for the Respondents have asked the court to dismiss the suit
> with costs. This court is inclined to strike out the suit rather than
> dismiss it as prayed. A court that lacks jurisdiction cannot determine the
> merits of the claim or grant substantive relief thereon. This also leaves
> the Claimant at liberty, subject to the applicable law and limitation
> periods to pursue the claim before the court vested with the requisite
> jurisdiction.
>
> 14. Accordingly, the suit is struck out with costs for want of jurisdiction.
> Costs of the Preliminary Objection and the Suit are assessed at Kshs. 5,000/=,
> payable by the Claimant to the Respondents.
>
> RULING DATED, SIGNED AND DELIVERED VIRTUALLY AT MALABA SMALL CLAIMS COURT
> THIS 31ST DAY OF AUGUST 2026.
>
> A.Z. OGANGE
> RESIDENT MAGISTRATE/ADJUDICATOR
> 31.08.2026
> ```
>
> **This chunk was never retrieved** because (a) its embedding is dominated by abstract procedural language (`"statutory exclusion is categorical"`, `"doctrine of precedent"`) rather than the concrete factual keywords in the user query, and (b) `MAX_CHUNKS_PER_COMPONENT = 1` with `firac_component = ""` for all chunks means only the single highest-scoring chunk from this case is passed to the LLM — which is Chunk 1 or 2, not this one.

---

## 3. Root-Cause Diagnosis

### A. Why the Jurisdictional Holding Was Inverted

**Retrieval / Chunking Factor:**

The court's operative holding is entirely contained in Chunk 4 (`2026SCC235KLR_3`, pages 2–3). The cross-encoder reranker scores chunks against the user's raw query string — something like *"employer suing storekeeper SCC recover KSh 185,000 cash concealed stock sheets M-Pesa"*. Chunks 1–2 are semantically closest to that query because they contain the exact claimant-side keywords:

- `"claim concerns monies allegedly concealed or stolen during his employment"` (Chunk 1)
- `"Claimant's documents permit recovery of stolen cash, goods and costs through civil proceedings"` (Chunk 1)
- `"Maisha Mapya Kenya Foundation"` — the pro-jurisdiction authority (Chunks 1 & 2)

Chunk 4, by contrast, uses abstract legal language (`"statutory exclusion is categorical"`, `"hierarchical doctrine of precedent"`, `"struck out with costs for want of jurisdiction"`) that scores lower against a factual query about recovery of cash.

Because `firac_component = ""` for all chunks, the retriever's `MAX_CHUNKS_PER_COMPONENT = 1` guard collapses all 4 chunks into a single group and picks the **one with the highest cross-encoder score** — which is Chunk 1 or 2 (the claimant's submission), not Chunk 4 (the holding). The final disposition is therefore never passed to the LLM.

**Prompt / Reasoning Factor:**

The LLM synthesis prompt ([`Retriever.py:123–158`](backend/Retriever.py#L123)) injects `winning_party` as a metadata label (`"Winning Party: Respondents"`) but then provides this instruction:

```text
TASK:
Provide a comprehensive analysis in exactly this format:

**KEY LEGAL PRINCIPLES:**
[2-3 sentences explaining the main legal rules or doctrines established or applied in this case]

**RELEVANCE TO YOUR QUERY:**
[3-4 sentences explaining specifically how this case relates to the user's query]
```

The prompt does **not** instruct the model to treat `winning_party` as the authoritative signal for who prevailed on the jurisdiction question. The `CASE FIRAC COMPONENTS` section simply pastes the retrieved chunk text — which contains the Claimant's *Maisha Mapya* argument and the Court's transitional hedging language, but not the dispositive paragraphs 10–14. The model reads the retrieved text, identifies the pro-jurisdiction argument as the main reasoning chain (because it occupies most of the visible text), and concludes that the SCC has jurisdiction — the exact inversion of the actual ruling.

The `winning_party = "Respondents"` field is present in the metadata table shown to the LLM but the prompt gives no instruction that "Respondents winning = SCC struck out the claim". Without that link, the model treats it as an incidental label rather than the outcome signal.

---

### B. Why the Electronic Evidence Intent Was Dropped

**Intent / Query Routing Factor:**

The Qclassifier ([`Qclassifier.py:423–637`](backend/Qclassifier.py#L423)) compresses any multi-intent query into a single `vector_query` string (line ~372 schema: `"vector_query": "string (comma-separated keywords)"`). There is no multi-query fan-out — the system does not decompose *"Can I sue in the SCC AND are my M-Pesa logs admissible?"* into two independent sub-queries and retrieve separately for each.

When the two-part prompt is embedded as one vector, the jurisdiction half dominates because it contains the most specific legal noun phrases (`"Small Claims Court"`, `"section 13(5)"`, `"employer-employee"`) which pull the centroid of the embedding toward the jurisdiction cluster. The evidential half (`"M-Pesa till logs"`, `"stock sheets"`, `"admissibility"`, `"Section 106B"`) contributes less mass and its signal is diluted.

**Retrieval Diversity Factor:**

With `TOP_N_PARENTS = 3` and no MMR or source-diversity constraint, the entire retrieval budget is consumed by whichever 3 parent documents the cross-encoder ranks highest. Because the jurisdiction query keywords match multiple chunks in `2026SCC235` (employment, cash concealment, SCC, employer) **and** the DB only has 78 vectors across 9 cases, `2026SCC235` likely occupies the top-1 slot with a large margin.

The electronic evidence precedents — *Ogembo v Yongo [2024] KEHC 15763* and *Republic v Barisa Wayu Mataguda [2011] KEHC 1481* — contain terms like `"Section 106B"`, `"electronic records"`, `"certificate"`, `"admissibility"`. These do not appear in the user's query at all (the user described their evidence as "M-Pesa till logs" and "signed daily stock sheets", not using the statutory certification vocabulary). The cross-encoder therefore scores them low relative to `2026SCC235` for this query. The evidentiary cases are silently dropped from the `top_3`.

---

## 4. Current System Prompt & Retrieval Code Snippets

### Current LLM Synthesis System Prompt (Verbatim — `Retriever.py:123–158`)

```text
You are a legal research assistant analyzing a court case for relevance to a user's query.

USER'S QUERY:
{query_info['vector_query']}

TARGET LEGAL AREAS:
{', '.join([f"{d.get('domain', 'Unknown')} ({d.get('confidence_score', 0):.1f})" for d in query_info['legal_domains'][:3]])}

CASE TO ANALYZE:
Case Name: {case_data['parent_document']}
Court: {case_data['metadata']['court_level']}
Judge: {case_data['metadata']['judge']}
Year: {case_data['metadata']['year']}
Legal Domain: {case_data['metadata']['legal_domain']}
Winning Party: {case_data['metadata']['winning_party']}

CASE FIRAC COMPONENTS:
{components_text}

TASK:
Provide a comprehensive analysis in exactly this format:

**CASE OVERVIEW:**
[2-3 sentences summarizing what this case is about - the parties, the dispute, and the outcome]

**KEY LEGAL PRINCIPLES:**
[2-3 sentences explaining the main legal rules or doctrines established or applied in this case]

**RELEVANCE TO YOUR QUERY:**
[3-4 sentences explaining specifically how this case relates to the user's query. Be concrete about which aspects are most relevant and why this case matters for their situation]

**CRITICAL FACTS:**
[2-3 sentences highlighting the most important factual elements that influenced the court's decision]

Keep your response clear, professional, and focused on practical legal insights.
Use plain language while maintaining legal accuracy.
```

### Retrieval Configuration (Verbatim — `Retriever.py:46–53, 681–694`)

```python
TOP_N_PARENTS          = 3     # line 47 — max distinct cases returned
MAX_CHUNKS_PER_COMPONENT = 1   # line 50 — max 1 chunk per FIRAC component per case
LOW_CONFIDENCE_THRESHOLD = 0.4 # line 53

# Vector query (line 681-688)
n_vec = min(50, collection.count())          # = 50 for 78-vector DB
vec_raw = collection.query(
    query_embeddings=[query_embedding],
    n_results=n_vec,
    include=['documents', 'metadatas', 'distances'],
)

# RRF fusion (line 694)
fused_ids = _rrf_fuse_c(vec_ranked_ids, bm25_ranked_ids, k=60, n=30)
# → 30 candidates, cross-encoder reranked, then grouped by parent, top 3 returned
```

### Classifier Query Compression (Verbatim — `Qclassifier.py:372`)

```json
{
  "retrieval_mode": "STRATEGIC_LEGAL_ANALYSIS",
  "intents": ["array — but never fan-out into separate sub-queries"],
  "vector_query": "single comma-separated keyword string for BOTH intents"
}
```

---

## 5. Prioritised Fix Recommendations

| Priority | Fix | File(s) | Why |
|---|---|---|---|
| **P0** | **Annotate `firac_component` during ingestion** using an LLM-assisted FIRAC tagger so the retriever can distinguish submission paragraphs from holding paragraphs. | `build_case_chunks.py` | Without FIRAC labels, `MAX_CHUNKS_PER_COMPONENT = 1` picks the highest-cosine chunk (usually the claimant's submission) instead of the Court's reasoning/conclusion. |
| **P0** | **Multi-query fan-out in Qclassifier:** decompose multi-intent prompts into N independent sub-queries, retrieve separately, merge results de-duplicating by `case_identifier`. | `Qclassifier.py`, `Retriever.py` | A single embedding centroid cannot serve two semantically distinct intents (jurisdiction vs. evidentiary admissibility) with a 3-parent budget. |
| **P1** | **Add a holding-extraction step:** after retrieving the best chunk from a case, also fetch its `next_chunk_id` sibling(s) so the final 1–2 paragraphs of a ruling are always included. | `Retriever.py` | The operative order is almost always in the last chunk. The 1-paragraph overlap is insufficient to surface it when the high-scoring chunk is in the middle of the document. |
| **P1** | **Explicit holding-inversion guard in the synthesis prompt:** add an instruction like *"If `winning_party` is Respondents, the Court ruled against the Claimant. Begin CASE OVERVIEW by stating the outcome before describing the arguments."* | `Retriever.py:123` | The LLM currently has no anchor instructing it to treat `winning_party` as the outcome, so it derives the holding from whatever chunk text dominates the context window. |
| **P2** | **Source-diversity (MMR) in RRF results:** after cross-encoder reranking, enforce that at most 1 slot per parent document is used in the top-3 until all parents are represented at least once. Already implied by `TOP_N_PARENTS` logic but needs to be enforced earlier in the ranking pipeline. | `Retriever.py:694–757` | 78 vectors / 9 cases means a single highly-matching case can dominate all 3 slots before less-obvious but relevant cases (e.g., *Ogembo v Yongo*) are considered. |
| **P2** | **Expand `SCCCases/` with more electronic-evidence rulings** (additional s.106B decisions) so the evidence-admissibility intent has candidate documents to retrieve. | `backend/SCCCases/` | *Ogembo v Yongo* and *Barisa Wayu Mataguda* are present but their s.106B content is thin relative to the 31-chunk *Safaricom v Ocean View* case which dominates the vector space. |

---

## 6. Evidence of All Fixes Required (Summary Table)

| Failure | Root Cause | Evidence |
|---|---|---|
| Holding inversion on `2026SCC235` | Chunk 4 (paragraphs 10–14, the operative striking-out order) was not retrieved; Chunks 1–2 (claimant's pro-jurisdiction submission) scored higher against the factual query | `firac_component = ""` for all 4 chunks; `MAX_CHUNKS_PER_COMPONENT = 1` forces a single chunk pick with no FIRAC distinction; Chunk 4 verbatim text confirmed above |
| LLM conflated *Maisha Mapya* argument with Court's holding | Synthesis prompt injects `winning_party` without instructing the model to treat it as the dispositive outcome signal | `Retriever.py:138` — `Winning Party: {winning_party}` appears in context but prompt task (`TASK:` section) makes no reference to it |
| Electronic evidence intent silently dropped | Two-intent query collapsed to one vector; `TOP_N_PARENTS = 3` consumed by jurisdiction case; no multi-query fan-out | `Qclassifier.py` output schema: single `vector_query` string; no sub-query decomposition in codebase |
| *Ogembo v Yongo* / *Barisa Wayu Mataguda* not surfaced | s.106B vocabulary (`"Section 106B"`, `"certificate"`, `"electronic record"`) absent from user query; cross-encoder scores them low | User described evidence as "M-Pesa till logs" and "stock sheets" — no statutory vocabulary to bridge to those cases without query expansion |
