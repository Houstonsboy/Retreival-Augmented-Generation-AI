# Kenya GraphRAG Legal Advisory System — Project Report

> **Scope:** Advocate-facing legal research tool for Kenyan contract law, grounded in the Kenya Law Reports (eKLR) corpus and the Kenya Evidence Act (Cap 80) / Law of Contract Act (Cap 23).

---

## 1. Project Overview

This system is the **advocate-facing tier** of a two-tiered GraphRAG legal advisory platform targeting Kenya's Small Claims Court ecosystem. It enables legal practitioners to submit natural language queries and receive **FIRAC-structured** (Facts, Issue, Rule, Application, Conclusion) legal analysis grounded in verifiable Kenyan case law — with zero hallucination risk from unsupported citations.

### The problem it solves

Kenya's public eKLR portal uses primitive keyword matching. It cannot:
- Filter cases by disputed legal issue or factual pattern
- Retrieve precedents by FIRAC structure
- Estimate quantum (damages) trends
- Surface counter-arguments or risk signals

Standard LLMs and Vector-RAG pipelines hallucinate 58–88% of the time on low-profile East African legal corpora (Dahl et al., 2024). A Kenyan contract dispute requires citing specific Acts, specific sections, and specific eKLR case numbers — a fabricated citation has direct legal consequences.

### The solution

A **hybrid retrieval architecture** combining:
1. **ChromaDB** — vector similarity search over FIRAC-chunked case law and statutory text
2. **Neo4j** — structured knowledge graph storing citation relationships, statute interpretations, and multi-hop legal reasoning paths

Both layers are constrained to **Kenyan contract law** (Law of Contract Act Cap 23, Evidence Act Cap 80, SCC Act 2016) with High Court and appellate decisions from 2021–2026, focused on Nairobi-region contract breaches, Chama transactions, and unpaid SME invoices.

---

## 2. Current System Architecture

### 2.1 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Embedding model | `intfloat/e5-base-v2` (768-dim) | Semantic embeddings for all text |
| Vector DB | ChromaDB (persistent) | FIRAC chunk storage + cosine similarity search |
| LLM backbone | `llama-3.3-70b-versatile` via Groq API | FIRAC extraction, query classification, response generation |
| Query classifier | Custom prompt (Qclassifier.py) | Routes queries to DIRECT_STATUTORY_LOOKUP / STRATEGIC_LEGAL_ANALYSIS / HYBRID |
| Graph DB | Neo4j Community 5.18.0 | Citation graph, statute relationships (in progress) |
| Backend | Flask (Python) | REST API layer |
| Frontend | React | Advocate-facing UI |

### 2.2 Data Pipeline (Current)

```
PDF (eKLR judgment)
        │
        ▼
  firac.py (Groq LLM)
  ┌────────────────────────────────────────┐
  │  Extracts: metadata + 5 FIRAC chunks   │
  └────────────────────────────────────────┘
        │
        ▼
  ingester.py
  ┌─────────────────────────────────────────────────────┐
  │  Embeds each chunk → stores in ChromaDB             │
  │  Collection: legal_summaries (LegalSummariesDB)     │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  Query (via frontend)
        │
        ▼
  Qclassifier.py → routing JSON
        │
        ├──► Retriever.py       → hits legal_summaries   → LLM summary
        └──► RulesRetriever.py  → hits master_rules      → statute chunks
```

### 2.3 Current Collection Stats

| Collection | Path | Count | Purpose |
|---|---|---|---|
| `legal_summaries` | `backend/LegalSummariesDB` | 40 chunks (~8 cases) | Case law FIRAC chunks |
| `master_rules` | `backend/constitutionlogic/MasterRulesDB` | 619 chunks | Constitution + Evidence Act |

---

## 3. Database Architecture (Current + Proposed)

The advocate-facing side uses **three data stores** working together:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ADVOCATE QUERY                                │
│                  "Can I enforce a WhatsApp contract?"               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    Qclassifier.py
                    (routing + entity extraction)
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                  ▼
    ChromaDB             ChromaDB            Neo4j
  legal_summaries       master_rules      Knowledge Graph
  (FIRAC case law)    (statutes/acts)    (citation network)
           │                 │                  │
           └─────────────────┴──────────────────┘
                             │
                    Retriever.py + RulesRetriever.py
                             │
                    LLM → FIRAC response
```

---

## 4. ChromaDB Schema

### Decision: One ChromaDB Instance, Two Collections

Both collections will live in a **single ChromaDB persistent store** at `backend/LegalDB/`. The current split between `LegalSummariesDB` and `constitutionlogic/MasterRulesDB` will be consolidated. This simplifies connection management and allows cross-collection metadata queries.

```
backend/LegalDB/
├── chroma.sqlite3
└── [uuid]/            ← managed by ChromaDB internals
```

---

### 4.1 Collection: `contract_law_cases`

**Purpose:** Stores FIRAC-chunked eKLR case law, scoped strictly to Kenyan contract law. Each case produces exactly 5 chunks (one per FIRAC component). This collection powers case precedent retrieval and win-probability analysis.

**Embedding:** `intfloat/e5-base-v2` (768-dim), cosine similarity, prefix `"passage: "` on ingest.

#### Full Field Specification

| Field | Type | Required | Example | Notes |
|---|---|---|---|---|
| **IDENTITY** | | | | |
| `case_identifier` | `str` | ✅ | `"Ogembo v Yongo [2024] KEHC 15763"` | Primary key — unique per case |
| `case_number` | `str` | ✅ | `"Civil Appeal E200 of 2023"` | Court registry number |
| `citation` | `str` | ✅ | `"[2024] KEHC 15763"` | Formal eKLR citation |
| `eKLR_url` | `str` | ✅ | `"http://kenyalaw.org/caselaw/cases/view/..."` | Direct eKLR link |
| **PARTIES** | | | | |
| `parties` | `str` | ✅ | `"John Ogembo v Mary Yongo"` | Full parties string |
| `plaintiff` | `str` | ✅ | `"John Ogembo"` | Parsed plaintiff name |
| `defendant` | `str` | ✅ | `"Mary Yongo"` | Parsed defendant name |
| **COURT** | | | | |
| `court_level` | `str` | ✅ | `"High Court of Kenya"` | Court that issued judgment |
| `court_station` | `str` | ✅ | `"Nairobi"` | Geographic court station |
| `judge` | `str` | ✅ | `"HM Nyaga J"` | Presiding judge(s) |
| **TEMPORAL** | | | | |
| `year` | `str` | ✅ | `"2024"` | Year of judgment (for ChromaDB filter) |
| `date_decided` | `str` | ✅ | `"2024-11-14"` | Full ISO date |
| **LEGAL CLASSIFICATION** | | | | |
| `legal_domain` | `str` | ✅ | `"Contract Law"` | Always "Contract Law" in this collection |
| `case_type` | `str` | ✅ | `"Civil Appeal"` | Appeal / Suit / Petition |
| `dispute_type` | `str` | ✅ | `"Breach of Contract"` | Breach / Misrepresentation / Unjust Enrichment / Quantum Meruit / Non-payment |
| `transaction_type` | `str` | ✅ | `"SME Invoice"` | WhatsApp contract / Chama loan / SME invoice / Formal written contract / Mobile money |
| **CONTRACT ELEMENTS** | | | | |
| `offer_present` | `str` | ✅ | `"true"` | Was offer identified in judgment |
| `acceptance_present` | `str` | ✅ | `"true"` | Was acceptance identified |
| `consideration_present` | `str` | ✅ | `"true"` | Was consideration identified |
| `intention_present` | `str` | ✅ | `"true"` | Intention to create legal relations |
| `contract_element_gap` | `str` | ⬜ | `"consideration"` | Which element was missing/disputed |
| **OUTCOME** | | | | |
| `winning_party` | `str` | ✅ | `"Plaintiff"` | `"Plaintiff"` / `"Defendant"` / `"Partial"` |
| `winning_party_name` | `str` | ✅ | `"John Ogembo"` | Name of winning party |
| `appeal_outcome` | `str` | ⬜ | `"Appeal Allowed"` | Appeal Allowed / Dismissed / Partially Allowed |
| **QUANTUM (STRUCTURED)** | | | | |
| `quantum_awarded` | `str` | ⬜ | `"500000"` | Amount in KES, stored as string for ChromaDB compat |
| `quantum_currency` | `str` | ⬜ | `"KES"` | Always KES for this corpus |
| `interest_rate` | `str` | ⬜ | `"14"` | Interest rate as percentage number string |
| `interest_from_date` | `str` | ⬜ | `"2024-01-01"` | Date interest runs from |
| `costs_awarded` | `str` | ⬜ | `"true"` | Whether costs were awarded |
| **STATUTES & CITATIONS** | | | | |
| `statutes_cited` | `str` | ⬜ | `"Law of Contract Act Cap 23; Evidence Act Cap 80"` | Semicolon-delimited statute list |
| `sections_cited` | `str` | ⬜ | `"Section 3; Section 106B"` | Specific sections cited |
| `cases_cited` | `str` | ⬜ | `"Muruatetu v Republic; Obado v Anayo"` | Semicolon-delimited case citations |
| **FIRAC** | | | | |
| `firac_component` | `str` | ✅ | `"facts"` | `facts` / `issues` / `rules` / `application` / `conclusion` |
| **EVIDENCE** | | | | |
| `digital_evidence_type` | `str` | ⬜ | `"WhatsApp"` | WhatsApp / Email / M-Pesa / Screenshot / None |
| `evidence_admitted` | `str` | ⬜ | `"true"` | Whether digital evidence was admitted |
| `section_106b_compliant` | `str` | ⬜ | `"false"` | Whether Section 106B certificate was present |
| **TRACEABILITY** | | | | |
| `document_hash` | `str` | ✅ | `"20a5c00bd0be..."` | SHA256 of source PDF text |
| `source_file_path` | `str` | ✅ | `"/backend/Repo/..."` | Absolute path to source PDF |
| `content_length` | `str` | ✅ | `"1132"` | Character count of this chunk |
| `ingested_at` | `str` | ✅ | `"2026-09-13 14:30:00"` | Ingestion timestamp |
| `embedding_model` | `str` | ✅ | `"intfloat/e5-base-v2"` | Model used to embed this chunk |

> **Note on types:** ChromaDB metadata values must be `str`, `int`, or `float`. All boolean-like fields (`offer_present`, `evidence_admitted`, etc.) are stored as the string `"true"` / `"false"` and parsed at retrieval time. Monetary amounts (`quantum_awarded`, `interest_rate`) are stored as numeric strings so they can be filtered with ChromaDB's `$gte` / `$lte` operators after casting.

#### Quantum as a Structured Field

`quantum_awarded` is stored as a numeric string (e.g., `"500000"`) rather than a natural language phrase (e.g., `"KES 500,000 with interest"`). This enables:

```python
# Filter cases with quantum > KES 100,000
collection.query(
    query_embeddings=[...],
    where={"quantum_awarded": {"$gte": "100000"}},
    n_results=10
)
```

The LLM extracts quantum during FIRAC processing. If the judgment mentions "Kshs 500,000", the ingester normalises it to `"500000"`. If quantum was not awarded or not determinable, the field is set to `"0"`. This makes it possible for the retriever to:
- Surface average quantum across similar dispute types
- Flag cases where quantum was reduced on appeal
- Support the paper's quantum estimation feature for advocates

---

### 4.2 Collection: `master_rules`

**Purpose:** Stores chunked constitutional provisions, the Kenya Evidence Act (Cap 80), and the Law of Contract Act (Cap 23). Powers the `DIRECT_STATUTORY_LOOKUP` retrieval mode.

**Embedding:** Same model. Shared ChromaDB instance.

#### Current Fields (Stable)

| Field | Type | Example | Notes |
|---|---|---|---|
| `article_number` | `str` | `"106B"` | Article or section number |
| `article_header` | `str` | `"Admissibility of electronic records"` | Section heading |
| `chapter` | `str` | `"SEVEN"` | Chapter identifier |
| `chapter_title` | `str` | `"ADMISSIBILITY OF EVIDENCE"` | Chapter heading |
| `part` | `str` | `"III"` | Part within chapter |
| `part_title` | `str` | `""` | Part heading |
| `source_document` | `str` | `"Evidence Act.pdf"` | Source file |
| `document_type` | `str` | `"evidence_act"` | `constitution` / `evidence_act` / `contract_act` |
| `chunk_index` | `str` | `"1"` | Chunk position within article |
| `total_chunks` | `str` | `"3"` | Total chunks for this article |
| `is_complete_article` | `str` | `"True"` | Whether article fits in one chunk |
| `article_length` | `str` | `"1450"` | Full article character count |
| `chunk_length` | `str` | `"730"` | This chunk's character count |
| `ingestion_timestamp` | `str` | `"1770036253"` | Unix timestamp |

#### Fields to Add

| Field | Type | Example | Notes |
|---|---|---|---|
| `cap` | `str` | `"Cap 80"` | Laws of Kenya chapter reference |
| `act_title` | `str` | `"Evidence Act"` | Short human-readable act name |
| `last_amended` | `str` | `"2019"` | Year of last amendment (if known) |
| `is_active` | `str` | `"true"` | Whether provision is currently in force |
| `related_articles` | `str` | `"78A; 106C"` | Semicolon-delimited related sections within same act |

---

## 5. Neo4j Knowledge Graph Schema

Neo4j stores the **relationship layer** that ChromaDB cannot express: which cases cite which other cases, which statutes govern which legal concepts, and which judicial holdings interpret which statutory provisions. This enables multi-hop traversal queries like:

> *"Find all High Court cases that cite Muruatetu AND interpret Section 106B AND awarded quantum above KES 200,000."*

---

### 5.1 Node Types

#### `(:Case)` Node

Mirrors the case-level metadata from ChromaDB but adds graph-specific properties.

| Property | Type | Example |
|---|---|---|
| `case_id` | `string` | `"KEHC-15763-2024"` |
| `case_identifier` | `string` | `"Ogembo v Yongo [2024] KEHC 15763"` |
| `case_number` | `string` | `"Civil Appeal E200 of 2023"` |
| `citation` | `string` | `"[2024] KEHC 15763"` |
| `eKLR_url` | `string` | URL to eKLR |
| `parties` | `string` | Full parties string |
| `plaintiff` | `string` | Plaintiff name |
| `defendant` | `string` | Defendant name |
| `court_level` | `string` | `"High Court of Kenya"` |
| `court_station` | `string` | `"Nairobi"` |
| `year` | `integer` | `2024` |
| `date_decided` | `string` | `"2024-11-14"` |
| `dispute_type` | `string` | `"Breach of Contract"` |
| `transaction_type` | `string` | `"SME Invoice"` |
| `winning_party` | `string` | `"Plaintiff"` |
| `quantum_awarded` | `integer` | `500000` |
| `interest_rate` | `float` | `14.0` |
| `costs_awarded` | `boolean` | `true` |
| `offer_present` | `boolean` | `true` |
| `acceptance_present` | `boolean` | `true` |
| `consideration_present` | `boolean` | `true` |
| `digital_evidence_type` | `string` | `"WhatsApp"` |
| `evidence_admitted` | `boolean` | `false` |
| `section_106b_compliant` | `boolean` | `false` |
| `chroma_case_identifier` | `string` | FK → ChromaDB `case_identifier` |

#### `(:Statute)` Node

| Property | Type | Example |
|---|---|---|
| `statute_id` | `string` | `"EvidenceAct-106B"` |
| `act_title` | `string` | `"Evidence Act"` |
| `cap` | `string` | `"Cap 80"` |
| `section_number` | `string` | `"106B"` |
| `section_header` | `string` | `"Admissibility of electronic records"` |
| `document_type` | `string` | `"evidence_act"` |
| `chroma_chunk_ids` | `string` | Semicolon-delimited ChromaDB IDs for this statute's chunks |

#### `(:LegalConcept)` Node

| Property | Type | Example |
|---|---|---|
| `concept_id` | `string` | `"offer"` |
| `name` | `string` | `"Offer"` |
| `category` | `string` | `"contract_element"` / `"doctrine"` / `"remedy"` / `"evidence_rule"` |
| `definition` | `string` | Short plain-language definition |

**Predefined concepts for contract law:**
- Contract elements: `offer`, `acceptance`, `consideration`, `intention`
- Doctrines: `quantum_meruit`, `unjust_enrichment`, `promissory_estoppel`, `frustration`, `mistake`, `misrepresentation`
- Remedies: `specific_performance`, `damages`, `rescission`, `injunction`
- Evidence rules: `section_106b_certificate`, `hearsay`, `best_evidence`

#### `(:Judge)` Node

| Property | Type | Example |
|---|---|---|
| `judge_id` | `string` | `"HM-Nyaga"` |
| `name` | `string` | `"HM Nyaga J"` |
| `court` | `string` | `"High Court of Kenya"` |
| `station` | `string` | `"Nairobi"` |

---

### 5.2 Relationship Types

| Relationship | From → To | Properties | Example |
|---|---|---|---|
| `CITES` | `(:Case)→(:Case)` | `cite_context: str` | Ogembo cites Muruatetu |
| `INTERPRETS` | `(:Case)→(:Statute)` | `interpretation: str`, `holding: str` | Case interprets Section 106B |
| `APPLIES` | `(:Case)→(:LegalConcept)` | `outcome: str` | Case applies quantum meruit doctrine |
| `OVERRULES` | `(:Case)→(:Case)` | `reason: str` | Appellate case overrules lower court |
| `DISTINGUISHES` | `(:Case)→(:Case)` | `distinction: str` | Case distinguishes on different facts |
| `FOLLOWS` | `(:Case)→(:Case)` | — | Case follows earlier precedent |
| `GOVERNS` | `(:Statute)→(:LegalConcept)` | — | Section 3 Cap 23 governs "offer" |
| `PRESIDED_OVER` | `(:Judge)→(:Case)` | — | Judge presided over case |
| `CROSS_REFERENCES` | `(:Statute)→(:Statute)` | — | Section 78A cross-references 106B |

---

### 5.3 Example Cypher Queries (Advocate Use Cases)

```cypher
-- 1. Find all cases that cite Section 106B and lost on digital evidence
MATCH (c:Case)-[:INTERPRETS]->(s:Statute {section_number: "106B"})
WHERE c.evidence_admitted = false
RETURN c.citation, c.parties, c.date_decided, c.quantum_awarded
ORDER BY c.date_decided DESC

-- 2. Multi-hop: find cases citing Ogembo v Yongo, and what they decided
MATCH (c:Case)-[:CITES]->(cited:Case {case_identifier: "Ogembo v Yongo [2024] KEHC 15763"})
RETURN c.citation, c.winning_party, c.quantum_awarded

-- 3. Average quantum for breach of contract in Nairobi, 2022-2024
MATCH (c:Case)
WHERE c.dispute_type = "Breach of Contract"
  AND c.court_station = "Nairobi"
  AND c.year >= 2022 AND c.year <= 2024
  AND c.quantum_awarded > 0
RETURN avg(c.quantum_awarded) as avg_quantum, count(c) as case_count

-- 4. Find cases applying quantum meruit with quantum above 200k
MATCH (c:Case)-[:APPLIES]->(concept:LegalConcept {name: "Quantum Meruit"})
WHERE c.quantum_awarded > 200000
RETURN c.citation, c.parties, c.quantum_awarded, c.winning_party

-- 5. Judge-specific ruling pattern (win rate for plaintiffs)
MATCH (j:Judge)-[:PRESIDED_OVER]->(c:Case)
WHERE j.name = "HM Nyaga J"
RETURN j.name,
       count(c) as total_cases,
       sum(CASE WHEN c.winning_party = 'Plaintiff' THEN 1 ELSE 0 END) as plaintiff_wins
```

---

### 5.4 ChromaDB ↔ Neo4j Sync

Every `(:Case)` node in Neo4j carries a `chroma_case_identifier` property that maps back to the `case_identifier` field in ChromaDB. Every `(:Statute)` node carries `chroma_chunk_ids`. This two-way pointer allows the retrieval engine to:

1. Use **Neo4j** to traverse the citation graph and identify relevant case IDs
2. Use those IDs to pull the full **FIRAC text chunks** from ChromaDB for LLM summarisation

The hybrid query pattern:

```python
# Step 1: Neo4j graph traversal → get relevant case identifiers
case_ids = neo4j.run("""
    MATCH (c:Case)-[:INTERPRETS]->(s:Statute {section_number: $section})
    RETURN c.chroma_case_identifier as id
""", section="106B")

# Step 2: ChromaDB vector search restricted to those cases
results = collection.query(
    query_embeddings=[query_vector],
    where={"case_identifier": {"$in": [r["id"] for r in case_ids]}},
    n_results=10
)
```

---

## 6. Data Ingestion Pipeline (Target State)

```
eKLR PDF
    │
    ▼
firac.py (Groq LLM)
    │  Extracts:
    │  - 5 FIRAC chunks
    │  - All metadata fields (including structured quantum)
    │  - Cases cited, statutes cited
    │
    ├──► ingester.py ──────────────────────────────────► ChromaDB
    │    - Embeds 5 chunks                               contract_law_cases
    │    - Stores all fields                             collection
    │    - Structured quantum (KES numeric string)
    │    - Contract elements (offer/acceptance/etc.)
    │
    └──► neo4j_ingester.py ────────────────────────────► Neo4j
         - Creates/merges (:Case) node                  Knowledge Graph
         - Creates (:Statute) nodes for cited acts
         - Creates (:LegalConcept) nodes
         - Creates CITES edges to cited cases
         - Creates INTERPRETS edges to statutes
         - Creates APPLIES edges to concepts
         - Links to (:Judge) node
```

---

## 7. Retrieval Flow (Target State)

```
User Query: "Can I recover payment for services rendered under a WhatsApp agreement?"
    │
    ▼
Qclassifier.py
    │  Outputs:
    │  - retrieval_mode: HYBRID_RETRIEVAL
    │  - legal_domains: [Contract Law]
    │  - entities: {statutes: ["Law of Contract Act Cap 23"],
    │               concepts: ["quantum meruit", "offer", "acceptance"]}
    │  - dispute_type: "Non-payment / Quantum Meruit"
    │  - transaction_type: "WhatsApp contract"
    │
    ├──► Neo4j traversal (Retriever.py)
    │    MATCH cases applying quantum meruit doctrine
    │    MATCH cases with WhatsApp evidence
    │    MATCH Section 3 Cap 23 (offer/acceptance)
    │    → Returns: [case_id_1, case_id_2, case_id_3, statute_id_1]
    │
    ├──► ChromaDB vector search (scoped to returned IDs)
    │    → Returns: FIRAC chunks for top 3 cases
    │
    ├──► RulesRetriever.py
    │    → Returns: Section 3 Cap 23 chunks + relevant Evidence Act sections
    │
    ▼
LLM (llama-3.3-70b-versatile via Groq)
    │  Constructs FIRAC response:
    │  - Facts: user's situation
    │  - Issue: enforceability of WhatsApp agreement + quantum recovery
    │  - Rules: Section 3 Cap 23 + cited cases
    │  - Application: how rules apply to user's facts
    │  - Conclusion: outcome prediction + quantum estimate
    │
    ▼
Frontend Response Widget
    - FIRAC structured output
    - Case cards (with eKLR links)
    - Statute references
    - Quantum estimate: "Similar cases awarded KES 120,000–500,000"
    - Win probability signal
```

---

## 8. What Exists vs. What Needs to Be Built

### Built ✅

| Component | Status |
|---|---|
| FIRAC extraction from PDF (firac.py) | Done |
| ChromaDB ingestion with metadata (ingester.py) | Done |
| Vector similarity retrieval (Retriever.py) | Done |
| Statutory lookup retrieval (RulesRetriever.py) | Done |
| Query classifier with routing modes (Qclassifier.py) | Done |
| Flask REST API (app.py) | Done |
| Neo4j installed and running (neo4j-community-5.18.0) | Installed |
| Legal dictionary knowledge graph (legal term nodes) | Done |
| `master_rules` ChromaDB collection (Constitution + Evidence Act) | Done |

### To Build 🔨

| Component | Priority | Notes |
|---|---|---|
| Migrate `LegalSummariesDB` + `MasterRulesDB` to single `LegalDB` path | High | Schema consolidation |
| Add new fields to ingester.py (quantum, contract elements, etc.) | High | Extend FIRAC extraction prompt |
| Neo4j ingester for (:Case) nodes (neo4j_ingester.py) | High | Needs Bolt connector |
| CITES / INTERPRETS / APPLIES edge creation | High | Depends on Neo4j ingester |
| Hybrid retrieval: Neo4j graph traversal → ChromaDB scoped search | High | Core GraphRAG feature |
| Law of Contract Act (Cap 23) into `master_rules` | Medium | Add to MasterRulesDB pipeline |
| Quantum estimation in LLM response | Medium | Aggregate from retrieved cases |
| Win-probability scoring | Medium | Based on dispute_type + judge + quantum |
| Counter-argument analysis | Low | Second LLM pass on retrieved cases |
| eKLR corpus expansion (target: 200+ contract law cases) | Ongoing | Currently ~8 cases |

---

## 9. Immediate Next Steps

1. **Extend the FIRAC extraction prompt** in `firac.py` to extract the new structured fields: `case_number`, `citation`, `plaintiff`, `defendant`, `court_station`, `date_decided`, `dispute_type`, `transaction_type`, `offer_present`, `acceptance_present`, `consideration_present`, `quantum_awarded`, `interest_rate`, `statutes_cited`, `cases_cited`, `digital_evidence_type`, `evidence_admitted`, `section_106b_compliant`.

2. **Update ingester.py** to write all new fields into ChromaDB metadata.

3. **Rename collection** from `legal_summaries` → `contract_law_cases` and consolidate both ChromaDB paths into `backend/LegalDB/`.

4. **Write `neo4j_ingester.py`** that reads from the enriched ChromaDB metadata and creates the corresponding Neo4j graph nodes and relationships.

5. **Update `Retriever.py`** to support Neo4j graph traversal as a pre-filter step before ChromaDB vector search.

6. **Ingest Law of Contract Act (Cap 23)** into the `master_rules` collection using the existing `EvidenceDigester` pattern.

---

## 10. References

- Dahl et al. (2024). *Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models.* Journal of Legal Analysis, 16(1).
- Kang et al. (2025). *Automating IRAC Analysis in Malaysian Contract Law using a Semi-Structured Knowledge Base.* AI and Law.
- Lewis et al. (2021). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401.
- Kenya Evidence Act (Cap 80), Section 106B.
- Law of Contract Act (Cap 23).
- Small Claims Court Act (2016).
