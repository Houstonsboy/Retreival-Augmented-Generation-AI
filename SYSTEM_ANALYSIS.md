  # Legal RAG System — Full Architecture Analysis

  ## Overview

  This is a **Kenyan Legal RAG (Retrieval-Augmented Generation)** system built with Flask + ChromaDB + Groq. It has two retrieval pipelines: a general RAG pipeline and a specialized legal query classification + retrieval pipeline that serves case law, constitutional provisions, and statutory text.

  ---

  ## Files & Their Roles

  | File | Role |
  |------|------|
  | `app.py` | Flask server, all API endpoints |
  | `Qclassifier.py` | Query intent classification via LLM, orchestrates case law retrieval |
  | `Retriever.py` | Case law semantic search against `LegalSummariesDB` |
  | `RulesRetriever.py` | Statutory provisions search against `MasterRulesDB` |
  | `pixe.py` | General RAG pipeline (hybrid search + reranking) against `ChunkDB` |
  | `CaseLaw/firac.py` | PDF ingestion — extracts FIRAC structure via LLM |
  | `CaseLaw/ingester.py` | Embeds FIRAC data into `LegalSummariesDB` |
  | `constitutionlogic/constitution.py` | Constitution PDF parser — LLM structures text into JSON |
  | `constitutionlogic/constitutionbreaker.py` | Article extraction + chunking |
  | `constitutionlogic/constitution_digester.py` | Embeds constitution chunks into `MasterRulesDB` |

  ---

  ## Models Used (Universal)

  | Model | Purpose | Temperature |
  |-------|---------|-------------|
  | `qwen/qwen3.8-27b` via Groq API | All LLM tasks (classification, extraction, summarization, answer generation) | 0.1 – 0.3 |
  | `intfloat/e5-base-v2` (HuggingFace) | All embeddings — 768-dimensional, used in all three ChromaDB collections | N/A |
  | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking step in the `pixe.py` general RAG pipeline only | N/A |

  > **API Key rotation**: `Qclassifier.py` loads up to 3 Groq keys (`GROQ_API_KEY`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3`) and rotates on 429 rate-limit errors.

  ---

  ## Vector Databases (ChromaDB)

  | Database Path | Collection Name | Contents |
  |--------------|----------------|---------|
  | `backend/ChunkDB/` | `ChunkDB` | Raw document chunks (used by `/api/chat`) |
  | `backend/LegalSummariesDB/` | `legal_summaries` | FIRAC-structured case law chunks |
  | `backend/constitutionlogic/MasterRulesDB/` | `master_rules` | Constitution of Kenya + Evidence Act article chunks |

  ---

  ## Pipeline 1: Legal Query Classification + Retrieval (`/api/qretrieve`)

  This is the primary legal intelligence pipeline. The request flow:

  ```
  POST /api/qretrieve
    ↓
  Qclassifier(user_query)
    ├── classify_legal_query()  →  Groq LLM (SYSTEM_PROMPT_CLASSIFIER)
    ├── normalize_for_retriever()
    └── retrieve_relevant_cases()  →  LegalSummariesDB
          └── generate_case_summary()  →  Groq LLM (per case)
    
    ↓ (if retrieval_mode is DIRECT_STATUTORY_LOOKUP or HYBRID_RETRIEVAL)

  retrieve_relevant_rules(classification)  →  MasterRulesDB

    ↓

  Return: { classification, retrieval: { results (cases), rules (statutes) } }
  ```

  Then separately:

  ```
  POST /api/generate-legal-answer
    ↓
  Build context from retrieval results
    ↓
  Groq LLM (mode-specific system prompt)
    ↓
  Return: { answer }
  ```

  ---

  ### Step 1 — Query Classification (`Qclassifier.py`)

  **Model**: `qwen/qwen3.8-27b`  
  **Temperature**: `0.1` (low for deterministic JSON output)  
  **Max tokens**: `1500`

  The classifier's single massive system prompt (`SYSTEM_PROMPT`) is ~420 lines and instructs the model to act as a **Senior Kenyan Legal Research Specialist** with a routing protocol. It is never answering the user's question — only classifying it.

  #### Retrieval Mode Detection (STEP 0)

  The LLM must classify every query into one of three modes:

  | Mode | Trigger |
  |------|---------|
  | `DIRECT_STATUTORY_LOOKUP` | User asks what an article/section says, explicit citation queries, factual verification |
  | `STRATEGIC_LEGAL_ANALYSIS` | Litigation strategy, doctrine application, case law requests, multi-party FIRAC analysis |
  | `HYBRID_RETRIEVAL` | Both — direct text lookup + how courts have interpreted/applied it |

  #### Output Schema

  ```json
  {
    "retrieval_mode": "DIRECT_STATUTORY_LOOKUP | STRATEGIC_LEGAL_ANALYSIS | HYBRID_RETRIEVAL",
    "intents": ["DIRECT_ARTICLE_RETRIEVAL", "STATUTORY_INTERPRETATION", ...],
    "target_components": ["STATUTORY_TEXT"] // or ["FACTS","ISSUES","RULES","APPLICATION","CONCLUSION"],
    "legal_domains": [
      { "domain": "Criminal Law", "confidence_score": 0.9, "applies_to": "CASE_LAW | STATUTORY_LOOKUP | BOTH" }
    ],
    "citation_analysis": {
      "explicit_citations": ["37"],
      "implied_subject": ["freedom of assembly"],
      "citation_confidence": "HIGH | MEDIUM | LOW | MISMATCH",
      "correction_signal": "..."
    },
    "strategy_critique": {
      "spatial": "...",   // Jurisdictional risks
      "doctrinal": "...", // Doctrine mismatches
      "temporal": "...",  // Limitation of Actions timelines
      "procedural": "..."  // Exhaustion doctrine, AG notice
    },
    "entities": {
      "statutes": { "primary": [...], "contextual": [...] },
      "article_numbers": ["37"],
      "section_numbers": [],
      "source_documents": ["Constitution of Kenya 2010"],
      "cases": [],
      "judges": []
    },
    "vector_query": "Article 37 Constitution Kenya",
    "vector_query_statutory": "...", // HYBRID mode only
    "vector_query_caselaw": "...",   // HYBRID mode only
    "retrieval_strategy": "STATUTORY_FIRST_THEN_CASELAW",
    "reasoning_summary": "..."
  }
  ```

  **Key classifier rules:**
  - Mode A: Uses a surgical minimal `vector_query` (under 10 words), skips strategic audit
  - Mode B: Dense keyword cluster, full 4-dimension strategy critique
  - Mode C: Dual-track output with separate statutory and case law query signals
  - The LLM is explicitly told to NOT answer the legal question — classifier only
  - JSON retry logic: on JSON parse failure or missing fields, the conversation continues with a correction message to the LLM (up to 3 retries)

  ---

  ### Step 2 — Case Law Retrieval (`Retriever.py`)

  **Database**: `LegalSummariesDB` (collection: `legal_summaries`)  
  **Embedding model**: `intfloat/e5-base-v2`  
  **Top N parents returned**: 3  
  **Semantic search pool**: 50 chunks

  **Scoring formula** per chunk:
  ```
  base_similarity = 1 / (1 + distance)
  domain_boost = 0.2–0.5 × confidence_score  (if legal domain matches)
  final_score = base_similarity + (multiplier × domain_boost)
  ```

  **Retrieval steps:**
  1. Parse classifier output, extract `vector_query_caselaw` (or fallback to `vector_query`)
  2. Load embedding model
  3. Embed query → search 50 candidates
  4. Filter to chunks matching `target_components` (FIRAC components like `RULES`, `APPLICATION`)
  5. Fall back to all chunks if component filter returns nothing
  6. Group chunks by `case_identifier` (parent document)
  7. Sort parents by `best_score`, take top 3
  8. For each parent, call `generate_case_summary()` via Groq LLM

  #### Case Summary Generation (per case)

  **Model**: `qwen/qwen3.8-27b`  
  **Temperature**: `0.3`  
  **Max tokens**: `1000`  
  **System prompt**: `"You are an expert Kenyan legal analyst who provides clear, insightful case summaries for lawyers and legal researchers."`

  The user prompt provides:
  - User's vector query
  - Target legal areas with confidence scores
  - Case name, court, judge, year, legal domain, winning party
  - Full FIRAC component chunks (Facts, Issues, Rules, Application, Conclusion)

  The LLM generates a structured analysis in four sections:
  - **CASE OVERVIEW** (2–3 sentences)
  - **KEY LEGAL PRINCIPLES** (2–3 sentences)
  - **RELEVANCE TO YOUR QUERY** (3–4 sentences, concrete)
  - **CRITICAL FACTS** (2–3 sentences)

  ---

  ### Step 3 — Statutory Rules Retrieval (`RulesRetriever.py`)

  **Triggered when**: `retrieval_mode` is `DIRECT_STATUTORY_LOOKUP` or `HYBRID_RETRIEVAL`, or `"RULES"` or `"STATUTORY_TEXT"` in `target_components`

  **Database**: `MasterRulesDB` (collection: `master_rules`)  
  **Embedding model**: `intfloat/e5-base-v2`  
  **Results**: Top 5 articles  

  **ChromaDB filter logic** (`build_where_filter()`):
  1. Always filters: `document_type IN ["constitution", "evidence_act"]`
  2. If `article_numbers` present → filters by `article_number` (direct lookup)
  3. If `section_numbers` present → also filters by `article_number` field
  4. If `source_documents` present → maps document name to PDF filename via `STATUTE_MAP`

  **Direct vs. semantic search**:
  - `DIRECT_STATUTORY_LOOKUP` with article numbers → `n_results=10` (constrained by metadata filter)
  - Otherwise → `n_results=30` (broader semantic search)

  Results are grouped by article (`document_type_articleNumber` key), sorted by score, top 5 returned with full text, preview, and PDF URL.

  ---

  ### Step 4 — Answer Generation (`/api/generate-legal-answer`)

  **Model**: `qwen/qwen3.8-27b`  
  **Temperature**: `0.3`  
  **Max tokens**: `1500`

  The system prompt switches based on `retrieval_mode`:

  #### Mode: `DIRECT_STATUTORY_LOOKUP`
  ```
  You are a Kenyan legal expert providing clear, accurate explanations of statutory provisions.
  1. Answer using retrieved statutory text as PRIMARY source
  2. Explain in plain language
  3. Use case law ONLY to illustrate interpretation
  4. Be concise and directly answer the question
  ```

  #### Mode: `HYBRID_RETRIEVAL`
  ```
  You are a Kenyan legal expert providing comprehensive legal analysis.
  1. Start with statutory provision(s)
  2. Then explain judicial interpretation
  3. Combine both for a complete answer
  Format: statutory basis → judicial interpretation → practical guidance
  ```

  #### Mode: `STRATEGIC_LEGAL_ANALYSIS` (default)
  ```
  You are a Kenyan legal expert providing strategic legal analysis.
  1. Use case law as PRIMARY source
  2. Reference relevant statutes from the cases
  3. Focus on practical application and legal strategy
  Format: how courts addressed this → legal principles applied → strategic guidance
  ```

  Context provided (injected into user prompt):
  - Top 3 statutory provisions (article number, header, full text, source)
  - Top 2 cases (case name, year, court, top 2 FIRAC chunks with component labels)

  ---

  ## Pipeline 2: General RAG (`/api/chat` → `pixe.py`)

  This is a simpler, document-agnostic RAG pipeline. It does NOT use the legal query classifier.

  ```
  POST /api/chat
    ↓
  run_rag_pipeline(query)
    ├── expand_query()  →  Groq LLM (adds synonyms/related terms)
    ├── hybrid_search()  →  70% semantic (e5-base-v2) + 30% TF-IDF keyword
    ├── rerank_chunks()  →  cross-encoder/ms-marco-MiniLM-L-6-v2
    ├── deduplicate_chunks()  →  >70% word overlap = remove
    └── generate_response()  →  Groq LLM
  ```

  **Database**: `ChunkDB` (collection: `ChunkDB`)  
  **Initial retrieval**: 10 candidates  
  **Final after reranking**: 3 chunks

  **Query expansion system prompt**:
  ```
  You are a helpful assistant that expands search queries with synonyms and related terms.
  ```

  **Response generation system prompt**:
  ```
  You are a helpful assistant that answers questions based ONLY on the provided context.
  Rules:
  1. Answer ONLY from context provided
  2. If context doesn't have the answer, say so
  3. DO NOT use training knowledge
  4. Break down complex questions step by step
  Reference chunks as [Chunk 1], [Chunk 2], etc.
  ```

  ---

  ## Ingestion Pipeline

  ### Case Law Ingestion (`CaseLaw/firac.py` + `CaseLaw/ingester.py`)

  ```
  PDF file
    ↓
  extract_text_from_pdf()  →  PyMuPDF (fitz) + Tesseract OCR fallback
    │   (OCR triggered if page has < 50 chars of direct text)
    ↓
  clean_text()  →  remove nav elements, OCR noise, normalize whitespace
    ↓
  extract_all_firac_combined()  →  Single Groq API call
    │   Model: qwen/qwen3.8-27b, temp=0.1, max_tokens=16000
    │   System: "Legal document analyst extracting structured FIRAC from Kenyan court judgments"
    │   Output: SECTION 0 (Metadata) + SECTION 1–5 (FIRAC)
    ↓
  Parse into: metadata, facts, issues, rules, application, conclusion
    ↓
  ingest_firac_data()  →  Embed each FIRAC component chunk
    │   Embedding: intfloat/e5-base-v2
    │   Store in LegalSummariesDB with metadata:
    │     case_identifier, firac_component, file_name, parties,
    │     court_level, judge, year, legal_domain, winning_party,
    │     source_file_path
    └──→ LegalSummariesDB / legal_summaries collection
  ```

  **FIRAC Combined Extraction Prompt** (simplified structure the LLM produces):

  ```
  SECTION 0: METADATA
    FILE NAME: ...
    PARTIES: ...
    COURT LEVEL: ...
    JUDGE: ...
    YEAR: ...
    LEGAL DOMAIN: ...
    WINNING PARTY: ... (plaintiff or defendant)

  SECTION 1: FACTS
    Background, chronological events, witness testimonies, 
    forensic evidence, prosecution case, defence case, 
    procedural facts

  SECTION 2: ISSUES
    Whether... questions the court must answer

  SECTION 3: RULES
    Binding legal principles (ratio decidendi), statutes cited,
    precedents applied — NOT dicta, NOT procedural background

  SECTION 4: APPLICATION
    How the court applied rules to facts, evidence evaluation,
    credibility assessments, reasoning per issue

  SECTION 5: CONCLUSION
    Holdings per issue, overall outcome, final orders
  ```

  **Metadata extraction per-component prompts** also exist individually (FACTS_EXTRACTION_PROMPT, ISSUES_EXTRACTION_PROMPT, RULE_EXTRACTION_PROMPT, APPLICATION_ANALYSIS_EXTRACTION_PROMPT, CONCLUSION_EXTRACTION_PROMPT) for standalone use.

  ---

  ### Constitution Ingestion (`constitutionlogic/constitution.py`)

  ```
  TheConstitutionOfKenya.pdf
    ↓
  extract_text_from_pdf()  →  PyMuPDF + Tesseract OCR fallback
    ↓
  LLM chunks the PDF into JSON  (Groq API, qwen/qwen3.8-27b)
    │   System Prompt: "Kenyan Constitutional Data Architect"
    │   Rules:
    │     - Hierarchy tracking: CHAPTER → PART → ARTICLE
    │     - Title-Above-Number rule (article title is the line BEFORE article number)
    │     - Content cleaning (strip "Constitution of Kenya", "[Rev. 2022]", page numbers)
    │     - Map each article to 1 of 10 legal domains
    │   Output per article:
    │     { source, chapter_no, chapter_title, part_no, part_title,
    │       article_no, article_title, legal_domain, summary, keywords, text_content }
    ↓
  flatten_articles_to_embeddable_chunks()  →  chunks with CHUNK_SIZE/CHUNK_OVERLAP
    ↓
  ConstitutionDigester.embed_chunk()  →  intfloat/e5-base-v2
    ↓
  MasterRulesDB / master_rules collection
    │   Metadata stored: document_type="constitution", article_number,
    │   article_header, chapter, chapter_title, part, source_document
  ```

  ---

  ## Endpoint Summary

  | Endpoint | Method | What it does |
  |----------|--------|-------------|
  | `/api/chat` | POST | General RAG via `pixe.py` → `ChunkDB` |
  | `/api/qretrieve` | POST | Legal classifier + case law + statutory retrieval |
  | `/api/generate-legal-answer` | POST | LLM answer generation from retrieval results |
  | `/api/digest` | POST | Ingest PDF(s) via FIRAC → `LegalSummariesDB` |
  | `/api/digest/constitution` | POST | Ingest Constitution PDF → `MasterRulesDB` |
  | `/api/firac` | POST | Extract FIRAC from PDF (no DB storage) |
  | `/api/ingest-firac` | POST | Store FIRAC data into DB |
  | `/api/case-pdf` | GET | Serve case PDF by `case_identifier` |
  | `/api/rule-pdf` | GET | Serve Constitution/Evidence Act PDF |
  | `/api/upload` | POST | Upload PDF to Repo directory |
  | `/api/files` | GET | List PDFs in Repo |

  ---

  ## Key Design Decisions & Observations

  ### What works well
  1. **Dual-mode retrieval**: Separating statutory (direct lookup) from case law (semantic) makes a lot of sense for legal research. Users asking "what does Article 37 say" get the statute; users asking "can I sue the government for X" get case law.
  2. **Single combined FIRAC API call**: Originally each FIRAC section was a separate API call (commented-out code shows this). Combining into one call reduces Groq rate-limiting.
  3. **Three Groq API key rotation**: Smart failover with per-key rate limit detection and a primary-key reset after exhaustion.
  4. **Component-based scoring**: Boosting chunks that match `target_components` (e.g., RULES, APPLICATION) over ones that don't is important — a "FACTS" chunk for a statutory query is noise.

  ### Architectural observations worth discussing
  1. **`pixe.py` vs `Retriever.py` duplication**: Two parallel retrieval pipelines exist. `pixe.py` uses hybrid search + cross-encoder reranking and queries `ChunkDB`. `Retriever.py` does simple semantic search against `LegalSummariesDB`. The more sophisticated retrieval methods in `pixe.py` (reranking, hybrid search) are NOT applied to the legal case law retrieval in `Retriever.py`.
  2. **Retriever.py loads the embedding model fresh on every request**: `SentenceTransformer(EMBEDDING_MODEL_NAME)` is called inside `retrieve_relevant_cases()` — this means it re-loads the 768-dim model on every API call. Same issue in `RulesRetriever.py`.
  3. **Temperature mismatch**: FIRAC extraction uses temp=0.1 (good, structured output). Case summaries use temp=0.3. The classifier uses temp=0.1 (good). The final answer generation uses temp=0.3. Consistent but worth discussing whether 0.3 is right for answer generation in a legal context.
  4. **No reranking in the legal retrieval path**: The legal pipeline (`/api/qretrieve`) uses only chromadb cosine distance + a manual scoring formula. The cross-encoder reranker used in `pixe.py` is absent here.
  5. **Strategy critique is informational only**: The classifier produces `strategy_critique` (jurisdiction flags, limitation period warnings) but this is returned to the frontend, not used to modify the retrieval or answer generation logic.
  6. **LLM-generated case summaries on every retrieval**: Every `/api/qretrieve` call generates 3 case summaries via Groq LLM. This adds latency (3 API calls) and cost to every query.

  ---

  ## Data Flow Diagram (Text)

  ```
  User Query
      │
      ▼
  POST /api/qretrieve
      │
      ├──▶ Qclassifier.py
      │       │  LLM: qwen/qwen3.8-27b (temp=0.1)
      │       │  → retrieval_mode (DIRECT/STRATEGIC/HYBRID)
      │       │  → vector_query, target_components, entities
      │       │
      │       ├──▶ Retriever.py (always)
      │       │       │  Embed: e5-base-v2
      │       │       │  DB: LegalSummariesDB/legal_summaries
      │       │       │  → Top 3 case law parents
      │       │       └──▶ generate_case_summary() × 3
      │       │               LLM: qwen/qwen3.8-27b (temp=0.3)
      │       │
      │       └──▶ (returned to app.py)
      │
      ├──▶ RulesRetriever.py (if DIRECT or HYBRID mode)
      │       │  Embed: e5-base-v2
      │       │  DB: MasterRulesDB/master_rules
      │       └──▶ Top 5 constitutional articles
      │
      └──▶ Response: { classification, retrieval: { results, rules } }

  User submits retrieval results for answer
      │
      ▼
  POST /api/generate-legal-answer
      │  LLM: qwen/qwen3.8-27b (temp=0.3)
      │  System: mode-specific legal expert prompt
      └──▶ Final legal answer
  ```

  ---

  ## Environment Variables Required

  ```
  GROQ_API_KEY       # Primary Groq key
  GROQ_API_KEY_2     # Optional: rate limit fallback
  GROQ_API_KEY_3     # Optional: rate limit fallback
  ```
