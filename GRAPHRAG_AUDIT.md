# Graph RAG Audit — Current Setup & Tech Stack

> **Purpose:** Document what Graph RAG infrastructure exists in the codebase, what is NOT yet wired into the live pipeline, and provide a complete tech stack overview for architecture review.
> **Status as of 2026-11-23:** Graph layer is fully built but entirely disconnected from the RAG pipeline. Zero Cypher queries reach the Python backend.

---

## 1. What "Graph RAG" Means in This Codebase

Standard RAG retrieves chunks by vector similarity. Graph RAG augments that by traversing a knowledge graph — when a term is retrieved, the graph walks its relationships (`SEE`, `CF`, `ALSO_TERMED`) to surface related concepts the query never mentioned. In this system, the intended graph layer sits on **Black's Law Dictionary 7th Edition** (~40,000 legal terms) stored in Neo4j, so a query mentioning "consideration" would also surface "quid pro quo", "adequate consideration", and "past consideration" through graph traversal — concepts an embedding alone might miss.

---

## 2. Graph Infrastructure — What Exists

### 2.1 Neo4j Server

| Item | Detail |
|---|---|
| Distribution | Neo4j Community 5.18.0 |
| Location | `neo4j-community-5.18.0/` (repo root) |
| Bolt port | `localhost:7687` |
| HTTP Browser | `localhost:7474` |
| Config | `neo4j-community-5.18.0/conf/neo4j.conf` |
| Current state | **NOT RUNNING** — last started 2026-02-23, shut down same day |
| Last log entry | `2026-02-23 14:17:01 INFO Stopped.` |

The server binary is installed and functional; it just needs `./bin/neo4j start` to come back up.

---

### 2.2 Java Graph Builder (`neo4j-java/`)

Two Java classes implement the graph population layer:

**`DictionaryKnowledgeGraph.java`** (235 lines)

Reads the exported JSON file and uses Cypher `MERGE` statements to build the graph:

```
Node type:    Term  { name, definition, page }
Relationships: SEE  →  (term A references term B)
               CF   →  (compare: term A vs term B)
               ALSO_TERMED → (synonym / alias)
```

Sample Cypher it emits:
```cypher
MERGE (t:Term {name: $name}) SET t.definition = $definition, t.page = $page
MERGE (a:Term {name: $from}) MERGE (b:Term {name: $to}) MERGE (a)-[:SEE]->(b)
```

**`Neo4jWriter.java`** (135 lines)

Generic reusable writer utility — `createNode()`, `createRelationship()`, `runWrite()`, `runRead()` — intended to be extended for future node types beyond dictionary terms (e.g., `Case`, `Statute`, `Section`, `Judge`).

**Run command:**
```bash
export NEO4J_PASSWORD=<password>
cd neo4j-java
./run_dictionary_graph.sh
# or: mvn compile exec:java -Dexec.mainClass="com.legal.rag.neo4j.DictionaryKnowledgeGraph"
```

---

### 2.3 Python Export Bridge (`export_blacklaw_to_json.py`)

This is the bridge between ChromaDB (Python) and Neo4j (Java). Because the Neo4j Java driver is not in `requirements.txt`, the graph build is a two-step offline process:

```
Step 1 (Python): ChromaDB black_law_dictionary → blacklaw_export_for_neo4j.json
Step 2 (Java):   blacklaw_export_for_neo4j.json → Neo4j graph (Term nodes + relationships)
```

The exported JSON is already on disk at:
`backend/Const_IntepretationLaws/blacklaw_export_for_neo4j.json`
— 40,061 entries, ~15 MB.

**Export JSON schema:**
```json
{
  "entries": [
    {
      "term": "consideration",
      "definition": "Something of value given by one party...",
      "page": 329,
      "metadata": {
        "see": ["adequate consideration", "past consideration"],
        "cf": ["nudum pactum"],
        "also_termed": ["quid pro quo"]
      }
    }
  ]
}
```

---

### 2.4 Black's Law Dictionary ChromaDB Collection

| Property | Value |
|---|---|
| Path | `backend/Const_IntepretationLaws/BlackLawDictionaryDB/` |
| Collection name | `black_law_dictionary` |
| Vector count | **40,061** |
| Embedding model | `intfloat/e5-base-v2` (768-dim) |
| Metadata keys | `term`, `definition`, `definition_length`, `page`, `source`, `document_type`, `ingestion_timestamp` |
| Cross-ref fields | `see`, `cf`, `also_termed` stored as comma-separated strings |
| Source document | Black's Law Dictionary 7th Edition (`BlackLaw7thEdition.pdf`) |

This collection is **also not connected to the main RAG pipeline** — no retriever calls it. It exists solely as the source for the Neo4j export.

---

### 2.5 Supporting Reference Materials (`Const_IntepretationLaws/`)

The directory also contains a set of Kenyan constitutional interpretation resources that form the intended semantic layer for future graph nodes beyond dictionary terms:

| File | Purpose |
|---|---|
| `BlackLaw7thEdition.pdf` | Source for 40,061 dictionary terms |
| `BlackLawDictionaryDB/` | ChromaDB vector store of dictionary |
| `blacklaw_export_for_neo4j.json` | Pre-generated Neo4j import file |
| `reader.py` | PDF reader used during ingestion |
| `testBlackLaw.py` | Metadata field checker / cross-reference validator |
| `metadata_field_report.json` | Sample cross-reference statistics |
| `135144-INTERPRETATION OF CHAPTER 6...pdf` | Constitutional interpretation texts |
| `Cecil Abungu Constitutional Interpretation.pdf` | Academic source |
| `Mumo Matemu CA Decision.pdf` | Key constitutional court ruling |
| `Wambora Supreme Court.pdf` | Supreme Court interpretation case |
| `Rose Wangui Mambo... 2020KEHC1870.pdf` | FIDA interested party ruling |
| `Judicial Commission... Goldenberg...pdf` | Landmark corruption/interpretation case |

---

## 3. What Is NOT Connected to the Pipeline (The Gap)

| Component | Exists? | Wired to pipeline? |
|---|---|---|
| Neo4j 5.18.0 server | ✅ installed | ❌ not running, no Python calls |
| `DictionaryKnowledgeGraph.java` | ✅ compiled | ❌ offline tool only |
| `Neo4jWriter.java` | ✅ built | ❌ no Java service exposed |
| `black_law_dictionary` ChromaDB | ✅ 40,061 vectors | ❌ never queried by any retriever |
| `blacklaw_export_for_neo4j.json` | ✅ on disk | ❌ only consumed by Java builder |
| Neo4j Python driver (`neo4j`) | ❌ not in requirements.txt | ❌ — |
| Cypher queries in Python | ❌ zero | ❌ — |
| Graph traversal in Retriever.py | ❌ none | ❌ — |
| Graph traversal in RulesRetriever.py | ❌ none | ❌ — |

**Zero lines of Python in `app.py`, `Retriever.py`, `RulesRetriever.py`, or `Qclassifier.py` reference Neo4j, Cypher, or the dictionary collection.**

---

## 4. Current RAG Pipeline — Full Tech Stack

### 4.1 Backend

| Layer | Technology | Version | Notes |
|---|---|---|---|
| Runtime | Python | 3.13 | |
| Web framework | Flask | 3.1.2 | with flask-cors 6.0.1 |
| **Vector store** | ChromaDB | 1.3.5 | persistent, cosine distance, 4 collections |
| **Embedding model** | `intfloat/e5-base-v2` | via sentence-transformers 5.1.2 | 768-dim; `"passage: "` prefix for docs, `"query: "` for queries |
| **Cross-encoder reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | via sentence-transformers | sigmoid-normalised scores |
| **Lexical retrieval** | BM25Okapi | rank-bm25 0.2.2 | in-memory, rebuilt per process start |
| **Fusion** | RRF (Reciprocal Rank Fusion) | custom (`k=60, n=30`) | fuses BM25 + vector rankings |
| **LLM** | `qwen/qwen3.8-27b` | Groq API | 16 API keys rotated; classifier temp=0.1, retriever temp=0.3 |
| **Intent classifier** | Qclassifier.py | custom | outputs `retrieval_mode`, `intents[]`, `vector_query` |
| PDF extraction | PyMuPDF (fitz) | 1.24.14 | coordinate-aware; bounding-box chunk metadata |
| PDF generation | fpdf2 | 2.9.0 | s.106B certificates + deficiency reports |
| OCR | RapidOCR + pytesseract | 3.10.0 / 0.3.13 | ONNX Runtime 1.23.2 (CPU) |
| Image handling | Pillow | 11.1.0 | |
| Scanned PDF routing | ocrmypdf | 16.5.0 | |
| DOCX ingestion | python-docx | 1.2.0 | paragraph-split, no coordinate data |
| Auth & history | Firebase Admin SDK | 7.7.0 | Firestore + Firebase Auth |
| ML inference | PyTorch | 2.9.1 | CPU inference for sentence-transformers |
| Transformers | Hugging Face Transformers | 4.57.1 | |
| OpenTelemetry | opentelemetry-api/sdk | 1.45.1 | OTLP gRPC export |
| Citation parsing | LangChain | 1.0.7 / langchain-core 1.0.6 | partial usage |

**ChromaDB Collections Summary:**

| Collection | DB Path | Vectors | Source |
|---|---|---|---|
| `case_law` | `backend/CaseLawDB/` | **78** | 11 SCC case PDFs/DOCX in `SCCCases/` |
| `statutes` | `backend/StatutesDB/` | **1,331** | Kenyan statute text via `build_statute_chunks.py` |
| `master_rules` | `backend/constitutionlogic/MasterRulesDB/` | **619** | Constitution of Kenya 2010 |
| `black_law_dictionary` | `backend/Const_IntepretationLaws/BlackLawDictionaryDB/` | **40,061** | Black's Law Dictionary 7th Ed *(not queried by pipeline)* |

---

### 4.2 Frontend

| Technology | Version | Role |
|---|---|---|
| Next.js | 16.0.3 | App framework (App Router) |
| React | 19.2.0 | UI |
| TypeScript | 5 | Type safety |
| Tailwind CSS | 4 | Styling |
| Firebase (client) | 10.12.0 | Auth + Firestore real-time |
| lucide-react | 0.562.0 | Icons |
| react-markdown | 10.1.0 | LLM response rendering |
| remark-gfm | 4.0.1 | GFM tables in markdown |
| react-pdf | 7.7.3 | In-browser PDF preview |
| eslint-config-next | 16.0.3 | Lint |

---

### 4.3 Graph / Neo4j Layer (Offline, Not Integrated)

| Technology | Version | Role |
|---|---|---|
| Neo4j Community | 5.18.0 | Graph database server |
| Java | (JVM) | Graph builder runtime |
| Maven | — | Build tool for `neo4j-java/` |
| neo4j-java-driver | 5.17.0 (bundled) | Bolt protocol client |
| Gson | — | JSON parsing in Java |
| Cypher | Neo4j DSL | Graph query language |

**Planned graph schema (from `DictionaryKnowledgeGraph.java`):**
```
(:Term {name, definition, page})
  -[:SEE]->(:Term)
  -[:CF]->(:Term)
  -[:ALSO_TERMED]->(:Term)
```

---

## 5. Envisioned Graph RAG Architecture (Not Yet Implemented)

Based on the existing code artifacts, the intended Graph RAG flow would be:

```
User query
    │
    ▼
Qclassifier.py  ──────────────────────────────────────────────────────────┐
(extract legal terms / entities from query)                               │
    │                                                                      │
    ▼                                                                      ▼
ChromaDB vector search                                        Neo4j graph traversal
(case_law / statutes / master_rules)                          MATCH (t:Term {name: $term})
    │                                                         -[:SEE|CF|ALSO_TERMED*1..2]→
    │                                                         (related:Term)
    │                                                              │
    └──────────────────── merge & deduplicate ────────────────────┘
                                  │
                                  ▼
                       Cross-encoder rerank (Retriever.py)
                                  │
                                  ▼
                       LLM synthesis (qwen/qwen3.8-27b via Groq)
                                  │
                                  ▼
                           Response to advocate
```

The key gap is **Step 2**: nothing currently calls Neo4j during a query, so legal term expansion (the graph traversal) never happens. The user query is embedded as-is; synonyms, related doctrines, and cross-referenced concepts are not surfaced unless they appear verbatim in the original query text.

---

## 6. What Needs to Happen to Activate Graph RAG

In order of dependency:

1. **Start Neo4j** — `cd neo4j-community-5.18.0 && ./bin/neo4j start`
2. **Load the graph** — `cd neo4j-java && ./run_dictionary_graph.sh` (reads existing `blacklaw_export_for_neo4j.json`, builds ~40,061 `Term` nodes and their `SEE`/`CF`/`ALSO_TERMED` edges)
3. **Add `neo4j` to `requirements.txt`** — the official Python driver (`neo4j>=5.18`)
4. **Write a `GraphExpander` utility in Python** — takes a list of legal terms extracted from the classifier output, runs a Cypher query (`MATCH (t:Term {name: $name})-[:SEE|CF|ALSO_TERMED]-(r) RETURN r.name`), returns expanded term list
5. **Hook into `Retriever.py`** — after Qclassifier extracts `intents[]` and `vector_query`, expand that keyword list via graph traversal before building the embedding, so the vector search is enriched with related terms
6. **Consider case-level graph** (future) — extend `Neo4jWriter.java` to add `(:Case)`, `(:Section)`, `(:Judge)` nodes and `CITES`, `INTERPRETS`, `OVERRULES` edges from the SCC case metadata already in ChromaDB

---

## 7. Summary

The Graph RAG layer is architecturally designed and physically built — Neo4j is installed, the Java builder is compiled, the dictionary is exported, and the ChromaDB vector store has 40,061 embeddings ready. The only thing missing is the Python glue that calls Neo4j at query time and feeds the expanded term set back into the retrieval pipeline. As of today, every query reaches the user with no graph traversal — the system is operating as a pure hybrid vector+BM25 RAG, with the graph layer dormant at `localhost:7687` (currently stopped).
