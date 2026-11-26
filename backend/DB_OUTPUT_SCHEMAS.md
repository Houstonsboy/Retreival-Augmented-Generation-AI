# Database Output Schemas

Three ingestion pipelines, three output shapes. Each section shows:
1. **Intermediate** — the Python dict produced by the extraction step before ChromaDB
2. **Stored** — the exact document + metadata written into ChromaDB
3. **Live example** — a real record from the running database

---

## Path 1 — Case Law → `legal_summaries`

**Pipeline:** `firac.py` → Groq LLM (FIRAC extraction) → `ingester.py` → ChromaDB

**Location:** `backend/LegalSummariesDB/` · collection: `legal_summaries`

### 1a. Intermediate: `run_firac()` return value

`firac.py` calls the Groq LLM once and parses the response into six plain-text fields.
This dict is the handoff to `ingester.py`.

```json
{
  "full_response": "SECTION 0: METADATA\nFILE NAME: ...\n\nSECTION 1: FACTS\n...",
  "metadata":    "FILE NAME: Muruatetu & another v Republic\nPARTIES: Francis Karioko Muruatetu...\nCOURT LEVEL: Supreme Court of Kenya\nJUDGE: MK KOOME, CJ, PM MWILU...\nYEAR: 2021\nLEGAL DOMAIN: Criminal Law\nWINNING PARTY: The appellants...",
  "facts":       "The appellants, Francis Karioko Muruatetu and Wilson Thirimbu Mwangi, were convicted of murder and sentenced to death. They challenged the legality of the mandatory nature of the death sentence...",
  "issues":      "The key issues the court had to decide included:\n1. Whether the decision of the Supreme Court finding that the mandatory nature of the death sentence...",
  "rules":       "The binding legal rules and principles applied by the court included:\n1. The Constitution of Kenya, 2010, particularly article 25...",
  "application": "The court applied these legal rules and principles to the established facts by:\n1. Analyzing the constitutionality of the mandatory death sentence...",
  "conclusion":  "The court concluded that:\n1. The decision in Muruatetu applies only to sentences of murder under sections 203 and 204 of the Penal Code..."
}
```

> `metadata` is a raw multiline string. `ingester.py` calls `parse_metadata()` to split it
> into a dict before writing to ChromaDB. No JSON at this stage — plain text sections.

### 1b. Stored: one ChromaDB document per FIRAC component

Each of the five FIRAC fields becomes a **separate vector** with the same case metadata attached.
One PDF therefore produces **5 documents** in the collection (8 cases = 40 total vectors).

**Document (the embedded text):**
```
The appellants, Francis Karioko Muruatetu and Wilson Thirimbu Mwangi, were convicted
of murder and sentenced to death. They challenged the legality of the mandatory nature
of the death sentence imposed under section 204 of the Penal Code...
```

**Metadata (stored alongside the vector):**
```json
{
  "case_identifier":  "Muruatetu & another v Republic; Katiba Institute & 5 others (Amicus Curiae)",
  "file_name":        "Muruatetu & another v Republic; Katiba Institute & 5 others (Amicus Curiae)",
  "firac_component":  "facts",
  "parties":          "Francis Karioko Muruatetu & another v Republic; Katiba Institute & 5 others (Amicus Curiae)",
  "court_level":      "Supreme Court of Kenya",
  "judge":            "MK KOOME, CJ, PM MWILU, DCJ & VP, MK IBRAHIM, SC WANJALA, I LENAOLA, N NDUNGU & W OUKO, SCJJ",
  "year":             "2021",
  "legal_domain":     "Criminal Law",
  "winning_party":    "The appellants, as the court found in their favor regarding the unconstitutionality of the mandatory death sentence for murder.",
  "content_length":   1132,
  "document_hash":    "20a5c00bd0be6ab5025c1d6e3d5ac160f187ad622d444f72000ad9ac78e3c810",
  "source_file_path": "/home/gift/Retreival-Augmented-Generation-AI/backend/Repo/Res_ipsa_loquitur/Muruatetu  another v Republic Katiba Institute  5 others (Amicus Curiae) (Petition 15  16of2015) 2021KESC31(KLR) (6July2021) (Directions).pdf",
  "ingested_at":      "2025-12-30 00:00:49",
  "embedding_model":  "intfloat/e5-base-v2"
}
```

**Document ID pattern:**
```
{sanitized_case_identifier}_{firac_component}_{unix_ms_timestamp}

Example:
Muruatetu___another_v_Republic__Katiba_Institute___facts_1767042049256
```

### 1c. All five chunks for the same case

| `firac_component` | `content_length` | Embedded text starts with… |
|---|---|---|
| `facts`       | 1132 | "The appellants, Francis Karioko Muruatetu…" |
| `issues`      | 1131 | "The key issues the court had to decide…" |
| `rules`       | 622  | "The binding legal rules and principles…" |
| `application` | 746  | "The court applied these legal rules…" |
| `conclusion`  | 848  | "The court concluded that…" |

### 1d. Metadata field reference

| Field | Type | Role |
|---|---|---|
| `case_identifier` | `str` | Primary key — used for dedup (`collection.delete(where={"case_identifier": ...})`) |
| `file_name` | `str` | Display name for the case |
| `firac_component` | `str` | `facts` / `issues` / `rules` / `application` / `conclusion` — used in query `where` filter |
| `parties` | `str` | Full party names — embedded in the query text |
| `court_level` | `str` | Filterable: `"Supreme Court of Kenya"`, `"Court of Appeal"`, `"High Court"` |
| `judge` | `str` | Filterable by judge name |
| `year` | `str` | Filterable: supports `$gte` / `$lte` range queries |
| `legal_domain` | `str` | Filterable: `"Criminal Law"`, `"Contract Law"`, etc. |
| `winning_party` | `str` | Embedded for semantic relevance |
| `content_length` | `int` | Char count of the embedded text |
| `document_hash` | `str` | SHA-256 of the original PDF text — change detection |
| `source_file_path` | `str` | Absolute path to original PDF |
| `ingested_at` | `str` | `"YYYY-MM-DD HH:MM:SS"` timestamp |
| `embedding_model` | `str` | Always `"intfloat/e5-base-v2"` |

---

## Path 2 — Statutes → `master_rules`

**Pipeline:** PDF → `constitutionbreaker.py` (article chunking) → `constitution.py` (Groq LLM JSON structuring) → `constitution_digester.py` → ChromaDB

**Location:** `backend/constitutionlogic/MasterRulesDB/` · collection: `master_rules`

### 2a. Intermediate: `constitution.py` LLM output

The LLM is prompted to return a **JSON array** — one object per constitutional article.
This is written to `constitution_embeddable_chunks.json` before embedding.

```json
[
  {
    "source":        "constitution2010",
    "chapter_no":    4,
    "chapter_title": "THE BILL OF RIGHTS",
    "part_no":       1,
    "part_title":    "GENERAL PROVISIONS TO THE BILL OF RIGHTS",
    "article_no":    20,
    "article_title": "Application of Bill of Rights.",
    "legal_domain":  "Fundamental Rights & Freedoms",
    "summary":       "Establishes that the Bill of Rights applies to all law and binds all State organs and persons.",
    "keywords":      ["Bill of Rights", "application", "State organs", "persons", "fundamental freedoms"],
    "text_content":  "20. (1) The Bill of Rights applies to all law and binds all State organs and all persons.\n(2) Every person shall enjoy the rights and fundamental freedoms in the Bill of Rights to the greatest extent consistent with the nature of the right or fundamental freedom..."
  }
]
```

> This intermediate JSON is the full structured representation of a statute article before it
> is chunked further (if over 1500 chars) and embedded by `constitution_digester.py`.

### 2b. Stored: one ChromaDB document per article chunk

Long articles are split into multiple chunks. Short articles produce exactly one chunk.

**Document (the embedded text):**
```
20. (1) The Bill of Rights applies to all law and binds all State organs and all persons.
(2) Every person shall enjoy the rights and fundamental freedoms in the Bill of Rights to
the greatest extent consistent with the nature of the right or fundamental freedom.
(3) In applying a provision of the Bill of Rights, a court shall...
```

**Metadata:**
```json
{
  "article_number":     "20",
  "article_header":     "Application of Bill of Rights.",
  "chapter":            "FOUR",
  "chapter_title":      "THE BILL OF RIGHTS",
  "part":               "1",
  "part_title":         "GENERAL PROVISIONS TO THE BILL OF RIGHTS",
  "source_document":    "TheConstitutionOfKenya.pdf",
  "document_type":      "constitution",
  "is_complete_article": false,
  "chunk_index":        1,
  "total_chunks":       2,
  "article_length":     1579,
  "chunk_length":       998,
  "ingestion_timestamp": 1770036263
}
```

**Document ID pattern:**
```
const_{source_document}_art{article_number}_chunk{chunk_index}_{content_hash_8chars}

Example:
const_TheConstitutionOfKenya.pdf_art20_chunk1_040dee31
```

### 2c. Single-chunk article example (Article 1)

**Document:**
```
1. (1) All sovereign power belongs to the people of Kenya and shall be exercised
only in accordance with this Constitution.
(2) The people may exercise their sovereign power either directly or through their
democratically elected representatives...
```

**Metadata:**
```json
{
  "article_number":     "1",
  "article_header":     "Sovereignty of the people.",
  "chapter":            "ONE",
  "chapter_title":      "SOVEREIGNTY OF THE PEOPLE AND SUPREMACY OF THIS CONSTITUTION",
  "part":               "",
  "part_title":         "",
  "source_document":    "TheConstitutionOfKenya.pdf",
  "document_type":      "constitution",
  "is_complete_article": true,
  "chunk_index":        1,
  "total_chunks":       1,
  "article_length":     730,
  "chunk_length":       730,
  "ingestion_timestamp": 1770036253
}
```

### 2d. Metadata field reference

| Field | Type | Role |
|---|---|---|
| `article_number` | `str` | Enables direct lookup by citation: `{"article_number": {"$eq": "47"}}` |
| `article_header` | `str` | Article title — embedded in the query text for semantic matching |
| `chapter` | `str` | Word form: `"ONE"`, `"FOUR"`, `"NINE"` — filterable |
| `chapter_title` | `str` | e.g. `"THE BILL OF RIGHTS"` — embedded for semantic matching |
| `part` | `str` | Part number as string, empty `""` if no part in that chapter |
| `part_title` | `str` | e.g. `"GENERAL PROVISIONS TO THE BILL OF RIGHTS"` |
| `source_document` | `str` | Filename of the source PDF — identifies which statute |
| `document_type` | `str` | Always `"constitution"` — filterable for routing |
| `is_complete_article` | `bool` | `true` if the full article fits in one chunk, `false` if split |
| `chunk_index` | `int` | 1-based chunk position within the article |
| `total_chunks` | `int` | Total chunks for this article |
| `article_length` | `int` | Char count of the full article |
| `chunk_length` | `int` | Char count of this specific chunk |
| `ingestion_timestamp` | `int` | Unix epoch timestamp |

---

## Path 3 — Raw Documents → `ChunkDB`

**Pipeline:** `digester.py` → `RecursiveCharacterTextSplitter` (1000 chars / 300 overlap) → ChromaDB

**Location:** `backend/ChunkDB/` · collection: `ChunkDB`

> **Note:** This is the legacy pipeline. It has no semantic structuring — text is split
> mechanically by character count. The `legal_summaries` and `master_rules` collections
> should be preferred for all legal query workloads. `ChunkDB` is retained for raw document
> search and the `/api/chat` fallback path through `pixe.py`.

### 3a. No intermediate structure

`digester.py` does not call an LLM. The transformation is:

```
PDF text  →  clean_text()  →  RecursiveCharacterTextSplitter  →  [chunk_0, chunk_1, ..., chunk_n]
```

Each chunk is embedded with the prefix `"passage: {chunk_text}"` before being stored.

### 3b. Stored: one ChromaDB document per text window

**Document (the embedded text):**
```
Copyright © 2020 NonDisclosureAgreement.com. All Rights Reserved. Page 1 of 2
NON-DISCLOSURE AGREEMENT (NDA)
This Nondisclosure Agreement or ("Agreement") has been entered into on the date of
________ and is by and between:
Party Disclosing Information: ________________________
...
```

**Metadata:**
```json
{
  "source":        "/home/gift/Retreival-Augmented-Generation-AI/Repo/Basic-Non-Disclosure-Agreement.pdf",
  "file_name":     "Basic-Non-Disclosure-Agreement.pdf",
  "file_type":     ".pdf",
  "file_hash":     "203788e953b24dddab8f335632ab258ce98d436349be26acedd929a511e34e4e",
  "file_size":     5696,
  "original_size": 5773,
  "chunk_id":      0,
  "chunk_index":   0,
  "total_chunks":  8,
  "chunk_size":    944,
  "start_index":   0,
  "cleaned":       true,
  "loaded_at":     "2025-11-09 19:41:56",
  "ingested_at":   "2025-11-09 19:42:06",
  "embedding_model": "intfloat/e5-base-v2"
}
```

**Document ID pattern:**
```
{file_stem}_chunk_{chunk_index}

Example:
Basic-Non-Disclosure-Agreement_chunk_0
```

### 3c. Metadata field reference

| Field | Type | Role |
|---|---|---|
| `source` | `str` | Absolute path — primary dedup key (`collection.delete(where={"source": ...})`) |
| `file_name` | `str` | Filename for display |
| `file_type` | `str` | `".pdf"` or `".txt"` |
| `file_hash` | `str` | SHA-256 of raw file bytes — skip re-ingestion if unchanged |
| `file_size` | `int` | Cleaned content char count |
| `original_size` | `int` | Raw content char count before cleaning |
| `chunk_id` | `int` | Same as `chunk_index` (legacy alias) |
| `chunk_index` | `int` | 0-based position in the file's chunk sequence |
| `total_chunks` | `int` | Total chunks produced from this file |
| `chunk_size` | `int` | Char count of this chunk (≤ 1000) |
| `start_index` | `int` | Character offset of this chunk in the cleaned document |
| `cleaned` | `bool` | Always `true` — text passed through `clean_text()` |
| `loaded_at` | `str` | `"YYYY-MM-DD HH:MM:SS"` — when text was extracted from PDF |
| `ingested_at` | `str` | `"YYYY-MM-DD HH:MM:SS"` — when the chunk was written to ChromaDB |
| `embedding_model` | `str` | Always `"intfloat/e5-base-v2"` |

---

## Summary: Collection Comparison

| | `legal_summaries` | `master_rules` | `ChunkDB` |
|---|---|---|---|
| **Location** | `backend/LegalSummariesDB/` | `backend/constitutionlogic/MasterRulesDB/` | `backend/ChunkDB/` |
| **Documents** | 40 | 619 | 704 |
| **Chunk unit** | FIRAC component | Constitutional article (or clause) | 1000-char sliding window |
| **LLM involved** | Yes — Groq FIRAC extraction | Yes — Groq JSON structuring | No |
| **Dedup key** | `case_identifier` | content hash in ID | `source` (file path) |
| **Primary filter fields** | `firac_component`, `legal_domain`, `court_level`, `year` | `article_number`, `chapter`, `document_type` | `file_type`, `cleaned` |
| **Embedding prefix** | `"passage: "` | `"passage: "` | `"passage: "` |
| **Embedding model** | `intfloat/e5-base-v2` (768-dim) | `intfloat/e5-base-v2` (768-dim) | `intfloat/e5-base-v2` (768-dim) |
| **Distance metric** | cosine | cosine | cosine |
| **Used by** | `Retriever.py` (case law queries) | `RulesRetriever.py` (statute queries) | `pixe.py` (raw document chat) |
