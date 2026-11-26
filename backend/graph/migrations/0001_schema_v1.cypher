// Schema v1 — empty graph with constraints and indexes for the Part 1 case-law design.
// All statements are idempotent (IF NOT EXISTS).
// Community edition only: uniqueness constraints and regular indexes.
// Property-existence and node-key constraints are Enterprise-only;
// required-property enforcement is handled by pydantic models in models.py.

// ---------- identity (uniqueness) ----------
CREATE CONSTRAINT case_key        IF NOT EXISTS FOR (n:Case)          REQUIRE n.case_key IS UNIQUE;
CREATE CONSTRAINT court_code      IF NOT EXISTS FOR (n:Court)         REQUIRE n.code IS UNIQUE;
CREATE CONSTRAINT judge_key       IF NOT EXISTS FOR (n:Judge)         REQUIRE n.judge_key IS UNIQUE;
CREATE CONSTRAINT opinion_key     IF NOT EXISTS FOR (n:Opinion)       REQUIRE n.opinion_key IS UNIQUE;
CREATE CONSTRAINT paragraph_key   IF NOT EXISTS FOR (n:Paragraph)     REQUIRE n.para_key IS UNIQUE;
CREATE CONSTRAINT issue_key       IF NOT EXISTS FOR (n:Issue)         REQUIRE n.issue_key IS UNIQUE;
CREATE CONSTRAINT principle_key   IF NOT EXISTS FOR (n:Principle)     REQUIRE n.principle_key IS UNIQUE;
CREATE CONSTRAINT provision_key   IF NOT EXISTS FOR (n:Provision)     REQUIRE n.provision_key IS UNIQUE;
CREATE CONSTRAINT statute_key     IF NOT EXISTS FOR (n:Statute)       REQUIRE n.statute_key IS UNIQUE;
CREATE CONSTRAINT evidence_code   IF NOT EXISTS FOR (n:EvidenceType)  REQUIRE n.code IS UNIQUE;
CREATE CONSTRAINT concept_key     IF NOT EXISTS FOR (n:Concept)       REQUIRE n.concept_key IS UNIQUE;
// Term is reserved for the dictionary layer (concept layer, Phase C4 of roadmap).
CREATE CONSTRAINT term_name       IF NOT EXISTS FOR (n:Term)          REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT source_doc_key  IF NOT EXISTS FOR (n:SourceDocument) REQUIRE n.doc_hash IS UNIQUE;
CREATE CONSTRAINT ingest_run_id   IF NOT EXISTS FOR (n:IngestRun)     REQUIRE n.run_id IS UNIQUE;
CREATE CONSTRAINT migration_ver   IF NOT EXISTS FOR (n:Migration)     REQUIRE n.version IS UNIQUE;

// ---------- lookup indexes ----------
CREATE INDEX case_citation   IF NOT EXISTS FOR (n:Case)      ON (n.citation);
CREATE INDEX case_year       IF NOT EXISTS FOR (n:Case)      ON (n.year);
CREATE INDEX case_court      IF NOT EXISTS FOR (n:Case)      ON (n.court_code);
CREATE INDEX case_in_corpus  IF NOT EXISTS FOR (n:Case)      ON (n.in_corpus);
CREATE INDEX para_case       IF NOT EXISTS FOR (n:Paragraph) ON (n.case_key, n.para_no);
CREATE INDEX provision_act   IF NOT EXISTS FOR (n:Provision) ON (n.statute_key, n.section);
CREATE INDEX principle_kind  IF NOT EXISTS FOR (n:Principle) ON (n.kind);

// ---------- full-text (keyword entry points, including aliases) ----------
// aliases_text is a space-joined copy of aliases[]; see models.py for the convention.
CREATE FULLTEXT INDEX case_names_ft      IF NOT EXISTS FOR (n:Case)      ON EACH [n.name, n.aliases_text];
CREATE FULLTEXT INDEX paragraph_text_ft  IF NOT EXISTS FOR (n:Paragraph) ON EACH [n.text];
CREATE FULLTEXT INDEX principle_text_ft  IF NOT EXISTS FOR (n:Principle) ON EACH [n.text];
CREATE FULLTEXT INDEX issue_text_ft      IF NOT EXISTS FOR (n:Issue)     ON EACH [n.question];

// ---------- vector index (768-dim, e5-base-v2, cosine) ----------
// Created now so the option exists when embeddings are added.
// OPEN QUESTION: whether paragraph embeddings live in Neo4j, in Chroma, or both
// is an open architecture decision — see backend/graph/README.md → Roadmap.
CREATE VECTOR INDEX paragraph_embedding IF NOT EXISTS
FOR (n:Paragraph) ON (n.embedding)
OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } };
