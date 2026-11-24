"""Pydantic models for graph nodes and edges.

Property names follow the Part 1 schema design.

Convention for aliases_text: a space-joined string copy of aliases[], kept in
sync with the list by the repository upsert. Neo4j full-text indexes require
string properties; the list form is used in application code.

Relationship types are documented as an enum below. Neo4j doesn't require them
to be declared; they serve as the authoritative list for the Python layer.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Relationship types (Part 1 design)
# ---------------------------------------------------------------------------

class RelType(str, Enum):
    # Document structure
    CONTAINS = "CONTAINS"
    HAS_PARAGRAPH = "HAS_PARAGRAPH"
    NEXT = "NEXT"
    HAS_OPINION = "HAS_OPINION"
    AUTHORED_BY = "AUTHORED_BY"
    IN_OPINION = "IN_OPINION"
    SUPPORTED_BY = "SUPPORTED_BY"
    QUOTES = "QUOTES"
    # Citation network
    CITES = "CITES"
    APPEAL_FROM = "APPEAL_FROM"
    # Court / jurisdiction
    DECIDED_IN = "DECIDED_IN"
    BINDS = "BINDS"
    DECIDED_BY = "DECIDED_BY"
    AT_STATION = "AT_STATION"
    # Holding / analysis
    HOLDS = "HOLDS"
    STATES = "STATES"
    STATES_OBITER = "STATES_OBITER"
    DECIDES = "DECIDES"
    ANSWERS = "ANSWERS"
    # Statute / provision
    INTERPRETS = "INTERPRETS"
    APPLIES = "APPLIES"
    MENTIONS = "MENTIONS"
    PART_OF = "PART_OF"
    MADE_UNDER = "MADE_UNDER"
    # Derivation / principle relations
    DERIVED_FROM = "DERIVED_FROM"
    QUALIFIES = "QUALIFIES"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    SUPPORTS = "SUPPORTS"
    BEARS_ON = "BEARS_ON"
    GROUNDED_IN = "GROUNDED_IN"
    # Argument / consideration
    ARGUED_BY_PARTY = "ARGUED_BY_PARTY"
    CONSIDERED = "CONSIDERED"
    # Evidence
    RULED_ON_EVIDENCE = "RULED_ON_EVIDENCE"
    INVOLVES = "INVOLVES"
    # SCC-specific
    CAUSE_OF_ACTION = "CAUSE_OF_ACTION"
    HAS_OUTCOME = "HAS_OUTCOME"
    AWARDED = "AWARDED"
    HAS_PARTY_TYPE = "HAS_PARTY_TYPE"
    # Seed reference
    EXCLUDED_BY = "EXCLUDED_BY"
    WITHIN = "WITHIN"


class CitationTreatment(str, Enum):
    FOLLOWED = "FOLLOWED"
    APPLIED = "APPLIED"
    CONSIDERED = "CONSIDERED"
    DISTINGUISHED = "DISTINGUISHED"
    DEPARTED_FROM = "DEPARTED_FROM"
    OVERRULED = "OVERRULED"
    QUOTED = "QUOTED"
    QUOTED_VIA = "QUOTED_VIA"
    CITED_BY_PARTY = "CITED_BY_PARTY"


# ---------------------------------------------------------------------------
# Provenance (mixed into every node)
# ---------------------------------------------------------------------------

class Provenance(BaseModel):
    source_doc_hash: str | None = None
    ingest_run_id: str | None = None
    # Format: "regex" | "llm:<model>@<prompt_version>" | "manual"
    extractor: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Node models
# ---------------------------------------------------------------------------

class CaseNode(BaseModel):
    case_key: str
    citation: str
    name: str
    year: int | None = None
    court_code: str | None = None
    # aliases[] is used in app code; aliases_text is kept in sync for full-text search.
    aliases: list[str] = Field(default_factory=list)
    aliases_text: str = ""
    in_corpus: bool = False
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("aliases_text", mode="before")
    @classmethod
    def _build_aliases_text(cls, v, info):
        aliases = (info.data or {}).get("aliases", [])
        if aliases and not v:
            return " ".join(aliases)
        return v or ""


class CourtNode(BaseModel):
    code: str
    name: str
    jurisdiction: str | None = None
    provenance: Provenance = Field(default_factory=Provenance)


class JudgeNode(BaseModel):
    judge_key: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


class OpinionNode(BaseModel):
    opinion_key: str
    case_key: str
    judge_key: str
    opinion_type: Literal["majority", "concurring", "dissenting", "sole"] = "sole"
    provenance: Provenance = Field(default_factory=Provenance)


class ParagraphNode(BaseModel):
    para_key: str
    case_key: str
    para_no: int
    text: str
    # embedding is populated by the embedder; None until then.
    # OPEN QUESTION: whether to store embeddings in Neo4j or Chroma — see README Roadmap.
    embedding: list[float] | None = None
    provenance: Provenance = Field(default_factory=Provenance)


class IssueNode(BaseModel):
    issue_key: str
    question: str
    provenance: Provenance = Field(default_factory=Provenance)


class PrincipleNode(BaseModel):
    principle_key: str
    text: str
    kind: str | None = None  # e.g. "ratio", "obiter", "rule"
    provenance: Provenance = Field(default_factory=Provenance)


class StatuteNode(BaseModel):
    statute_key: str
    name: str
    short_name: str | None = None
    provenance: Provenance = Field(default_factory=Provenance)


class ProvisionNode(BaseModel):
    provision_key: str
    statute_key: str
    section: str
    text: str | None = None
    provenance: Provenance = Field(default_factory=Provenance)


class EvidenceTypeNode(BaseModel):
    code: str
    label: str
    provenance: Provenance = Field(default_factory=Provenance)


class ConceptNode(BaseModel):
    concept_key: str
    label: str
    provenance: Provenance = Field(default_factory=Provenance)


class TermNode(BaseModel):
    # Reserved for the dictionary layer (concept layer, Phase C4 of roadmap).
    name: str
    definition: str | None = None
    page: int | None = None
    see: list[str] = Field(default_factory=list)
    cf: list[str] = Field(default_factory=list)
    also_termed: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


class SourceDocumentNode(BaseModel):
    doc_hash: str
    file_name: str
    file_type: str | None = None
    provenance: Provenance = Field(default_factory=Provenance)


class IngestRunNode(BaseModel):
    run_id: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: str = "pending"
    provenance: Provenance = Field(default_factory=Provenance)


# ---------------------------------------------------------------------------
# Edge model
# ---------------------------------------------------------------------------

class GraphEdge(BaseModel):
    type: RelType
    src_key: str
    dst_key: str
    # Additional properties (e.g. treatment, para, binding for CITES)
    props: dict[str, Any] = Field(default_factory=dict)
