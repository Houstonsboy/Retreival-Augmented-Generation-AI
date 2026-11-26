"""Pydantic models for the case-law ingest pipeline."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal
from pydantic import BaseModel, Field


class CaseMeta(BaseModel):
    case_key: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    citation: str = ""
    court_code: str = ""
    court_level: str = ""
    station: str = ""
    case_number: str = ""
    decision_date: date | None = None
    decision_type: Literal["Judgment", "Ruling", "Order", "Unknown"] = "Unknown"
    judges: list[str] = Field(default_factory=list)
    parties: str = ""
    appeal_from: str = ""
    source_file: str = ""
    doc_hash: str = ""


class QuoteSpan(BaseModel):
    start: int
    end: int
    attributed_to_raw: str = ""


class Paragraph(BaseModel):
    para_key: str
    case_key: str
    para_no: int
    para_sub: str = ""
    para_numbering: Literal["natural", "synthetic"] = "natural"
    text: str
    opinion_author: str = ""
    opinion_kind: Literal["lead", "concurring", "dissent", "sole", "unknown"] = "unknown"
    domains: list[str] = Field(default_factory=list)
    has_quote: bool = False
    quote_spans: list[QuoteSpan] = Field(default_factory=list)
    quote_sources: str = ""
    page_start: int = 0
    page_end: int = 0
    text_repaired: bool = False
    firac: str = ""


class ParsedDocument(BaseModel):
    meta: CaseMeta
    paragraphs: list[Paragraph]
    ingest_run_id: str = ""


class CitationRef(BaseModel):
    citation_id: str
    raw_text: str
    case_name: str = ""
    year: int | None = None
    para_key: str = ""


class ProvisionRef(BaseModel):
    provision_id: str
    statute: str = ""
    section: str = ""
    para_key: str = ""


class DeterministicFindings(BaseModel):
    citations: list[CitationRef] = Field(default_factory=list)
    provisions: list[ProvisionRef] = Field(default_factory=list)
    outcome_raw: str = ""
    outcome_code: str = ""
    outcome_unresolved: bool = False
    awards_raw: list[str] = Field(default_factory=list)
