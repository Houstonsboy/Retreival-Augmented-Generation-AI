"""Deterministic (regex-only) extraction from parsed paragraphs.

Extracts citations, provisions, outcomes, awards, and cheap header edges.
Never calls an LLM.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NamedTuple

from rapidfuzz import fuzz

from .models import CitationRef, DeterministicFindings, ParsedDocument, ProvisionRef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Citation regexes (B3)
# ---------------------------------------------------------------------------

# [YYYY] COURT NUMBER (KLR) — primary Kenya citation
_BRACKET_COURT_RE = re.compile(
    r"\[(?P<year>\d{4})\]\s*"
    r"(?P<court>KEHC|KECA|KESC|KEELC|KEELRC|SCC|KEMC|EACA|UGHC)\s*"
    r"(?P<number>\d+)"
    r"(?:\s*\([^)]*\))*",
    re.IGNORECASE,
)
# [YYYY] eKLR
_EKLR_RE = re.compile(r"\[(?P<year>\d{4})\]\s*eKLR", re.IGNORECASE)
# (YYYY) n? EA n  or  [YYYY] EA n
_EA_RE = re.compile(
    r"[\[\(](?P<year>\d{4})[\]\)]\s*(?:\d+\s+)?EA\s+(?P<number>\d+)",
    re.IGNORECASE,
)
# (YYYY) KLR n
_KLR_RE = re.compile(r"\((?P<year>\d{4})\)\s*(?P<number>\d+)\s*KLR\b", re.IGNORECASE)
# English reports: AC, QB, WLR, All ER
_ENG_RE = re.compile(
    r"\[(?P<year>\d{4})\]\s*(?:\d+\s+)?(?P<rep>AC|QB|WLR|All\s+ER|EWCA|UKHL|UKSC)\s+(?P<number>\d+)",
    re.IGNORECASE,
)
# Civil Appeal No. N of YYYY (unreported)
_UNREPORTED_RE = re.compile(
    r"Civil\s+Appeal\s+No\.?\s*(?P<number>\d+)\s+of\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
# Case name pattern: "X v/vs/versus Y"
_CASE_NAME_RE = re.compile(
    r"(?P<name>[A-Z][A-Za-z\s&().,'\-]{3,60}?)"
    r"\s+(?:v\.?|vs\.?|versus)\s+"
    r"(?P<respondent>[A-Z][A-Za-z\s&().,'\-]{2,50}?)"
    r"(?=\s*[\[\(,\.]|\s+\d{4}|\s*$)",
)


def _extract_preceding_name(text: str, citation_start: int) -> str:
    """Extract the case name immediately before a citation in the text."""
    preceding = text[max(0, citation_start - 150):citation_start]
    m = _CASE_NAME_RE.search(preceding)
    if m:
        return f"{m.group('name').strip()} v {m.group('respondent').strip()}"
    return ""


def extract_citations(para_key: str, text: str) -> list[CitationRef]:
    """Extract all citation references from a paragraph."""
    refs: list[CitationRef] = []
    seen_ids: set[str] = set()

    def add(citation_id: str, raw_text: str, case_name: str, year: int | None) -> None:
        if citation_id not in seen_ids:
            seen_ids.add(citation_id)
            refs.append(CitationRef(
                citation_id=citation_id,
                raw_text=raw_text,
                case_name=case_name,
                year=year,
                para_key=para_key,
            ))

    for m in _BRACKET_COURT_RE.finditer(text):
        court = m.group("court").upper()
        year = int(m.group("year"))
        number = m.group("number")
        cid = f"{court}-{year}-{number}"
        name = _extract_preceding_name(text, m.start())
        add(cid, m.group(0), name, year)

    for m in _EKLR_RE.finditer(text):
        year = int(m.group("year"))
        name = _extract_preceding_name(text, m.start())
        slug = re.sub(r"[^\w]", "-", name.lower())[:40] if name else "unknown"
        cid = f"eKLR-{year}-{slug}"
        add(cid, m.group(0), name, year)

    for m in _EA_RE.finditer(text):
        year = int(m.group("year"))
        number = m.group("number")
        name = _extract_preceding_name(text, m.start())
        cid = f"EA-{year}-{number}"
        add(cid, m.group(0), name, year)

    for m in _KLR_RE.finditer(text):
        year = int(m.group("year"))
        number = m.group("number")
        name = _extract_preceding_name(text, m.start())
        cid = f"KLR-{year}-{number}"
        add(cid, m.group(0), name, year)

    for m in _ENG_RE.finditer(text):
        year = int(m.group("year"))
        rep = re.sub(r"\s+", "", m.group("rep")).upper()
        number = m.group("number")
        name = _extract_preceding_name(text, m.start())
        cid = f"{rep}-{year}-{number}"
        add(cid, m.group(0), name, year)

    for m in _UNREPORTED_RE.finditer(text):
        year = int(m.group("year"))
        number = m.group("number")
        name = _extract_preceding_name(text, m.start())
        cid = f"UNREPORTED-CA-{year}-{number}"
        add(cid, m.group(0), name, year)

    return refs


# ---------------------------------------------------------------------------
# Provision regexes (B3)
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"[Ss]ections?\s*(?P<sec>\d+[A-Z]?)"
    r"(?P<subs>(?:\(\d+\)|\([a-z]\)|\([ivxl]+\))*)"
    r"(?:\s+of\s+(?:the\s+)?(?P<act>[A-Z][A-Za-z\s&]+?(?:Act|Rules|Regulations|Code|Order)))?",
    re.IGNORECASE,
)
_SECTION_SHORT_RE = re.compile(
    r"\bS\.?\s*(?P<sec>\d+[A-Z]?)"
    r"(?P<subs>(?:\(\d+\)|\([a-z]\))*)"
    r"(?:\s+of\s+the\s+(?P<act>[A-Z][A-Za-z\s&]+?(?:Act|Rules)))?",
)
_ORDER_RULE_RE = re.compile(
    r"Order\s+(?P<order>[IVXL\d]+)\s+[Rr]ule\s+(?P<rule>\d+)(?P<sub>\(\w\))?",
    re.IGNORECASE,
)
_RULE_OF_RE = re.compile(
    r"[Rr]ule\s+(?P<rule>\d+)(?P<sub>\(\d+\)|\([a-z]\))?(?:\s+of\s+(?:the\s+)?(?P<rules>[A-Z][A-Za-z\s]+?(?:Rules)))",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(
    r"[Aa]rticle\s+(?P<art>\d+)"
    r"(?P<subs>(?:\(\d+\)|\([a-z]\))*)"
    r"(?:\s+of\s+(?:the\s+)?(?P<doc>Constitution|[A-Z][A-Za-z\s]+?))?",
    re.IGNORECASE,
)

_ACT_ALIASES: dict[str, str] = {
    "evidence act": "evidence-act",
    "the evidence act": "evidence-act",
    "civil procedure act": "civil-procedure-act",
    "civil procedure rules": "civil-procedure-rules",
    "small claims courts act": "scc-act",
    "small claims court act": "scc-act",
    "constitution of kenya": "constitution",
    "constitution": "constitution",
    "environment and land court act": "elc-act",
    "arbitration act": "arbitration-act",
    "judicature act": "judicature-act",
    "computer misuse and cybercrimes act": "cmca",
}

_DEFAULT_ACT_STACK: list[str] = []


def _resolve_act(act_text: str | None, context_act: str | None) -> str:
    if act_text:
        lower = act_text.lower().strip()
        for alias, slug in _ACT_ALIASES.items():
            if alias in lower or lower in alias:
                return slug
        return re.sub(r"\s+", "-", lower)
    if context_act:
        return context_act
    return "unknown-act"


def _normalise_subs(subs: str) -> str:
    return subs.replace("(", "/").replace(")", "").strip("/")


def extract_provisions(para_key: str, text: str) -> list[ProvisionRef]:
    refs: list[ProvisionRef] = []
    seen_ids: set[str] = set()
    last_act: str | None = None

    def add(statute: str, section: str) -> None:
        provision_id = f"{statute}/{section}"
        if provision_id not in seen_ids:
            seen_ids.add(provision_id)
            refs.append(ProvisionRef(
                provision_id=provision_id,
                statute=statute,
                section=section,
                para_key=para_key,
            ))

    for m in _SECTION_RE.finditer(text):
        sec = m.group("sec")
        subs = _normalise_subs(m.group("subs") or "")
        act = _resolve_act(m.group("act"), last_act)
        if m.group("act"):
            last_act = act
        section = f"s{sec}" + (f"/{subs}" if subs else "")
        add(act, section)

    for m in _SECTION_SHORT_RE.finditer(text):
        sec = m.group("sec")
        subs = _normalise_subs(m.group("subs") or "")
        act = _resolve_act(m.group("act"), last_act)
        if m.group("act"):
            last_act = act
        section = f"s{sec}" + (f"/{subs}" if subs else "")
        add(act, section)

    for m in _ORDER_RULE_RE.finditer(text):
        order = m.group("order")
        rule = m.group("rule")
        sub = (m.group("sub") or "").strip("()")
        section = f"Order{order}/rule{rule}" + (f"/{sub}" if sub else "")
        add("civil-procedure-rules", section)

    for m in _RULE_OF_RE.finditer(text):
        rule = m.group("rule")
        sub_raw = (m.group("sub") or "").strip("()")
        rules_act = _resolve_act(m.group("rules"), None)
        section = f"rule{rule}" + (f"/{sub_raw}" if sub_raw else "")
        add(rules_act, section)

    for m in _ARTICLE_RE.finditer(text):
        art = m.group("art")
        subs = _normalise_subs(m.group("subs") or "")
        doc_raw = m.group("doc") or "Constitution"
        act = _resolve_act(doc_raw, None)
        section = f"Art{art}" + (f"/{subs}" if subs else "")
        add(act, section)

    return refs


# ---------------------------------------------------------------------------
# Outcome and award extraction
# ---------------------------------------------------------------------------

_OUTCOME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bsuit\s+is\s+dismissed\b", re.IGNORECASE), "DISMISSED"),
    (re.compile(r"\bcase\s+is\s+dismissed\b", re.IGNORECASE), "DISMISSED"),
    (re.compile(r"\bappeal\s+is\s+dismissed\b", re.IGNORECASE), "DISMISSED"),
    (re.compile(r"\bapplication\s+is\s+dismissed\b", re.IGNORECASE), "DISMISSED"),
    (re.compile(r"\bsuit\s+is\s+struck\s+out\b", re.IGNORECASE), "STRUCK_OUT"),
    (re.compile(r"\bcase\s+struck\s+out\b", re.IGNORECASE), "STRUCK_OUT"),
    (re.compile(r"\bappeal\s+(?:is\s+)?allowed\b", re.IGNORECASE), "ALLOWED"),
    (re.compile(r"\bapplication\s+(?:is\s+)?allowed\b", re.IGNORECASE), "ALLOWED"),
    (re.compile(r"\bpartly\s+allowed\b", re.IGNORECASE), "PARTLY_ALLOWED"),
    (re.compile(r"\bset\s+aside\b.*\bgranted\b", re.IGNORECASE), "SET_ASIDE_GRANTED"),
    (re.compile(r"\bset\s+aside\b.*\brefused\b", re.IGNORECASE), "SET_ASIDE_REFUSED"),
    (re.compile(r"\brefused?\b", re.IGNORECASE), "REFUSED"),
    (re.compile(r"\bgranted\b", re.IGNORECASE), "GRANTED"),
    (re.compile(r"\bobjection\s+upheld\b", re.IGNORECASE), "OBJECTION_UPHELD"),
    (re.compile(r"\bobjection\s+dismissed\b|\bobjection\s+overruled\b", re.IGNORECASE), "OBJECTION_DISMISSED"),
]

_AWARD_AMOUNT_RE = re.compile(
    r"(?:costs?\s+(?:assessed|taxed|awarded|of|fixed)\s+at\s+)?Kshs?\.?\s*(?P<amount>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_OWN_COSTS_RE = re.compile(r"each\s+party\s+(?:to\s+)?bear\s+(?:their\s+own|own)\s+costs?", re.IGNORECASE)


def extract_outcome(paragraphs_text: str) -> tuple[str, str, bool]:
    """Return (outcome_raw, outcome_code, unresolved)."""
    orders_re = re.compile(
        r"(?:ORDERS?|JUDGMENT|RULING)[:\s]*\n?((?:.|\n){0,1000}?)(?:\n\n|\Z)",
        re.IGNORECASE,
    )
    orders_text = ""
    m = orders_re.search(paragraphs_text[-3000:])
    if m:
        orders_text = m.group(0)

    search_text = orders_text or paragraphs_text[-2000:]
    for pat, code in _OUTCOME_PATTERNS:
        if pat.search(search_text):
            return search_text[:200], code, False

    return search_text[:200], "", True


def extract_awards(text: str) -> list[str]:
    awards = []
    if _OWN_COSTS_RE.search(text):
        awards.append("costs:own")
    for m in _AWARD_AMOUNT_RE.finditer(text):
        raw = m.group("amount").replace(",", "")
        awards.append(f"kshs:{raw}")
    return awards


# ---------------------------------------------------------------------------
# Entity resolution / deduplication (B3)
# ---------------------------------------------------------------------------

class _CaseRecord:
    def __init__(self, cid: str, name: str, year: int | None):
        self.citation_id = cid
        self.name = name
        self.year = year
        self.aliases: list[str] = []


_KNOWN_MERGES: list[tuple[set[str], str]] = [
    (
        {"mataguda", "matuguda", "matriguda", "berisa", "barisa"},
        "KEHC-2011-1481",
    ),
    (
        {"giulla", "giella", "cassman-brown"},
        "eKLR-1973-giella-v-cassman-brown",
    ),
    (
        {"civil-case-490-of-2013", "nonny"},
        "KEHC-2014-6468",
    ),
]


def _normalise_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+v\.?\s+", "-v-", text)
    text = re.sub(r"[^\w\-]", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def resolve_citation_ids(
    refs: list[CitationRef],
    corpus_case_keys: set[str],
) -> list[CitationRef]:
    """
    Merge duplicate citations and apply known alias mappings.
    Returns the deduplicated, resolved list.
    """
    resolved: list[CitationRef] = []
    id_map: dict[str, str] = {}

    for ref in refs:
        cid = ref.citation_id
        name_norm = _normalise_for_match(ref.case_name)

        # Apply hard-coded merges
        for name_tokens, target_key in _KNOWN_MERGES:
            if any(tok in name_norm or tok in cid.lower() for tok in name_tokens):
                logger.info(
                    "Merge %s → %s (name_norm=%s)",
                    cid, target_key, name_norm,
                )
                id_map[cid] = target_key
                cid = target_key
                break

        # Corpus case-key override: if deterministic id matches a known key, use it
        if cid in corpus_case_keys:
            pass  # already exact
        else:
            # Fuzzy match by name against corpus keys if year matches
            best_score = 0
            best_key = cid
            for ck in corpus_case_keys:
                ck_name = _normalise_for_match(ck)
                score = fuzz.token_set_ratio(name_norm, ck_name)
                year_part = str(ref.year) if ref.year else ""
                if score >= 90 and year_part and year_part in ck:
                    if score > best_score:
                        best_score = score
                        best_key = ck
            if best_key != cid:
                logger.info(
                    "Fuzzy merge %s → %s (score=%d, name=%s)",
                    cid, best_key, best_score, ref.case_name,
                )
                id_map[cid] = best_key
                cid = best_key

        resolved.append(CitationRef(
            citation_id=cid,
            raw_text=ref.raw_text,
            case_name=ref.case_name,
            year=ref.year,
            para_key=ref.para_key,
        ))

    return resolved


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def run_deterministic(doc: ParsedDocument) -> DeterministicFindings:
    """Run all deterministic extraction over a parsed document."""
    all_citations: list[CitationRef] = []
    all_provisions: list[ProvisionRef] = []
    full_text_parts: list[str] = []

    for para in doc.paragraphs:
        text = para.text
        full_text_parts.append(text)
        cits = extract_citations(para.para_key, text)
        provs = extract_provisions(para.para_key, text)
        all_citations.extend(cits)
        all_provisions.extend(provs)
        if cits:
            logger.debug(
                "%s: found %d citation(s), %d provision(s)",
                para.para_key, len(cits), len(provs),
            )

    full_text = "\n\n".join(full_text_parts)
    outcome_raw, outcome_code, unresolved = extract_outcome(full_text)
    awards = extract_awards(full_text)

    return DeterministicFindings(
        citations=all_citations,
        provisions=all_provisions,
        outcome_raw=outcome_raw,
        outcome_code=outcome_code,
        outcome_unresolved=unresolved,
        awards_raw=awards,
    )
