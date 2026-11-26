"""Canonical key builders for graph nodes.

All keys are deterministic so the same real-world entity always gets the same
graph node regardless of which pipeline run created it.
"""
from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace to hyphens."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text


def normalise_name(name: str) -> str:
    """Normalise a case or party name for alias matching.

    Lowercases, strips punctuation, collapses 'v'/'vs'/'versus' and common
    spelling variants so Matuguda/Matriguda/Mataguda all resolve to the same
    normal form.
    """
    text = unicodedata.normalize("NFC", name).lower()
    text = re.sub(r"\bversus\b|\bvs\.?\b", "v", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Court code extraction
# ---------------------------------------------------------------------------

# Abbreviated court codes that appear inside citation brackets.
_COURT_ABBREVS = {
    "KEHC": "KEHC",   # High Court of Kenya
    "KECA": "KECA",   # Court of Appeal of Kenya
    "KESC": "KESC",   # Supreme Court of Kenya
    "KEMC": "KEMC",   # Magistrate Court of Kenya
    "KESC": "KESC",
}

_CITATION_RE = re.compile(
    r"(?<!\w)\[?(?P<year>\d{4})\]?\s*(?P<court>KE[A-Z]{2,5}|SCC)"
    r"\s*(?P<number>\d+)(?!\w)", re.IGNORECASE,
)
_EKLR_RE = re.compile(r"\[(?P<year>\d{4})\]\s*eKLR", re.IGNORECASE)


def case_key(citation: str, name: str = "") -> str:
    """Return a canonical case key from a citation string.

    Examples:
        "[2024] KEHC 15763 (KLR)" -> "KEHC-2024-15763"
        "[2011] eKLR"              -> "eKLR-2011-<slug-of-name>"
        "[2019] KECA 42 (KLR)"    -> "KECA-2019-42"
    """
    citation = citation.strip()
    m = _CITATION_RE.search(citation)
    if m:
        court = m.group("court").upper()
        year = m.group("year")
        number = m.group("number")
        return f"{court}-{year}-{number}"

    # eKLR citations have no case number
    m2 = _EKLR_RE.search(citation)
    if m2:
        year = m2.group("year")
        name_slug = _slug(name) if name else "unknown"
        return f"eKLR-{year}-{name_slug}"

    # Fallback: year + slug of citation text
    year_m = re.search(r"\b(\d{4})\b", citation)
    year = year_m.group(1) if year_m else "0000"
    return f"UNKNOWN-{year}-{_slug(citation) or _slug(name) or 'case'}"


def provision_key(statute: str, section: str) -> str:
    """Return a canonical provision key.

    Example: provision_key("Evidence Act", "s.106B(4)(d)") -> "evidence-act/s106B/4/d"
    """
    act_slug = _slug(statute)
    # Normalise "s.106B(4)(d)" -> "s106B/4/d"
    sec = section.strip()
    sec = re.sub(r"^s\.", "s", sec)                 # s.106B -> s106B
    sec = re.sub(r"\(([^)]+)\)", r"/\1", sec)       # (4)(d) -> /4/d
    sec = re.sub(r"[^a-zA-Z0-9/]", "", sec)         # strip remaining punctuation
    return f"{act_slug}/{sec}"


def para_key(ck: str, para_no: int) -> str:
    """Return a canonical paragraph key.

    Example: para_key("KEHC-2024-15763", 44) -> "KEHC-2024-15763#p44"
    """
    return f"{ck}#p{para_no}"


def judge_key(name: str) -> str:
    """Return a canonical judge key."""
    return f"judge-{_slug(name)}"


def court_code(name: str) -> str:
    """Return a canonical court code from a court name."""
    upper = name.strip().upper()
    for abbrev in _COURT_ABBREVS:
        if abbrev in upper:
            return _COURT_ABBREVS[abbrev]
    return _slug(name).upper()
