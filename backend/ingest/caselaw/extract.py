"""Text extraction and repair for SCC case-law judgments.

Handles PDF (via PyMuPDF) and DOCX (via python-docx).
Outputs a ParsedDocument with repaired paragraphs.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import date
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ligature repair
# ---------------------------------------------------------------------------

# Word-boundary replacements applied before any other processing.
# Format: (bad_form, corrected_form, context_pattern_or_None)
# When context_pattern is set, the replacement only fires inside that match.
_LIGATURE_SIMPLE = [
    (r"\bcerticate\b",      "certificate"),
    (r"\bnding\b",          "finding"),
    (r"\bsatised\b",        "satisfied"),
    (r"\boending\b",        "offending"),
    (r"\bjustication\b",    "justification"),
    (r"\badavit\b",         "affidavit"),
    (r"\bplainti\b",        "plaintiff"),
    (r"\bdierent\b",        "different"),
    (r"\baected\b",         "affected"),
    (r"\bsuered\b",         "suffered"),
    (r"\becient\b",         "efficient"),
    (r"\bdenes\b",          "defines"),
    (r"\bconrm\b",          "confirm"),
    (r"\bspecic\b",         "specific"),
    (r"\bsignicant\b",      "significant"),
    (r"\bsucient\b",        "sufficient"),
    (r"\beect\b",           "effect"),
    (r"\boce\b",            "office"),
    (r"\bofce\b",           "office"),
    (r"\brst\b",            "first"),
    (r"\bnal\b",            "final"),
    (r"\bnancial\b",        "financial"),
    (r"\bprot\b",           "profit"),
    (r"\bcondential\b",     "confidential"),
    (r"\bfactual\b",        "factual"),   # keep — real word; here as no-op sentinel
]

# "led" is a real word — only replace in legal filing contexts.
_LED_PATTERN = re.compile(
    r"\bled\s+(?=a\b|on\b|in\s+court|submissions|by\s+way|an?\s+application|"
    r"an?\s+notice|an?\s+affidavit|documents?|pleadings?)",
    re.IGNORECASE,
)

_COMPILED_SIMPLE = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in _LIGATURE_SIMPLE]


def repair_ligatures(text: str, log_prefix: str = "") -> tuple[str, int]:
    """Apply ligature repairs; return (repaired_text, replacement_count)."""
    count = 0

    def _sub(m: re.Match, rep: str) -> str:
        nonlocal count
        count += 1
        return rep

    for pattern, replacement in _COMPILED_SIMPLE:
        text = pattern.sub(lambda m, r=replacement: _sub(m, r), text)

    def _led_sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        return "filed " + m.group(0)[len("led "):]

    text = _LED_PATTERN.sub(_led_sub, text)
    if count and log_prefix:
        logger.debug("%s: %d ligature replacements", log_prefix, count)
    return text, count


# ---------------------------------------------------------------------------
# Page furniture stripping
# ---------------------------------------------------------------------------

_FURNITURE_PATTERNS = [
    re.compile(r"https?://\S+kenyalaw\S*", re.IGNORECASE),
    re.compile(r"Judgment SCCCOMM No[^\n]+?pg\.\s*\d+", re.IGNORECASE),
    re.compile(r"Hon\s+[A-Z]\.[A-Z]\s+\w+-[A-Z]\.[A-Z]\s+pg\.\s*\d+", re.IGNORECASE),
    re.compile(r"(?m)^\s*\d{1,3}\s*$"),
    re.compile(r"I certify that this is a true copy[^\n]*", re.IGNORECASE),
    re.compile(r"(?m)^Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"(?m)^Signed:?[^\n]{0,60}$"),
]


def strip_furniture(text: str) -> str:
    for pat in _FURNITURE_PATTERNS:
        text = pat.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Unknown-token check
# ---------------------------------------------------------------------------

def _load_wordlist() -> set[str]:
    words: set[str] = set()
    legal_extras = {
        "affidavit", "plaintiff", "defendant", "claimant", "respondent",
        "appellant", "petitioner", "injunction", "subrogation", "arbitration",
        "certiorari", "mandamus", "habeas", "corpus", "firac", "statute",
        "tribunal", "deponent", "pleading", "interlocutory", "contemnor",
        "judicature", "adjudication", "litigant", "overruled", "obiter",
        "dictum", "ratio", "decidendi", "amicus", "curiae", "ipso", "facto",
        "prima", "facie", "locus", "standi", "res", "judicata", "estoppel",
        "subrogate", "mpesa", "ksh", "kshs", "nairobi", "mombasa", "kisumu",
        "kenya", "kenyan", "eklr", "kehc", "keca", "kesc", "keelc", "kemc",
        "unfiled", "refiles", "malpractice", "overruling", "para",
        "defraud", "unfair", "repay", "repayment", "cctv", "dvd", "sms",
        "whatsapp", "upto", "electro", "telecom", "smartphone",
        # Accept Kenyan proper names as unknown (they will be flagged but not cause test failures)
    }
    words.update(legal_extras)
    wordlist_path = Path("/usr/share/dict/words")
    if wordlist_path.exists():
        with wordlist_path.open() as f:
            for line in f:
                words.add(line.strip().lower())
    return words


_WORDLIST: set[str] | None = None


def check_unknown_tokens(text: str, threshold: float = 0.005) -> list[str]:
    """Return top-30 unknown tokens if unknown fraction > threshold."""
    global _WORDLIST
    if _WORDLIST is None:
        _WORDLIST = _load_wordlist()
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    if not tokens:
        return []
    unknown = [t for t in tokens if t.lower() not in _WORDLIST]
    fraction = len(unknown) / len(tokens)
    if fraction > threshold:
        from collections import Counter
        top = [t for t, _ in Counter(unknown).most_common(30)]
        logger.warning(
            "High unknown-token rate %.1f%% (%d/%d) — top-30: %s",
            fraction * 100, len(unknown), len(tokens), top,
        )
        return top
    return []


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

_BROKEN_FI_PAIRS = re.compile(
    r"\b(certi|satis|justi|quali|recti|noti|clari|rati|veri|magni|modi|certi|testi|grati|arti|bene"
    r"|identi|spec|signi|ef|diver|horri|magni|rec)\s+(cate|cates|cated|cating|faction|fied|fier"
    r"|fication|fications|fy|fying|fe|fer|fers|fice|fices|ft|fth|nal|nds|nding|ndings"
    r"|le|led|les|ling|nd|nds|cient|ciency|ciencies|cent|cult|cult)\b",
    re.IGNORECASE,
)


def _rejoin_fi_fragments(text: str) -> str:
    """Rejoin words broken at the fi-ligature position by Kenya Law PDFs."""
    return _BROKEN_FI_PAIRS.sub(lambda m: m.group(1) + "fi" + m.group(2), text)


def extract_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_no, line_text), ...] preserving page numbers."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    doc = fitz.open(str(path))
    from legal_text import pdf_ligature_map

    lines: list[tuple[int, str]] = []
    glyphs = pdf_ligature_map("\n".join(page.get_text() for page in doc))
    for page in doc:
        page_text = page.get_text("text").translate(glyphs)
        # Rejoin fi-ligature fragments before line splitting
        page_text = _rejoin_fi_fragments(page_text)
        for line in page_text.splitlines():
            lines.append((page.number + 1, line))
    doc.close()
    return lines


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

def extract_docx(path: Path) -> list[tuple[int, str]]:
    """Return [(0, line_text), ...] — DOCX has no page numbers."""
    from docx import Document

    doc = Document(str(path))
    lines: list[tuple[int, str]] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines.append((0, text))
    return lines


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

_SCC_CASE_NUMBER_RE = re.compile(
    r"(?P<year>\d{4})(?P<court>KEHC|KECA|KESC|KEELC|KEELRC|SCC|KEMC)"
    r"(?P<number>\d+)\(?KLR\)?",
    re.IGNORECASE,
)
_BRACKET_CITATION_RE = re.compile(
    r"\[(?P<year>\d{4})\]\s*(?P<court>KEHC|KECA|KESC|KEELC|KEELRC|SCC|KEMC|EACA)\s*(?P<number>\d+)",
    re.IGNORECASE,
)
_EKLR_RE = re.compile(r"\[(?P<year>\d{4})\]\s*eKLR", re.IGNORECASE)

_DATE_RE = re.compile(
    r"(?P<day>\d{1,2})\s+(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

_COURT_LEVEL_MAP = {
    "KEHC": ("HC", "High Court of Kenya"),
    "KECA": ("CoA", "Court of Appeal of Kenya"),
    "KESC": ("SC", "Supreme Court of Kenya"),
    "KEELC": ("ELC", "Environment and Land Court"),
    "KEELRC": ("ELRC", "Employment and Labour Relations Court"),
    "SCC": ("SCC", "Small Claims Court"),
    "KEMC": ("MC", "Magistrate Court of Kenya"),
}

_JUDGE_RE = re.compile(
    r"(?:BEFORE|CORAM)\s*[:\-]?\s*(?:HON\.?\s*)?(?:JUSTICE\s*)?([A-Z][A-Z\s\.\-,]+?)(?:\s*(?:J\.A\.|J\.|JA\b|JUDGE|MAGISTRATE))",
    re.IGNORECASE,
)
_JUDGE_SPLIT_RE = re.compile(r"(?:,|\band\b|&)", re.IGNORECASE)

_STATION_RE = re.compile(
    r"(?:AT|IN)\s+([A-Z][A-Z\s]+?)(?:\s*REGISTRY|\s*$)",
    re.IGNORECASE,
)
_DECISION_TYPE_RE = re.compile(r"\b(JUDGMENT|RULING|ORDER)\b", re.IGNORECASE)


def _parse_citation_from_filename(filename: str) -> tuple[str, str, int | None]:
    """Return (court_code, case_number, year) from filename."""
    m = _SCC_CASE_NUMBER_RE.search(filename)
    if m:
        court = m.group("court").upper()
        return court, m.group("number"), int(m.group("year"))
    return "", "", None


def _build_case_key_from_filename(filename: str) -> str:
    """Build case_key from filename citation pattern."""
    m = _SCC_CASE_NUMBER_RE.search(filename)
    if m:
        return f"{m.group('court').upper()}-{m.group('year')}-{m.group('number')}"
    return ""


def parse_decision_date(text: str) -> date | None:
    m = _DATE_RE.search(text)
    if not m:
        return None
    try:
        return date(int(m.group("year")), _MONTH_MAP[m.group("month").lower()], int(m.group("day")))
    except (ValueError, KeyError):
        return None


def parse_judges(header_text: str) -> list[str]:
    judges: list[str] = []
    for m in _JUDGE_RE.finditer(header_text):
        raw = m.group(1).strip()
        for part in _JUDGE_SPLIT_RE.split(raw):
            name = re.sub(r"\s+", " ", part).strip().rstrip(",")
            if name and len(name) > 2:
                judges.append(name)
    return judges or []


def parse_station(header_text: str) -> str:
    m = _STATION_RE.search(header_text)
    if m:
        return m.group(1).strip().title()
    return ""


def parse_decision_type(text: str) -> str:
    m = _DECISION_TYPE_RE.search(text)
    if m:
        word = m.group(1).capitalize()
        if word in ("Judgment", "Ruling", "Order"):
            return word
    return "Unknown"


# ---------------------------------------------------------------------------
# Header block extraction (first N lines)
# ---------------------------------------------------------------------------

def extract_header_block(lines: list[tuple[int, str]], max_lines: int = 60) -> str:
    """Return first max_lines as a single string for header parsing."""
    return "\n".join(line for _, line in lines[:max_lines])


# ---------------------------------------------------------------------------
# Paragraph segmentation
# ---------------------------------------------------------------------------

_NUMBERED_PARA_RE = re.compile(r"^\s*(\d{1,3})\.\s+\S")
_OPINION_HEADING_RE = re.compile(
    r"^\s*(JUDGMENT|RULING)\s+OF\s+([A-Z][A-Z\s\.\-]+?)(?:,\s*J\.?A?\.?)?$",
    re.IGNORECASE,
)
_LONG_PARA_MAX = 2500

SegmentedPara = NamedTuple("SegmentedPara", [
    ("para_no", int),
    ("para_sub", str),
    ("numbering", str),
    ("opinion_author", str),
    ("opinion_kind", str),
    ("text", str),
    ("page_start", int),
    ("page_end", int),
])


def _split_long_paragraph(
    para_no: int, sub: str, numbering: str,
    opinion_author: str, opinion_kind: str,
    text: str, page_start: int, page_end: int,
) -> list[SegmentedPara]:
    if len(text) <= _LONG_PARA_MAX:
        return [SegmentedPara(para_no, sub, numbering, opinion_author, opinion_kind, text, page_start, page_end)]

    parts: list[SegmentedPara] = []
    suffix_ord = ord("a")
    remaining = text
    sub_idx = 0
    while remaining:
        if len(remaining) <= _LONG_PARA_MAX:
            suffix = chr(suffix_ord + sub_idx) if sub_idx > 0 else sub
            parts.append(SegmentedPara(
                para_no, suffix or chr(suffix_ord + sub_idx),
                numbering, opinion_author, opinion_kind,
                remaining, page_start, page_end,
            ))
            break
        split_at = _LONG_PARA_MAX
        for sep in (". ", "! ", "? ", "\n"):
            idx = remaining.rfind(sep, 0, _LONG_PARA_MAX)
            if idx > _LONG_PARA_MAX // 2:
                split_at = idx + len(sep)
                break
        suffix = chr(suffix_ord + sub_idx)
        parts.append(SegmentedPara(
            para_no, suffix,
            numbering, opinion_author, opinion_kind,
            remaining[:split_at].strip(), page_start, page_end,
        ))
        remaining = remaining[split_at:].strip()
        sub_idx += 1
    return parts


def segment_paragraphs(
    lines: list[tuple[int, str]],
    case_key: str,
) -> list[SegmentedPara]:
    """Split lines into paragraphs; detect opinions; handle unnumbered judgments."""
    full_text = "\n".join(line for _, line in lines)
    has_numbered = bool(_NUMBERED_PARA_RE.search(full_text))

    if has_numbered:
        return _segment_numbered(lines)
    else:
        return _segment_unnumbered(lines)


def _segment_numbered(lines: list[tuple[int, str]]) -> list[SegmentedPara]:
    result: list[SegmentedPara] = []
    current_no = 0
    current_lines: list[str] = []
    current_page_start = lines[0][0] if lines else 0
    current_page_end = current_page_start
    current_opinion_author = ""
    current_opinion_kind = "unknown"

    def flush(para_no: int, page_start: int, page_end: int,
              opinion_author: str, opinion_kind: str) -> None:
        text = " ".join(" ".join(current_lines).split()).strip()
        if text:
            for seg in _split_long_paragraph(
                para_no, "", "natural", opinion_author, opinion_kind,
                text, page_start, page_end,
            ):
                result.append(seg)

    for page_no, line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect opinion heading
        om = _OPINION_HEADING_RE.match(stripped)
        if om:
            flush(current_no, current_page_start, current_page_end,
                  current_opinion_author, current_opinion_kind)
            current_opinion_author = om.group(2).strip()
            current_opinion_kind = "unknown"
            current_lines = []
            current_page_start = page_no
            current_page_end = page_no
            continue

        nm = _NUMBERED_PARA_RE.match(stripped)
        if nm:
            flush(current_no, current_page_start, current_page_end,
                  current_opinion_author, current_opinion_kind)
            current_no = int(nm.group(1))
            current_lines = [stripped]
            current_page_start = page_no
            current_page_end = page_no
        else:
            if not result and not current_lines:
                # Header text before para 1
                current_no = 0
            current_lines.append(stripped)
            current_page_end = page_no

    flush(current_no, current_page_start, current_page_end,
          current_opinion_author, current_opinion_kind)
    return result


def _segment_unnumbered(lines: list[tuple[int, str]]) -> list[SegmentedPara]:
    """Blank-line paragraph boundaries; assign synthetic sequential numbers."""
    result: list[SegmentedPara] = []
    current_lines: list[str] = []
    current_page_start = lines[0][0] if lines else 0
    current_page_end = current_page_start
    current_opinion_author = ""
    current_opinion_kind = "unknown"
    para_counter = 0

    def flush() -> None:
        nonlocal para_counter
        text = " ".join(current_lines).strip()
        if not text:
            return
        om = _OPINION_HEADING_RE.match(text)
        if om:
            return
        for seg in _split_long_paragraph(
            para_counter, "", "synthetic", current_opinion_author, current_opinion_kind,
            text, current_page_start, current_page_end,
        ):
            result.append(seg)
        para_counter += 1

    for page_no, line in lines:
        stripped = line.strip()
        om = _OPINION_HEADING_RE.match(stripped)
        if om:
            flush()
            current_opinion_author = om.group(2).strip()
            current_opinion_kind = "unknown"
            current_lines = []
            current_page_start = page_no
            current_page_end = page_no
            continue

        if not stripped:
            if current_lines:
                flush()
                current_lines = []
                current_page_start = page_no
                current_page_end = page_no
        else:
            current_lines.append(stripped)
            current_page_end = page_no

    if current_lines:
        flush()

    return result


# ---------------------------------------------------------------------------
# Quote-span detection
# ---------------------------------------------------------------------------

_QUOTE_INTRO_RE = re.compile(
    r"(?:stated|held|observed that|found|noted|remarked)[:\s]*[\"“]",
    re.IGNORECASE,
)
_QUOTE_BLOCK_RE = re.compile(
    r"[\"“](.{20,500}?)[\"”]",
    re.DOTALL,
)
_ATTRIBUTION_RE = re.compile(
    r"\b(?:in\s+)?([A-Z][a-z]+(?:\s+v\.?\s+[A-Z][a-z]+)?)\s*(?:\[|\(|\d{4})",
)


def detect_quote_spans(text: str) -> list[dict]:
    spans = []
    for m in _QUOTE_BLOCK_RE.finditer(text):
        start = m.start()
        # Try to find attribution from text just before the quote
        preceding = text[max(0, start - 200):start]
        attr_m = _ATTRIBUTION_RE.search(preceding)
        attributed_to_raw = attr_m.group(1) if attr_m else ""
        spans.append({
            "start": start,
            "end": m.end(),
            "attributed_to_raw": attributed_to_raw,
        })
    return spans


# ---------------------------------------------------------------------------
# Document hash
# ---------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Domain detection (simple keyword)
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Evidence Law",      ["106b", "certificate", "electronic evidence", "admissibility", "s.106"]),
    ("Defamation",        ["defamation", "libel", "slander", "reputation"]),
    ("Civil Procedure",   ["default judgment", "set aside", "order ix", "service"]),
    ("Arbitration",       ["arbitration", "arbitral award", "arb clause"]),
    ("Insurance",         ["insurer", "subrogation", "insurance policy", "indemnity"]),
    ("Employment Law",    ["employment", "labour", "unfair dismissal", "wrongful termination"]),
    ("Land Law",          ["land", "environment", "elc", "title deed"]),
    ("Intellectual Property", ["intellectual property", "trademark", "copyright", "passing off"]),
    ("Contract Law",      ["contract", "agreement", "breach", "offer", "consideration"]),
    ("Tort",              ["negligence", "tort", "duty of care", "damages"]),
]


def detect_domains(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(kw in text_lower for kw in keywords):
            found.append(domain)
    return found or ["General"]
