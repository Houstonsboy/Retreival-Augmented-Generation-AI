"""
Post-generation citation verification.

Extracts every section/article/rule/Cap/case reference from generated text,
checks each against the corpus of relevant sources, logs unsupported claims to
logs/hallucinations.jsonl, attempts one regeneration, then strips remaining bad
sentences as a last resort.
"""

import re
import json
import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

LOGS_DIR = Path(__file__).resolve().parent / "logs"
HALL_LOG = LOGS_DIR / "hallucinations.jsonl"

# ── Extraction patterns ──────────────────────────────────────────────────────

_SEC_RE = re.compile(
    r'\b(?:section|sec|s)\.?\s*(\d+[A-Z]?(?:\([^)]+\))*)',
    re.IGNORECASE,
)
_ART_RE = re.compile(
    r'\barticle\s+(\d+[A-Z]?(?:\([^)]+\))*)',
    re.IGNORECASE,
)
_RULE_RE = re.compile(
    r'\brule\s+(\d+[A-Z]?(?:\([^)]+\))*)',
    re.IGNORECASE,
)
_CAP_RE = re.compile(r'\bcap\.?\s*(\d+[A-Z]?)\b', re.IGNORECASE)
_KLR_RE = re.compile(r'\d{4}(?:KEHC|KECA|SCC|KEELC)\d+\(KLR\)')
_CASE_RE = re.compile(
    r'\b([A-Z][^\s,;(]{1,30}(?:\s+[A-Z&][^\s,;(]{0,30}){0,4})'
    r'\s+v\.?\s+'
    r'([A-Z][^\s,;(]{1,30}(?:\s+[A-Z&][^\s,;(]{0,30}){0,4})\b'
)


def _core(ref: str) -> str:
    """'39(1)(a)' → '39'; 'article 163' → '163'."""
    digits = re.sub(r'[^\d]', '', ref.split('(')[0])
    return digits


def extract_citations(text: str) -> Dict[str, List[str]]:
    return {
        "sections":  [m.group(1) for m in _SEC_RE.finditer(text)],
        "articles":  [m.group(1) for m in _ART_RE.finditer(text)],
        "rules":     [m.group(1) for m in _RULE_RE.finditer(text)],
        "caps":      [m.group(1) for m in _CAP_RE.finditer(text)],
        "klr":       _KLR_RE.findall(text),
        "cases":     [f"{m.group(1)} v {m.group(2)}" for m in _CASE_RE.finditer(text)],
    }


# ── Build valid-reference set from source objects ────────────────────────────

def _valid_refs(sources: list) -> Dict[str, set]:
    v: Dict[str, set] = {
        "sections": set(), "articles": set(), "rules": set(),
        "caps": set(), "klr": set(), "cases": set(),
    }
    for s in sources:
        an = str(s.get("article_number", "")).strip()
        if an:
            n = _core(an)
            v["sections"].add(n)
            v["articles"].add(n)

        label = s.get("document_label", "")
        cap_m = _CAP_RE.search(label)
        if cap_m:
            v["caps"].add(cap_m.group(1))

        for text in [
            s.get("full_text", ""),
            s.get("text_preview", ""),
        ]:
            for m in _SEC_RE.finditer(text):
                v["sections"].add(_core(m.group(1)))
            for m in _ART_RE.finditer(text):
                v["articles"].add(_core(m.group(1)))

        cn = s.get("case_name", "")
        if cn:
            v["cases"].add(cn.lower())
            klr_m = _KLR_RE.search(cn)
            if klr_m:
                v["klr"].add(klr_m.group(0))

        for chunk in s.get("chunks", []):
            ct = chunk.get("text", "")
            for m in _KLR_RE.finditer(ct):
                v["klr"].add(m.group(0))
            for m in _CASE_RE.finditer(ct):
                v["cases"].add(f"{m.group(1)} v {m.group(2)}".lower())
    return v


def _supported(ref: str, ctype: str, valid: Dict[str, set]) -> bool:
    if ctype in ("sections", "articles", "rules"):
        n = _core(ref)
        return n in valid["sections"] or n in valid["articles"]
    if ctype == "caps":
        return ref in valid["caps"]
    if ctype == "klr":
        return ref in valid["klr"]
    if ctype == "cases":
        rl = ref.lower()
        return any(rl in v or v in rl for v in valid["cases"])
    return False


# ── Logging ──────────────────────────────────────────────────────────────────

def log_hallucination(query: str, claim: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    with open(HALL_LOG, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "query": query,
            "unsupported_claim": claim,
        }) + "\n")


# ── Sentence-level stripping ─────────────────────────────────────────────────

def _strip_sentences(text: str, bad_labels: List[str]) -> str:
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    kept = []
    for sent in sentences:
        drop = False
        for label in bad_labels:
            # Extract numeric / name core from label like "Section 39" or "KLR 2024KEHC1234"
            parts = label.split(" ", 1)
            needle = parts[-1] if len(parts) > 1 else label
            n = _core(needle)
            if n and re.search(rf'\b{re.escape(n)}\b', sent):
                drop = True
                break
            if needle.lower() in sent.lower():
                drop = True
                break
        if not drop:
            kept.append(sent)
    return " ".join(kept)


# ── Main entry point ─────────────────────────────────────────────────────────

def verify_and_clean(
    analysis_text: str,
    relevant_sources: list,
    query: str,
    regen_fn: Optional[Callable[[List[str]], Optional[str]]],
) -> Dict:
    """
    Verify citations in analysis_text against relevant_sources.

    Returns {"text": cleaned_text, "unverified_claims_removed": [labels]}.
    """
    if not analysis_text.strip():
        return {"text": analysis_text, "unverified_claims_removed": []}

    valid = _valid_refs(relevant_sources)
    cits = extract_citations(analysis_text)

    unsupported: List[str] = []
    for ctype, refs in cits.items():
        for ref in refs:
            if not _supported(ref, ctype, valid):
                type_label = ctype[:-1].upper() if ctype not in ("klr", "cases") else ctype.upper()
                unsupported.append(f"{type_label} {ref}")

    if not unsupported:
        return {"text": analysis_text, "unverified_claims_removed": []}

    for claim in unsupported:
        log_hallucination(query, claim)

    # Attempt one regeneration
    regen_text: Optional[str] = None
    if regen_fn:
        try:
            regen_text = regen_fn(unsupported)
        except Exception:
            regen_text = None

    if regen_text:
        valid2 = _valid_refs(relevant_sources)
        cits2 = extract_citations(regen_text)
        still_bad: List[str] = []
        for ctype, refs in cits2.items():
            for ref in refs:
                if not _supported(ref, ctype, valid2):
                    type_label = ctype[:-1].upper() if ctype not in ("klr", "cases") else ctype.upper()
                    still_bad.append(f"{type_label} {ref}")
        if not still_bad:
            return {"text": regen_text, "unverified_claims_removed": []}
        cleaned = _strip_sentences(regen_text, still_bad)
        return {"text": cleaned, "unverified_claims_removed": still_bad}

    # Last resort: strip from original
    cleaned = _strip_sentences(analysis_text, unsupported)
    return {"text": cleaned, "unverified_claims_removed": unsupported}
