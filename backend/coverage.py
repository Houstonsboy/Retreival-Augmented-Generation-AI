"""
Coverage check: which Acts named in the query are in our corpus?

Uses manifest.json (statutes) and constitution_fallback aliases.
Returns {"in_corpus": [...], "not_in_corpus": [...]}.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "processed" / "statutes" / "manifest.json"
)

_ALIAS_MAP: Dict[str, str] = {}   # lower alias → canonical title


def _load_alias_map() -> Dict[str, str]:
    global _ALIAS_MAP
    if _ALIAS_MAP:
        return _ALIAS_MAP
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    m: Dict[str, str] = {}
    for act in manifest.get("acts", []):
        title = act["act_title"]
        m[title.lower()] = title
        for alias in act.get("aliases", []):
            m[alias.lower()] = title
    const = manifest.get("constitution_fallback", {})
    canonical = "Constitution of Kenya 2010"
    for alias in const.get("aliases", []):
        m[alias.lower()] = canonical
    _ALIAS_MAP = m
    return m


def _norm(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).lower().strip()


def check_coverage(classification: dict) -> Dict[str, List[str]]:
    """
    Given a classifier output dict, list which named Acts are in/out of corpus.

    Reads entities.statutes.{primary,contextual} and entities.source_documents.
    Returns {"in_corpus": [...canonical titles...], "not_in_corpus": [...as named...]}.
    """
    alias_map = _load_alias_map()

    entities = classification.get("entities", {})
    statutes = entities.get("statutes", {})
    raw: List[str] = []
    if isinstance(statutes, dict):
        raw.extend(statutes.get("primary", []))
        raw.extend(statutes.get("contextual", []))
    elif isinstance(statutes, list):
        raw.extend(statutes)
    raw.extend(entities.get("source_documents", []))

    # Deduplicate preserving order
    seen: set = set()
    unique: List[str] = []
    for act in raw:
        act = act.strip()
        if act and act.lower() not in seen:
            seen.add(act.lower())
            unique.append(act)

    in_corpus: List[str] = []
    not_in_corpus: List[str] = []

    for act_name in unique:
        lower = act_name.lower()
        if lower in alias_map:
            in_corpus.append(alias_map[lower])
            continue

        # Normalised partial match (handles "Arbitration Act, Cap 49")
        norm = _norm(act_name)
        matched = False
        for alias, title in alias_map.items():
            alias_n = _norm(alias)
            if norm and alias_n and (norm in alias_n or alias_n in norm):
                in_corpus.append(title)
                matched = True
                break

        if not matched:
            not_in_corpus.append(act_name)

    return {
        "in_corpus": list(dict.fromkeys(in_corpus)),
        "not_in_corpus": list(dict.fromkeys(not_in_corpus)),
    }
