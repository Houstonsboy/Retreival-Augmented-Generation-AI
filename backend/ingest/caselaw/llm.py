"""LLM extraction — 2 Groq calls per judgment (B4).

Calls are cached by doc_hash + prompt_version in ingest/caselaw/cache/.
A rerun with the same inputs costs 0 API calls.
Model: qwen/qwen3.8-27b at T=0, fixed seed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "cache"
_CACHE_DIR.mkdir(exist_ok=True)

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = "qwen/qwen3-8b"
_PROMPT_VERSION_1 = "v1.0"
_PROMPT_VERSION_2 = "v1.0"

FIRAC_ROLES = ["facts", "issue", "rule", "analysis", "conclusion", "orders",
               "header", "submissions_claimant", "submissions_respondent"]
CITATION_TREATMENTS = ["FOLLOWED", "APPLIED", "CONSIDERED", "DISTINGUISHED",
                        "DEPARTED_FROM", "OVERRULED", "QUOTED", "CITED_BY_PARTY"]
PROVISION_MODES = ["INTERPRETS", "APPLIES", "MENTIONS"]
EVIDENCE_TYPES = ["whatsapp", "sms", "mpesa", "email", "photo", "cctv",
                  "dvd", "social_post", "document"]
EVIDENCE_RESULTS = ["ADMITTED", "EXCLUDED", "EXCLUDED_WITH_LEAVE_TO_CURE",
                    "ACCEPTED_FOR_SERVICE", "ACCEPTED_TO_LINK_IDENTITY", "RELIED_ON"]
OUTCOME_CODES = ["ALLOWED", "DISMISSED", "STRUCK_OUT", "SET_ASIDE_REFUSED",
                 "SET_ASIDE_GRANTED", "OBJECTION_UPHELD", "OBJECTION_DISMISSED",
                 "PARTLY_ALLOWED", "GRANTED", "REFUSED"]
CLAIM_TYPE_CODES = ["money_lent", "sale_goods_services", "property_damage_tort",
                    "personal_injury", "subrogation_recovery", "employment_founded",
                    "defamation", "bailment", "other"]
PARTY_TYPE_CODES = ["individual", "company", "insurer", "employer", "government"]
PRINCIPLE_RELATIONS = ["QUALIFIES", "CONFLICTS_WITH", "SUPPORTS"]
CONTRACT_ELEMENTS = list("ABCDEFGHIJK")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(doc_hash: str, prompt_version: str, call_no: int) -> str:
    raw = f"{doc_hash}:{prompt_version}:{call_no}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def _cache_load(doc_hash: str, prompt_version: str, call_no: int) -> dict | None:
    path = _cache_path(_cache_key(doc_hash, prompt_version, call_no))
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupt cache entry %s — ignoring", path.name)
    return None


def _cache_save(doc_hash: str, prompt_version: str, call_no: int, data: dict) -> None:
    path = _cache_path(_cache_key(doc_hash, prompt_version, call_no))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def _groq_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    return key


def _call_groq(messages: list[dict], max_tokens: int = 4096) -> dict:
    import requests

    headers = {
        "Authorization": f"Bearer {_groq_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _MODEL,
        "messages": messages,
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(3):
        try:
            resp = requests.post(_GROQ_API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = 20 * (attempt + 1)
                logger.warning("Rate limited — waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("LLM response parse error: %s", exc)
            raise
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning("Groq call failed (%s) — retry %d", exc, attempt + 1)
            time.sleep(5)
    raise RuntimeError("All Groq retries exhausted")


# ---------------------------------------------------------------------------
# Call 1 — paragraph roles + citation treatment
# ---------------------------------------------------------------------------

_CALL1_SYSTEM = """You are a legal extraction engine. Output ONLY valid JSON matching the schema provided.
No prose, no markdown. Follow these rules:
- Citations appearing in paragraphs summarising a party's submissions → treatment CITED_BY_PARTY (unless the court later adopts them).
- DEPARTED_FROM and OVERRULED require explicit language (e.g. "depart from", "overrule", "no longer good law").
- If unsure of treatment, use CONSIDERED.
- For provisions, choose the mode that best reflects the paragraph: INTERPRETS (court explains meaning), APPLIES (court applies to facts), MENTIONS (reference only).
- evidence_rulings: only include if the paragraph contains a ruling on admissibility or use of a specific evidence type.
- outcome and claim_type: provide at the judgment level (not per paragraph). outcome.para is the paragraph number where the final order appears."""

_CALL1_SCHEMA = """{
  "paragraphs": [
    {
      "para": "p44",
      "firac": "facts|issue|rule|analysis|conclusion|orders|header|submissions_claimant|submissions_respondent",
      "citations": [
        {"citation_id": "<from deterministic list>", "treatment": "FOLLOWED|APPLIED|CONSIDERED|DISTINGUISHED|DEPARTED_FROM|OVERRULED|QUOTED|CITED_BY_PARTY", "reason_span": "<=15 words copied from paragraph"}
      ],
      "provisions": [
        {"provision_id": "<from list>", "mode": "INTERPRETS|APPLIES|MENTIONS"}
      ],
      "evidence_rulings": [
        {"type": "whatsapp|sms|mpesa|email|photo|cctv|dvd|social_post|document", "result": "ADMITTED|EXCLUDED|EXCLUDED_WITH_LEAVE_TO_CURE|ACCEPTED_FOR_SERVICE|ACCEPTED_TO_LINK_IDENTITY|RELIED_ON"}
      ]
    }
  ],
  "outcome": {"code": "ALLOWED|DISMISSED|STRUCK_OUT|SET_ASIDE_REFUSED|SET_ASIDE_GRANTED|OBJECTION_UPHELD|OBJECTION_DISMISSED|PARTLY_ALLOWED|GRANTED|REFUSED", "para": "p51"},
  "claim_type": "money_lent|sale_goods_services|property_damage_tort|personal_injury|subrogation_recovery|employment_founded|defamation|bailment|other",
  "party_types": ["individual", "company", "insurer", "employer", "government"]
}"""


def call1_paragraph_roles(
    doc_hash: str,
    case_key: str,
    paragraphs: list[dict],
    det_citations: list[dict],
    det_provisions: list[dict],
) -> dict:
    """LLM call 1: paragraph FIRAC roles, citation treatments, evidence rulings."""
    cached = _cache_load(doc_hash, _PROMPT_VERSION_1, 1)
    if cached is not None:
        logger.info("%s call1: cache hit", case_key)
        return cached

    para_lines = "\n".join(
        f"p{p['para_no']}: {p['text'][:500]}" for p in paragraphs
    )
    cit_list = "\n".join(
        f"  {c['citation_id']} ({c.get('case_name','')})" for c in det_citations
    )
    prov_list = "\n".join(
        f"  {p['provision_id']}" for p in det_provisions
    )

    user_msg = f"""Case: {case_key}

PARAGRAPHS (truncated to 500 chars each):
{para_lines}

CITATIONS FOUND BY REGEX (use only these IDs):
{cit_list or '  (none)'}

PROVISIONS FOUND BY REGEX (use only these IDs):
{prov_list or '  (none)'}

Output schema:
{_CALL1_SCHEMA}

Extract for ALL paragraphs listed. Omit citations/provisions/evidence_rulings arrays if empty."""

    messages = [
        {"role": "system", "content": _CALL1_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    result = _call_groq(messages, max_tokens=8192)
    _cache_save(doc_hash, _PROMPT_VERSION_1, 1, result)
    logger.info("%s call1: completed, %d paragraphs", case_key, len(result.get("paragraphs", [])))
    return result


# ---------------------------------------------------------------------------
# Call 2 — Issues and Principles
# ---------------------------------------------------------------------------

_CALL2_SYSTEM = """You are a legal knowledge extraction engine. Output ONLY valid JSON.
Rules:
- Match to existing issues/principles by ID when possible.
- New items use null for the existing id field and provide proposed text.
- status for new items is always "proposed".
- role: "ratio" = binding holding; "obiter" = observation not essential to decision.
- paras: list of paragraph IDs supporting this principle.
- derived_from: list of provision_ids (from the provisions list).
- bears_on: list of ContractElement codes A-K.
- relations: cross-links to OTHER principles; cite target_principle_id from the existing list."""

_CALL2_SCHEMA = """{
  "issues": [
    {"issue_id": "<existing id or null>", "proposed_question": "<only if no existing id fits>"}
  ],
  "principles": [
    {
      "principle_id": "<existing id or null>",
      "proposed_text": "<neutral one-sentence rule if new>",
      "role": "ratio|obiter",
      "paras": ["p41"],
      "answers_issue": "<issue_id or proposed question>",
      "derived_from": ["<provision_id>"],
      "bears_on": ["A"],
      "relations": [
        {"type": "QUALIFIES|CONFLICTS_WITH|SUPPORTS", "target_principle_id": "<existing>", "para": "p38"}
      ]
    }
  ],
  "concepts": ["adverse_inference", "subrogation"]
}"""


def call2_issues_principles(
    doc_hash: str,
    case_key: str,
    analysis_paragraphs: list[dict],
    existing_issues: list[dict],
    existing_principles: list[dict],
    det_provisions: list[dict],
) -> dict:
    """LLM call 2: issues and principles from analysis/conclusion paragraphs."""
    cached = _cache_load(doc_hash, _PROMPT_VERSION_2, 2)
    if cached is not None:
        logger.info("%s call2: cache hit", case_key)
        return cached

    para_lines = "\n".join(
        f"p{p['para_no']}: {p['text'][:600]}" for p in analysis_paragraphs
    )
    issue_list = "\n".join(
        f"  {i['issue_id']}: {i['question']}" for i in existing_issues
    )
    principle_list = "\n".join(
        f"  {p['principle_id']}: {p.get('text','')[:120]}"
        for p in existing_principles[:40]
    )
    prov_list = "\n".join(f"  {p['provision_id']}" for p in det_provisions)

    user_msg = f"""Case: {case_key}

ANALYSIS/CONCLUSION PARAGRAPHS:
{para_lines}

EXISTING ISSUES (match by ID):
{issue_list or '  (none yet)'}

EXISTING PRINCIPLES (match or relate by ID):
{principle_list or '  (none yet)'}

PROVISIONS FOUND IN THIS CASE:
{prov_list or '  (none)'}

CONTRACT ELEMENTS: A=Offer B=Acceptance C=Consideration D=Intention E=Capacity F=Legality G=Certainty H=Performance/Breach I=Damages J=SpecificRelief K=Limitation

Output schema:
{_CALL2_SCHEMA}

Extract issues and principles from this judgment's analysis/conclusion sections only."""

    messages = [
        {"role": "system", "content": _CALL2_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    result = _call_groq(messages, max_tokens=8192)
    _cache_save(doc_hash, _PROMPT_VERSION_2, 2, result)
    logger.info("%s call2: completed", case_key)
    return result


# ---------------------------------------------------------------------------
# Grounding and validation (B5)
# ---------------------------------------------------------------------------

def validate_call1(result: dict, paragraphs: list[dict], det_cit_ids: set[str],
                   det_prov_ids: set[str]) -> dict:
    """
    Reject items that fail grounding checks. Returns cleaned result.
    Items that fail are logged and dropped; the rest are kept.
    """
    from rapidfuzz import fuzz as _fuzz

    para_map = {f"p{p['para_no']}{p.get('para_sub','')}": p["text"] for p in paragraphs}
    clean_paras = []

    for para_data in result.get("paragraphs", []):
        para_id = para_data.get("para", "")
        if para_id not in para_map:
            logger.warning("Grounding: para %s not found — dropping", para_id)
            continue

        para_text = para_map[para_id]
        clean_cits = []
        for cit in para_data.get("citations", []):
            cit_id = cit.get("citation_id", "")
            if cit_id not in det_cit_ids:
                logger.warning("Grounding: citation %s not in deterministic list — dropping", cit_id)
                continue
            reason = cit.get("reason_span", "")
            if reason:
                score = _fuzz.partial_ratio(reason.lower(), para_text.lower())
                if score < 90:
                    logger.warning(
                        "Grounding: reason_span '%s' not found in %s (score=%d) — dropping",
                        reason[:30], para_id, score,
                    )
                    continue
            clean_cits.append(cit)

        clean_provs = []
        for prov in para_data.get("provisions", []):
            pid = prov.get("provision_id", "")
            if pid not in det_prov_ids:
                logger.warning("Grounding: provision %s not in deterministic list — dropping", pid)
                continue
            clean_provs.append(prov)

        para_data["citations"] = clean_cits
        para_data["provisions"] = clean_provs
        clean_paras.append(para_data)

    result["paragraphs"] = clean_paras
    return result


def validate_call2(result: dict, existing_principle_ids: set[str]) -> dict:
    """Validate call 2 output: drop principles with no paras or missing relation targets."""
    clean_principles = []
    for p in result.get("principles", []):
        if not p.get("paras"):
            logger.warning("Grounding: principle '%s' has no supporting paras — dropping",
                           p.get("proposed_text", "?")[:50])
            continue
        clean_relations = []
        for rel in p.get("relations", []):
            target = rel.get("target_principle_id", "")
            if target and target not in existing_principle_ids:
                logger.warning("Grounding: relation target %s not in existing — dropping", target)
                continue
            clean_relations.append(rel)
        p["relations"] = clean_relations
        clean_principles.append(p)
    result["principles"] = clean_principles
    return result
