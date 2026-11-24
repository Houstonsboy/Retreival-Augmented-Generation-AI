"""Main ingest pipeline for SCC case-law judgments.

Orchestrates:
  1. Text extraction + ligature repair (extract.py)
  2. Header parsing with overrides (extract.py + overrides.yaml)
  3. Paragraph segmentation (extract.py)
  4. Deterministic extraction — citations, provisions, outcomes (deterministic.py)
  5. LLM extraction — FIRAC roles, citation treatment, issues/principles (llm.py)
  6. Chroma case_law_v2 indexing (chroma_index.py)
  7. Neo4j graph writes (graph/repository.py) — skipped when GRAPH_ENABLED=false

CLI: python -m ingest.caselaw run [--case KEY] [--dry-run] [--no-llm]
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
import yaml
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .extract import (
    extract_pdf, extract_docx, repair_ligatures, strip_furniture,
    extract_header_block, segment_paragraphs, detect_quote_spans,
    detect_domains, check_unknown_tokens, parse_decision_date, parse_judges,
    parse_station, parse_decision_type, file_hash,
)
from .models import CaseMeta, Paragraph, ParsedDocument, QuoteSpan
from .deterministic import run_deterministic, resolve_citation_ids
from .llm import call1_paragraph_roles, call2_issues_principles, validate_call1, validate_call2

logger = logging.getLogger(__name__)

_SCCASES_DIR = Path(__file__).resolve().parents[2] / "SCCCases"
_OVERRIDES_PATH = Path(__file__).parent / "overrides.yaml"
_ISSUES_PATH = Path(__file__).parent / "issues.yaml"
_REVIEW_QUEUE_PATH = Path(__file__).parent / "review_queue.yaml"
_BASE_DIR = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Load overrides
# ---------------------------------------------------------------------------

def _load_overrides() -> dict[str, dict]:
    if not _OVERRIDES_PATH.exists():
        return {}
    with _OVERRIDES_PATH.open() as f:
        data = yaml.safe_load(f)
    result = {}
    for entry in data.get("cases", []):
        result[entry["source_file"]] = entry
    return result


def _load_issues() -> list[dict]:
    if not _ISSUES_PATH.exists():
        return []
    with _ISSUES_PATH.open() as f:
        data = yaml.safe_load(f)
    return data.get("issues", [])


# ---------------------------------------------------------------------------
# Parse a single file into a ParsedDocument
# ---------------------------------------------------------------------------

def parse_file(path: Path, overrides: dict[str, dict]) -> ParsedDocument:
    filename = path.name
    ext = path.suffix.lower()

    # Extract raw lines
    if ext == ".pdf":
        raw_lines = extract_pdf(path)
    elif ext in (".docx", ".doc"):
        raw_lines = extract_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Repair ligatures + strip furniture
    repaired_lines: list[tuple[int, str]] = []
    total_replacements = 0
    for page_no, line in raw_lines:
        repaired, count = repair_ligatures(line, log_prefix=filename)
        total_replacements += count
        stripped = strip_furniture(repaired)
        if stripped:
            repaired_lines.append((page_no, stripped))

    if total_replacements:
        logger.info("%s: %d total ligature replacements", filename, total_replacements)

    full_text = "\n".join(line for _, line in repaired_lines)
    unknown_tokens = check_unknown_tokens(full_text)
    if unknown_tokens:
        logger.warning("%s: unknown tokens after repair: %s", filename, unknown_tokens[:30])

    # Header parsing
    override = overrides.get(filename, {})
    header_block = extract_header_block(repaired_lines)

    def _pick(key: str, parsed_val: Any) -> Any:
        return override.get(key, parsed_val)

    # Build case_key from filename
    from .extract import _build_case_key_from_filename
    case_key_from_file = _build_case_key_from_filename(filename)

    doc_hash_val = file_hash(path)

    # Decision date
    dd_raw = override.get("decision_date")
    if dd_raw:
        if isinstance(dd_raw, str):
            try:
                decision_date = date.fromisoformat(dd_raw)
            except ValueError:
                decision_date = parse_decision_date(dd_raw)
        else:
            decision_date = dd_raw
    else:
        decision_date = parse_decision_date(header_block)

    judges_parsed = parse_judges(header_block)
    station_parsed = parse_station(header_block)
    dtype_parsed = parse_decision_type(header_block)

    meta = CaseMeta(
        case_key=_pick("case_key", case_key_from_file) or f"UNKNOWN-{doc_hash_val[:8]}",
        name=_pick("name", filename.replace(".pdf", "").replace(".docx", "")[:80]),
        aliases=list(override.get("aliases", [])),
        citation=_pick("citation", ""),
        court_code=_pick("court_code", ""),
        court_level=_pick("court_level", ""),
        station=_pick("station", station_parsed),
        case_number=_pick("case_number", ""),
        decision_date=decision_date,
        decision_type=_pick("decision_type", dtype_parsed),
        judges=list(override.get("judges", judges_parsed)),
        parties=_pick("parties", ""),
        appeal_from=_pick("appeal_from", ""),
        source_file=filename,
        doc_hash=doc_hash_val,
    )

    # Paragraph segmentation
    segs = segment_paragraphs(repaired_lines, meta.case_key)

    # Domain detection from full text
    doc_domains = detect_domains(full_text)
    override_domains = list(override.get("domains", []))

    paragraphs: list[Paragraph] = []
    for seg in segs:
        para_key_str = f"{meta.case_key}#p{seg.para_no}"
        if seg.para_sub and seg.para_sub not in ("", "natural", "synthetic"):
            para_key_str = f"{meta.case_key}#p{seg.para_no}{seg.para_sub}"

        para_text = seg.text
        quote_spans_raw = detect_quote_spans(para_text)
        has_quote = bool(quote_spans_raw)
        quote_sources = " | ".join(
            qs["attributed_to_raw"] for qs in quote_spans_raw if qs["attributed_to_raw"]
        )
        para_domains = override_domains or detect_domains(para_text) or doc_domains

        paragraphs.append(Paragraph(
            para_key=para_key_str,
            case_key=meta.case_key,
            para_no=seg.para_no,
            para_sub=seg.para_sub,
            para_numbering=seg.numbering,
            text=para_text,
            opinion_author=seg.opinion_author,
            opinion_kind=seg.opinion_kind,
            domains=para_domains,
            has_quote=has_quote,
            quote_spans=[QuoteSpan(**qs) for qs in quote_spans_raw],
            quote_sources=quote_sources,
            page_start=seg.page_start,
            page_end=seg.page_end,
            text_repaired=total_replacements > 0,
        ))

    logger.info(
        "%s → %s: %d paragraphs",
        filename, meta.case_key, len(paragraphs),
    )
    return ParsedDocument(meta=meta, paragraphs=paragraphs)


# ---------------------------------------------------------------------------
# Append to review queue
# ---------------------------------------------------------------------------

def _append_review_queue(proposed_items: list[dict]) -> None:
    existing: list[dict] = []
    if _REVIEW_QUEUE_PATH.exists():
        with _REVIEW_QUEUE_PATH.open() as f:
            existing = yaml.safe_load(f) or []
    existing.extend(proposed_items)
    with _REVIEW_QUEUE_PATH.open("w") as f:
        yaml.dump(existing, f, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Graph writes (B6)
# ---------------------------------------------------------------------------

def _write_graph(doc: ParsedDocument, findings, llm1: dict, llm2: dict,
                 ingest_run_id: str, repo) -> None:
    from graph.keys import judge_key as _judge_key, provision_key as _prov_key

    case_key = doc.meta.case_key
    provenance = {
        "source_doc_hash": doc.meta.doc_hash,
        "ingest_run_id": ingest_run_id,
        "extractor": "header",
    }

    # Case node
    repo.upsert_node("Case", "case_key", {
        "case_key": case_key,
        "citation": doc.meta.citation,
        "name": doc.meta.name,
        "aliases": doc.meta.aliases,
        "aliases_text": " ".join(doc.meta.aliases),
        "in_corpus": True,
        "year": doc.meta.decision_date.year if doc.meta.decision_date else None,
        "court_code": doc.meta.court_code,
        "decision_type": doc.meta.decision_type,
        "decision_date": str(doc.meta.decision_date) if doc.meta.decision_date else "",
        **provenance,
    })

    # Judges → DECIDED_BY
    for judge_name in doc.meta.judges:
        jk = _judge_key(judge_name)
        repo.upsert_node("Judge", "judge_key", {"judge_key": jk, "name": judge_name, **provenance})
        repo.upsert_edge_by_keys("Case", "case_key", case_key,
                                 "DECIDED_BY", "Judge", "judge_key", jk, {})

    # Station → AT_STATION
    if doc.meta.station:
        station_key = f"station-{doc.meta.station.lower().replace(' ', '-')}"
        repo.upsert_node("Station", "station_key", {
            "station_key": station_key, "name": doc.meta.station, **provenance,
        })
        repo.upsert_edge_by_keys("Case", "case_key", case_key,
                                 "AT_STATION", "Station", "station_key", station_key, {})

    # Court → DECIDED_IN
    if doc.meta.court_code:
        repo.upsert_edge_by_keys("Case", "case_key", case_key,
                                 "DECIDED_IN", "Court", "code", doc.meta.court_code, {})

    # SourceDocument → CONTAINS → Case
    repo.upsert_node("SourceDocument", "doc_hash", {
        "doc_hash": doc.meta.doc_hash,
        "file_name": doc.meta.source_file,
        "file_type": Path(doc.meta.source_file).suffix.lstrip("."),
        **provenance,
    })
    repo.upsert_edge_by_keys("SourceDocument", "doc_hash", doc.meta.doc_hash,
                              "CONTAINS", "Case", "case_key", case_key, {})

    # Paragraphs
    para_keys = [p.para_key for p in doc.paragraphs]
    llm1_para_map = {
        p_data.get("para", ""): p_data
        for p_data in llm1.get("paragraphs", [])
    }

    for i, para in enumerate(doc.paragraphs):
        pid_short = f"p{para.para_no}{para.para_sub}"
        llm_para = llm1_para_map.get(pid_short, {})
        firac = llm_para.get("firac", para.firac or "")

        repo.upsert_node("Paragraph", "para_key", {
            "para_key": para.para_key,
            "case_key": case_key,
            "para_no": para.para_no,
            "para_sub": para.para_sub,
            "para_numbering": para.para_numbering,
            "text": para.text,
            "firac": firac,
            "opinion_author": para.opinion_author,
            "opinion_kind": para.opinion_kind,
            "has_quote": para.has_quote,
            "quote_spans": str([qs.model_dump() for qs in para.quote_spans]),
            "page_start": para.page_start,
            "page_end": para.page_end,
            "source_doc_hash": doc.meta.doc_hash,
            "ingest_run_id": ingest_run_id,
            "extractor": "header",
        })
        repo.upsert_edge_by_keys("Case", "case_key", case_key,
                                 "HAS_PARAGRAPH", "Paragraph", "para_key", para.para_key, {})

        if i > 0:
            repo.upsert_edge_by_keys("Paragraph", "para_key", para_keys[i - 1],
                                     "NEXT", "Paragraph", "para_key", para.para_key, {})

        # Citation edges
        for cit_data in llm_para.get("citations", []):
            cit_id = cit_data.get("citation_id", "")
            treatment = cit_data.get("treatment", "CONSIDERED")
            reason = cit_data.get("reason_span", "")
            # Ensure stub case exists
            repo.upsert_node("Case", "case_key", {
                "case_key": cit_id, "in_corpus": False,
                "citation": cit_id, "name": cit_id,
                "extractor": "regex",
            })
            repo.upsert_edge_by_keys("Paragraph", "para_key", para.para_key,
                                     "CITES", "Case", "case_key", cit_id,
                                     {"treatment": treatment, "reason_span": reason,
                                      "confidence": 0.9, "source_doc_hash": doc.meta.doc_hash})

        # Provision edges
        for prov_data in llm_para.get("provisions", []):
            pid = prov_data.get("provision_id", "")
            mode = prov_data.get("mode", "MENTIONS")
            parts = pid.split("/", 1)
            stat = parts[0] if parts else pid
            sec = parts[1] if len(parts) > 1 else ""
            repo.upsert_node("Provision", "provision_key", {
                "provision_key": pid, "statute_key": stat, "section": sec,
                "extractor": "regex",
            })
            repo.upsert_edge_by_keys("Paragraph", "para_key", para.para_key,
                                     mode, "Provision", "provision_key", pid,
                                     {"source_doc_hash": doc.meta.doc_hash})

        # Evidence rulings
        for ev in llm_para.get("evidence_rulings", []):
            ev_type = ev.get("type", "")
            ev_result = ev.get("result", "")
            if ev_type:
                repo.upsert_edge_by_keys("Paragraph", "para_key", para.para_key,
                                         "RULED_ON_EVIDENCE", "EvidenceType", "code", ev_type,
                                         {"result": ev_result, "source_doc_hash": doc.meta.doc_hash})

    # Outcome
    outcome_data = llm1.get("outcome", {})
    if not outcome_data.get("code") and findings.outcome_code:
        outcome_data = {"code": findings.outcome_code}
    if outcome_data.get("code"):
        repo.upsert_edge_by_keys("Case", "case_key", case_key,
                                 "HAS_OUTCOME", "Outcome", "code", outcome_data["code"], {})

    # Claim type
    claim_type = llm1.get("claim_type", "")
    if claim_type:
        repo.upsert_edge_by_keys("Case", "case_key", case_key,
                                 "CAUSE_OF_ACTION", "ClaimType", "code", claim_type, {})

    # Party types
    for pt in llm1.get("party_types", []):
        repo.upsert_edge_by_keys("Case", "case_key", case_key,
                                 "HAS_PARTY_TYPE", "PartyType", "code", pt, {})

    # Issues and Principles (LLM call 2)
    _write_issues_principles(case_key, doc.meta.doc_hash, llm2, ingest_run_id, repo)


def _write_issues_principles(case_key: str, doc_hash: str, llm2: dict,
                              ingest_run_id: str, repo) -> None:
    proposed_items: list[dict] = []

    for issue_data in llm2.get("issues", []):
        iid = issue_data.get("issue_id")
        if not iid:
            proposed_q = issue_data.get("proposed_question", "")
            if not proposed_q:
                continue
            slug = proposed_q[:50].lower()
            import re
            slug = re.sub(r"\W+", "-", slug).strip("-")
            iid = f"issue:proposed-{slug}"
            proposed_items.append({
                "type": "issue",
                "issue_id": iid,
                "question": proposed_q,
                "status": "proposed",
                "case_key": case_key,
            })
        repo.upsert_node("Issue", "issue_key", {
            "issue_key": iid,
            "question": issue_data.get("proposed_question", ""),
            "source_doc_hash": doc_hash,
            "ingest_run_id": ingest_run_id,
            "extractor": f"llm:qwen/qwen3-8b@v1.0",
        })
        repo.upsert_edge_by_keys("Case", "case_key", case_key,
                                 "DECIDES", "Issue", "issue_key", iid, {})

    for prin_data in llm2.get("principles", []):
        pid = prin_data.get("principle_id")
        if not pid:
            text = prin_data.get("proposed_text", "")
            if not text:
                continue
            import re, hashlib
            pid = "principle:" + hashlib.sha256(text.encode()).hexdigest()[:12]
            proposed_items.append({
                "type": "principle",
                "principle_id": pid,
                "text": text,
                "role": prin_data.get("role", "obiter"),
                "case_key": case_key,
                "paras": prin_data.get("paras", []),
                "answers_issue": prin_data.get("answers_issue", ""),
                "status": "proposed",
            })

        prov = {
            "principle_key": pid,
            "text": prin_data.get("proposed_text", ""),
            "kind": prin_data.get("role", "obiter"),
            "status": "proposed" if not prin_data.get("principle_id") else "approved",
            "source_doc_hash": doc_hash,
            "ingest_run_id": ingest_run_id,
            "extractor": f"llm:qwen/qwen3-8b@v1.0",
        }
        repo.upsert_node("Principle", "principle_key", prov)

        # STATES edge from supporting paragraphs
        for para_ref in prin_data.get("paras", []):
            para_key_full = f"{case_key}#{para_ref}"
            repo.upsert_edge_by_keys("Paragraph", "para_key", para_key_full,
                                     "STATES", "Principle", "principle_key", pid,
                                     {"role": prin_data.get("role", "obiter")})

        # ANSWERS edge
        answers = prin_data.get("answers_issue")
        if answers:
            repo.upsert_edge_by_keys("Principle", "principle_key", pid,
                                     "ANSWERS", "Issue", "issue_key", answers, {})

        # DERIVED_FROM provisions
        for prov_id in prin_data.get("derived_from", []):
            repo.upsert_edge_by_keys("Principle", "principle_key", pid,
                                     "DERIVED_FROM", "Provision", "provision_key", prov_id, {})

        # ContractElement BEARS_ON
        for ce_code in prin_data.get("bears_on", []):
            repo.upsert_edge_by_keys("Principle", "principle_key", pid,
                                     "BEARS_ON", "ContractElement", "code", ce_code, {})

        # Cross-principle relations
        for rel in prin_data.get("relations", []):
            rel_type = rel.get("type", "")
            target = rel.get("target_principle_id", "")
            if rel_type and target:
                repo.upsert_edge_by_keys("Principle", "principle_key", pid,
                                         rel_type, "Principle", "principle_key", target, {})

    # Concepts
    for concept in llm2.get("concepts", []):
        import re
        ckey = "concept:" + re.sub(r"\W+", "-", concept.lower()).strip("-")
        repo.upsert_node("Concept", "concept_key", {
            "concept_key": ckey, "label": concept,
            "source_doc_hash": doc_hash, "extractor": "llm:qwen/qwen3-8b@v1.0",
        })
        repo.upsert_edge_by_keys("Case", "case_key", case_key,
                                 "INVOLVES", "Concept", "concept_key", ckey, {})

    if proposed_items:
        _append_review_queue(proposed_items)
        logger.info(
            "%s: %d proposed items added to review_queue.yaml",
            case_key, len(proposed_items),
        )


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_pipeline(
    case_filter: str | None = None,
    dry_run: bool = False,
    no_llm: bool = False,
    ingest_run_id: str | None = None,
) -> dict:
    """
    Run the full ingest pipeline.

    Returns a summary dict with counts and any blocked items.
    """
    if ingest_run_id is None:
        ingest_run_id = str(uuid.uuid4())[:8]

    overrides = _load_overrides()
    issues = _load_issues()

    scc_dir = _SCCASES_DIR
    if not scc_dir.exists():
        raise FileNotFoundError(f"SCCCases directory not found: {scc_dir}")

    all_files = sorted(scc_dir.glob("*"))
    all_files = [f for f in all_files if f.suffix.lower() in (".pdf", ".docx", ".doc")]

    # Sort oldest first (by decision year from filename)
    import re as _re
    def _year_sort(p: Path) -> int:
        m = _re.search(r"(\d{4})", p.name)
        return int(m.group(1)) if m else 9999
    all_files.sort(key=_year_sort)

    if case_filter:
        all_files = [f for f in all_files if case_filter.lower() in f.name.lower()]

    parsed_docs: list[ParsedDocument] = []
    corpus_case_keys: set[str] = set()

    # First pass: parse all files
    for path in all_files:
        logger.info("Parsing: %s", path.name)
        try:
            doc = parse_file(path, overrides)
            parsed_docs.append(doc)
            corpus_case_keys.add(doc.meta.case_key)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", path.name, exc)

    # Deterministic extraction
    all_findings = {}
    for doc in parsed_docs:
        findings = run_deterministic(doc)
        # Resolve citation IDs against corpus
        findings.citations = resolve_citation_ids(findings.citations, corpus_case_keys)
        all_findings[doc.meta.case_key] = findings

    # Chroma indexing
    firac_map: dict[str, str] = {}
    llm1_results: dict[str, dict] = {}
    llm2_results: dict[str, dict] = {}
    groq_calls_used = 0

    if not no_llm:
        existing_principles: list[dict] = []

        for doc in parsed_docs:
            findings = all_findings[doc.meta.case_key]
            det_cit_ids = {c.citation_id for c in findings.citations}
            det_prov_ids = {p.provision_id for p in findings.provisions}

            # Call 1
            det_cits_dicts = [c.model_dump() for c in findings.citations]
            det_provs_dicts = [p.model_dump() for p in findings.provisions]
            paras_dicts = [{"para_no": p.para_no, "para_sub": p.para_sub, "text": p.text}
                           for p in doc.paragraphs]

            from pathlib import Path as _Path
            cache_check_path = _Path(__file__).parent / "cache"
            import hashlib as _hash
            cache_key_check = _hash.sha256(
                f"{doc.meta.doc_hash}:v1.0:1".encode()
            ).hexdigest()[:16]
            cache_hit_1 = (_Path(__file__).parent / "cache" / f"{cache_key_check}.json").exists()

            result1 = call1_paragraph_roles(
                doc.meta.doc_hash, doc.meta.case_key,
                paras_dicts, det_cits_dicts, det_provs_dicts,
            )
            if not cache_hit_1:
                groq_calls_used += 1

            result1 = validate_call1(result1, paras_dicts, det_cit_ids, det_prov_ids)
            llm1_results[doc.meta.case_key] = result1

            # Build FIRAC map
            for p_data in result1.get("paragraphs", []):
                pid = p_data.get("para", "")
                firac_map[f"{doc.meta.case_key}#{pid}"] = p_data.get("firac", "")

            # Call 2 — only analysis/conclusion paragraphs
            analysis_paras = [
                {"para_no": p.para_no, "para_sub": p.para_sub, "text": p.text}
                for p in doc.paragraphs
                if firac_map.get(p.para_key, "") in ("analysis", "conclusion", "rule", "")
            ][:30]  # cap at 30 to stay within token budget

            cache_key_check2 = _hash.sha256(
                f"{doc.meta.doc_hash}:v1.0:2".encode()
            ).hexdigest()[:16]
            cache_hit_2 = (_Path(__file__).parent / "cache" / f"{cache_key_check2}.json").exists()

            result2 = call2_issues_principles(
                doc.meta.doc_hash, doc.meta.case_key,
                analysis_paras, issues, existing_principles,
                det_provs_dicts,
            )
            if not cache_hit_2:
                groq_calls_used += 1

            existing_principle_ids = {p.get("principle_id", "") for p in existing_principles}
            result2 = validate_call2(result2, existing_principle_ids)
            llm2_results[doc.meta.case_key] = result2

            # Accumulate approved principles for subsequent cases
            for p_data in result2.get("principles", []):
                if p_data.get("principle_id"):
                    existing_principles.append({
                        "principle_id": p_data["principle_id"],
                        "text": p_data.get("proposed_text", ""),
                    })

    if not dry_run:
        # Index into Chroma case_law_v2
        from pathlib import Path as _Path
        from config import CASELAW_DB_DIR

        db_v2_dir = _Path(str(CASELAW_DB_DIR) + "_v2")
        from .chroma_index import build_case_law_v2
        total_indexed = build_case_law_v2(parsed_docs, firac_map, db_v2_dir,
                                           ingest_run_id=ingest_run_id)
        logger.info("Chroma case_law_v2: %d records", total_indexed)

        # Graph writes (skipped if GRAPH_ENABLED=false)
        graph_enabled = os.getenv("GRAPH_ENABLED", "false").lower() in ("1", "true", "yes")
        if graph_enabled:
            from graph.client import get_client
            from graph.repository import Neo4jRepository
            from graph.migrations.runner import run_pending

            client = get_client()
            run_pending(client)
            repo = _GraphRepoAdapter(Neo4jRepository(client))

            # Register IngestRun node
            client.write(
                "MERGE (r:IngestRun {run_id: $rid}) SET r.started_at = datetime()",
                rid=ingest_run_id,
            )

            for doc in parsed_docs:
                # Delete previous edges for this doc before rewriting
                client.write(
                    "MATCH (n {source_doc_hash: $h})-[r]-() DELETE r",
                    h=doc.meta.doc_hash,
                )
                llm1 = llm1_results.get(doc.meta.case_key, {})
                llm2 = llm2_results.get(doc.meta.case_key, {})
                findings = all_findings[doc.meta.case_key]
                _write_graph(doc, findings, llm1, llm2, ingest_run_id, repo)

            client.write(
                "MERGE (r:IngestRun {run_id: $rid}) SET r.finished_at = datetime(), r.status = 'done'",
                rid=ingest_run_id,
            )
            logger.info("Graph writes complete.")
        else:
            logger.info("GRAPH_ENABLED=false — skipping Neo4j writes.")

    return {
        "cases": len(parsed_docs),
        "total_paragraphs": sum(len(d.paragraphs) for d in parsed_docs),
        "groq_calls_used": groq_calls_used,
        "ingest_run_id": ingest_run_id,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# Thin adapter: adds upsert_edge_by_keys to the existing repository
# ---------------------------------------------------------------------------

class _GraphRepoAdapter:
    def __init__(self, inner):
        self._inner = inner

    def upsert_node(self, label, key_field, node):
        self._inner.upsert_node(label, key_field, node)

    def upsert_edge_by_keys(
        self,
        src_label: str, src_key: str, src_val: str,
        rel_type: str,
        dst_label: str, dst_key: str, dst_val: str,
        props: dict,
    ) -> None:
        client = self._inner._c
        props_str = ", ".join(f"r.{k} = ${k}" for k in props)
        set_clause = f"SET {props_str}" if props else ""
        query = (
            f"MATCH (a:{src_label} {{{src_key}: $sv}}) "
            f"MATCH (b:{dst_label} {{{dst_key}: $dv}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) {set_clause}"
        )
        client.write(query, sv=src_val, dv=dst_val, **props)
