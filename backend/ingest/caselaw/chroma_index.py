"""Build the case_law_v2 Chroma collection (Part A4).

One record per paragraph. ID = para_key (e.g. KEHC-2024-15763#p44).
Embedding prefix: "passage: " (same as the existing case_law collection).
Model: intfloat/e5-base-v2 (same).
Metadata keys include full backward-compat aliases for Retriever.py.

Does NOT overwrite case_law until A5 regression check passes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EMBED_MODEL = "intfloat/e5-base-v2"
_EMBED_PREFIX = "passage: "
_BATCH_SIZE = 64

_COURT_LEVEL_MAP = {
    "KEHC": "High Court",
    "KECA": "Court of Appeal",
    "KESC": "Supreme Court",
    "KEELC": "ELC",
    "KEELRC": "ELRC",
    "SCC": "Small Claims Court",
    "KEMC": "Magistrate Court",
}


def _get_collection(db_dir: Path, collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _make_metadata(
    para,
    meta,
    para_keys: list[str],
    idx: int,
    firac_role: str = "",
) -> dict[str, Any]:
    """
    Build a flat Chroma metadata dict with both new keys and backward-compat aliases.
    """
    # Next/prev para_key for stitching compatibility
    prev_key = para_keys[idx - 1] if idx > 0 else ""
    next_key = para_keys[idx + 1] if idx < len(para_keys) - 1 else ""

    # Map court_code to human-readable court_level
    court_level = _COURT_LEVEL_MAP.get(meta.court_code, meta.court_level or meta.court_code)

    # Primary domain for backward compat
    primary_domain = para.domains[0] if para.domains else "General"

    return {
        # --- New canonical keys ---
        "case_key":        meta.case_key,
        "para_no":         para.para_no,
        "para_sub":        para.para_sub,
        "para_numbering":  para.para_numbering,
        "case_name":       meta.name,
        "citation":        meta.citation,
        "court_code":      meta.court_code,
        "court_level":     court_level,
        "station":         meta.station,
        "decision_date":   str(meta.decision_date) if meta.decision_date else "",
        "year":            str(meta.decision_date.year) if meta.decision_date else "",
        "decision_type":   meta.decision_type,
        "judges":          " | ".join(meta.judges),
        "opinion_author":  para.opinion_author,
        "opinion_kind":    para.opinion_kind,
        "domains":         " | ".join(para.domains),
        "has_quote":       "true" if para.has_quote else "false",
        "quote_sources":   para.quote_sources,
        "page_start":      para.page_start,
        "page_end":        para.page_end,
        "source_file":     meta.source_file,
        "doc_hash":        meta.doc_hash,
        "ingest_run_id":   para.para_key.split("#")[0],  # filled properly in pipeline
        "text_repaired":   "true" if para.text_repaired else "false",
        "firac":           firac_role or para.firac,

        # --- Backward-compat aliases for Retriever.py ---
        "file_name":       meta.source_file,
        "parties":         meta.parties,
        "legal_domain":    primary_domain,
        "winning_party":   "",
        "firac_component": firac_role or para.firac,
        "chunk_index":     para.para_no,
        "case_identifier": meta.case_key,
        "prev_chunk_id":   prev_key,
        "next_chunk_id":   next_key,
    }


def build_case_law_v2(
    parsed_docs: list,
    firac_map: dict[str, str],
    db_dir: Path,
    collection_name: str = "case_law_v2",
    ingest_run_id: str = "",
) -> int:
    """
    Index all parsed documents into Chroma case_law_v2.

    parsed_docs: list of ParsedDocument
    firac_map:   {para_key: firac_role} from LLM call 1
    Returns total records indexed.
    """
    embedder = SentenceTransformer(_EMBED_MODEL)
    collection = _get_collection(db_dir, collection_name)

    total = 0
    for doc in parsed_docs:
        meta = doc.meta
        paras = doc.paragraphs
        para_keys = [p.para_key for p in paras]

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for idx, para in enumerate(paras):
            firac_role = firac_map.get(para.para_key, para.firac or "")
            md = _make_metadata(para, meta, para_keys, idx, firac_role)
            if ingest_run_id:
                md["ingest_run_id"] = ingest_run_id

            ids.append(para.para_key)
            documents.append(para.text)
            metadatas.append(md)

        # Embed in batches
        for i in range(0, len(documents), _BATCH_SIZE):
            batch_ids = ids[i:i + _BATCH_SIZE]
            batch_docs = documents[i:i + _BATCH_SIZE]
            batch_metas = metadatas[i:i + _BATCH_SIZE]
            prefixed = [_EMBED_PREFIX + d for d in batch_docs]
            embeddings = embedder.encode(prefixed, show_progress_bar=False).tolist()

            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
            )
            total += len(batch_ids)
            logger.info(
                "%s: indexed paragraphs %d–%d / %d",
                meta.case_key, i, min(i + _BATCH_SIZE, len(documents)), len(documents),
            )

    logger.info("case_law_v2: total %d records indexed", total)
    return total


def switch_collection(config_path: Path, old_name: str, new_name: str) -> None:
    """
    Rewrite CASELAW_COLLECTION_NAME in config.py from old_name to new_name.
    Call only after A5 regression check passes.
    """
    text = config_path.read_text()
    updated = text.replace(
        f'CASELAW_COLLECTION_NAME = "{old_name}"',
        f'CASELAW_COLLECTION_NAME = "{new_name}"',
    )
    if updated == text:
        raise ValueError(f"Could not find CASELAW_COLLECTION_NAME = '{old_name}' in {config_path}")
    config_path.write_text(updated)
    logger.info("Switched collection: %s → %s in %s", old_name, new_name, config_path)
