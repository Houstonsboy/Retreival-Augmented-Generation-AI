#!/usr/bin/env python3
"""
Step 4 — Build case chunks for embedding.

Reads  data/processed/cases/cases_meta.json       (one entry per file)
Reads  backend/SCCCases/                           (PDFs and DOCX)
Writes data/processed/cases/cases_embed.jsonl
Writes backend/locations.db                        (chunk_locations table)

Uses PyMuPDF (fitz) for coordinate-aware PDF extraction so that page numbers
and bounding boxes are captured per chunk. DOCX files fall back to paragraph-
split text with no coordinate data.

Usage:
    cd backend
    python build_case_chunks.py
    python build_case_chunks.py --meta ../data/processed/cases/cases_meta.json
"""

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCC_CASES_DIR = BASE_DIR / "SCCCases"
DEFAULT_META = BASE_DIR.parent / "data" / "processed" / "cases" / "cases_meta.json"
DEFAULT_OUTPUT = BASE_DIR.parent / "data" / "processed" / "cases" / "cases_embed.jsonl"
LOCATIONS_DB = BASE_DIR / "locations.db"

MAX_CHARS_PER_CHUNK = 3000
OVERLAP_PARAGRAPHS = 1
_SCANNED_CHARS_PER_PAGE = 50   # avg chars/page below this → treat as scanned
_HEADER_Y_MARGIN = 54          # points from page edge; short blocks here are headers/footers
_COLLECTION = "case_law"


# ─── SQLite sidecar ───────────────────────────────────────────────────────────

def _init_locations_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_locations (
            chunk_id         TEXT PRIMARY KEY,
            collection       TEXT NOT NULL,
            source_file_path TEXT NOT NULL,
            served_file_path TEXT,
            page_start       INTEGER,
            page_end         INTEGER,
            boxes_json       TEXT,
            page_dims_json   TEXT,
            label            TEXT,
            ocr              INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def _upsert_location(conn, chunk_id, source_path, served_path,
                     page_start, page_end, boxes, page_dims, label, ocr):
    conn.execute(
        """INSERT OR REPLACE INTO chunk_locations
           (chunk_id, collection, source_file_path, served_file_path,
            page_start, page_end, boxes_json, page_dims_json, label, ocr)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk_id, _COLLECTION,
            str(source_path), str(served_path),
            page_start, page_end,
            json.dumps(boxes), json.dumps(page_dims),
            label, int(ocr),
        ),
    )


# ─── Scanned-PDF detection + OCR ─────────────────────────────────────────────

def _is_scanned_pdf(path: Path) -> bool:
    """Return True when pypdf extracts too little text to be a native-text PDF."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        if not reader.pages:
            return False
        total = sum(len(page.extract_text() or "") for page in reader.pages)
        return (total / len(reader.pages)) < _SCANNED_CHARS_PER_PAGE
    except Exception:
        return False


def _maybe_ocr_pdf(path: Path):
    """
    If *path* looks scanned, run ocrmypdf --skip-text and return
    (ocr_output_path, ocr=True). Never overwrites the original.
    Returns (path, False) when OCR is not needed or not available.
    """
    if not _is_scanned_pdf(path):
        return path, False

    ocr_path = path.with_name(path.stem + "_ocr" + path.suffix)
    if ocr_path.exists():
        print(f"    [OCR] reusing existing copy: {ocr_path.name}")
        return ocr_path, True

    print(f"    [OCR] no text layer detected in {path.name}, running ocrmypdf...")
    try:
        result = subprocess.run(
            ["ocrmypdf", "--skip-text", "--quiet", str(path), str(ocr_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"    [OCR] ocrmypdf failed (rc={result.returncode}): "
                f"{result.stderr.strip()[:200]}",
                file=sys.stderr,
            )
            return path, False
        print(f"    [OCR] saved OCR copy → {ocr_path.name}")
        return ocr_path, True
    except FileNotFoundError:
        print(
            "    [OCR] ocrmypdf not installed; skipping OCR "
            "(pip install ocrmypdf  +  apt install tesseract-ocr)",
            file=sys.stderr,
        )
        return path, False


# ─── PyMuPDF coordinate-aware extraction ─────────────────────────────────────

def _is_header_block(text: str, y0: float, y1: float, page_h: float) -> bool:
    t = text.strip()
    if not t:
        return True
    if t.isdigit():         # bare page number
        return True
    near_edge = y0 < _HEADER_Y_MARGIN or y1 > page_h - _HEADER_Y_MARGIN
    return near_edge and len(t) < 120


def extract_paragraphs_pdf(path: Path) -> list:
    """
    Use PyMuPDF to extract text blocks with page number and bounding-box data.
    Returns list of dicts: {text, page, bbox, page_w, page_h}.
    Running headers/footers and bare page numbers are dropped.
    """
    try:
        import fitz
    except ImportError:
        print("ERROR: PyMuPDF not installed. Run: pip install PyMuPDF", file=sys.stderr)
        sys.exit(1)

    doc = fitz.open(str(path))
    paragraphs = []

    for page_num, page in enumerate(doc):
        pw = page.rect.width
        ph = page.rect.height
        data = page.get_text("dict")

        for block in data.get("blocks", []):
            if block.get("type") != 0:  # 0 = text, 1 = image
                continue

            block_text = " ".join(
                span["text"]
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ).strip()

            if not block_text:
                continue

            x0, y0, x1, y1 = block["bbox"]

            if _is_header_block(block_text, y0, y1, ph):
                continue

            paragraphs.append({
                "text": block_text,
                "page": page_num,
                "bbox": [x0, y0, x1, y1],
                "page_w": pw,
                "page_h": ph,
            })

    doc.close()
    return paragraphs


# ─── Chunk building (coord-aware) ────────────────────────────────────────────

def _build_chunk_record(paras: list) -> dict:
    text = "\n\n".join(p["text"] for p in paras)
    pages = [p["page"] for p in paras]
    page_start = min(pages)
    page_end = max(pages)
    boxes = [{"page": p["page"], "bbox": p["bbox"]} for p in paras]
    page_dims = {}
    for p in paras:
        k = str(p["page"])
        if k not in page_dims:
            page_dims[k] = [p["page_w"], p["page_h"]]
    return {
        "text": text,
        "page_start": page_start,
        "page_end": page_end,
        "boxes": boxes,
        "page_dims": page_dims,
        "label": "",
    }


def split_paras_into_chunks(paragraphs: list, max_chars: int = MAX_CHARS_PER_CHUNK,
                             overlap_paras: int = OVERLAP_PARAGRAPHS) -> list:
    if not paragraphs:
        return []

    chunks = []
    current: list = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para["text"]) + 2
        if current and current_len + para_len > max_chars:
            chunks.append(_build_chunk_record(current))
            current = current[-overlap_paras:] if overlap_paras else []
            current_len = sum(len(p["text"]) + 2 for p in current)
        current.append(para)
        current_len += para_len

    if current:
        chunks.append(_build_chunk_record(current))

    return chunks or [_build_chunk_record(paragraphs)]


# ─── DOCX (text-only, no coords) ─────────────────────────────────────────────

def extract_text_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        print("ERROR: python-docx not installed. Run: pip install python-docx", file=sys.stderr)
        sys.exit(1)
    doc = Document(str(path))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paras)


def split_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK,
                      overlap_paras: int = OVERLAP_PARAGRAPHS) -> list:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current: list = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para) + 2
        if current and current_len + para_len > max_chars:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paras:] if overlap_paras else []
            current_len = sum(len(p) + 2 for p in current)
        current.append(para)
        current_len += para_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]


# ─── Misc helpers ─────────────────────────────────────────────────────────────

def safe_chunk_id(case_identifier: str, index: int) -> str:
    """Generate a stable, ChromaDB-safe chunk ID."""
    safe = case_identifier.replace("(", "").replace(")", "").replace(" ", "_")
    return f"{safe}_{index}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build case chunks for embedding")
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cases-dir", default=str(SCC_CASES_DIR))
    args = parser.parse_args()

    meta_path = Path(args.meta)
    output_path = Path(args.output)
    cases_dir = Path(args.cases_dir)

    if not meta_path.exists():
        print(f"ERROR: cases_meta.json not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta_list = json.load(f)

    print(f"Loaded {len(meta_list)} entries from {meta_path.name}")

    unfilled = [
        e["file_name_on_disk"] for e in meta_list
        if not e.get("exclude") and not e.get("legal_domain")
    ]
    if unfilled:
        print("\nWARNING: the following non-excluded entries have an empty legal_domain.")
        print("Fill data/processed/cases/cases_meta.json before the full embed run.")
        for fn in unfilled:
            print(f"  {fn}")
        print()

    conn = _init_locations_db(LOCATIONS_DB)
    all_chunks = []

    for entry in meta_list:
        if entry.get("exclude", False):
            print(f"  Skipping (excluded): {entry['case_identifier']}")
            continue

        original_path = cases_dir / entry["file_name_on_disk"]
        if not original_path.exists():
            print(f"  WARNING: file not found, skipping: {original_path}", file=sys.stderr)
            continue

        suffix = original_path.suffix.lower()
        print(f"  Extracting: {original_path.name}")

        ocr_used = False
        coord_chunks = None  # list[dict] with text + coord data

        if suffix == ".pdf":
            file_path, ocr_used = _maybe_ocr_pdf(original_path)
            paragraphs = extract_paragraphs_pdf(file_path)
            if not paragraphs:
                print(
                    f"  WARNING: no text paragraphs from {file_path.name}"
                    f"{' (even after OCR)' if ocr_used else ''} — skipping",
                    file=sys.stderr,
                )
                continue
            coord_chunks = split_paras_into_chunks(paragraphs)

        elif suffix == ".docx":
            file_path = original_path
            full_text = extract_text_docx(file_path)
            if not full_text.strip():
                print(f"  WARNING: no text extracted from {file_path.name} — skipping",
                      file=sys.stderr)
                continue
            coord_chunks = [
                {"text": t, "page_start": None, "page_end": None,
                 "boxes": [], "page_dims": {}, "label": ""}
                for t in split_into_chunks(full_text)
            ]

        else:
            print(f"  WARNING: unsupported extension '{suffix}', skipping", file=sys.stderr)
            continue

        doc_hash = hashlib.sha256(
            "".join(c["text"] for c in coord_chunks).encode("utf-8")
        ).hexdigest()[:16]
        chunk_total = len(coord_chunks)

        for i, cd in enumerate(coord_chunks):
            chunk_id = safe_chunk_id(entry["case_identifier"], i)
            page_start = cd["page_start"]
            page_end = cd["page_end"]

            chunk = {
                "chunk_id": chunk_id,
                "prev_chunk_id": "",
                "next_chunk_id": "",

                # Compatibility metadata — keys Retriever.py reads verbatim
                "case_identifier": entry["case_identifier"],
                "firac_component": "",
                "file_name": entry["file_name"],
                "parties": entry.get("parties", ""),
                "court_level": entry.get("court_level", ""),
                "judge": entry.get("judge", ""),
                "year": entry.get("year", ""),
                "legal_domain": entry.get("legal_domain", ""),
                "winning_party": entry.get("winning_party", ""),
                "source_file_path": str(file_path.resolve()),

                # Additional fields
                "document_hash": doc_hash,
                "chunk_index": i,
                "chunk_total": chunk_total,
                "content_length": len(cd["text"]),
                "page_start": page_start if page_start is not None else "",
                "page_end": page_end if page_end is not None else "",
                "label": cd.get("label", ""),
                "ocr": ocr_used,
                "text": cd["text"],
            }
            all_chunks.append(chunk)

            # Write bbox data to SQLite sidecar (only when coords available)
            if cd.get("boxes"):
                _upsert_location(
                    conn, chunk_id,
                    source_path=original_path,
                    served_path=file_path,  # OCR copy if ocr_used, else same as original
                    page_start=page_start,
                    page_end=page_end,
                    boxes=cd["boxes"],
                    page_dims=cd["page_dims"],
                    label=cd.get("label", ""),
                    ocr=ocr_used,
                )

    # Link prev/next within each case
    by_case: dict = defaultdict(list)
    for chunk in all_chunks:
        by_case[chunk["case_identifier"]].append(chunk)

    for chunks in by_case.values():
        for i, chunk in enumerate(chunks):
            chunk["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else ""
            chunk["next_chunk_id"] = chunks[i + 1]["chunk_id"] if i < len(chunks) - 1 else ""

    conn.commit()
    conn.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\nWrote {len(all_chunks)} chunks → {output_path}")
    print(f"Location data      → {LOCATIONS_DB}")

    cases_seen = set(c["case_identifier"] for c in all_chunks)
    for case_id in sorted(cases_seen):
        n = sum(1 for c in all_chunks if c["case_identifier"] == case_id)
        print(f"  {case_id}: {n} chunks")


if __name__ == "__main__":
    main()
