#!/usr/bin/env python3
"""
Step 3 — Build statute chunks for embedding.

Reads  data/processed/statutes/sections.jsonl
Writes data/processed/statutes/sections_embed.jsonl

- Tokenises each section with the real e5-base-v2 tokenizer.
- Leaves sections at or under 480 tokens as-is.
- Splits sections over 480 tokens into overlapping 480-token windows
  (50-token overlap, stride 430).
- Writes full compatibility metadata so RulesRetriever.py needs no
  changes to how it reads chunk fields.
- Sets prev_chunk_id / next_chunk_id within each section group for
  overlap-stitching at retrieval time.

Usage:
    cd backend
    python build_statute_chunks.py
    python build_statute_chunks.py --input ../data/processed/statutes/sections.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR.parent / "data" / "processed" / "statutes" / "sections.jsonl"
DEFAULT_OUTPUT = BASE_DIR.parent / "data" / "processed" / "statutes" / "sections_embed.jsonl"

MODEL_NAME = "intfloat/e5-base-v2"
MAX_TOKENS = 480
OVERLAP_TOKENS = 50
STRIDE = MAX_TOKENS - OVERLAP_TOKENS  # 430


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers", file=sys.stderr)
        sys.exit(1)


def split_text(text: str, tokenizer, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS):
    """
    Tokenise text (no prefix, no special tokens) and return a list of
    decoded sub-strings, each within max_tokens tokens.
    Consecutive windows share `overlap` tokens.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text], len(ids)

    stride = max_tokens - overlap
    windows = []
    i = 0
    while i < len(ids):
        end = min(i + max_tokens, len(ids))
        decoded = tokenizer.decode(
            ids[i:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        windows.append(decoded)
        if end == len(ids):
            break
        i += stride
    return windows, max_tokens  # all windows are capped at max_tokens


def safe_str(v) -> str:
    if v is None:
        return ""
    return str(v)


def make_chunk(record: dict, text: str, token_count: int, chunk_index: int, chunk_total: int, chunk_id: str) -> dict:
    """Build one output chunk dict from a sections.jsonl record."""
    is_complete = chunk_total == 1

    return {
        # --- Identity ---
        "chunk_id": chunk_id,
        "prev_chunk_id": "",       # set in post-processing
        "next_chunk_id": "",       # set in post-processing

        # --- Native section fields (kept verbatim for downstream scripts) ---
        "act_slug": record["act_slug"],
        "act_title": record["act_title"],
        "act_version": safe_str(record.get("act_version")),
        "frbr_work_id": safe_str(record.get("frbr_work_id")),
        "breadcrumb": safe_str(record.get("breadcrumb")),
        "section_num": safe_str(record.get("section_num")),
        "section_heading": safe_str(record.get("section_heading")),
        "part_num": safe_str(record.get("part_num")),
        "part_heading": safe_str(record.get("part_heading")),
        "chapter_num": safe_str(record.get("chapter_num")),
        "chapter_heading": safe_str(record.get("chapter_heading")),
        "is_schedule": bool(record.get("is_schedule", False)),
        "schedule_title": safe_str(record.get("schedule_title")),
        "status": safe_str(record.get("status", "current")),
        "references": ",".join(
            ref.get("target_href", "") for ref in (record.get("references") or [])
            if isinstance(ref, dict)
        ),

        # --- Text (the string that gets embedded) ---
        "text": text,
        "token_count": token_count,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "is_complete_section": is_complete,

        # --- Compatibility keys: names RulesRetriever.py already reads ---
        "article_number": safe_str(record.get("section_num")),
        "article_header": safe_str(record.get("section_heading")),
        "part": safe_str(record.get("part_num")),
        "part_title": safe_str(record.get("part_heading")),
        "chapter": safe_str(record.get("chapter_num")),
        "chapter_title": safe_str(record.get("chapter_heading")),
        "source_document": record["act_title"],   # matched by STATUTE_MAP → act_title
        "document_type": record["act_slug"],       # grouping key + DOC_TYPE_LABELS lookup

        # --- Compatibility keys from original constitution digester schema ---
        "total_chunks": chunk_total,
        "is_complete_article": is_complete,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = load_tokenizer()

    # --- Read input ---
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Read {len(records)} input records from {input_path.name}")

    # --- Group records by (act_slug, section_num) so we can link
    #     parser-level pre-splits as well as our own sub-splits ---
    # Each record is already a portion of a section (chunk_index within chunk_total).
    # We process each record independently for tokenisation, then link all
    # resulting output chunks within the same section group.

    # section_key → ordered list of output chunks (in document order)
    section_output: dict = defaultdict(list)

    total_over_limit = 0
    total_splits = 0
    all_output_order = []  # (section_key, chunk) in processing order

    for record in records:
        section_key = (record["act_slug"], safe_str(record.get("act_version")), safe_str(record.get("section_num")))
        original_chunk_id = record["chunk_id"]
        original_chunk_index = record.get("chunk_index", 0)

        text = (record.get("embedding_text") or record.get("text") or "").strip()
        if not text:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        actual_tokens = len(ids)

        if actual_tokens <= MAX_TOKENS:
            chunk = make_chunk(record, text, actual_tokens, original_chunk_index, record.get("chunk_total", 1), original_chunk_id)
            section_output[section_key].append((original_chunk_index, 0, chunk))
            all_output_order.append(chunk)
        else:
            total_over_limit += 1
            windows, _ = split_text(text, tokenizer)
            total_splits += len(windows)
            print(f"  Split {original_chunk_id!r}: {actual_tokens} tokens → {len(windows)} sub-chunks")

            for sub_i, window_text in enumerate(windows):
                sub_tokens = len(tokenizer.encode(window_text, add_special_tokens=False))
                sub_id = f"{original_chunk_id}__c{sub_i}"
                chunk = make_chunk(record, window_text, sub_tokens, original_chunk_index, record.get("chunk_total", 1), sub_id)
                chunk["chunk_index"] = sub_i
                chunk["chunk_total"] = len(windows)
                chunk["is_complete_section"] = len(windows) == 1
                chunk["is_complete_article"] = len(windows) == 1
                chunk["total_chunks"] = len(windows)
                section_output[section_key].append((original_chunk_index, sub_i, chunk))
                all_output_order.append(chunk)

    print(f"\n{total_over_limit} records exceeded {MAX_TOKENS} tokens and were split into {total_splits} sub-chunks.")

    # --- Link prev/next within each section group ---
    # Sort by (original_chunk_index, sub_index) to get document order.
    for section_key, items in section_output.items():
        items_sorted = sorted(items, key=lambda x: (x[0], x[1]))
        chunks = [item[2] for item in items_sorted]
        for i, chunk in enumerate(chunks):
            chunk["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else ""
            chunk["next_chunk_id"] = chunks[i + 1]["chunk_id"] if i < len(chunks) - 1 else ""

    # --- Write output in the same order records were processed ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for chunk in all_output_order:
            f.write(json.dumps(chunk) + "\n")

    print(f"Wrote {len(all_output_order)} chunks → {output_path}")


if __name__ == "__main__":
    main()
