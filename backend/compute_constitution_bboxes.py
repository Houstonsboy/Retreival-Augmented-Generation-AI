"""
Compute bounding boxes for constitution chunks.

The page_start values in locations.db are unreliable (they pointed to the ToC).
This script searches ALL pages to find the best matching page, then does
word-level bbox extraction on that page.

Strategy:
1. Build a full-text index of all 166 pages (done once).
2. For each chunk, find the best-matching page via fuzz.partial_ratio.
3. On the best page, do sliding-window word alignment.
4. Store boxes + match_score + correct page_start in locations.db.
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import List, Tuple

import fitz
from rapidfuzz import fuzz

DB_PATH = Path(__file__).parent / "locations.db"
CONSTITUTION_PDF = Path(__file__).parent / "MasterRules" / "TheConstitutionOfKenya.pdf"
MR_CHROMA = Path(__file__).parent / "constitutionlogic" / "MasterRulesDB"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def align_chunk_on_page(
    chunk_text: str,
    words: List[Tuple],
    page_num: int,
) -> Tuple[List[dict], float]:
    """
    Sliding-window fuzzy alignment of chunk_text against word list from one page.
    Returns (boxes, score).
    """
    norm_chunk = _normalize(chunk_text)
    chunk_tokens = norm_chunk.split()
    n = len(chunk_tokens)
    if n == 0 or not words:
        return [], 0.0

    norm_words = [_normalize(w[4]) for w in words]
    W = len(norm_words)
    if W == 0:
        return [], 0.0

    best_score = 0.0
    best_start = 0
    best_len = n

    win_lo = max(1, int(n * 0.7))
    win_hi = min(W, int(n * 1.4))

    for wlen in range(win_lo, win_hi + 1):
        for start in range(0, W - wlen + 1):
            candidate = " ".join(norm_words[start : start + wlen])
            score = fuzz.ratio(norm_chunk, candidate) / 100.0
            if score > best_score:
                best_score = score
                best_start = start
                best_len = wlen
            if best_score >= 0.99:
                break
        if best_score >= 0.99:
            break

    if best_score < 0.35:
        return [], best_score

    matched = words[best_start : best_start + best_len]
    x0 = min(w[0] for w in matched)
    y0 = min(w[1] for w in matched)
    x1 = max(w[2] for w in matched)
    y1 = max(w[3] for w in matched)
    return [{"page": page_num, "bbox": [x0, y0, x1, y1]}], best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not CONSTITUTION_PDF.exists():
        print(f"ERROR: Constitution PDF not found: {CONSTITUTION_PDF}")
        return

    doc = fitz.open(str(CONSTITUTION_PDF))
    n_pages = len(doc)
    print(f"Constitution PDF: {n_pages} pages")

    # Build full-text index for all pages (for fast page-level search)
    print("Building page text index...")
    page_texts: List[str] = []
    page_words: List[List[Tuple]] = []
    for pg in range(n_pages):
        page = doc[pg]
        words = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
        page_words.append(words)
        page_texts.append(_normalize(" ".join(w[4] for w in words)))
    print(f"  Indexed {n_pages} pages.")

    con = sqlite3.connect(str(DB_PATH))

    # Add columns if missing
    cols = {c[1] for c in con.execute("PRAGMA table_info(chunk_locations)").fetchall()}
    for col_def in [("match_score", "REAL"), ("best_page", "INTEGER")]:
        if col_def[0] not in cols:
            con.execute(f"ALTER TABLE chunk_locations ADD COLUMN {col_def[0]} {col_def[1]}")
    con.commit()

    rows = con.execute("""
        SELECT chunk_id, page_start, page_end, boxes_json
        FROM chunk_locations WHERE collection='master_rules'
        ORDER BY chunk_id
    """).fetchall()

    # Load chunk texts from Chroma
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(MR_CHROMA), settings=Settings(anonymized_telemetry=False))
    col_chroma = client.get_collection("master_rules")

    id_to_row = {r[0]: r for r in rows}
    chunk_ids = [r[0] for r in rows]

    all_scores: List[Tuple[str, float]] = []
    below_threshold: List[Tuple[str, float]] = []
    computed = 0
    skipped = 0

    BATCH = 100
    for i in range(0, len(chunk_ids), BATCH):
        batch_ids = chunk_ids[i : i + BATCH]
        result = col_chroma.get(ids=batch_ids, include=["documents"])
        for cid, text in zip(result["ids"], result["documents"]):
            _, ps, pe, bj = id_to_row[cid]
            existing = json.loads(bj) if bj else []
            if existing and not args.overwrite:
                skipped += 1
                continue

            norm_chunk = _normalize(text)

            # Step 1: Find best page via partial_ratio (fast)
            page_scores = []
            for pg in range(n_pages):
                if not page_texts[pg]:
                    continue
                s = fuzz.partial_ratio(norm_chunk, page_texts[pg]) / 100.0
                page_scores.append((s, pg))

            if not page_scores:
                con.execute(
                    "UPDATE chunk_locations SET boxes_json=?, match_score=? WHERE chunk_id=? AND collection='master_rules'",
                    ("[]", 0.0, cid),
                )
                all_scores.append((cid, 0.0))
                below_threshold.append((cid, 0.0))
                computed += 1
                continue

            page_scores.sort(reverse=True)
            # Try top 3 candidate pages with full word alignment
            best_boxes: List[dict] = []
            best_score = 0.0
            best_pg = page_scores[0][1]

            for _, pg in page_scores[:3]:
                boxes, score = align_chunk_on_page(text, page_words[pg], pg)
                if score > best_score:
                    best_score = score
                    best_boxes = boxes
                    best_pg = pg
                if best_score >= 0.99:
                    break

            all_scores.append((cid, best_score))
            if best_score < args.threshold:
                below_threshold.append((cid, best_score))
                best_boxes = []

            con.execute(
                """UPDATE chunk_locations
                   SET boxes_json=?, match_score=?, best_page=?, page_start=?, page_end=?
                   WHERE chunk_id=? AND collection='master_rules'""",
                (json.dumps(best_boxes), best_score, best_pg, best_pg + 1, best_pg + 1, cid),
            )
            computed += 1

        con.commit()
        print(f"  {min(i + BATCH, len(chunk_ids))}/{len(chunk_ids)} processed...")

    con.close()
    doc.close()

    total = len(rows)
    matched_n = sum(1 for _, s in all_scores if s >= args.threshold)
    match_rate = matched_n / max(len(all_scores), 1)
    avg_score = sum(s for _, s in all_scores) / max(len(all_scores), 1)

    print(f"\n=== Constitution bbox alignment complete ===")
    print(f"Total:    {total}")
    print(f"Computed: {computed}  |  Skipped: {skipped}")
    print(f"Match rate (>= {args.threshold}): {match_rate:.1%}  ({matched_n}/{len(all_scores)})")
    print(f"Average score: {avg_score:.3f}")
    print(f"Below {args.threshold}: {len(below_threshold)}")
    if below_threshold:
        print("  Worst chunks:")
        for cid, s in sorted(below_threshold, key=lambda x: x[1])[:15]:
            print(f"    {cid}: {s:.3f}")


if __name__ == "__main__":
    main()
