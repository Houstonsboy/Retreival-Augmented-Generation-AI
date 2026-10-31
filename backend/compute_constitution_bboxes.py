"""
Fast constitution bbox alignment via article-number keyword search.

Strategy:
1. For each chunk, extract its article number from the chunk_id (constitution_art{N}_{idx})
2. Find pages containing the article pattern "^N." or "^N\." or article header keywords
3. On the matching page, do word-level sliding-window alignment
4. Fall back to first-occurrence search if article-number search fails

This is ~100x faster than full partial_ratio over all pages.
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional

import fitz
from rapidfuzz import fuzz

DB_PATH = Path(__file__).parent / "locations.db"
CONSTITUTION_PDF = Path(__file__).parent / "MasterRules" / "TheConstitutionOfKenya.pdf"
MR_CHROMA = Path(__file__).parent / "constitutionlogic" / "MasterRulesDB"


def _normalize(text: str) -> str:
    return re.sub(r"[\s\xa0]+", " ", text.lower()).strip()


def _get_page_words(doc: fitz.Document, page_num: int) -> List[Tuple]:
    return [(w[0], w[1], w[2], w[3], w[4]) for w in doc[page_num].get_text("words")]


def _get_page_text(doc: fitz.Document, page_num: int) -> str:
    return _normalize(doc[page_num].get_text())


def find_article_page(doc: fitz.Document, article_num: str, page_texts: List[str]) -> Optional[int]:
    """
    Find the page containing a given article number.
    Skips TOC pages (pages before the preamble, typically pages 1-12).
    """
    # Article numbers in the PDF look like "N." at the start of a line
    patterns = [
        re.compile(r"\b" + re.escape(article_num) + r"\s*[\.\xa0]", re.IGNORECASE),
        re.compile(r"\b" + re.escape(article_num) + r"\s*\(1\)", re.IGNORECASE),
    ]
    # Skip first 12 pages (TOC)
    for pg in range(12, len(page_texts)):
        if not page_texts[pg]:
            continue
        raw = doc[pg].get_text()
        for pat in patterns:
            if pat.search(raw):
                # Confirm article content (not just a TOC reference) by checking
                # that the next few chars look like article text, not just a page number
                m = pat.search(raw)
                if m:
                    context = raw[m.start():m.start() + 60]
                    # Context should have some lowercase words (article text), not just "N. Title"
                    if len(context.split()) >= 4:
                        return pg
    return None


def align_on_page(chunk_text: str, words: List[Tuple], page_num: int) -> Tuple[List[dict], float]:
    """Sliding-window fuzzy alignment. Returns (boxes, score)."""
    norm_chunk = _normalize(chunk_text)
    chunk_tokens = norm_chunk.split()
    n = len(chunk_tokens)
    if n == 0 or not words:
        return [], 0.0

    norm_words = [_normalize(w[4]) for w in words]
    W = len(norm_words)
    if W == 0:
        return [], 0.0

    # Use token_set_ratio for the best page-level match to select window
    page_text = " ".join(norm_words)
    page_score = fuzz.partial_ratio(norm_chunk, page_text) / 100.0
    if page_score < 0.45:
        return [], page_score

    # Find best window with ratio
    best_score = 0.0
    best_start = 0
    best_len = n
    win_lo = max(1, int(n * 0.7))
    win_hi = min(W, int(n * 1.5))

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
        return [], page_score  # return page-level score as informational

    matched = words[best_start : best_start + best_len]
    x0 = min(w[0] for w in matched)
    y0 = min(w[1] for w in matched)
    x1 = max(w[2] for w in matched)
    y1 = max(w[3] for w in matched)
    return [{"page": page_num, "bbox": [x0, y0, x1, y1]}], best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not CONSTITUTION_PDF.exists():
        print(f"ERROR: {CONSTITUTION_PDF}")
        return

    doc = fitz.open(str(CONSTITUTION_PDF))
    n_pages = len(doc)
    print(f"Constitution PDF: {n_pages} pages")

    print("Building page text index...")
    page_texts: List[str] = [_normalize(doc[pg].get_text()) for pg in range(n_pages)]
    page_words: List[Optional[List[Tuple]]] = [None] * n_pages
    print(f"  Done. Pages with text: {sum(1 for t in page_texts if t)}/{n_pages}")

    con = sqlite3.connect(str(DB_PATH), timeout=30)
    # Add columns if missing
    cols = {c[1] for c in con.execute("PRAGMA table_info(chunk_locations)").fetchall()}
    for col_name, col_type in [("match_score", "REAL"), ("best_page", "INTEGER")]:
        if col_name not in cols:
            con.execute(f"ALTER TABLE chunk_locations ADD COLUMN {col_name} {col_type}")
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

    # Build article-to-first-page cache
    art_page_cache: dict = {}

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

            # Extract article number from chunk_id: constitution_art{N}_{idx}
            m = re.match(r"constitution_art(\d+)_", cid)
            article_num = m.group(1) if m else None

            best_pg: Optional[int] = None

            if article_num:
                if article_num not in art_page_cache:
                    art_page_cache[article_num] = find_article_page(doc, article_num, page_texts)
                best_pg = art_page_cache[article_num]

            # Fallback: search a few pages around page_start
            if best_pg is None and ps:
                # Try ±3 pages from recorded page_start
                candidates = list(range(max(0, ps - 4), min(n_pages, ps + 4)))
                best_score_candidate = 0.0
                for pg in candidates:
                    if not page_texts[pg]:
                        continue
                    score = fuzz.partial_ratio(_normalize(text[:300]), page_texts[pg]) / 100.0
                    if score > best_score_candidate:
                        best_score_candidate = score
                        best_pg = pg

            if best_pg is None:
                all_scores.append((cid, 0.0))
                below_threshold.append((cid, 0.0))
                con.execute(
                    "UPDATE chunk_locations SET boxes_json=?, match_score=? WHERE chunk_id=? AND collection='master_rules'",
                    ("[]", 0.0, cid),
                )
                computed += 1
                continue

            # Lazy-load words for that page
            if page_words[best_pg] is None:
                page_words[best_pg] = _get_page_words(doc, best_pg)

            words = page_words[best_pg]
            boxes, score = align_on_page(text, words, best_pg)
            all_scores.append((cid, score))

            if score < args.threshold:
                below_threshold.append((cid, score))
                boxes = []

            con.execute(
                """UPDATE chunk_locations
                   SET boxes_json=?, match_score=?, best_page=?, page_start=?, page_end=?
                   WHERE chunk_id=? AND collection='master_rules'""",
                (json.dumps(boxes), score, best_pg, best_pg + 1, best_pg + 1, cid),
            )
            computed += 1

        con.commit()
        print(f"  {min(i + BATCH, len(chunk_ids))}/{len(chunk_ids)}...")

    con.close()
    doc.close()

    total = len(rows)
    matched_n = sum(1 for _, s in all_scores if s >= args.threshold)
    match_rate = matched_n / max(len(all_scores), 1)
    avg_score = sum(s for _, s in all_scores) / max(len(all_scores), 1)

    print(f"\n=== Constitution bbox alignment ===")
    print(f"Total: {total}, Computed: {computed}, Skipped: {skipped}")
    print(f"Match rate (>={args.threshold}): {match_rate:.1%} ({matched_n}/{len(all_scores)})")
    print(f"Average score: {avg_score:.3f}")
    print(f"Below {args.threshold}: {len(below_threshold)}")
    for cid, s in sorted(below_threshold, key=lambda x: x[1])[:20]:
        print(f"  {cid}: {s:.3f}")


if __name__ == "__main__":
    main()
