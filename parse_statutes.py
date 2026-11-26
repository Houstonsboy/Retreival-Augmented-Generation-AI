#!/usr/bin/env python3
"""
Parse Kenyan statute HTML (Akoma Ntoso / Laws.Africa) into clean JSONL records
for GraphRAG ingestion.

Outputs written to data/processed/statutes/:
  sections.jsonl     — embedding units, one per section (chunked if > MAX_TOKENS)
  structure.jsonl    — Neo4j nodes (Act, Part, Chapter, Section, Schedule, DefinedTerm)
                       and edges (HAS_PART, HAS_CHAPTER, HAS_SECTION, HAS_SCHEDULE,
                       NEXT_SECTION, DEFINES)
  references.jsonl   — cross-reference REFERS_TO edges
  definitions.jsonl  — defined terms from Interpretation / Definitions sections

Usage:
  python parse_statutes.py
  python parse_statutes.py --slug evidence_act
  python parse_statutes.py --validate
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag

# ── Paths ─────────────────────────────────────────────────────────────────────
MANIFEST_PATH = Path("data/raw_statutes/manifest.json")
RAW_DIR       = Path("data/raw_statutes")
OUTPUT_DIR    = Path("data/processed/statutes")

# ── Thresholds ────────────────────────────────────────────────────────────────
MAX_TOKENS   = 800
TOKEN_FACTOR = 1.35   # word-count → token estimate for Kenyan legal text

# ── Compiled class patterns ───────────────────────────────────────────────────
CLS_ACT         = re.compile(r"\bakn-act\b")
CLS_BODY        = re.compile(r"\bakn-body\b")
CLS_CHAPTER     = re.compile(r"\bakn-chapter\b")
CLS_PART        = re.compile(r"\bakn-part\b")
CLS_SECTION     = re.compile(r"\bakn-section\b")
CLS_SUBSECTION  = re.compile(r"\bakn-subsection\b")
CLS_PARAGRAPH   = re.compile(r"\bakn-paragraph\b")
CLS_CONTENT     = re.compile(r"\bakn-content\b")
CLS_SUBPART     = re.compile(r"\bakn-subpart\b")
CLS_ATTACHMENTS = re.compile(r"\bakn-attachments\b")
CLS_ATTACHMENT  = re.compile(r"\bakn-attachment\b")
CLS_REMARK      = re.compile(r"\bakn-remark\b")
CLS_CROSSHEAD   = re.compile(r"\bakn-crossHeading\b")
CLS_REF         = re.compile(r"\bakn-ref\b")
CLS_TABLE       = re.compile(r"\bakn-table\b")
CLS_TR          = re.compile(r"\bakn-tr\b")
CLS_CELL        = re.compile(r"\b(akn-th|akn-td)\b")

# ── Text patterns ─────────────────────────────────────────────────────────────
STATUS_RE = re.compile(
    r"\b(repealed|spent|deleted|omitted|expired)\b", re.IGNORECASE
)
SECTION_HDG_RE = re.compile(r"^(\d+[A-Za-z]*)\s*\.\s*(.*)")
STRUCT_NUM_RE  = re.compile(
    r"(?:part|chapter|division)\s+([IVXLCDM\d]+)", re.IGNORECASE
)
STRUCT_TITLE_RE = re.compile(
    r"(?:part|chapter|division)\s+[IVXLCDM\d]+\s*[-–—]\s*(.*)",
    re.IGNORECASE,
)
DEFINITION_RE = re.compile(
    r'"([^"]{1,80})"\s+(?:means|includes|has the meaning|shall mean|shall include)',
    re.IGNORECASE,
)
SAFE_RE = re.compile(r"[^\w]")


# ── Utilities ─────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * TOKEN_FACTOR))


def detect_status(heading: str) -> str:
    m = STATUS_RE.search(heading)
    if not m:
        return "current"
    return "spent" if m.group(1).lower() in ("spent", "expired") else "repealed"


def safe_id(s: str) -> str:
    return SAFE_RE.sub("_", s)


# ── HTML → text ───────────────────────────────────────────────────────────────

def table_to_markdown(tbl: Tag) -> str:
    rows = []
    for tr in tbl.find_all(class_=CLS_TR):
        cells = [
            td.get_text(" ", strip=True).replace("|", "\\|")
            for td in tr.find_all(class_=CLS_CELL)
        ]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    ncols = rows[0].count("|") - 1
    sep   = "| " + " | ".join(["---"] * ncols) + " |"
    return "\n".join([rows[0], sep] + rows[1:]) if len(rows) > 1 else rows[0]


def element_text(el: Tag, include_remarks: bool) -> str:
    """
    Recursively extract legal text from an AKN element.
    - h2/h3/h4: skipped (caller prepends the heading line explicitly)
    - akn-table: converted to Markdown
    - akn-remark: wrapped in [...] if include_remarks, else dropped
    - akn-crossHeading: surrounded by newlines
    - everything else: recurse and collect text
    """
    parts: list[str] = []

    for child in el.children:
        if isinstance(child, NavigableString):
            t = str(child)
            if t.strip():
                parts.append(t)
            continue
        if not isinstance(child, Tag):
            continue

        tag = child.name
        cls = " ".join(child.get("class", []))

        if tag in ("h2", "h3", "h4"):
            continue
        elif "akn-table" in cls or tag == "table":
            md = table_to_markdown(child)
            if md:
                parts.append("\n" + md + "\n")
        elif "akn-remark" in cls:
            if include_remarks:
                parts.append(f" [{child.get_text(' ', strip=True)}]")
        elif "akn-crossHeading" in cls:
            parts.append(f"\n{child.get_text(' ', strip=True)}\n")
        else:
            parts.append(element_text(child, include_remarks))

    raw = "".join(parts)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def section_texts(heading_line: str, sec_el: Tag) -> tuple[str, str]:
    """
    Returns (full_text, emb_text).
    full_text includes amendment remarks; emb_text omits them.
    The heading line is prepended to both.
    """
    full = heading_line + "\n" + element_text(sec_el, include_remarks=True)
    emb  = heading_line + "\n" + element_text(sec_el, include_remarks=False)
    return full.strip(), emb.strip()


def get_refs(el: Tag, chunk_id: str) -> list[dict]:
    return [
        {
            "source_chunk_id": chunk_id,
            "target_href": ref.get("href", ""),
            "target_text": ref.get_text(" ", strip=True),
            "resolved": ref.get("href", "").startswith("/akn/"),
        }
        for ref in el.find_all(class_=CLS_REF)
        if ref.get("href", "")
    ]


# ── Definitions ───────────────────────────────────────────────────────────────

def extract_definitions(
    section_num: str, heading: str, full_text: str,
    act_slug: str, chunk_id: str,
) -> list[dict]:
    if not re.search(r"\b(interpretation|definitions?)\b", heading, re.IGNORECASE):
        return []
    matches = list(DEFINITION_RE.finditer(full_text))
    defs = []
    for i, m in enumerate(matches):
        term  = m.group(1).strip()
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else min(start + 700, len(full_text))
        defs.append({
            "chunk_id":       chunk_id,
            "act_slug":       act_slug,
            "section_num":    section_num,
            "term":           term,
            "definition_text": full_text[start:end].strip(),
        })
    return defs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_section(base_id: str, template: dict, sec_el: Tag) -> list[dict]:
    """
    If token_count <= MAX_TOKENS return single chunk.
    Otherwise split at akn-subsection (or akn-paragraph) boundaries.
    Each chunk gets chunk_id = base_id + f"__{index}".
    """
    if template["token_count"] <= MAX_TOKENS:
        c = {**template, "chunk_id": base_id + "__0", "chunk_index": 0, "chunk_total": 1}
        return [c]

    # Locate split boundaries
    splits = sec_el.find_all(class_=CLS_SUBSECTION, recursive=False)
    if not splits:
        inner = sec_el.find(class_=CLS_CONTENT)
        if inner:
            splits = inner.find_all(class_=CLS_SUBSECTION, recursive=False)
    if not splits:
        splits = sec_el.find_all(class_=CLS_PARAGRAPH, recursive=False)

    if len(splits) < 2:
        c = {**template, "chunk_id": base_id + "__0", "chunk_index": 0, "chunk_total": 1}
        return [c]

    chunks: list[dict] = []
    buf_full: list[str] = []
    buf_emb:  list[str] = []

    def flush() -> None:
        if not buf_full:
            return
        idx    = len(chunks)
        cid    = f"{base_id}__{idx}"
        f_text = "\n".join(buf_full)
        e_text = "\n".join(buf_emb)
        chunks.append({
            **template,
            "chunk_id":       cid,
            "text":           f_text,
            "embedding_text": template["breadcrumb"] + "\n" + e_text,
            "token_count":    estimate_tokens(e_text),
            "chunk_index":    idx,
            "chunk_total":    -1,
            "references":     [],
        })
        buf_full.clear()
        buf_emb.clear()

    for sub in splits:
        ft = element_text(sub, include_remarks=True).strip()
        et = element_text(sub, include_remarks=False).strip()
        if estimate_tokens("\n".join(buf_emb + [et])) > MAX_TOKENS and buf_emb:
            flush()
        buf_full.append(ft)
        buf_emb.append(et)

    flush()

    total = len(chunks)
    for c in chunks:
        c["chunk_total"] = total

    if not chunks:
        return [{**template, "chunk_id": base_id + "__0", "chunk_index": 0, "chunk_total": 1}]

    # Put all section-level refs in chunk 0 only
    chunks[0]["references"] = template["references"]
    return chunks


# ── Body walker ───────────────────────────────────────────────────────────────

def walk_act(body_el: Tag, akn_root: Tag, act_meta: dict) -> tuple[list, list, list, list]:
    """
    Walk akn-body → (chapter →)? (part →)? section.
    Returns (sections, structure, references, definitions).
    """
    slug    = act_meta["slug"]
    version = act_meta["version"]
    act_id  = f"act__{slug}__{version}"

    sections:    list[dict] = []
    structure:   list[dict] = [{
        "type":  "node",
        "label": "Act",
        "id":    act_id,
        "properties": {
            "slug":         slug,
            "title":        act_meta["title"],
            "version":      version,
            "frbr_work_id": act_meta.get("work", ""),
            "role":         act_meta.get("role", "core"),
        },
    }]
    references:  list[dict] = []
    definitions: list[dict] = []
    prev_sec_id: Optional[str] = None

    # ── Section processor (closure) ──────────────────────────────────────────
    def process_section(
        sec_el: Tag,
        chapter_num: Optional[str], chapter_heading: Optional[str],
        part_num:    Optional[str], part_heading:    Optional[str],
    ) -> None:
        nonlocal prev_sec_id

        h_el = sec_el.find(["h3", "h2"])
        if not h_el:
            return
        raw_h = h_el.get_text(" ", strip=True)
        m = SECTION_HDG_RE.match(raw_h)
        if not m:
            return

        sec_num   = m.group(1)
        sec_title = m.group(2).strip(" .")
        status    = detect_status(raw_h)

        bc_parts: list[str] = []
        if chapter_heading:
            bc_parts.append(chapter_heading)
        if part_heading:
            bc_parts.append(part_heading)
        bc_parts.append(f"Section {sec_num}")
        breadcrumb = " / ".join(bc_parts)

        base_id   = f"{slug}__{version}__s{safe_id(sec_num)}"
        full_text, emb_text = section_texts(raw_h, sec_el)
        tok = estimate_tokens(emb_text)

        template: dict = {
            "chunk_id":       "",
            "act_slug":       slug,
            "act_title":      act_meta["title"],
            "act_version":    version,
            "frbr_work_id":   act_meta.get("work", ""),
            "breadcrumb":     breadcrumb,
            "section_num":    sec_num,
            "section_heading": sec_title,
            "part_num":       part_num,
            "part_heading":   part_heading,
            "chapter_num":    chapter_num,
            "chapter_heading": chapter_heading,
            "is_schedule":    False,
            "schedule_title": None,
            "status":         status,
            "text":           full_text,
            "embedding_text": breadcrumb + "\n" + emb_text,
            "token_count":    tok,
            "chunk_index":    0,
            "chunk_total":    1,
            "references":     get_refs(sec_el, base_id + "__0"),
        }

        chunks = chunk_section(base_id, template, sec_el)
        sections.extend(chunks)
        references.extend(chunks[0].get("references", []))
        definitions.extend(
            extract_definitions(sec_num, sec_title, full_text, slug, chunks[0]["chunk_id"])
        )

        sec_node_id = f"section__{slug}__{safe_id(sec_num)}"
        structure.append({
            "type":  "node",
            "label": "Section",
            "id":    sec_node_id,
            "properties": {
                "num":     sec_num,
                "heading": sec_title,
                "status":  status,
                "act_slug": slug,
            },
        })
        structure.append({"type": "edge", "label": "HAS_SECTION",
                           "from": act_id, "to": sec_node_id})
        if prev_sec_id:
            structure.append({"type": "edge", "label": "NEXT_SECTION",
                               "from": prev_sec_id, "to": sec_node_id})
        prev_sec_id = sec_node_id

    # ── Part processor ───────────────────────────────────────────────────────
    def process_part(
        part_el: Tag,
        chapter_num: Optional[str], chapter_heading: Optional[str],
    ) -> None:
        p_h    = part_el.find("h2")
        p_text = p_h.get_text(" ", strip=True) if p_h else ""
        nm     = STRUCT_NUM_RE.search(p_text)
        tm     = STRUCT_TITLE_RE.search(p_text)
        p_num   = nm.group(1) if nm else ""
        p_title = tm.group(1).strip() if tm else p_text

        part_node_id = f"part__{slug}__{safe_id(p_num)}"
        structure.append({"type": "node", "label": "Part", "id": part_node_id,
                           "properties": {"num": p_num, "title": p_title, "act_slug": slug}})
        structure.append({"type": "edge", "label": "HAS_PART",
                           "from": act_id, "to": part_node_id})

        # Some Acts (e.g. Limitation Act Part III) use akn-subpart (A, B, C)
        # between the part and section level.
        subparts = part_el.find_all(class_=CLS_SUBPART, recursive=False)
        if subparts:
            for sp_el in subparts:
                for sec_el in sp_el.find_all(class_=CLS_SECTION, recursive=False):
                    process_section(sec_el, chapter_num, chapter_heading, p_num, p_title)
        else:
            for sec_el in part_el.find_all(class_=CLS_SECTION, recursive=False):
                process_section(sec_el, chapter_num, chapter_heading, p_num, p_title)

    # ── Structural dispatch ──────────────────────────────────────────────────
    chapters = body_el.find_all(class_=CLS_CHAPTER, recursive=False)

    if chapters:
        # Evidence Act: chapter → (part →)? section
        for chap_el in chapters:
            ch_h    = chap_el.find("h2")
            ch_text = ch_h.get_text(" ", strip=True) if ch_h else ""
            cn      = STRUCT_NUM_RE.search(ch_text)
            ct      = STRUCT_TITLE_RE.search(ch_text)
            ch_num   = cn.group(1) if cn else ""
            ch_title = ct.group(1).strip() if ct else ch_text

            chap_node_id = f"chapter__{slug}__{safe_id(ch_num)}"
            structure.extend([
                {"type": "node", "label": "Chapter", "id": chap_node_id,
                 "properties": {"num": ch_num, "title": ch_title, "act_slug": slug}},
                {"type": "edge", "label": "HAS_CHAPTER",
                 "from": act_id, "to": chap_node_id},
            ])

            parts = chap_el.find_all(class_=CLS_PART, recursive=False)
            if parts:
                for part_el in parts:
                    process_part(part_el, ch_num, ch_title)
            else:
                for sec_el in chap_el.find_all(class_=CLS_SECTION, recursive=False):
                    process_section(sec_el, ch_num, ch_title, None, None)
    else:
        parts = body_el.find_all(class_=CLS_PART, recursive=False)
        if parts:
            for part_el in parts:
                process_part(part_el, None, None)
        else:
            for sec_el in body_el.find_all(class_=CLS_SECTION, recursive=False):
                process_section(sec_el, None, None, None, None)

    # ── Schedules ────────────────────────────────────────────────────────────
    atts_el = akn_root.find(class_=CLS_ATTACHMENTS)
    if atts_el:
        for i, att in enumerate(
            atts_el.find_all(class_=CLS_ATTACHMENT, recursive=False), 1
        ):
            h_el      = att.find(["h1", "h2", "h3"])
            sch_title = h_el.get_text(" ", strip=True) if h_el else f"Schedule {i}"
            sch_num   = str(i)
            sch_id    = f"{slug}__{version}__sch{sch_num}__0"

            full_t = element_text(att, include_remarks=True).strip()
            emb_t  = element_text(att, include_remarks=False).strip()

            sections.append({
                "chunk_id":        sch_id,
                "act_slug":        slug,
                "act_title":       act_meta["title"],
                "act_version":     version,
                "frbr_work_id":    act_meta.get("work", ""),
                "breadcrumb":      sch_title,
                "section_num":     f"SCH{sch_num}",
                "section_heading": sch_title,
                "part_num":        None,
                "part_heading":    None,
                "chapter_num":     None,
                "chapter_heading": None,
                "is_schedule":     True,
                "schedule_title":  sch_title,
                "status":          "current",
                "text":            full_t,
                "embedding_text":  sch_title + "\n" + emb_t,
                "token_count":     estimate_tokens(emb_t),
                "chunk_index":     0,
                "chunk_total":     1,
                "references":      get_refs(att, sch_id),
            })
            sch_node_id = f"schedule__{slug}__{sch_num}"
            structure.extend([
                {"type": "node", "label": "Schedule", "id": sch_node_id,
                 "properties": {"num": sch_num, "title": sch_title, "act_slug": slug}},
                {"type": "edge", "label": "HAS_SCHEDULE",
                 "from": act_id, "to": sch_node_id},
            ])

    # ── DEFINES edges ────────────────────────────────────────────────────────
    for d in definitions:
        t_hash       = hashlib.md5(d["term"].lower().encode()).hexdigest()[:8]
        term_node_id = f"term__{slug}__{t_hash}"
        sec_node_id  = f"section__{slug}__{safe_id(d['section_num'])}"
        structure.extend([
            {"type": "node", "label": "DefinedTerm", "id": term_node_id,
             "properties": {"term": d["term"], "act_slug": slug}},
            {"type": "edge", "label": "DEFINES",
             "from": sec_node_id, "to": term_node_id},
        ])

    return sections, structure, references, definitions


# ── Per-Act entry point ───────────────────────────────────────────────────────

def parse_act(html_path: Path, entry: dict) -> dict:
    print(f"  {entry['slug']} ...", end=" ", flush=True)
    html  = html_path.read_text(encoding="utf-8", errors="replace")
    soup  = BeautifulSoup(html, "lxml")

    akn_root = soup.find(class_=CLS_ACT)
    if not akn_root:
        print("WARN: akn-act not found")
        return {"sections": [], "structure": [], "references": [], "definitions": []}

    body_el = akn_root.find(class_=CLS_BODY)
    if not body_el:
        print("WARN: akn-body not found")
        return {"sections": [], "structure": [], "references": [], "definitions": []}

    act_meta = {
        "slug":    entry["slug"],
        "title":   entry["title"],
        "version": entry.get("version", "unknown"),
        "work":    entry.get("work", ""),
        "role":    entry.get("role", "core"),
    }

    secs, struct, refs, defs = walk_act(body_el, akn_root, act_meta)
    print(f"{len(secs)} chunks  {len(defs)} defs  {len(refs)} refs")
    return {"sections": secs, "structure": struct, "references": refs, "definitions": defs}


# ── I/O ───────────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"    wrote {len(records):>6,} records  →  {path}")


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(manifest: list, output_dir: Path) -> None:
    import random

    sec_path = output_dir / "sections.jsonl"
    if not sec_path.exists():
        print("[!] sections.jsonl not found — run parser first")
        return

    records = [
        json.loads(line)
        for line in sec_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print("\n" + "=" * 64)
    print("VALIDATION REPORT")
    print("=" * 64)

    by_slug: dict[str, list] = {}
    for r in records:
        by_slug.setdefault(r["act_slug"], []).append(r)
    m_by_slug = {m["slug"]: m for m in manifest}

    # 1 ── Section count vs manifest
    print("\n[1] Section counts:")
    for slug in sorted(by_slug):
        parsed   = sum(1 for r in by_slug[slug]
                       if not r["is_schedule"] and r["chunk_index"] == 0)
        expected = m_by_slug.get(slug, {}).get("section_count", -1)
        flag     = "OK" if parsed == expected else f"DIFF  manifest={expected}"
        print(f"  {slug:<48}  parsed={parsed:>3}  [{flag}]")

    # 2 ── Chrome leakage
    print("\n[2] Chrome leakage (checking first 300 chars of embedding_text):")
    # Use tightly-scoped phrases unlikely to appear in statutory text
    CHROME = {"navbar", "sign in", "download pdf", "cookie policy", "table of contents"}
    leaks  = [r["chunk_id"] for r in records
              for w in CHROME if w in r["embedding_text"].lower()[:300]]
    print(f"  {'OK — none detected' if not leaks else '[!] ' + str(leaks[:5])}")

    # 3 ── Data quality
    print("\n[3] Data quality:")
    all_ids = [r["chunk_id"] for r in records]
    dupes   = [cid for cid in set(all_ids) if all_ids.count(cid) > 1]
    empty   = [r["chunk_id"] for r in records if not r["text"].strip()]
    no_num  = [r["chunk_id"] for r in records if not r["section_num"]]
    print(f"  Duplicate chunk_ids : {dupes[:5] or 'none'}")
    print(f"  Empty text          : {empty[:5] or 'none'}")
    print(f"  Missing section_num : {no_num[:5] or 'none'}")

    # 4 ── Spot-checks
    print("\n[4] Spot-checks:")
    CHECKS = [
        ("limitation_of_actions_act", "4",    "contract",           "Limitation s.4"),
        ("evidence_act",              "106B",  "electronic",         "Evidence s.106B"),
        ("small_claims_court_act",    "12",    "jurisdiction",       "SCC s.12"),
    ]
    for slug, num, kw, label in CHECKS:
        hits = [r for r in by_slug.get(slug, [])
                if r["section_num"] == num and r["chunk_index"] == 0]
        if not hits:
            print(f"  [!] {label}: NOT FOUND")
        elif kw.lower() not in hits[0]["text"].lower():
            print(f"  [!] {label}: found but '{kw}' absent from text")
        else:
            preview = hits[0]["text"][:120].replace("\n", " ")
            print(f"  OK  {label}: {preview}…")

    # 5 ── Random samples
    print("\n[5] Random samples (3 per Act, first 4 Acts):")
    for slug in sorted(by_slug)[:4]:
        recs    = by_slug[slug]
        samples = random.sample(recs, min(3, len(recs)))
        print(f"\n  ── {slug} ──")
        for s in samples:
            print(f"    {s['chunk_id']}  tok={s['token_count']}  status={s['status']}")
            print(f"    bc: {s['breadcrumb']}")
            print(f"    {s['text'][:90].replace(chr(10), ' ')}…")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--slug",     help="Parse only this Act slug")
    ap.add_argument("--validate", action="store_true",
                    help="Validation report on existing output; no parsing")
    args = ap.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"[!] Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if args.validate:
        run_validation(manifest, OUTPUT_DIR)
        return

    targets = (
        [m for m in manifest if m["slug"] == args.slug] if args.slug else manifest
    )
    if not targets:
        print(f"[!] Slug '{args.slug}' not in manifest")
        sys.exit(1)

    all_sec, all_struct, all_refs, all_defs = [], [], [], []

    print(f"[*] Parsing {len(targets)} Acts\n")
    for entry in targets:
        html_path = RAW_DIR / entry["file"]
        if not html_path.exists():
            print(f"  [!] Missing file: {html_path}")
            continue
        result = parse_act(html_path, entry)
        all_sec.extend(result["sections"])
        all_struct.extend(result["structure"])
        all_refs.extend(result["references"])
        all_defs.extend(result["definitions"])

    print(f"\n[*] Writing to {OUTPUT_DIR}/")
    write_jsonl(OUTPUT_DIR / "sections.jsonl",    all_sec)
    write_jsonl(OUTPUT_DIR / "structure.jsonl",   all_struct)
    write_jsonl(OUTPUT_DIR / "references.jsonl",  all_refs)
    write_jsonl(OUTPUT_DIR / "definitions.jsonl", all_defs)

    print("\n[*] Validation")
    run_validation(manifest, OUTPUT_DIR)


if __name__ == "__main__":
    main()
