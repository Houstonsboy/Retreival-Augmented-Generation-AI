#!/usr/bin/env python3
"""
Download Kenyan statutes from new.kenyalaw.org (Akoma Ntoso HTML) for the
Small Claims Court GraphRAG pipeline.

Output:
    data/raw_statutes/<slug>__<version>.html   raw HTML per Act
    data/raw_statutes/manifest.json            provenance + validation info

Usage:
    python download_statutes.py           # skip files already downloaded
    python download_statutes.py --force   # re-download everything
"""
import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_DIR = Path("./data/raw_statutes")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
CRAWL_DELAY = 1.5   # seconds between requests
TIMEOUT = 30        # large Acts (CPA, Evidence Act) can be slow

BASE = "https://new.kenyalaw.org/akn/ke/act"

# role: "core" = substantive/procedural law the system reasons with
#       "exclusion" = used to detect claims outside SCC jurisdiction
STATUTE_TARGETS = [
    # ---- Core ----
    {"slug": "small_claims_court_act", "title": "Small Claims Court Act",
     "role": "core", "url": f"{BASE}/2016/2/eng@2022-12-31"},
    {"slug": "law_of_contract_act", "title": "Law of Contract Act",
     "role": "core", "url": f"{BASE}/1960/43/eng@2022-12-31"},
    {"slug": "sale_of_goods_act", "title": "Sale of Goods Act",
     "role": "core", "url": f"{BASE}/1930/33/eng@1964-11-03"},
    {"slug": "consumer_protection_act", "title": "Consumer Protection Act",
     "role": "core", "url": f"{BASE}/2012/46/eng@2022-12-31"},
    {"slug": "limitation_of_actions_act", "title": "Limitation of Actions Act",
     "role": "core", "url": f"{BASE}/1968/21/eng@2022-12-31"},
    {"slug": "evidence_act", "title": "Evidence Act",
     "role": "core", "url": f"{BASE}/1963/46/eng@2023-12-11"},
    {"slug": "civil_procedure_act", "title": "Civil Procedure Act",
     "role": "core", "url": f"{BASE}/1924/3/eng@2022-12-31"},
    {"slug": "bills_of_exchange_act", "title": "Bills of Exchange Act",
     "role": "core", "url": f"{BASE}/1927/7/eng@2022-12-31"},
    {"slug": "kenya_information_and_communications_act",
     "title": "Kenya Information and Communications Act",
     "role": "core", "url": f"{BASE}/1998/2/eng@2022-12-31"},
    {"slug": "stamp_duty_act", "title": "Stamp Duty Act",
     "role": "core", "url": f"{BASE}/1958/31/eng@2026-07-01"},
    {"slug": "judicature_act", "title": "Judicature Act",
     "role": "core", "url": f"{BASE}/1967/16/eng@2023-12-11"},
    {"slug": "small_claims_court_rules", "title": "Small Claims Court (Procedure) Rules",
     "role": "core", "url": "https://new.kenyalaw.org/akn/ke/act/ln/2019/145/eng@2022-12-31"},

    # ---- Jurisdiction exclusions / redirects ----
    {"slug": "employment_act", "title": "Employment Act",
     "role": "exclusion", "url": f"{BASE}/2007/11/eng@2024-04-26"},
    {"slug": "landlord_and_tenant_act",
     "title": "Landlord and Tenant (Shops, Hotels and Catering Establishments) Act",
     "role": "exclusion", "url": f"{BASE}/1965/13/eng@2022-12-31"},
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

AKN_URL_RE = re.compile(r"/akn/(ke/act(?:/[a-z]+)?/\d{4}/\d+)/(\w+)@(\d{4}-\d{2}-\d{2})")
SECTION_RE = re.compile(r'class="[^"]*\bakn-section\b')


def build_session() -> requests.Session:
    """Session with retries on rate limits and transient server errors."""
    retry = Retry(
        total=4,
        backoff_factor=2,  # 2s, 4s, 8s, 16s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update(HEADERS)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def parse_akn_url(url: str) -> dict:
    """Extract the FRBR work id, language and expression (version) date."""
    m = AKN_URL_RE.search(url)
    if not m:
        return {"work": None, "language": None, "version": "unknown"}
    work, lang, version = m.groups()
    return {"work": work, "language": lang, "version": version}


def dedupe(targets: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for t in targets:
        if t["url"] in seen:
            print(f"[~] Skipping duplicate URL for {t['title']}")
            continue
        seen.add(t["url"])
        unique.append(t)
    return unique


def download_statutes(force: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    manifest = []

    targets = dedupe(STATUTE_TARGETS)
    print(f"[*] Downloading {len(targets)} statutes to '{OUTPUT_DIR}'\n")

    for i, item in enumerate(targets, 1):
        meta = parse_akn_url(item["url"])
        file_path = OUTPUT_DIR / f"{item['slug']}__{meta['version']}.html"
        entry = {**item, **meta, "file": file_path.name}

        print(f"[{i}/{len(targets)}] {item['title']} ({meta['version']})")

        if file_path.exists() and not force:
            html = file_path.read_text(encoding="utf-8", errors="replace")
            entry["status"] = "cached"
            print("    [=] Already downloaded, skipping (use --force to refetch)")
        else:
            try:
                resp = session.get(item["url"], timeout=TIMEOUT)
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "")
                if "html" not in ctype:
                    raise ValueError(f"unexpected Content-Type: {ctype}")
                file_path.write_bytes(resp.content)  # keep server encoding intact
                html = resp.content.decode(resp.encoding or "utf-8", errors="replace")
                entry["status"] = "downloaded"
                entry["fetched_at"] = datetime.now(timezone.utc).isoformat()
                print(f"    [✓] Saved {len(resp.content):,} bytes")
            except Exception as e:
                entry["status"] = "failed"
                entry["error"] = str(e)
                manifest.append(entry)
                print(f"    [!] Failed: {e}")
                time.sleep(CRAWL_DELAY)
                continue
            time.sleep(CRAWL_DELAY)

        # Validate that the structured Akoma Ntoso markup is actually present.
        section_count = len(SECTION_RE.findall(html))
        entry["section_count"] = section_count
        entry["sha256"] = hashlib.sha256(html.encode("utf-8")).hexdigest()
        if section_count == 0:
            entry["warning"] = "no akn-section markup found"
            print("    [?] WARNING: no 'akn-section' elements found. The page may "
                  "load its body dynamically; inspect the file before parsing.")
        else:
            print(f"    [i] {section_count} sections detected")

        manifest.append(entry)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ok = sum(e["status"] in ("downloaded", "cached") for e in manifest)
    print(f"\n[*] Done: {ok}/{len(manifest)} available. Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download files that already exist")
    args = parser.parse_args()
    download_statutes(force=args.force)
