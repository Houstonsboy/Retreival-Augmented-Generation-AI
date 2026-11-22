"""
Standalone test: OCR + process + analyze the two WhatsApp screenshots.

Run from the backend/ directory:
    python test_scc_pipeline.py

Prints the full analysis JSON (or the exact error from Groq).
"""

import json
import os
import sys
from pathlib import Path

# Ensure the backend package root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from SCCVision.service import extract_text
from SCCVision.preprocess import PreprocessOptions
from SCCVision.analyst.pipeline import analyze_case
from SCCVision.analyst.enrich import _MAX_MESSAGES, _MAX_MSG_TEXT, _MAX_TOTAL_CHARS
from groq_client import GroqPayloadTooLargeError, GroqRateLimitError

IMAGES_DIR = Path(__file__).resolve().parent.parent / "SCCImages"

# Pick the two WhatsApp Nov-8 screenshots; adjust if you want different ones
IMAGE_FILES = [
    IMAGES_DIR / "WhatsApp Image 2026-11-08 at 20.11.54.jpeg",
    IMAGES_DIR / "WhatsApp Image 2026-11-08 at 20.12.06.jpeg",
]

OPTIONS = PreprocessOptions(upscale=True, invert=True)


def ocr_image(path: Path) -> dict:
    print(f"  OCR: {path.name} ...", end=" ", flush=True)
    data = path.read_bytes()
    result = extract_text(data, OPTIONS, path.name)
    n_msgs = len(result.get("messages") or [])
    print(f"done ({n_msgs} messages)")
    return result


def build_evidence_item(ocr: dict, path: Path, idx: int) -> dict:
    """Minimal evidence item matching what /sccvision/evidence/process returns."""
    return {
        "evidence_id": f"test-{idx}",
        "type": "chat_screenshot",
        "file_name": path.name,
        "sha256": None,
        "ocr": {
            "contact": ocr.get("contact") or {},
            "messages": ocr.get("messages") or [],
            "system_events": ocr.get("system_events") or [],
        },
        "parsed": None,
        "user_edits": {},
    }


def main():
    print("=== SCC Vision Pipeline Test ===\n")
    print(f"Payload caps: {_MAX_MESSAGES} msgs, {_MAX_MSG_TEXT} chars/msg, {_MAX_TOTAL_CHARS} total chars\n")

    # 1. OCR each image
    print("Step 1: OCR")
    evidence_items = []
    for i, img_path in enumerate(IMAGE_FILES):
        if not img_path.exists():
            print(f"  SKIP (not found): {img_path}")
            continue
        ocr = ocr_image(img_path)
        evidence_items.append(build_evidence_item(ocr, img_path, i))

    if not evidence_items:
        print("No images found — check IMAGE_FILES paths.")
        sys.exit(1)

    print(f"\n  Total evidence items: {len(evidence_items)}")
    total_msgs = sum(len(e["ocr"]["messages"]) for e in evidence_items)
    print(f"  Total raw messages across all screenshots: {total_msgs}")

    # 2. Build a minimal case file
    case_file = {
        "case_context": {"user_role": "claimant", "user_account": ""},
        "evidence": evidence_items,
        "user_provided_facts": [],
        "rejected_facts": [],
        "inferred_facts": [],
        "requests": [],
        "analyses": [],
    }

    # 3. Estimate payload size before sending
    payload_json = json.dumps(case_file, ensure_ascii=False)
    print(f"\n  Case file JSON size: {len(payload_json):,} bytes ({len(payload_json)//1024} KB)")

    # 4. Run analysis
    print("\nStep 2: Analyze (calling Groq) ...")
    try:
        result = analyze_case(case_file)
        print("\n=== ANALYSIS RESULT ===\n")
        print(json.dumps(result.get("result"), indent=2, ensure_ascii=False))

        verdict = (result.get("result") or {}).get("verdict")
        summary = (result.get("result") or {}).get("summary")
        print(f"\nVerdict : {verdict}")
        print(f"Summary : {summary}")
        print("\n=== Full output written to test_output.json ===")
        Path("test_output.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    except GroqPayloadTooLargeError as e:
        print(f"\n[413] Groq payload too large:\n  {e}")
    except GroqRateLimitError as e:
        print(f"\n[429] Groq rate limit:\n  {e}")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
