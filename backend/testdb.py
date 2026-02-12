"""
Vector Database Export Script – MasterRulesDB & LegalSummariesDB

Dumps both vector databases into separate timestamped JSON files
so you can inspect how each DB is structured and populated.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
import json
from datetime import datetime

# Configuration
BASE_DIR = Path(__file__).resolve().parent

# Master Rules DB (Constitution + Evidence Act)
MASTER_RULES_DB_DIR = BASE_DIR / "constitutionlogic" / "MasterRulesDB"
MASTER_RULES_COLLECTION = "master_rules"

# Legal Summaries DB (case law FIRAC chunks)
LEGAL_SUMMARIES_DB_DIR = BASE_DIR / "LegalSummariesDB"
LEGAL_SUMMARIES_COLLECTION = "legal_summaries"

# Max characters of chunk text to show before truncating (set to None for full text)
CHUNK_TEXT_PREVIEW_LEN = 400

# Output directory for both exports (same folder level for easy finding)
EXPORT_DIR = BASE_DIR / "Database"

# Use a single timestamp for both exports (same run)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_RULES_OUTPUT = EXPORT_DIR / f"master_rules_db_export_{TIMESTAMP}.json"
LEGAL_SUMMARIES_OUTPUT = EXPORT_DIR / f"legal_summaries_db_export_{TIMESTAMP}.json"


def _truncate_text(text: str, max_len: int | None) -> tuple[str, int, bool]:
    """Return (preview, full_length, is_truncated)."""
    if max_len is None or len(text) <= max_len:
        return (text, len(text), False)
    return (text[:max_len] + "...", len(text), True)


def dump_master_rules_db() -> dict:
    """Dump MasterRulesDB to a structure suitable for JSON export."""
    results = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "database_path": str(MASTER_RULES_DB_DIR),
            "collection_name": MASTER_RULES_COLLECTION,
            "output_file": str(MASTER_RULES_OUTPUT),
            "status": "unknown",
            "embeddings_included": False,
        },
        "chunks": [],
        "summary": {},
    }

    if not MASTER_RULES_DB_DIR.exists():
        results["export_info"]["status"] = "error"
        results["export_info"]["error"] = f"Database not found at {MASTER_RULES_DB_DIR}"
        return results

    try:
        client = chromadb.PersistentClient(
            path=str(MASTER_RULES_DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=MASTER_RULES_COLLECTION)
        # Don't load embeddings to keep export size manageable
        all_results = collection.get(include=["documents", "metadatas"])

        ids = all_results["ids"]
        documents = all_results["documents"]
        metadatas = all_results["metadatas"]
        total = len(ids)

        for idx in range(total):
            doc = documents[idx] if idx < len(documents) else ""
            text_preview, full_length, is_truncated = _truncate_text(
                doc, CHUNK_TEXT_PREVIEW_LEN
            )
            chunk_data = {
                "index": idx + 1,
                "id": ids[idx],
                "text": text_preview,
                "text_length": full_length,
                "text_truncated": is_truncated,
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
            }
            results["chunks"].append(chunk_data)

        results["summary"]["total_chunks"] = total
        results["summary"]["unique_article_numbers"] = len(
            set(
                c["metadata"].get("article_number")
                for c in results["chunks"]
                if c.get("metadata")
            )
        )
        results["summary"]["unique_sources"] = len(
            set(
                c["metadata"].get("source_document", "unknown")
                for c in results["chunks"]
                if c.get("metadata")
            )
        )
        results["summary"]["avg_text_length"] = (
            sum(c["text_length"] for c in results["chunks"]) / total if total else 0
        )
        results["export_info"]["status"] = "success"
    except Exception as e:
        results["export_info"]["status"] = "error"
        results["export_info"]["error"] = str(e)
        results["export_info"]["error_type"] = type(e).__name__

    return results


def dump_legal_summaries_db() -> dict:
    """Dump LegalSummariesDB to a structure suitable for JSON export."""
    results = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "database_path": str(LEGAL_SUMMARIES_DB_DIR),
            "collection_name": LEGAL_SUMMARIES_COLLECTION,
            "output_file": str(LEGAL_SUMMARIES_OUTPUT),
            "status": "unknown",
            "embeddings_included": False,
        },
        "chunks": [],
        "summary": {},
    }

    if not LEGAL_SUMMARIES_DB_DIR.exists():
        results["export_info"]["status"] = "error"
        results["export_info"]["error"] = f"Database not found at {LEGAL_SUMMARIES_DB_DIR}"
        return results

    try:
        client = chromadb.PersistentClient(
            path=str(LEGAL_SUMMARIES_DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=LEGAL_SUMMARIES_COLLECTION)
        all_results = collection.get(include=["documents", "metadatas"])

        ids = all_results["ids"]
        documents = all_results["documents"]
        metadatas = all_results["metadatas"]
        total = len(ids)

        for idx in range(total):
            doc = documents[idx] if idx < len(documents) else ""
            text_preview, full_length, is_truncated = _truncate_text(
                doc, CHUNK_TEXT_PREVIEW_LEN
            )
            chunk_data = {
                "index": idx + 1,
                "id": ids[idx],
                "text": text_preview,
                "text_length": full_length,
                "text_truncated": is_truncated,
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
            }
            results["chunks"].append(chunk_data)

        results["summary"]["total_chunks"] = total
        results["summary"]["unique_case_identifiers"] = len(
            set(
                c["metadata"].get("case_identifier")
                for c in results["chunks"]
                if c.get("metadata")
            )
        )
        results["summary"]["unique_firac_components"] = sorted(
            set(
                c["metadata"].get("firac_component", "")
                for c in results["chunks"]
                if c.get("metadata")
            )
        )
        results["summary"]["avg_text_length"] = (
            sum(c["text_length"] for c in results["chunks"]) / total if total else 0
        )
        results["export_info"]["status"] = "success"
    except Exception as e:
        results["export_info"]["status"] = "error"
        results["export_info"]["error"] = str(e)
        results["export_info"]["error_type"] = type(e).__name__

    return results


def main():
    """Dump both Master Rules and Legal Summaries DBs to timestamped JSON files."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("VECTOR DB EXPORT – MasterRulesDB & LegalSummariesDB")
    print("=" * 80)
    print(f"Output folder: {EXPORT_DIR}")
    print(f"Timestamp for this run: {TIMESTAMP}")
    print(f"Text preview length: {CHUNK_TEXT_PREVIEW_LEN} chars (set CHUNK_TEXT_PREVIEW_LEN to None for full text)")
    print()

    # ---- Master Rules DB ----
    print("📜 [1/2] Exporting Master Rules DB...")
    master_data = dump_master_rules_db()
    with open(MASTER_RULES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)

    if master_data["export_info"]["status"] == "success":
        print(f"   ✓ Saved to: {MASTER_RULES_OUTPUT}")
        print(f"   Total chunks: {master_data['summary'].get('total_chunks', 0)}")
        print(f"   Unique articles/sources: {master_data['summary'].get('unique_article_numbers', 'N/A')}")
    else:
        print(f"   ❌ {master_data['export_info'].get('error', 'Unknown error')}")
        print(f"   Details written to: {MASTER_RULES_OUTPUT}")

    print()

    # ---- Legal Summaries DB ----
    print("📚 [2/2] Exporting Legal Summaries DB...")
    legal_data = dump_legal_summaries_db()
    with open(LEGAL_SUMMARIES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(legal_data, f, indent=2, ensure_ascii=False)

    if legal_data["export_info"]["status"] == "success":
        print(f"   ✓ Saved to: {LEGAL_SUMMARIES_OUTPUT}")
        print(f"   Total chunks: {legal_data['summary'].get('total_chunks', 0)}")
        print(f"   Unique cases: {legal_data['summary'].get('unique_case_identifiers', 'N/A')}")
        print(f"   FIRAC components: {legal_data['summary'].get('unique_firac_components', [])}")
    else:
        print(f"   ❌ {legal_data['export_info'].get('error', 'Unknown error')}")
        print(f"   Details written to: {LEGAL_SUMMARIES_OUTPUT}")

    print()
    print("=" * 80)
    print("✅ Export complete. Inspect the JSON files to see how each DB looks.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
