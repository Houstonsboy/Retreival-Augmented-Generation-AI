"""
Vector Database Test Script – MasterRulesDB

Outputs ALL chunks from MasterRulesDB to a JSON file for verification.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
import json
from datetime import datetime

# Configuration
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "constitutionlogic"
CHROMA_COLLECTION_NAME = "master_rules"

# Output file configuration
OUTPUT_FILE = BASE_DIR / f"master_rules_db_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Max characters of chunk text to show before truncating (set to None for full text)
CHUNK_TEXT_PREVIEW_LEN = 400  # Change to None to export full text


def test_master_rules_db_all():
    """Output ALL chunks from MasterRulesDB to JSON file for verification."""

    results = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "database_path": str(CHROMA_DB_DIR),
            "collection_name": CHROMA_COLLECTION_NAME,
            "status": "unknown"
        },
        "chunks": [],
        "summary": {}
    }

    print("\n" + "="*80)
    print("🔍 MASTER RULES DB – FULL COLLECTION EXPORT TO JSON")
    print("="*80)

    if not CHROMA_DB_DIR.exists():
        error_msg = f"Database not found at {CHROMA_DB_DIR}"
        print(f"\n❌ ERROR: {error_msg}")
        print("Run your embedding scripts first.")
        results["export_info"]["status"] = "error"
        results["export_info"]["error"] = error_msg
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n❌ Error details saved to: {OUTPUT_FILE}")
        return

    print(f"✓ Database directory: {CHROMA_DB_DIR}")
    results["export_info"]["database_exists"] = True

    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"✓ Collection: {CHROMA_COLLECTION_NAME}")
        results["export_info"]["collection_found"] = True

        # Get all items from the collection
        print("\n📊 Retrieving all chunks from the collection...")
        all_results = collection.get(
            include=["documents", "metadatas", "embeddings"]
        )

        total_chunks = len(all_results["ids"])
        print(f"✓ Found {total_chunks} chunks")
        
        results["summary"]["total_chunks"] = total_chunks
        results["export_info"]["status"] = "success"

        # Process each chunk
        print(f"\n📝 Processing chunks...")
        for idx, (chunk_id, document, metadata, embedding) in enumerate(
            zip(
                all_results["ids"],
                all_results["documents"],
                all_results["metadatas"],
                all_results["embeddings"]
            ),
            start=1
        ):
            # Truncate text if needed
            if CHUNK_TEXT_PREVIEW_LEN and len(document) > CHUNK_TEXT_PREVIEW_LEN:
                text_preview = document[:CHUNK_TEXT_PREVIEW_LEN] + "..."
                is_truncated = True
                full_length = len(document)
            else:
                text_preview = document
                is_truncated = False
                full_length = len(document)

            chunk_data = {
                "index": idx,
                "id": chunk_id,
                "text": text_preview,
                "text_length": full_length,
                "text_truncated": is_truncated,
                "metadata": metadata,
                "embedding_dimensions": len(embedding) if embedding else 0,
                "has_embedding": embedding is not None
            }

            results["chunks"].append(chunk_data)

            # Progress indicator
            if idx % 10 == 0 or idx == total_chunks:
                print(f"  Processed {idx}/{total_chunks} chunks...", end='\r')

        print(f"\n✓ All {total_chunks} chunks processed")

        # Add summary statistics
        results["summary"]["chunks_with_embeddings"] = sum(
            1 for chunk in results["chunks"] if chunk["has_embedding"]
        )
        results["summary"]["unique_sources"] = len(set(
            chunk["metadata"].get("source", "unknown") 
            for chunk in results["chunks"] 
            if chunk["metadata"]
        ))
        results["summary"]["avg_text_length"] = (
            sum(chunk["text_length"] for chunk in results["chunks"]) / total_chunks
            if total_chunks > 0 else 0
        )

        # Write to JSON file
        print(f"\n💾 Writing results to JSON file...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ SUCCESS!")
        print(f"📄 Results saved to: {OUTPUT_FILE}")
        print(f"\n📊 Summary:")
        print(f"   Total chunks: {results['summary']['total_chunks']}")
        print(f"   Chunks with embeddings: {results['summary']['chunks_with_embeddings']}")
        print(f"   Unique sources: {results['summary']['unique_sources']}")
        print(f"   Average text length: {results['summary']['avg_text_length']:.0f} characters")
        print("\n" + "="*80)

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ ERROR: {error_msg}")
        results["export_info"]["status"] = "error"
        results["export_info"]["error"] = error_msg
        results["export_info"]["error_type"] = type(e).__name__

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n❌ Error details saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    test_master_rules_db_all()