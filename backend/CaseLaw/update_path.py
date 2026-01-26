# backend/CaseLaw/update_paths_to_absolute.py
"""
Update source_file_path metadata to absolute paths for existing chunks.
NO re-embedding, NO FIRAC extraction - just metadata updates.
"""
import chromadb
from pathlib import Path

# Configuration (same as ingester.py)
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "LegalSummariesDB"
CHROMA_COLLECTION_NAME = "legal_summaries"
RES_IPSA_LOQUITUR_DIR = BASE_DIR.parent.parent / "repo" / "Res_Ipsa_loquitur"


def update_paths_to_absolute():
    """
    Update source_file_path metadata to absolute paths for all existing chunks.
    This does NOT re-embed or re-extract FIRAC - only updates metadata.
    """
    print("=" * 80)
    print("🔄 UPDATING SOURCE PATHS TO ABSOLUTE (METADATA ONLY)")
    print("=" * 80)
    print("⚠️  This will NOT re-embed or call any APIs - just updates metadata\n")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        print("❌ Collection not found. Nothing to update.")
        return
    
    # Get all chunks
    print("📥 Fetching all chunks from database...")
    all_data = collection.get(include=["metadatas"])
    
    if not all_data.get("ids"):
        print("❌ No chunks found in database.")
        return
    
    chunk_ids = all_data["ids"]
    metadatas = all_data["metadatas"]
    
    print(f"Found {len(chunk_ids)} chunk(s) to check.\n")
    
    # Track statistics
    updated_count = 0
    already_absolute_count = 0
    not_found_count = 0
    empty_path_count = 0
    
    # Process each chunk
    for chunk_id, metadata in zip(chunk_ids, metadatas):
        if not metadata:
            continue
        
        current_path = metadata.get("source_file_path", "")
        
        # Skip if empty
        if not current_path:
            empty_path_count += 1
            continue
        
        # Check if already absolute
        current_path_obj = Path(current_path)
        if current_path_obj.is_absolute():
            already_absolute_count += 1
            continue
        
        # Try to resolve to absolute path
        try:
            # First, try to find the file relative to RES_IPSA_LOQUITUR_DIR
            if RES_IPSA_LOQUITUR_DIR.exists():
                potential_path = RES_IPSA_LOQUITUR_DIR / current_path
                if potential_path.exists():
                    absolute_path = str(potential_path.resolve())
                else:
                    # Try just the filename
                    filename = Path(current_path).name
                    potential_path = RES_IPSA_LOQUITUR_DIR / filename
                    if potential_path.exists():
                        absolute_path = str(potential_path.resolve())
                    else:
                        # Can't find file, but make path absolute anyway
                        absolute_path = str((RES_IPSA_LOQUITUR_DIR / current_path).resolve())
            else:
                # Base dir doesn't exist, but make it absolute relative to current working dir
                absolute_path = str(Path(current_path).resolve())
            
            # Update metadata (ONLY metadata, not embeddings!)
            updated_metadata = {**metadata, "source_file_path": absolute_path}
            collection.update(
                ids=[chunk_id],
                metadatas=[updated_metadata]
            )
            
            updated_count += 1
            if updated_count <= 5:  # Show first few updates
                print(f"  ✓ Updated: {chunk_id[:50]}...")
                print(f"    {current_path[:60]}... → {absolute_path[:60]}...")
            
        except Exception as e:
            not_found_count += 1
            if not_found_count <= 3:
                print(f"  ⚠️  Could not update {chunk_id[:50]}...: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ UPDATE COMPLETE")
    print("=" * 80)
    print(f"📊 Summary:")
    print(f"   • Updated to absolute: {updated_count} chunk(s)")
    print(f"   • Already absolute: {already_absolute_count} chunk(s)")
    print(f"   • Empty path (skipped): {empty_path_count} chunk(s)")
    print(f"   • Errors: {not_found_count} chunk(s)")
    print(f"   • Total processed: {len(chunk_ids)} chunk(s)")
    print(f"\n💡 No embeddings were modified - only metadata updated!")


if __name__ == "__main__":
    update_paths_to_absolute()