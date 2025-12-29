"""
Clear Vector Database Script

Safely clears all data from the ChromaDB vector database.
Use this when you want to re-embed documents from scratch.
"""

import chromadb
from pathlib import Path
import sys

# Configuration (should match ingester.py)
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "LegalSummariesDB"
CHROMA_COLLECTION_NAME = "legal_summaries"


def clear_vector_db(confirm=True):
    """
    Clear all data from the vector database.
    
    Args:
        confirm: If True, asks for user confirmation before deleting
    """
    
    print("\n" + "="*60)
    print("🗑️  CLEAR VECTOR DATABASE")
    print("="*60)
    
    # Check if database exists
    if not CHROMA_DB_DIR.exists():
        print(f"\n✓ Database directory doesn't exist: {CHROMA_DB_DIR}")
        print("Nothing to clear - database is already empty!")
        return
    
    print(f"Database directory: {CHROMA_DB_DIR}")
    
    # Connect to database
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"✓ Connected to collection: {CHROMA_COLLECTION_NAME}")
    except Exception as e:
        print(f"\n⚠️  Collection '{CHROMA_COLLECTION_NAME}' doesn't exist")
        print("Nothing to clear!")
        return
    
    # Get current count
    results = collection.get()
    current_count = len(results["ids"])
    
    print(f"\n📊 Current database contents:")
    print(f"   • Total chunks: {current_count}")
    
    if current_count == 0:
        print("\n✓ Database is already empty!")
        return
    
    # Count cases
    cases = set()
    for metadata in results["metadatas"]:
        case_id = metadata.get("case_identifier", "unknown")
        cases.add(case_id)
    
    print(f"   • Unique cases: {len(cases)}")
    for case in sorted(cases):
        case_chunks = sum(1 for m in results["metadatas"] 
                         if m.get("case_identifier") == case)
        print(f"      - {case} ({case_chunks} chunks)")
    
    # Ask for confirmation
    if confirm:
        print(f"\n⚠️  WARNING: This will DELETE all {current_count} chunks!")
        print("This action cannot be undone.")
        response = input("\nType 'yes' to confirm deletion: ").strip().lower()
        
        if response != 'yes':
            print("\n❌ Deletion cancelled.")
            return
    
    # Delete all data
    print(f"\n🗑️  Deleting all data from collection...")
    try:
        # Delete the collection entirely
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        print(f"✓ Collection '{CHROMA_COLLECTION_NAME}' deleted")
        
        # Recreate empty collection
        client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Created fresh empty collection '{CHROMA_COLLECTION_NAME}'")
        
    except Exception as e:
        print(f"\n❌ ERROR during deletion: {e}")
        return
    
    print(f"\n" + "="*60)
    print("✅ DATABASE CLEARED SUCCESSFULLY")
    print("="*60)
    print("\nYou can now run ingestion again to embed fresh data.")
    print()


def clear_specific_case(case_identifier):
    """
    Clear data for a specific case only.
    
    Args:
        case_identifier: The case identifier to remove
    """
    
    print("\n" + "="*60)
    print(f"🗑️  CLEAR SPECIFIC CASE: {case_identifier}")
    print("="*60)
    
    # Connect to database
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"\n❌ ERROR: Could not connect to database: {e}")
        return
    
    # Check if case exists
    results = collection.get(
        where={"case_identifier": case_identifier}
    )
    
    if not results["ids"]:
        print(f"\n⚠️  Case '{case_identifier}' not found in database")
        return
    
    chunk_count = len(results["ids"])
    print(f"\nFound {chunk_count} chunks for case '{case_identifier}'")
    
    # Ask for confirmation
    response = input(f"\nType 'yes' to delete this case: ").strip().lower()
    
    if response != 'yes':
        print("\n❌ Deletion cancelled.")
        return
    
    # Delete the case
    collection.delete(where={"case_identifier": case_identifier})
    
    print(f"\n✅ Case '{case_identifier}' deleted successfully!")
    print(f"   Removed {chunk_count} chunks")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--force":
            # Clear without confirmation
            clear_vector_db(confirm=False)
        elif sys.argv[1] == "--case":
            # Clear specific case
            if len(sys.argv) < 3:
                print("Usage: python clear_vectordb.py --case <case_identifier>")
                sys.exit(1)
            case_id = " ".join(sys.argv[2:])
            clear_specific_case(case_id)
        elif sys.argv[1] == "--help":
            print("\nUsage:")
            print("  python clear_vectordb.py              # Clear all (with confirmation)")
            print("  python clear_vectordb.py --force      # Clear all (no confirmation)")
            print("  python clear_vectordb.py --case <id>  # Clear specific case")
            print()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Normal mode with confirmation
        clear_vector_db(confirm=True)