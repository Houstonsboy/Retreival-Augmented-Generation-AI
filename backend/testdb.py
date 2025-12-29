"""
Simple Vector Database Test Script

Quick check to verify if documents are embedded and stored correctly.
"""

import chromadb
from pathlib import Path

# Configuration (should match ingester.py)
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "LegalSummariesDB"
CHROMA_COLLECTION_NAME = "legal_summaries"


def test_vector_db():
    """Simple test to check if documents are in the database."""
    
    print("\n" + "="*60)
    print("🔍 VECTOR DATABASE QUICK TEST")
    print("="*60)
    
    # Check if database directory exists
    if not CHROMA_DB_DIR.exists():
        print(f"\n❌ ERROR: Database not found at {CHROMA_DB_DIR}")
        print("Have you run the ingestion yet?")
        return
    
    print(f"✓ Database directory found: {CHROMA_DB_DIR}")
    
    # Connect to database
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"✓ Connected to collection: {CHROMA_COLLECTION_NAME}")
    except Exception as e:
        print(f"\n❌ ERROR: Could not connect to database")
        print(f"Details: {e}")
        return
    
    # Get all data
    results = collection.get(include=["metadatas", "documents"])
    
    total_chunks = len(results["ids"])
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS")
    print(f"{'='*60}")
    print(f"\n✅ Total chunks in database: {total_chunks}")
    
    if total_chunks == 0:
        print("\n⚠️  Database is EMPTY - No documents embedded yet!")
        print("\nNext steps:")
        print("1. Make sure you've run firac.py to extract FIRAC data")
        print("2. Run ingester.py to embed and store the data")
        return
    
    # Count unique cases
    cases = set()
    for metadata in results["metadatas"]:
        case_id = metadata.get("case_identifier", "unknown")
        cases.add(case_id)
    
    print(f"📁 Unique cases: {len(cases)}")
    
    # Count by FIRAC component
    components = {
        "facts": 0,
        "issues": 0,
        "rules": 0,
        "application": 0,
        "conclusion": 0
    }
    
    for metadata in results["metadatas"]:
        comp = metadata.get("firac_component", "unknown")
        if comp in components:
            components[comp] += 1
    
    print(f"\n📝 Chunks by component:")
    for comp, count in components.items():
        status = "✓" if count > 0 else "✗"
        print(f"   {status} {comp.capitalize()}: {count}")
    
    # List all cases with metadata
    print(f"\n📋 Cases in database:")
    for i, case in enumerate(sorted(cases), 1):
        # Count components for this case
        case_chunks = [m for m in results["metadatas"] 
                      if m.get("case_identifier") == case]
        print(f"   {i}. {case} ({len(case_chunks)} chunks)")
        
        # Show metadata for first chunk of this case (as sample)
        if case_chunks:
            sample_meta = case_chunks[0]
            print(f"      Metadata sample:")
            print(f"        • File Name: {sample_meta.get('file_name', 'N/A')}")
            print(f"        • Parties: {sample_meta.get('parties', 'N/A')[:60]}..." if len(sample_meta.get('parties', '')) > 60 else f"        • Parties: {sample_meta.get('parties', 'N/A')}")
            print(f"        • Court Level: {sample_meta.get('court_level', 'N/A')}")
            print(f"        • Judge: {sample_meta.get('judge', 'N/A')}")
            print(f"        • Year: {sample_meta.get('year', 'N/A')}")
            print(f"        • Legal Domain: {sample_meta.get('legal_domain', 'N/A')}")
            print(f"        • Winning Party: {sample_meta.get('winning_party', 'N/A')}")
    
    # Check for embeddings
    results_with_embeddings = collection.get(
        include=["embeddings"], 
        limit=1
    )
    
    if len(results_with_embeddings["embeddings"]) > 0:
        embedding_dim = len(results_with_embeddings["embeddings"][0])
        print(f"\n✅ Embeddings present (dimension: {embedding_dim})")
    else:
        print(f"\n⚠️  WARNING: No embeddings found!")
    
    # Completeness check
    print(f"\n{'='*60}")
    print(f"🔍 COMPLETENESS CHECK")
    print(f"{'='*60}")
    
    incomplete_cases = []
    cases_without_metadata = []
    
    for case in cases:
        case_chunks = [m for m in results["metadatas"] 
                      if m.get("case_identifier") == case]
        case_components = [m.get("firac_component") for m in case_chunks]
        
        # Check if all 5 components are present
        missing = []
        for comp in ["facts", "issues", "rules", "application", "conclusion"]:
            if comp not in case_components:
                missing.append(comp)
        
        if missing:
            incomplete_cases.append((case, missing))
        
        # Check if metadata fields are populated
        sample_meta = case_chunks[0] if case_chunks else {}
        metadata_fields = ['file_name', 'parties', 'court_level', 'judge', 'year', 'legal_domain', 'winning_party']
        missing_metadata = [field for field in metadata_fields if not sample_meta.get(field) or sample_meta.get(field) == ""]
        if missing_metadata:
            cases_without_metadata.append((case, missing_metadata))
    
    if incomplete_cases:
        print(f"\n⚠️  Found {len(incomplete_cases)} incomplete case(s):")
        for case, missing in incomplete_cases:
            print(f"   • {case}: missing {', '.join(missing)}")
    else:
        print(f"\n✅ All cases are COMPLETE (all 5 FIRAC components present)")
    
    # Metadata completeness check
    if cases_without_metadata:
        print(f"\n⚠️  Found {len(cases_without_metadata)} case(s) with missing metadata fields:")
        for case, missing_fields in cases_without_metadata:
            print(f"   • {case}: missing {', '.join(missing_fields)}")
    else:
        print(f"\n✅ All cases have complete metadata (file_name, parties, court_level, judge, year, legal_domain, winning_party)")
    
    print(f"\n{'='*60}")
    print("✅ TEST COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_vector_db()