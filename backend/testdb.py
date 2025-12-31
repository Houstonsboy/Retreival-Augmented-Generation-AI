"""
Enhanced Vector Database Test Script

Displays parent documents and all their chunks with complete metadata.
"""

import chromadb
from pathlib import Path

# Configuration (should match ingester.py)
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "LegalSummariesDB"
CHROMA_COLLECTION_NAME = "legal_summaries"


def test_vector_db():
    """Enhanced test to check documents and their chunks in detail."""
    
    print("\n" + "="*80)
    print("🔍 VECTOR DATABASE DETAILED TEST")
    print("="*80)
    
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
    
    print(f"\n{'='*80}")
    print(f"📊 OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"\n✅ Total chunks in database: {total_chunks}")
    
    if total_chunks == 0:
        print("\n⚠️  Database is EMPTY - No documents embedded yet!")
        print("\nNext steps:")
        print("1. Make sure you've run firac.py to extract FIRAC data")
        print("2. Run ingester.py to embed and store the data")
        return
    
    # Organize chunks by case
    cases_data = {}
    for idx, chunk_id in enumerate(results["ids"]):
        metadata = results["metadatas"][idx]
        document = results["documents"][idx]
        case_id = metadata.get("case_identifier", "unknown")
        
        if case_id not in cases_data:
            cases_data[case_id] = []
        
        cases_data[case_id].append({
            "id": chunk_id,
            "metadata": metadata,
            "document": document
        })
    
    print(f"📁 Unique cases (parent documents): {len(cases_data)}")
    
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
    
    print(f"\n📝 Total chunks by FIRAC component:")
    for comp, count in components.items():
        status = "✓" if count > 0 else "✗"
        print(f"   {status} {comp.capitalize()}: {count}")
    
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
    
    # Detailed display of each parent document and its chunks
    print(f"\n{'='*80}")
    print(f"📚 DETAILED BREAKDOWN: PARENTS AND CHUNKS")
    print(f"{'='*80}")
    
    for case_num, (case_id, chunks) in enumerate(sorted(cases_data.items()), 1):
        print(f"\n{'─'*80}")
        print(f"📄 PARENT DOCUMENT #{case_num}: {case_id}")
        print(f"{'─'*80}")
        print(f"Total chunks: {len(chunks)}\n")
        
        # Sort chunks by FIRAC component for logical display
        component_order = ["facts", "issues", "rules", "application", "conclusion"]
        sorted_chunks = sorted(chunks, key=lambda x: component_order.index(x["metadata"].get("firac_component", "unknown")) 
                              if x["metadata"].get("firac_component") in component_order else 999)
        
        for chunk_num, chunk_data in enumerate(sorted_chunks, 1):
            metadata = chunk_data["metadata"]
            document_text = chunk_data["document"]
            chunk_id = chunk_data["id"]
            
            print(f"  🔹 CHUNK {chunk_num}/{len(chunks)}")
            print(f"  {'┄'*76}")
            print(f"  Chunk ID: {chunk_id}")
            print(f"  FIRAC Component: {metadata.get('firac_component', 'N/A').upper()}")
            print(f"\n  📋 Metadata:")
            print(f"    • Case Identifier: {metadata.get('case_identifier', 'N/A')}")
            print(f"    • File Name: {metadata.get('file_name', 'N/A')}")
            print(f"    • Parties: {metadata.get('parties', 'N/A')}")
            print(f"    • Court Level: {metadata.get('court_level', 'N/A')}")
            print(f"    • Judge: {metadata.get('judge', 'N/A')}")
            print(f"    • Year: {metadata.get('year', 'N/A')}")
            print(f"    • Legal Domain: {metadata.get('legal_domain', 'N/A')}")
            print(f"    • Winning Party: {metadata.get('winning_party', 'N/A')}")
            
            # Display document text (truncated if too long)
            print(f"\n  📝 Document Text:")
            if len(document_text) > 300:
                print(f"    {document_text[:300]}...")
                print(f"    (... truncated, full length: {len(document_text)} characters)")
            else:
                print(f"    {document_text}")
            
            print()  # Blank line between chunks
    
    # Completeness check
    print(f"\n{'='*80}")
    print(f"🔍 COMPLETENESS CHECK")
    print(f"{'='*80}")
    
    incomplete_cases = []
    cases_without_metadata = []
    
    for case_id, chunks in cases_data.items():
        case_components = [chunk["metadata"].get("firac_component") for chunk in chunks]
        
        # Check if all 5 components are present
        missing = []
        for comp in ["facts", "issues", "rules", "application", "conclusion"]:
            if comp not in case_components:
                missing.append(comp)
        
        if missing:
            incomplete_cases.append((case_id, missing))
        
        # Check if metadata fields are populated
        sample_meta = chunks[0]["metadata"] if chunks else {}
        metadata_fields = ['file_name', 'parties', 'court_level', 'judge', 'year', 'legal_domain', 'winning_party']
        missing_metadata = [field for field in metadata_fields 
                          if not sample_meta.get(field) or sample_meta.get(field) == ""]
        if missing_metadata:
            cases_without_metadata.append((case_id, missing_metadata))
    
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
        print(f"\n✅ All cases have complete metadata")
    
    print(f"\n{'='*80}")
    print("✅ TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_vector_db()