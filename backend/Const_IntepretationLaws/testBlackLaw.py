#!/usr/bin/env python3
"""
Detailed Metadata Field Checker
Shows exactly what metadata fields exist in your ChromaDB and displays examples.
"""

from pathlib import Path
import chromadb
from chromadb.config import Settings
import json

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "BlackLawDictionaryDB"
CHROMA_COLLECTION_NAME = "black_law_dictionary"

def main():
    print("\n" + "="*80)
    print("DETAILED METADATA FIELD CHECKER")
    print("="*80 + "\n")
    
    # Connect
    print("🗄️  Connecting to ChromaDB...")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    print(f"✓ Connected! Total entries: {collection.count():,}\n")
    
    # Get sample entries
    print("📥 Fetching 20 sample entries...\n")
    result = collection.get(
        limit=20,
        include=['metadatas', 'documents']
    )
    
    # Collect all field names
    all_fields = set()
    for metadata in result['metadatas']:
        all_fields.update(metadata.keys())
    
    print("="*80)
    print("AVAILABLE METADATA FIELDS")
    print("="*80 + "\n")
    
    for field in sorted(all_fields):
        # Count how many entries have this field
        count = sum(1 for m in result['metadatas'] if field in m and m[field])
        percentage = 100 * count / len(result['metadatas'])
        print(f"  • {field:<20} (present in {count}/{len(result['metadatas'])} entries, {percentage:.0f}%)")
    
    print("\n" + "="*80)
    print("CROSS-REFERENCE STATISTICS")
    print("="*80 + "\n")
    
    has_see = sum(1 for m in result['metadatas'] if 'see' in m and m.get('see'))
    has_cf = sum(1 for m in result['metadatas'] if 'cf' in m and m.get('cf'))
    has_also = sum(1 for m in result['metadatas'] if 'also_termed' in m and m.get('also_termed'))
    
    print(f"Entries with 'see' field:        {has_see}/{len(result['metadatas'])} ({100*has_see/len(result['metadatas']):.1f}%)")
    print(f"Entries with 'cf' field:         {has_cf}/{len(result['metadatas'])} ({100*has_cf/len(result['metadatas']):.1f}%)")
    print(f"Entries with 'also_termed' field: {has_also}/{len(result['metadatas'])} ({100*has_also/len(result['metadatas']):.1f}%)")
    
    # Show detailed examples
    print("\n" + "="*80)
    print("DETAILED EXAMPLES")
    print("="*80 + "\n")
    
    examples_shown = 0
    for i in range(len(result['ids'])):
        metadata = result['metadatas'][i]
        
        # Only show entries with cross-references
        has_cross_ref = (
            (metadata.get('see') and metadata['see']) or
            (metadata.get('cf') and metadata['cf']) or
            (metadata.get('also_termed') and metadata['also_termed'])
        )
        
        if has_cross_ref and examples_shown < 5:
            print(f"📌 Entry {examples_shown + 1}:")
            print(f"   Term: {metadata.get('term', 'Unknown')}")
            print(f"   Page: {metadata.get('page', 'N/A')}")
            print(f"   ID: {result['ids'][i]}")
            
            if metadata.get('see'):
                print(f"   ✓ See: {metadata['see']}")
            if metadata.get('cf'):
                print(f"   ✓ Cf.: {metadata['cf']}")
            if metadata.get('also_termed'):
                print(f"   ✓ Also termed: {metadata['also_termed']}")
            
            # Show document text
            doc = result['documents'][i]
            print(f"   Document text:")
            # Split and show first few lines
            lines = doc.split('\n')
            for line in lines[:5]:
                if line.strip():
                    print(f"     {line[:100]}")
            if len(doc) > 500:
                print(f"     ... (truncated, full length: {len(doc)} chars)")
            
            print()
            examples_shown += 1
    
    if examples_shown == 0:
        print("⚠️  NO ENTRIES WITH CROSS-REFERENCES FOUND IN SAMPLE!\n")
        print("This could mean:")
        print("  1. Cross-references are rare in your dictionary")
        print("  2. OCR is not capturing 'See', 'Cf.', 'Also termed' text")
        print("  3. The entries checked happen to not have cross-references")
        print("\n💡 Try checking more entries or later pages where cross-refs might be more common")
    else:
        print(f"✅ Found {examples_shown} examples with cross-references!")
    
    # Save detailed report
    report = {
        'total_entries_in_db': collection.count(),
        'sample_size': len(result['metadatas']),
        'available_fields': sorted(list(all_fields)),
        'cross_reference_stats': {
            'has_see': has_see,
            'has_cf': has_cf,
            'has_also_termed': has_also,
            'percentage_with_see': 100 * has_see / len(result['metadatas']),
            'percentage_with_cf': 100 * has_cf / len(result['metadatas']),
            'percentage_with_also_termed': 100 * has_also / len(result['metadatas'])
        },
        'sample_entries': [
            {
                'term': result['metadatas'][i].get('term', 'Unknown'),
                'page': result['metadatas'][i].get('page', 0),
                'id': result['ids'][i],
                'see': result['metadatas'][i].get('see', ''),
                'cf': result['metadatas'][i].get('cf', ''),
                'also_termed': result['metadatas'][i].get('also_termed', ''),
                'has_cross_references': bool(
                    result['metadatas'][i].get('see') or
                    result['metadatas'][i].get('cf') or
                    result['metadatas'][i].get('also_termed')
                )
            }
            for i in range(len(result['metadatas']))
        ]
    }
    
    report_path = BASE_DIR / "metadata_field_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n🚨 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()