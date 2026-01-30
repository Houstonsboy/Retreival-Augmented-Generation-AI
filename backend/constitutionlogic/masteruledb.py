"""
Enhanced Vector Database Export Script for Master Rules Database

Exports all constitution articles and chunks with complete metadata to JSON file.
"""

import chromadb
from pathlib import Path
import json
from datetime import datetime

# Configuration (should match constitution_digester.py)
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "MasterRulesDB"
CHROMA_COLLECTION_NAME = "master_rules"

# Output JSON file in same directory as script
OUTPUT_FILE = BASE_DIR / f"master_rules_db_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def export_vector_db_to_json():
    """Export complete vector database to JSON file with full metadata."""
    
    export_data = {
        "export_metadata": {
            "timestamp": datetime.now().isoformat(),
            "database_path": str(CHROMA_DB_DIR),
            "collection_name": CHROMA_COLLECTION_NAME,
            "export_status": "unknown"
        },
        "statistics": {},
        "articles": {},
        "all_chunks": [],
        "structure_analysis": {},
        "completeness_check": {},
        "retrieval_test": {}
    }
    
    print("\n" + "="*80)
    print("🔍 MASTER RULES VECTOR DATABASE EXPORT TO JSON")
    print("="*80)
    
    # Check if database directory exists
    if not CHROMA_DB_DIR.exists():
        error_msg = f"Database not found at {CHROMA_DB_DIR}"
        print(f"\n❌ ERROR: {error_msg}")
        print("Have you run the constitution digester yet?")
        export_data["export_metadata"]["export_status"] = "error"
        export_data["export_metadata"]["error"] = error_msg
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"Error details saved to: {OUTPUT_FILE}")
        return
    
    print(f"✓ Database directory found: {CHROMA_DB_DIR}")
    export_data["export_metadata"]["database_exists"] = True
    
    # Connect to database
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"✓ Connected to collection: {CHROMA_COLLECTION_NAME}")
        export_data["export_metadata"]["collection_connected"] = True
    except Exception as e:
        error_msg = f"Could not connect to database: {str(e)}"
        print(f"\n❌ ERROR: {error_msg}")
        export_data["export_metadata"]["export_status"] = "error"
        export_data["export_metadata"]["error"] = error_msg
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"Error details saved to: {OUTPUT_FILE}")
        return
    
    # Get all data
    print("\n📊 Retrieving all data from database...")
    results = collection.get(include=["metadatas", "documents", "embeddings"])
    
    total_chunks = len(results["ids"])
    
    print(f"✓ Retrieved {total_chunks} chunks")
    
    # Overall Statistics
    export_data["statistics"]["total_chunks"] = total_chunks
    
    if total_chunks == 0:
        export_data["export_metadata"]["export_status"] = "warning"
        export_data["export_metadata"]["warning"] = "Database is EMPTY"
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\n⚠️  Database is EMPTY - Exported to: {OUTPUT_FILE}")
        return
    
    # Organize chunks by article
    print("\n📚 Organizing chunks by article...")
    articles_data = {}
    for idx, chunk_id in enumerate(results["ids"]):
        metadata = results["metadatas"][idx]
        document = results["documents"][idx]
        embedding = results["embeddings"][idx] if results["embeddings"] is not None and len(results["embeddings"]) > 0 else None
        article_num = metadata.get("article_number", "unknown")
        
        if article_num not in articles_data:
            articles_data[article_num] = []
        
        chunk_info = {
            "chunk_id": chunk_id,
            "metadata": metadata,
            "document_text": document,
            "document_length": len(document),
            "has_embedding": embedding is not None,
            "embedding_dimensions": len(embedding) if embedding is not None else 0
        }
        
        articles_data[article_num].append(chunk_info)
        export_data["all_chunks"].append(chunk_info)
    
    export_data["statistics"]["unique_articles"] = len(articles_data)
    print(f"✓ Found {len(articles_data)} unique articles")
    
    # Count by document structure
    print("\n📑 Analyzing document structure...")
    structures = {
        "chapters": set(),
        "parts": set(),
        "articles": set()
    }
    
    for metadata in results["metadatas"]:
        chapter = metadata.get("chapter", "")
        part = metadata.get("part", "")
        article = metadata.get("article_number", "")
        
        if chapter:
            structures["chapters"].add(chapter)
        if part:
            structures["parts"].add(part)
        if article:
            structures["articles"].add(article)
    
    export_data["structure_analysis"] = {
        "total_chapters": len(structures["chapters"]),
        "chapters": sorted(list(structures["chapters"])),
        "total_parts": len(structures["parts"]),
        "parts": sorted(list(structures["parts"])),
        "total_articles": len(structures["articles"]),
        "article_numbers": sorted([a for a in structures["articles"] if str(a).isdigit()], key=int) + 
                          sorted([a for a in structures["articles"] if not str(a).isdigit()])
    }
    print(f"✓ Analyzed structure: {len(structures['chapters'])} chapters, {len(structures['parts'])} parts")
    
    # Count complete vs partial articles
    print("\n📊 Analyzing article completeness...")
    complete_articles = 0
    partial_articles = 0
    for article_num, chunks in articles_data.items():
        is_complete = any(chunk["metadata"].get("is_complete_article", False) 
                         for chunk in chunks)
        if is_complete:
            complete_articles += 1
        else:
            partial_articles += 1
    
    export_data["statistics"]["complete_articles"] = complete_articles
    export_data["statistics"]["partial_articles"] = partial_articles
    print(f"✓ Complete articles: {complete_articles}, Partial articles: {partial_articles}")
    
    # Chunk size statistics
    print("\n📏 Calculating chunk size statistics...")
    chunk_lengths = [len(doc_text) for doc_text in results["documents"]]
    if chunk_lengths:
        export_data["statistics"]["chunk_sizes"] = {
            "average_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_length": min(chunk_lengths),
            "max_length": max(chunk_lengths),
            "total_characters": sum(chunk_lengths)
        }
        print(f"✓ Average chunk size: {export_data['statistics']['chunk_sizes']['average_length']:.0f} characters")
    
    # Check for embeddings
    print("\n🔢 Checking embeddings...")
    results_with_embeddings = collection.get(include=["embeddings"], limit=1)
    
    if len(results_with_embeddings["embeddings"]) > 0:
        embedding_dim = len(results_with_embeddings["embeddings"][0])
        export_data["statistics"]["embeddings"] = {
            "present": True,
            "dimensions": embedding_dim
        }
        print(f"✓ Embeddings present (dimension: {embedding_dim})")
    else:
        export_data["statistics"]["embeddings"] = {
            "present": False,
            "dimensions": 0
        }
        print(f"⚠️  WARNING: No embeddings found!")
    
    # Organize articles with sorted chunks
    print("\n📄 Organizing articles with sorted chunks...")
    
    def article_sort_key(article_item):
        article_num = article_item[0]
        try:
            return int(article_num)
        except (ValueError, TypeError):
            return float('inf')
    
    sorted_articles = sorted(articles_data.items(), key=article_sort_key)
    
    for article_id, chunks in sorted_articles:
        # Sort chunks by index
        sorted_chunks = sorted(chunks, key=lambda x: int(x["metadata"].get("chunk_index", 0)))
        
        export_data["articles"][str(article_id)] = {
            "article_number": article_id,
            "total_chunks": len(sorted_chunks),
            "chunks": sorted_chunks
        }
    
    print(f"✓ Organized {len(export_data['articles'])} articles")
    
    # Completeness check
    print("\n🔍 Running completeness check...")
    articles_missing_metadata = []
    metadata_fields_to_check = [
        'article_number', 'article_header', 'source_document',
        'chunk_index', 'total_chunks', 'chunk_length'
    ]
    
    for article_id, chunks in articles_data.items():
        for chunk_data in chunks:
            metadata = chunk_data["metadata"]
            missing_fields = []
            
            for field in metadata_fields_to_check:
                if field not in metadata or metadata[field] == "":
                    missing_fields.append(field)
            
            if missing_fields:
                articles_missing_metadata.append({
                    'article': article_id,
                    'chunk_id': chunk_data['chunk_id'],
                    'missing_fields': missing_fields
                })
    
    export_data["completeness_check"] = {
        "total_issues": len(articles_missing_metadata),
        "issues": articles_missing_metadata[:50],  # Limit to first 50
        "all_complete": len(articles_missing_metadata) == 0
    }
    
    if articles_missing_metadata:
        print(f"⚠️  Found {len(articles_missing_metadata)} chunk(s) with missing metadata")
    else:
        print(f"✅ All chunks have complete metadata")
    
    # Check chunk sequence consistency
    print("\n📊 Analyzing chunk sequences...")
    sequence_issues = []
    for article_id, chunks in sorted_articles[:50]:  # Check first 50 articles
        if not chunks:
            continue
            
        chunk_indices = []
        for chunk in chunks:
            idx = chunk["metadata"].get("chunk_index")
            if isinstance(idx, str) and idx.isdigit():
                chunk_indices.append(int(idx))
            elif isinstance(idx, int):
                chunk_indices.append(idx)
            else:
                chunk_indices.append(None)
        
        total_chunks_meta = chunks[0]["metadata"].get("total_chunks")
        total_chunks_meta = int(total_chunks_meta) if isinstance(total_chunks_meta, str) and total_chunks_meta.isdigit() else total_chunks_meta
        
        issues = []
        if None in chunk_indices:
            issues.append("Some chunks missing chunk_index")
        elif total_chunks_meta and len(chunks) != total_chunks_meta:
            issues.append(f"Expected {total_chunks_meta} chunks, found {len(chunks)}")
        else:
            expected_indices = list(range(1, len(chunks) + 1))
            if chunk_indices != expected_indices:
                issues.append(f"Chunk indices not sequential")
        
        if issues:
            sequence_issues.append({
                "article_id": article_id,
                "issues": issues
            })
    
    export_data["completeness_check"]["sequence_issues"] = {
        "total_articles_checked": min(50, len(sorted_articles)),
        "articles_with_issues": len(sequence_issues),
        "issues": sequence_issues
    }
    
    if sequence_issues:
        print(f"⚠️  Found {len(sequence_issues)} articles with sequence issues")
    else:
        print(f"✅ All checked articles have proper sequencing")
    
    # Sample retrieval test
    print("\n🧪 Running retrieval test...")
    if total_chunks > 0:
        try:
            sample_ids = results["ids"][:3]
            retrieved = collection.get(
                ids=sample_ids,
                include=["metadatas", "documents"]
            )
            
            export_data["retrieval_test"]["basic_retrieval"] = {
                "success": True,
                "chunks_retrieved": len(retrieved['ids'])
            }
            
            # Try a similarity search
            if len(results["documents"]) > 0:
                query_text = results["documents"][0][:100]
                query_results = collection.query(
                    query_texts=[query_text],
                    n_results=3,
                    include=["metadatas", "documents", "distances"]
                )
                
                if query_results and len(query_results["ids"][0]) > 0:
                    export_data["retrieval_test"]["similarity_search"] = {
                        "success": True,
                        "top_result_article": query_results['metadatas'][0][0].get('article_number'),
                        "top_result_distance": query_results['distances'][0][0]
                    }
                    print(f"✅ Similarity search works")
                else:
                    export_data["retrieval_test"]["similarity_search"] = {
                        "success": False,
                        "error": "No results returned"
                    }
            
            print(f"✅ Retrieval test passed")
        except Exception as e:
            export_data["retrieval_test"]["error"] = str(e)
            print(f"⚠️  Retrieval test failed: {e}")
    
    # Set final status
    export_data["export_metadata"]["export_status"] = "success"
    
    # Write to JSON file
    print(f"\n💾 Writing data to JSON file...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ SUCCESS!")
        print(f"📄 Data exported to: {OUTPUT_FILE}")
        print(f"\n📊 Export Summary:")
        print(f"   • Total chunks: {export_data['statistics']['total_chunks']}")
        print(f"   • Unique articles: {export_data['statistics']['unique_articles']}")
        print(f"   • Complete articles: {export_data['statistics']['complete_articles']}")
        print(f"   • Partial articles: {export_data['statistics']['partial_articles']}")
        print(f"   • File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
        print(f"\n{'='*80}")
        print("✅ EXPORT COMPLETE")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR writing JSON file: {e}")
        export_data["export_metadata"]["export_status"] = "error"
        export_data["export_metadata"]["write_error"] = str(e)


if __name__ == "__main__":
    export_vector_db_to_json()