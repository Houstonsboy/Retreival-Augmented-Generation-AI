"""
Enhanced Vector Database Test Script for Master Rules Database

Displays constitution articles and all their chunks with complete metadata.
"""

import chromadb
from pathlib import Path

# Configuration (should match constitution_digester.py)
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "MasterRulesDB"
CHROMA_COLLECTION_NAME = "master_rules"


def test_vector_db():
    """Enhanced test to check constitution articles and their chunks in detail."""
    
    print("\n" + "="*80)
    print("🔍 MASTER RULES VECTOR DATABASE DETAILED TEST")
    print("="*80)
    
    # Check if database directory exists
    if not CHROMA_DB_DIR.exists():
        print(f"\n❌ ERROR: Database not found at {CHROMA_DB_DIR}")
        print("Have you run the constitution digester yet?")
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
        print("\n⚠️  Database is EMPTY - No constitution chunks embedded yet!")
        print("\nNext steps:")
        print("1. Make sure you've run constitution_parser.py to parse the constitution")
        print("2. Run constitution_digester.py to embed and store the chunks")
        return
    
    # Organize chunks by article
    articles_data = {}
    for idx, chunk_id in enumerate(results["ids"]):
        metadata = results["metadatas"][idx]
        document = results["documents"][idx]
        article_num = metadata.get("article_number", "unknown")
        
        if article_num not in articles_data:
            articles_data[article_num] = []
        
        articles_data[article_num].append({
            "id": chunk_id,
            "metadata": metadata,
            "document": document
        })
    
    print(f"📚 Unique articles: {len(articles_data)}")
    
    # Count by document structure
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
    
    print(f"\n📑 Document Structure:")
    print(f"   • Chapters: {len(structures['chapters'])}")
    print(f"   • Parts: {len(structures['parts'])}")
    print(f"   • Articles: {len(structures['articles'])}")
    
    # Count complete vs partial articles
    complete_articles = 0
    partial_articles = 0
    for article_num, chunks in articles_data.items():
        # Check if any chunk is marked as complete article
        is_complete = any(chunk["metadata"].get("is_complete_article", False) 
                         for chunk in chunks)
        if is_complete:
            complete_articles += 1
        else:
            partial_articles += 1
    
    print(f"\n📊 Article Completeness:")
    print(f"   • Complete articles (single chunk): {complete_articles}")
    print(f"   • Partial articles (multiple chunks): {partial_articles}")
    
    # Check chunk sizes - FIXED THIS LINE
    chunk_lengths = [len(doc_text) for doc_text in results["documents"]]
    if chunk_lengths:
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        max_length = max(chunk_lengths)
        min_length = min(chunk_lengths)
        print(f"\n📏 Chunk Size Statistics:")
        print(f"   • Average length: {avg_length:.0f} characters")
        print(f"   • Min length: {min_length} characters")
        print(f"   • Max length: {max_length} characters")
    
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
    
    # Detailed display of each article and its chunks
    print(f"\n{'='*80}")
    print(f"📚 DETAILED BREAKDOWN: ARTICLES AND CHUNKS")
    print(f"{'='*80}")
    
    # Sort articles numerically
    def article_sort_key(article_item):
        article_num = article_item[0]
        try:
            # Try to convert to integer for numeric sorting
            return int(article_num)
        except (ValueError, TypeError):
            # If not a number, put at the end
            return float('inf')
    
    sorted_articles = sorted(articles_data.items(), key=article_sort_key)
    
    # Limit to showing first 5 articles to avoid too much output
    max_articles_to_show = 5
    articles_shown = 0
    
    for article_id, chunks in sorted_articles:
        articles_shown += 1
        if articles_shown > max_articles_to_show:
            remaining = len(sorted_articles) - max_articles_to_show
            print(f"\n... Skipping {remaining} more articles to avoid excessive output ...")
            break
            
        print(f"\n{'─'*80}")
        print(f"📄 ARTICLE #{articles_shown}: Article {article_id}")
        print(f"{'─'*80}")
        print(f"Total chunks: {len(chunks)}\n")
        
        # Sort chunks by index
        sorted_chunks = sorted(chunks, key=lambda x: int(x["metadata"].get("chunk_index", 0)))
        
        for chunk_num, chunk_data in enumerate(sorted_chunks, 1):
            metadata = chunk_data["metadata"]
            document_text = chunk_data["document"]
            chunk_id = chunk_data["id"]
            
            print(f"  🔹 CHUNK {chunk_num}/{len(chunks)}")
            print(f"  {'┄'*76}")
            print(f"  Chunk ID: {chunk_id[:50]}...")
            print(f"  Article Header: {metadata.get('article_header', 'N/A')}")
            print(f"\n  📋 Metadata:")
            print(f"    • Article Number: {metadata.get('article_number', 'N/A')}")
            print(f"    • Chapter: {metadata.get('chapter', 'N/A')}")
            if metadata.get('chapter_title'):
                print(f"      Title: {metadata.get('chapter_title')}")
            print(f"    • Part: {metadata.get('part', 'N/A')}")
            if metadata.get('part_title'):
                print(f"      Title: {metadata.get('part_title')}")
            print(f"    • Source Document: {metadata.get('source_document', 'N/A')}")
            print(f"    • Total Chunks in Article: {metadata.get('total_chunks', 'N/A')}")
            print(f"    • Chunk Index: {metadata.get('chunk_index', 'N/A')}")
            print(f"    • Article Length: {metadata.get('article_length', 'N/A')} chars")
            print(f"    • Chunk Length: {metadata.get('chunk_length', 'N/A')} chars")
            print(f"    • Is Complete Article: {metadata.get('is_complete_article', False)}")
            print(f"    • Document Type: {metadata.get('document_type', 'N/A')}")
            
            # Display document text (truncated if too long)
            print(f"\n  📝 Document Text:")
            if len(document_text) > 500:
                # Show first 250 and last 250 characters
                preview = document_text[:250] + "..." + document_text[-250:]
                print(f"    {preview}")
                print(f"    (... truncated, full length: {len(document_text)} characters)")
            else:
                print(f"    {document_text}")
            
            print()  # Blank line between chunks
    
    # Completeness check
    print(f"\n{'='*80}")
    print(f"🔍 COMPLETENESS CHECK")
    print(f"{'='*80}")
    
    # Check for missing metadata fields
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
                    'chunk': chunk_data['id'],
                    'missing_fields': missing_fields
                })
    
    if articles_missing_metadata:
        print(f"\n⚠️  Found {len(articles_missing_metadata)} chunk(s) with missing metadata fields:")
        for item in articles_missing_metadata[:10]:  # Show first 10 only
            print(f"   • Article {item['article']}, Chunk {item['chunk'][:20]}...")
            print(f"     Missing: {', '.join(item['missing_fields'])}")
        
        if len(articles_missing_metadata) > 10:
            print(f"     ... and {len(articles_missing_metadata) - 10} more")
    else:
        print(f"\n✅ All chunks have complete required metadata")
    
    # Check chunk sequence consistency
    print(f"\n📊 Chunk Sequence Analysis:")
    sequence_issues = []
    for article_id, chunks in sorted_articles[:20]:  # Check first 20 articles
        if not chunks:
            continue
            
        chunk_indices = []
        for chunk in chunks:
            idx = chunk["metadata"].get("chunk_index")
            # Convert to int if possible
            if isinstance(idx, str) and idx.isdigit():
                chunk_indices.append(int(idx))
            elif isinstance(idx, int):
                chunk_indices.append(idx)
            else:
                chunk_indices.append(None)
        
        # Get total chunks from first chunk's metadata
        total_chunks_meta = chunks[0]["metadata"].get("total_chunks")
        total_chunks_meta = int(total_chunks_meta) if isinstance(total_chunks_meta, str) and total_chunks_meta.isdigit() else total_chunks_meta
        
        issues = []
        if None in chunk_indices:
            issues.append("Some chunks missing chunk_index")
        elif total_chunks_meta and len(chunks) != total_chunks_meta:
            issues.append(f"Expected {total_chunks_meta} chunks, found {len(chunks)}")
        else:
            # Check if indices are sequential
            expected_indices = list(range(1, len(chunks) + 1))
            if chunk_indices != expected_indices:
                issues.append(f"Chunk indices not sequential: {chunk_indices}")
        
        if issues:
            sequence_issues.append((article_id, issues))
    
    if sequence_issues:
        for article_id, issues in sequence_issues[:5]:  # Show first 5 issues
            for issue in issues:
                print(f"   ⚠️  Article {article_id}: {issue}")
        if len(sequence_issues) > 5:
            print(f"   ... and {len(sequence_issues) - 5} more articles with sequence issues")
    else:
        print(f"   ✅ All checked articles have proper chunk sequencing")
    
    # Sample retrieval test
    print(f"\n{'='*80}")
    print(f"🧪 SAMPLE RETRIEVAL TEST")
    print(f"{'='*80}")
    
    if total_chunks > 0:
        try:
            # Try to retrieve a few chunks
            sample_ids = results["ids"][:3]
            retrieved = collection.get(
                ids=sample_ids,
                include=["metadatas", "documents"]
            )
            
            print(f"\n✅ Successfully retrieved {len(retrieved['ids'])} sample chunks")
            
            # Try a similarity search
            if len(results["documents"]) > 0:
                query_text = results["documents"][0][:100]  # Use first 100 chars of first document
                query_results = collection.query(
                    query_texts=[query_text],
                    n_results=3,
                    include=["metadatas", "documents", "distances"]
                )
                
                if query_results and len(query_results["ids"][0]) > 0:
                    print(f"✅ Similarity search works")
                    print(f"   Top result: Article {query_results['metadatas'][0][0].get('article_number')}")
                    print(f"   Distance: {query_results['distances'][0][0]:.4f}")
                else:
                    print(f"⚠️  Similarity search returned no results")
        except Exception as e:
            print(f"⚠️  Retrieval test failed: {e}")
    else:
        print("⚠️  No chunks available for retrieval test")
    
    print(f"\n{'='*80}")
    print("✅ MASTER RULES DATABASE TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_vector_db()