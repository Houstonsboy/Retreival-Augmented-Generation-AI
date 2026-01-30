"""
Rules Retrieval Script for MasterRulesDB

Searches Constitution and Evidence Act when "RULES" is in target_components.
Retrieves relevant statutory provisions based on vector query and filters.
"""

import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional
import urllib.parse

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "constitutionlogic" / "MasterRulesDB"
CHROMA_COLLECTION_NAME = "master_rules"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

# Statute mapping
STATUTE_MAP = {
    "Constitution of Kenya": "TheConstitutionOfKenya.pdf",
    "Constitution": "TheConstitutionOfKenya.pdf",
    "Evidence Act": "Evidence Act.pdf",
    "Evidence": "Evidence Act.pdf"
}

# Document type labels
DOC_TYPE_LABELS = {
    "constitution": "Constitution of Kenya",
    "evidence_act": "Evidence Act"
}


def load_classifier_output(input_data: Any) -> Dict:
    """Load and parse classifier output from various input types."""
    if isinstance(input_data, dict):
        return input_data
    
    if isinstance(input_data, str):
        if input_data.strip().startswith('{'):
            try:
                return json.loads(input_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")
        
        # Try to load from file path
        file_path = Path(input_data)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"File not found: {input_data}")
    
    raise ValueError(f"Unsupported input type: {type(input_data)}")


def extract_query_params(classifier_output: Dict) -> Dict[str, Any]:
    """Extract key parameters from classifier output."""
    return {
        'vector_query': classifier_output.get('vector_query', ''),
        'target_components': classifier_output.get('target_components', []),
        'legal_domains': classifier_output.get('legal_domains', []),
        'entities': classifier_output.get('entities', {}),
        'intents': classifier_output.get('intents', [])
    }


def build_where_filter(query_params: Dict) -> Dict:
    """Build ChromaDB where filter based on query parameters."""
    where_filter = {}
    
    # Always filter for both constitution and evidence_act
    where_filter["document_type"] = {"$in": ["constitution", "evidence_act"]}
    
    # Optional: Filter by specific statute if mentioned
    entities = query_params.get('entities', {})
    statutes = entities.get('statutes', [])
    
    if statutes:
        # Map statute names to actual PDF filenames
        source_docs = []
        for statute in statutes:
            mapped_doc = STATUTE_MAP.get(statute)
            if mapped_doc:
                source_docs.append(mapped_doc)
        
        if source_docs:
            where_filter["source_document"] = {"$in": source_docs}
    
    return where_filter


def calculate_rule_score(distance: float, chunk_metadata: Dict, legal_domains: List[Dict]) -> float:
    """Calculate relevance score for a rule chunk."""
    base_similarity = 1 / (1 + distance)
    
    # Optional: Boost based on legal domain match
    # For now, keep it simple - just use similarity
    return base_similarity


def group_by_article(results: Dict, scores: List[float]) -> Dict:
    """Group chunks by article (multiple chunks can belong to same article)."""
    articles = {}
    
    for idx, chunk_id in enumerate(results['ids'][0]):
        metadata = results['metadatas'][0][idx]
        doc_text = results['documents'][0][idx]
        score = scores[idx]
        
        # Create unique article key: document_type + article_number
        doc_type = metadata.get('document_type', 'unknown')
        article_num = metadata.get('article_number', 'N/A')
        article_key = f"{doc_type}_{article_num}"
        
        if article_key not in articles:
            articles[article_key] = {
                'document_type': doc_type,
                'article_number': article_num,
                'article_header': metadata.get('article_header', 'N/A'),
                'chapter': metadata.get('chapter', 'N/A'),
                'chapter_title': metadata.get('chapter_title', 'N/A'),
                'part': metadata.get('part', ''),
                'part_title': metadata.get('part_title', ''),
                'source_document': metadata.get('source_document', ''),
                'chunks': [],
                'best_score': 0.0,
                'chunk_ids': []
            }
        
        articles[article_key]['chunks'].append({
            'text': doc_text,
            'score': score,
            'chunk_id': chunk_id
        })
        
        articles[article_key]['chunk_ids'].append(chunk_id)
        
        if score > articles[article_key]['best_score']:
            articles[article_key]['best_score'] = score
    
    return articles


def format_rules_output(articles: Dict, query_params: Dict, stats: Dict) -> Dict:
    """Format rules results in frontend-compatible structure."""
    
    # Sort articles by best score
    sorted_articles = sorted(
        articles.values(),
        key=lambda x: x['best_score'],
        reverse=True
    )[:5]  # Top 5 articles
    
    results = []
    for rank, article in enumerate(sorted_articles, 1):
        # Combine all chunks for full article text
        # Sort chunks by score to show most relevant first
        sorted_chunks = sorted(article['chunks'], key=lambda x: x['score'], reverse=True)
        full_text = " ".join([chunk['text'] for chunk in sorted_chunks])
        
        # Get preview (first 500 chars)
        text_preview = full_text[:500]
        if len(full_text) > 500:
            text_preview += "..."
        
        # Build PDF URL
        pdf_url = f"/api/rule-pdf?doc_type={article['document_type']}&source={urllib.parse.quote(article['source_document'])}"
        
        results.append({
            'rank': rank,
            'document_type': article['document_type'],
            'document_label': DOC_TYPE_LABELS.get(article['document_type'], article['document_type']),
            'article_number': article['article_number'],
            'article_header': article['article_header'],
            'chapter': article['chapter'],
            'chapter_title': article['chapter_title'],
            'part': article['part'] if article['part'] else None,
            'part_title': article['part_title'] if article['part_title'] else None,
            'source_document': article['source_document'],
            'score': round(article['best_score'], 3),
            'text_preview': text_preview,
            'full_text': full_text,
            'chunk_count': len(article['chunks']),
            'pdf_url': pdf_url
        })
    
    return {
        'results': results,
        'count': len(results),
        'query_summary': {
            'vector_query': query_params['vector_query'],
            'filtered_statutes': query_params['entities'].get('statutes', []),
            'search_type': 'statutory_provisions'
        },
        'stats': stats
    }


def retrieve_relevant_rules(classifier_output: Any, verbose: bool = True) -> Dict:
    """
    Main rules retrieval function.
    Searches MasterRulesDB for relevant Constitution and Evidence Act provisions.
    
    Args:
        classifier_output: Qclassifier output (dict, JSON string, or file path)
        verbose: Whether to print progress messages
        
    Returns:
        Dict with results in frontend-compatible format
    """
    
    if verbose:
        print("\n" + "="*80)
        print("📜 STATUTORY RULES RETRIEVAL")
        print("="*80)
    
    # ---- STEP 1: Parse Input ----
    if verbose:
        print("\n[1/6] Parsing classifier output...")
    
    classifier_data = load_classifier_output(classifier_output)
    query_params = extract_query_params(classifier_data)
    
    if verbose:
        print(f"  ✓ Vector Query: {query_params['vector_query'][:60]}...")
        print(f"  ✓ Intents: {', '.join(query_params['intents'])}")
        if query_params['entities'].get('statutes'):
            print(f"  ✓ Target Statutes: {', '.join(query_params['entities']['statutes'])}")
    
    # ---- STEP 2: Load Embedding Model ----
    if verbose:
        print("\n[2/6] Loading embedding model...")
    
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        if verbose:
            print(f"  ✓ Loaded: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")
    
    # ---- STEP 3: Connect to MasterRulesDB ----
    if verbose:
        print("\n[3/6] Connecting to MasterRulesDB...")
    
    if not CHROMA_DB_DIR.exists():
        raise FileNotFoundError(f"MasterRulesDB not found at {CHROMA_DB_DIR}")
    
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        collection_count = collection.count()
        if verbose:
            print(f"  ✓ Connected to collection: {CHROMA_COLLECTION_NAME}")
            print(f"  ✓ Total documents: {collection_count}")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to ChromaDB: {e}")
    
    # ---- STEP 4: Build Filters ----
    if verbose:
        print("\n[4/6] Building search filters...")
    
    where_filter = build_where_filter(query_params)
    if verbose:
        print(f"  ✓ Filters: {json.dumps(where_filter, indent=2)}")
    
    # ---- STEP 5: Embed Query & Search ----
    if verbose:
        print("\n[5/6] Embedding query and searching...")
    
    query_embedding = embedding_model.encode(query_params['vector_query']).tolist()
    
    if verbose:
        print(f"  ✓ Embedding dimension: {len(query_embedding)}")
    
    # Search with filters
    results = collection.query(
        query_embeddings=[query_embedding],
        where=where_filter,
        n_results=30,  # Get more chunks, will group by article
        include=["documents", "metadatas", "distances"]
    )
    
    if verbose:
        print(f"  ✓ Retrieved {len(results['ids'][0])} chunks")
    
    # ---- STEP 6: Group and Format Results ----
    if verbose:
        print("\n[6/6] Grouping by article and formatting...")
    
    # Calculate scores
    scores = [calculate_rule_score(dist, results['metadatas'][0][idx], query_params['legal_domains']) 
              for idx, dist in enumerate(results['distances'][0])]
    
    # Group by article
    articles = group_by_article(results, scores)
    
    if verbose:
        print(f"  ✓ Grouped into {len(articles)} unique articles")
    
    # Compile stats
    stats = {
        'total_chunks_retrieved': len(results['ids'][0]),
        'unique_articles': len(articles),
        'top_articles_returned': min(5, len(articles)),
        'database': 'MasterRulesDB',
        'collection': CHROMA_COLLECTION_NAME
    }
    
    # Format output
    output = format_rules_output(articles, query_params, stats)
    
    if verbose:
        print(f"\n✓ Retrieval complete! Found {output['count']} relevant articles")
        print("="*80 + "\n")
    
    return output


def display_rules_results(output: Dict):
    """Pretty print rules retrieval results."""
    print("\n" + "="*80)
    print("📊 STATUTORY RULES RESULTS")
    print("="*80)
    
    # Query Summary
    print("\n🔎 Query Summary:")
    print(f"  Vector Query: {output['query_summary']['vector_query'][:70]}...")
    if output['query_summary']['filtered_statutes']:
        print(f"  Filtered Statutes: {', '.join(output['query_summary']['filtered_statutes'])}")
    
    # Stats
    print(f"\n📈 Statistics:")
    print(f"  Total Chunks Retrieved: {output['stats']['total_chunks_retrieved']}")
    print(f"  Unique Articles: {output['stats']['unique_articles']}")
    print(f"  Top Articles Returned: {output['stats']['top_articles_returned']}")
    
    # Results
    print(f"\n📜 Top {len(output['results'])} Articles:\n")
    
    for result in output['results']:
        print(f"{'─'*80}")
        print(f"[{result['rank']}] {result['document_label']} - Article {result['article_number']}")
        print(f"    {result['article_header']}")
        print(f"\n    📍 Location:")
        print(f"       Chapter: {result['chapter']} - {result['chapter_title']}")
        if result['part']:
            print(f"       Part: {result['part']} - {result['part_title']}")
        print(f"\n    📊 Relevance Score: {result['score']:.3f}")
        print(f"    📝 Chunks: {result['chunk_count']}")
        print(f"\n    📄 Text Preview:")
        print(f"       {result['text_preview'][:300]}...")
        print()
    
    print("="*80 + "\n")


# ============================================================================
# TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    # Sample test with a rule-focused query
    sample_classifier_output = {
        "intents": ["RULE_SEARCH"],
        "target_components": ["RULES"],
        "legal_domains": [
            {"domain": "Constitutional Law", "confidence_score": 0.9}
        ],
        "entities": {
            "statutes": ["Constitution of Kenya"],
            "cases": [],
            "judges": []
        },
        "vector_query": "fair trial rights, access to justice, due process, Article 47",
        "reasoning_summary": "Testing statutory retrieval for constitutional rights"
    }
    
    try:
        print("Testing MasterRulesDB Retrieval...")
        results = retrieve_relevant_rules(sample_classifier_output, verbose=True)
        display_rules_results(results)
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
