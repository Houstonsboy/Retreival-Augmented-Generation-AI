"""
Basic RAG Retrieval Script for Legal Summaries Database

Receives query classifier JSON and retrieves top 3 relevant parent documents
with supporting evidence from target FIRAC components.
"""

import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "LegalSummariesDB"
CHROMA_COLLECTION_NAME = "legal_summaries"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

# How many parent documents to return
TOP_N_PARENTS = 3

# How many chunks to show per component per parent (max)
MAX_CHUNKS_PER_COMPONENT = 1

# Minimum score threshold to warn about low confidence
LOW_CONFIDENCE_THRESHOLD = 0.5


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_classifier_output(input_data: Any) -> Dict:
    """
    Load and parse query classifier output.
    
    Accepts:
    - Dict (already parsed JSON)
    - JSON string
    - File path to JSON file
    
    Returns parsed dictionary.
    """
    # If already a dict, return as-is
    if isinstance(input_data, dict):
        return input_data
    
    # If string, try parsing as JSON first
    if isinstance(input_data, str):
        # Check if it's a file path
        if Path(input_data).exists() and Path(input_data).suffix == '.json':
            with open(input_data, 'r') as f:
                return json.load(f)
        else:
            # Try parsing as JSON string
            try:
                return json.loads(input_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")
    
    raise ValueError(f"Unsupported input type: {type(input_data)}")


def extract_query_params(classifier_output: Dict) -> Dict[str, Any]:
    """
    Extract key parameters from classifier output.
    
    Returns:
    {
        'vector_query': str,
        'target_components': List[str],
        'legal_domains': List[Dict],
        'entities': Dict,
        'adverse_signals': List[str]
    }
    """
    return {
        'vector_query': classifier_output.get('vector_query', ''),
        'target_components': classifier_output.get('target_components', []),
        'legal_domains': classifier_output.get('legal_domains', []),
        'entities': classifier_output.get('entities', {}),
        'adverse_signals': classifier_output.get('adverse_retrieval_signals', [])
    }


def calculate_domain_boost(chunk_metadata: dict, legal_domains: list[dict]) -> float:
    """
    Calculate domain matching boost score.

    Direct matches dominate over partial matches. 
    Boost is capped at 1.0.
    """
    chunk_domain = chunk_metadata.get('legal_domain', '').lower()
    if not chunk_domain:
        return 0.0
    
    boost = 0.0
    for domain_info in legal_domains:
        query_domain = domain_info.get('domain', '').lower()
        confidence = domain_info.get('confidence_score', 0.5)
        
        if chunk_domain == query_domain:
            # Exact match dominates: count full confidence
            boost += confidence
        elif query_domain in chunk_domain or chunk_domain in query_domain:
            # Partial match contributes less (weighted by half)
            boost += confidence * 0.5

        # Stop early if we already reach the cap
        if boost >= 1.0:
            boost = 1.0
            break

    return min(boost, 1.0)



def calculate_chunk_score(distance: float, chunk_metadata: Dict, 
                         target_components: List[str], legal_domains: List[Dict]) -> Dict:
    """
    Calculate comprehensive score for a chunk.
    
    Returns:
    {
        'final_score': float,
        'base_similarity': float,
        'component_match': bool,
        'domain_boost': float,
        'match_reasons': List[str]
    }
    """
    # Convert ChromaDB distance to similarity (lower distance = higher similarity)
    base_similarity = 1 / (1 + distance)
    
    # Check component match
    chunk_component = chunk_metadata.get('firac_component', '').upper()
    component_match = chunk_component in [c.upper() for c in target_components]
    
    # NEW LOGIC: Separate Procedural vs. Substantive
    # Boost "Land/Civil/Criminal" (Substance) higher than "Constitutional" (Instrument)
    substantive_domains = ['property law', 'land law', 'criminal law', 'civil law', 'family law']
    
    domain_boost = calculate_domain_boost(chunk_metadata, legal_domains)
    
    chunk_domain = chunk_metadata.get('legal_domain', '').lower()
    
    # Boost substances over instruments
    if any(sd in chunk_domain for sd in substantive_domains):
        multiplier = 0.5 # Give more weight to the facts
    else:
        multiplier = 0.2 # Give less weight to the filing type
        
    final_score = base_similarity + (multiplier * domain_boost)
    
    # Build match reasons
    match_reasons = []
    if base_similarity > 0.7:
        match_reasons.append("High semantic similarity")
    elif base_similarity > 0.5:
        match_reasons.append("Moderate semantic similarity")
    
    if component_match:
        match_reasons.append(f"Target component match ({chunk_component})")
    
    if domain_boost > 0:
        match_reasons.append(f"Legal domain overlap (boost: +{domain_boost:.2f})")
    
    return {
        'final_score': final_score,
        'base_similarity': base_similarity,
        'component_match': component_match,
        'domain_boost': domain_boost,
        'match_reasons': match_reasons if match_reasons else ["Basic relevance"]
    }


def group_by_parent(results: Dict, scores: List[Dict], target_components: List[str]) -> Dict:
    """
    Group chunks by parent document (case_identifier).
    
    Returns:
    {
        'parent_case_name': {
            'chunks': [...],
            'best_score': float,
            'evidence_count': int,
            'metadata': {...}
        }
    }
    """
    parents = {}
    
    for idx, chunk_id in enumerate(results['ids']):
        metadata = results['metadatas'][idx]
        document = results['documents'][idx]
        score_info = scores[idx]
        
        parent_id = metadata.get('case_identifier', 'Unknown Case')
        
        # Initialize parent if first time seeing it
        if parent_id not in parents:
            parents[parent_id] = {
                'chunks': [],
                'best_score': 0.0,
                'evidence_count': 0,
                'metadata': {
                    'file_name': metadata.get('file_name', 'N/A'),
                    'parties': metadata.get('parties', 'N/A'),
                    'court_level': metadata.get('court_level', 'N/A'),
                    'judge': metadata.get('judge', 'N/A'),
                    'year': metadata.get('year', 'N/A'),
                    'legal_domain': metadata.get('legal_domain', 'N/A'),
                    'winning_party': metadata.get('winning_party', 'N/A')
                }
            }
        
        # Add chunk
        chunk_data = {
            'chunk_id': chunk_id,
            'component': metadata.get('firac_component', 'N/A').upper(),
            'text': document,
            'score': score_info['final_score'],
            'base_similarity': score_info['base_similarity'],
            'domain_boost': score_info['domain_boost'],
            'match_reasons': score_info['match_reasons']
        }
        
        parents[parent_id]['chunks'].append(chunk_data)
        
        # Update best score
        if score_info['final_score'] > parents[parent_id]['best_score']:
            parents[parent_id]['best_score'] = score_info['final_score']
        
        # Count evidence from target components
        if score_info['component_match']:
            parents[parent_id]['evidence_count'] += 1
    
    return parents


def select_best_chunks_per_parent(parent_data: Dict, target_components: List[str]) -> List[Dict]:
    """
    Select the best chunk(s) per component for a parent document.
    
    Returns max MAX_CHUNKS_PER_COMPONENT chunks per component type.
    """
    chunks_by_component = {}
    
    # Group chunks by component
    for chunk in parent_data['chunks']:
        component = chunk['component']
        if component not in chunks_by_component:
            chunks_by_component[component] = []
        chunks_by_component[component].append(chunk)
    
    # Select best chunks per component (prioritize target components)
    selected_chunks = []
    
    # First, get chunks from target components
    for component in target_components:
        comp_upper = component.upper()
        if comp_upper in chunks_by_component:
            # Sort by score and take top N
            sorted_chunks = sorted(chunks_by_component[comp_upper], 
                                  key=lambda x: x['score'], 
                                  reverse=True)
            selected_chunks.extend(sorted_chunks[:MAX_CHUNKS_PER_COMPONENT])
    
    return selected_chunks


def format_output(top_parents: List[tuple], query_params: Dict, stats: Dict) -> Dict:
    """
    Format final output in structured JSON.
    """
    results = []
    
    for rank, (parent_id, parent_data) in enumerate(top_parents, 1):
        # Select best chunks to display
        evidence = select_best_chunks_per_parent(parent_data, query_params['target_components'])
        
        results.append({
            'rank': rank,
            'parent_document': parent_id,
            'relevance_score': round(parent_data['best_score'], 3),
            'metadata': parent_data['metadata'],
            'supporting_evidence': [
                {
                    'component': chunk['component'],
                    'score': round(chunk['score'], 3),
                    'base_similarity': round(chunk['base_similarity'], 3),
                    'domain_boost': round(chunk['domain_boost'], 3),
                    'text_preview': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                    'match_reasons': ', '.join(chunk['match_reasons']),
                    'chunk_id': chunk['chunk_id']
                }
                for chunk in evidence
            ],
            'evidence_count': parent_data['evidence_count']
        })
    
    return {
        'query_summary': {
            'vector_query': query_params['vector_query'],
            'target_components': query_params['target_components'],
            'primary_domains': [
                f"{d.get('domain', 'Unknown')} ({d.get('confidence_score', 0):.1f})"
                for d in query_params['legal_domains'][:2]  # Show top 2
            ]
        },
        'results': results,
        'retrieval_stats': stats
    }


# ============================================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================================

def retrieve_relevant_cases(classifier_output: Any, verbose: bool = True) -> Dict:
    """
    Main retrieval function.
    
    Args:
        classifier_output: Query classifier JSON (dict, string, or file path)
        verbose: If True, print progress messages
    
    Returns:
        Structured dict with top 3 parent documents and supporting evidence
    """
    
    if verbose:
        print("\n" + "="*80)
        print("🔍 LEGAL RAG RETRIEVAL")
        print("="*80)
    
    # ---- STEP 1: Parse Input ----
    if verbose:
        print("\n[1/7] Parsing classifier output...")
    
    classifier_data = load_classifier_output(classifier_output)
    query_params = extract_query_params(classifier_data)
    
    if verbose:
        print(f"  ✓ Vector Query: {query_params['vector_query'][:60]}...")
        print(f"  ✓ Target Components: {', '.join(query_params['target_components'])}")
        print(f"  ✓ Legal Domains: {len(query_params['legal_domains'])} domain(s)")
    
    # ---- STEP 2: Load Embedding Model ----
    if verbose:
        print("\n[2/7] Loading embedding model...")
    
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        if verbose:
            print(f"  ✓ Loaded: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")
    
    # ---- STEP 3: Connect to Vector DB ----
    if verbose:
        print("\n[3/7] Connecting to vector database...")
    
    if not CHROMA_DB_DIR.exists():
        raise FileNotFoundError(f"Database not found at {CHROMA_DB_DIR}")
    
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        if verbose:
            print(f"  ✓ Connected to: {CHROMA_COLLECTION_NAME}")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to database: {e}")
    
    # ---- STEP 4: Embed Query ----
    if verbose:
        print("\n[4/7] Embedding vector query...")
    
    query_embedding = embedding_model.encode(query_params['vector_query']).tolist()
    
    if verbose:
        print(f"  ✓ Embedding dimension: {len(query_embedding)}")
    
    # ---- STEP 5: Semantic Search ----
    if verbose:
        print("\n[5/7] Performing semantic search...")
    
    # Query for top 50 candidates (broad net)
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=50,
        include=['documents', 'metadatas', 'distances']
    )
    
    # Flatten results (ChromaDB returns nested lists)
    results = {
        'ids': search_results['ids'][0],
        'documents': search_results['documents'][0],
        'metadatas': search_results['metadatas'][0],
        'distances': search_results['distances'][0]
    }
    
    if verbose:
        print(f"  ✓ Found {len(results['ids'])} candidate chunks")
    
    # ---- STEP 6: Filter and Score ----
    if verbose:
        print("\n[6/7] Filtering by target components and scoring...")
    
    # Calculate scores for all chunks
    all_scores = []
    for idx in range(len(results['ids'])):
        score_info = calculate_chunk_score(
            results['distances'][idx],
            results['metadatas'][idx],
            query_params['target_components'],
            query_params['legal_domains']
        )
        all_scores.append(score_info)
    
    # Filter to only target components
    filtered_results = {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}
    filtered_scores = []
    
    for idx in range(len(results['ids'])):
        if all_scores[idx]['component_match']:
            filtered_results['ids'].append(results['ids'][idx])
            filtered_results['documents'].append(results['documents'][idx])
            filtered_results['metadatas'].append(results['metadatas'][idx])
            filtered_results['distances'].append(results['distances'][idx])
            filtered_scores.append(all_scores[idx])
    
    if verbose:
        print(f"  ✓ After component filter: {len(filtered_results['ids'])} chunks")
    
    # Handle edge case: no matches
    if len(filtered_results['ids']) == 0:
        if verbose:
            print("\n  ⚠️  WARNING: No chunks found in target components!")
            print("  Falling back to top overall matches...")
        # Use all results instead
        filtered_results = results
        filtered_scores = all_scores
    
    # Group by parent document
    parents = group_by_parent(filtered_results, filtered_scores, query_params['target_components'])
    
    if verbose:
        print(f"  ✓ Grouped into {len(parents)} unique parent documents")
    
    # Sort parents by best score
    sorted_parents = sorted(parents.items(), key=lambda x: x[1]['best_score'], reverse=True)
    
    # Take top N
    top_parents = sorted_parents[:TOP_N_PARENTS]
    
    # ---- STEP 7: Format Output ----
    if verbose:
        print(f"\n[7/7] Formatting results...")
    
    stats = {
        'total_candidates': len(results['ids']),
        'after_component_filter': len(filtered_results['ids']),
        'unique_parents_found': len(parents),
        'returned_parents': len(top_parents)
    }
    
    output = format_output(top_parents, query_params, stats)
    
    # Check for low confidence
    if top_parents and top_parents[0][1]['best_score'] < LOW_CONFIDENCE_THRESHOLD:
        output['warning'] = "Low confidence results - consider refining query"
    
    if verbose:
        print(f"  ✓ Returning top {len(top_parents)} parent documents")
        print("\n" + "="*80)
        print("✅ RETRIEVAL COMPLETE")
        print("="*80 + "\n")
    
    return output


# ============================================================================
# DISPLAY FUNCTION
# ============================================================================

def display_results(output: Dict):
    """
    Pretty-print retrieval results to console.
    """
    print("\n" + "="*80)
    print("📊 RETRIEVAL RESULTS")
    print("="*80)
    
    # Query Summary
    print("\n🔎 Query Summary:")
    print(f"  Vector Query: {output['query_summary']['vector_query'][:70]}...")
    print(f"  Target Components: {', '.join(output['query_summary']['target_components'])}")
    print(f"  Primary Domains: {', '.join(output['query_summary']['primary_domains'])}")
    
    # Warning if present
    if 'warning' in output:
        print(f"\n⚠️  {output['warning']}")
    
    # Results
    print("\n" + "="*80)
    print("🏆 TOP MATCHING CASES")
    print("="*80)
    
    for result in output['results']:
        print(f"\n{'─'*80}")
        print(f"#{result['rank']} | {result['parent_document']}")
        print(f"{'─'*80}")
        print(f"📈 Relevance Score: {result['relevance_score']:.3f}")
        print(f"📋 Court: {result['metadata']['court_level']} | Judge: {result['metadata']['judge']}")
        print(f"📅 Year: {result['metadata']['year']} | Domain: {result['metadata']['legal_domain']}")
        print(f"⚖️  Winning Party: {result['metadata']['winning_party']}")
        print(f"📊 Evidence Chunks: {result['evidence_count']} from target components")
        
        print(f"\n  🔍 Supporting Evidence:")
        for evidence in result['supporting_evidence']:
            print(f"\n    [{evidence['component']}] Score: {evidence['score']:.3f}")
            print(f"    Match: {evidence['match_reasons']}")
            print(f"    Text: {evidence['text_preview']}")
            print(f"    (Chunk ID: {evidence['chunk_id']})")
    
    # Stats
    print("\n" + "="*80)
    print("📈 Retrieval Statistics")
    print("="*80)
    stats = output['retrieval_stats']
    print(f"  Total candidates searched: {stats['total_candidates']}")
    print(f"  After component filtering: {stats['after_component_filter']}")
    print(f"  Unique parents found: {stats['unique_parents_found']}")
    print(f"  Parents returned: {stats['returned_parents']}")
    print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Example usage - replace with actual classifier output
    
    # You can pass classifier output in several ways:
    
    # Option 1: As a dictionary
    sample_classifier_output = {
        "intents": ["OUTCOME_ANALYSIS", "RULE_SEARCH"],
        "target_components": ["RULES", "APPLICATION", "CONCLUSION"],
        "legal_domains": [
            {"domain": "Criminal Law", "confidence_score": 0.9},
            {"domain": "Constitutional Law", "confidence_score": 0.7}
        ],
        "entities": {
            "statutes": ["Penal Code", "Constitution of Kenya"],
            "cases": [],
            "judges": []
        },
        "adverse_retrieval_signals": ["circumstantial evidence"],
        "vector_query": "murder, circumstantial evidence, Section 204 Penal Code, mandatory death sentence",
        "reasoning_summary": "Testing basic retrieval..."
    }
    
    # Option 2: As a JSON file path
    # sample_classifier_output = "path/to/classifier_output.json"
    
    # Option 3: As a JSON string
    # sample_classifier_output = '{"target_components": ["RULES"], ...}'
    
    try:
        # Run retrieval
        results = retrieve_relevant_cases(sample_classifier_output, verbose=True)
        
        # Display results
        display_results(results)
        
        # Optionally save to file
        save_option = input("\nSave results to JSON file? (y/n): ").strip().lower()
        if save_option == 'y':
            output_file = BASE_DIR / "retrieval_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)