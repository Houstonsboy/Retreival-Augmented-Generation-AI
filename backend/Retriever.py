"""
Enhanced RAG Retrieval Script for Legal Summaries Database

Receives query classifier JSON and retrieves top 3 relevant parent documents
with LLM-generated case summaries based on FIRAC components.
"""

import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional
import sys
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "LegalSummariesDB"
CHROMA_COLLECTION_NAME = "legal_summaries"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# How many parent documents to return
TOP_N_PARENTS = 3

# How many chunks to show per component per parent (max)
MAX_CHUNKS_PER_COMPONENT = 1

# Minimum score threshold to warn about low confidence
LOW_CONFIDENCE_THRESHOLD = 0.5


# ============================================================================
# LLM SUMMARY GENERATION
# ============================================================================

def generate_case_summary(case_data: Dict, query_info: Dict, verbose: bool = True) -> Dict:
    """
    Generate an LLM-powered summary of a case based on its FIRAC components.
    
    Args:
        case_data: Dictionary containing case metadata and chunks
        query_info: Dictionary containing query details
        verbose: Print progress messages
    
    Returns:
        Dictionary with 'summary' and 'relevance_explanation' fields
    """
    if verbose:
        print(f"  🤖 Generating AI summary for: {case_data['parent_document']}")
    
    # Prepare FIRAC components text
    components_text = ""
    for chunk in case_data['chunks']:
        component = chunk['component']
        text = chunk['text']
        components_text += f"\n\n[{component}]\n{text}"
    
    # Construct prompt
    prompt = f"""You are a legal research assistant analyzing a court case for relevance to a user's query.

USER'S QUERY:
{query_info['vector_query']}

TARGET LEGAL AREAS:
{', '.join([f"{d.get('domain', 'Unknown')} ({d.get('confidence_score', 0):.1f})" for d in query_info['legal_domains'][:3]])}

CASE TO ANALYZE:
Case Name: {case_data['parent_document']}
Court: {case_data['metadata']['court_level']}
Judge: {case_data['metadata']['judge']}
Year: {case_data['metadata']['year']}
Legal Domain: {case_data['metadata']['legal_domain']}
Winning Party: {case_data['metadata']['winning_party']}

CASE FIRAC COMPONENTS:
{components_text}

TASK:
Provide a comprehensive analysis in exactly this format:

**CASE OVERVIEW:**
[2-3 sentences summarizing what this case is about - the parties, the dispute, and the outcome]

**KEY LEGAL PRINCIPLES:**
[2-3 sentences explaining the main legal rules or doctrines established or applied in this case]

**RELEVANCE TO YOUR QUERY:**
[3-4 sentences explaining specifically how this case relates to the user's query. Be concrete about which aspects are most relevant and why this case matters for their situation]

**CRITICAL FACTS:**
[2-3 sentences highlighting the most important factual elements that influenced the court's decision]

Keep your response clear, professional, and focused on practical legal insights. Use plain language while maintaining legal accuracy."""

    # Make API call
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Kenyan legal analyst who provides clear, insightful case summaries for lawyers and legal researchers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        summary = result['choices'][0]['message']['content'].strip()
        
        if verbose:
            print(f"    ✓ Summary generated ({len(summary)} characters)")
        
        return {
            'summary': summary,
            'generated': True
        }
        
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"    ⚠️  API call failed: {str(e)}")
        
        # Fallback to simple concatenation
        fallback = f"**Case:** {case_data['parent_document']}\n\n"
        fallback += f"**Domain:** {case_data['metadata']['legal_domain']}\n\n"
        fallback += "**Key Components:**\n"
        for chunk in case_data['chunks'][:3]:
            fallback += f"\n[{chunk['component']}] {chunk['text'][:200]}...\n"
        
        return {
            'summary': fallback,
            'generated': False,
            'error': str(e)
        }


# ============================================================================
# HELPER FUNCTIONS (keeping existing ones)
# ============================================================================

def load_classifier_output(input_data: Any) -> Dict:
    """Load and parse query classifier output."""
    if isinstance(input_data, dict):
        return input_data
    
    if isinstance(input_data, str):
        if Path(input_data).exists() and Path(input_data).suffix == '.json':
            with open(input_data, 'r') as f:
                return json.load(f)
        else:
            try:
                return json.loads(input_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")
    
    raise ValueError(f"Unsupported input type: {type(input_data)}")


def extract_query_params(classifier_output: Dict) -> Dict[str, Any]:
    """Extract key parameters from classifier output."""
    return {
        'vector_query': classifier_output.get('vector_query', ''),
        'target_components': classifier_output.get('target_components', []),
        'legal_domains': classifier_output.get('legal_domains', []),
        'entities': classifier_output.get('entities', {}),
        'adverse_signals': classifier_output.get('adverse_retrieval_signals', [])
    }


def calculate_domain_boost(chunk_metadata: dict, legal_domains: list[dict]) -> float:
    """Calculate domain matching boost score."""
    chunk_domain = chunk_metadata.get('legal_domain', '').lower()
    if not chunk_domain:
        return 0.0
    
    boost = 0.0
    for domain_info in legal_domains:
        query_domain = domain_info.get('domain', '').lower()
        confidence = domain_info.get('confidence_score', 0.5)
        
        if chunk_domain == query_domain:
            boost += confidence
        elif query_domain in chunk_domain or chunk_domain in query_domain:
            boost += confidence * 0.5

        if boost >= 1.0:
            boost = 1.0
            break

    return min(boost, 1.0)


def calculate_chunk_score(distance: float, chunk_metadata: Dict, 
                         target_components: List[str], legal_domains: List[Dict]) -> Dict:
    """Calculate comprehensive score for a chunk."""
    base_similarity = 1 / (1 + distance)
    
    chunk_component = chunk_metadata.get('firac_component', '').upper()
    component_match = chunk_component in [c.upper() for c in target_components]
    
    substantive_domains = ['property law', 'land law', 'criminal law', 'civil law', 'family law']
    domain_boost = calculate_domain_boost(chunk_metadata, legal_domains)
    chunk_domain = chunk_metadata.get('legal_domain', '').lower()
    
    if any(sd in chunk_domain for sd in substantive_domains):
        multiplier = 0.5
    else:
        multiplier = 0.2
        
    final_score = base_similarity + (multiplier * domain_boost)
    
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
    """Group chunks by parent document (case_identifier)."""
    parents = {}
    
    for idx, chunk_id in enumerate(results['ids']):
        metadata = results['metadatas'][idx]
        document = results['documents'][idx]
        score_info = scores[idx]
        
        parent_id = metadata.get('case_identifier', 'Unknown Case')
        
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
        
        if score_info['final_score'] > parents[parent_id]['best_score']:
            parents[parent_id]['best_score'] = score_info['final_score']
        
        if score_info['component_match']:
            parents[parent_id]['evidence_count'] += 1
    
    return parents


def select_best_chunks_per_parent(parent_data: Dict, target_components: List[str]) -> List[Dict]:
    """Select the best chunk(s) per component for a parent document."""
    chunks_by_component = {}
    
    for chunk in parent_data['chunks']:
        component = chunk['component']
        if component not in chunks_by_component:
            chunks_by_component[component] = []
        chunks_by_component[component].append(chunk)
    
    selected_chunks = []
    
    for component in target_components:
        comp_upper = component.upper()
        if comp_upper in chunks_by_component:
            sorted_chunks = sorted(chunks_by_component[comp_upper], 
                                  key=lambda x: x['score'], 
                                  reverse=True)
            selected_chunks.extend(sorted_chunks[:MAX_CHUNKS_PER_COMPONENT])
    
    return selected_chunks


def format_output(top_parents: List[tuple], query_params: Dict, stats: Dict, verbose: bool = True) -> Dict:
    """
    Format final output with LLM-generated summaries in frontend-compatible structure.
    
    Returns structure matching: data.retrieval.results[0].ai_summary
    """
    results = []
    
    if verbose:
        print("\n" + "="*80)
        print("🤖 GENERATING AI CASE SUMMARIES")
        print("="*80)
    
    for rank, (parent_id, parent_data) in enumerate(top_parents, 1):
        # Prepare case data for LLM
        case_info = {
            'parent_document': parent_id,
            'metadata': parent_data['metadata'],
            'chunks': parent_data['chunks']
        }
        
        # Generate LLM summary
        summary_result = generate_case_summary(case_info, query_params, verbose)
        
        # Select best chunks for evidence
        evidence_chunks = select_best_chunks_per_parent(parent_data, query_params['target_components'])
        
        # Format result in FRONTEND-COMPATIBLE STRUCTURE
        results.append({
            'rank': rank,
            'case_name': parent_id,  # Frontend expects 'case_name'
            'parent_document': parent_id,
            'score': round(parent_data['best_score'], 3),  # Frontend expects 'score'
            
            # PRIMARY FIELD FOR AI RESPONSE WIDGET
            'ai_summary': summary_result['summary'],
            
            # METADATA FOR CLASSIFICATION WIDGET
            'metadata': {
                'file_name': parent_data['metadata']['file_name'],
                'parties': parent_data['metadata']['parties'],
                'court_level': parent_data['metadata']['court_level'],
                'judge': parent_data['metadata']['judge'],
                'year': parent_data['metadata']['year'],
                'legal_domain': parent_data['metadata']['legal_domain'],
                'winning_party': parent_data['metadata']['winning_party']
            },
            
            # MATCHED COMPONENTS FOR EVIDENCE WIDGET
            'matched_components': list(set([chunk['component'] for chunk in evidence_chunks])),
            
            # BEST CHUNK TEXT FOR EVIDENCE WIDGET PREVIEW
            'best_chunk': evidence_chunks[0]['text'] if evidence_chunks else '',
            
            # DETAILED CHUNKS FOR EVIDENCE WIDGET
            'chunks': [
                {
                    'component': chunk['component'],
                    'text': chunk['text'],
                    'score': round(chunk['score'], 3),
                    'base_similarity': round(chunk['base_similarity'], 3),
                    'domain_boost': round(chunk['domain_boost'], 3),
                    'match_reasons': ', '.join(chunk['match_reasons']),
                    'chunk_id': chunk['chunk_id']
                }
                for chunk in evidence_chunks
            ],
            
            # ADDITIONAL METADATA
            'evidence_count': parent_data['evidence_count'],
            'summary_generated': summary_result.get('generated', False),
            'relevance_score': round(parent_data['best_score'], 3)
        })
    
    # Return structure that frontend expects: data.retrieval.results
    return {
        'results': results,  # THIS IS THE KEY FIELD FRONTEND USES
        'query_summary': {
            'vector_query': query_params['vector_query'],
            'target_components': query_params['target_components'],
            'primary_domains': [
                f"{d.get('domain', 'Unknown')} ({d.get('confidence_score', 0):.1f})"
                for d in query_params['legal_domains'][:2]
            ]
        },
        'retrieval_stats': stats
    }


# ============================================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================================

def retrieve_relevant_cases(classifier_output: Any, verbose: bool = True) -> Dict:
    """
    Main retrieval function with LLM-enhanced summaries.
    Returns data in frontend-compatible format.
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
    
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=50,
        include=['documents', 'metadatas', 'distances']
    )
    
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
    
    all_scores = []
    for idx in range(len(results['ids'])):
        score_info = calculate_chunk_score(
            results['distances'][idx],
            results['metadatas'][idx],
            query_params['target_components'],
            query_params['legal_domains']
        )
        all_scores.append(score_info)
    
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
    
    if len(filtered_results['ids']) == 0:
        if verbose:
            print("\n  ⚠️  WARNING: No chunks found in target components!")
            print("  Falling back to top overall matches...")
        filtered_results = results
        filtered_scores = all_scores
    
    parents = group_by_parent(filtered_results, filtered_scores, query_params['target_components'])
    
    if verbose:
        print(f"  ✓ Grouped into {len(parents)} unique parent documents")
    
    sorted_parents = sorted(parents.items(), key=lambda x: x[1]['best_score'], reverse=True)
    top_parents = sorted_parents[:TOP_N_PARENTS]
    
    # ---- STEP 7: Format Output with LLM Summaries (FRONTEND-COMPATIBLE) ----
    if verbose:
        print(f"\n[7/7] Formatting results with AI summaries...")
    
    stats = {
        'total_candidates': len(results['ids']),
        'after_component_filter': len(filtered_results['ids']),
        'unique_parents_found': len(parents),
        'returned_parents': len(top_parents)
    }
    
    output = format_output(top_parents, query_params, stats, verbose)
    
    if top_parents and top_parents[0][1]['best_score'] < LOW_CONFIDENCE_THRESHOLD:
        output['warning'] = "Low confidence results - consider refining query"
    
    if verbose:
        print(f"\n  ✓ Generated summaries for {len(top_parents)} cases")
        print("\n" + "="*80)
        print("✅ RETRIEVAL COMPLETE")
        print("="*80 + "\n")
    
    return output


# ============================================================================
# DISPLAY FUNCTION (Updated for new structure)
# ============================================================================

def display_results(output: Dict):
    """Pretty-print retrieval results with AI summaries and detailed chunk information."""
    print("\n" + "="*80)
    print("📊 RETRIEVAL RESULTS")
    print("="*80)
    
    # Query Summary
    print("\n🔎 Query Summary:")
    print(f"  Vector Query: {output['query_summary']['vector_query'][:70]}...")
    print(f"  Target Components: {', '.join(output['query_summary']['target_components'])}")
    print(f"  Primary Domains: {', '.join(output['query_summary']['primary_domains'])}")
    
    if 'warning' in output:
        print(f"\n⚠️  {output['warning']}")
    
    # Results
    print("\n" + "="*80)
    print("🏆 TOP MATCHING CASES")
    print("="*80)
    
    for result in output['results']:
        print(f"\n{'─'*80}")
        print(f"#{result['rank']} | {result['case_name']}")
        print(f"{'─'*80}")
        print(f"📈 Relevance Score: {result['score']:.3f}")
        print(f"📋 Court: {result['metadata']['court_level']} | Judge: {result['metadata']['judge']}")
        print(f"📅 Year: {result['metadata']['year']} | Domain: {result['metadata']['legal_domain']}")
        print(f"⚖️  Winning Party: {result['metadata']['winning_party']}")
        
        print(f"\n{'─'*80}")
        print("🤖 AI-GENERATED CASE ANALYSIS")
        print(f"{'─'*80}")
        print(result['ai_summary'])
        
        if not result.get('summary_generated', False):
            print("\n⚠️  Note: AI summary generation failed")
        
        print(f"\n{'─'*80}")
        print(f"📊 SUPPORTING EVIDENCE ({result['evidence_count']} chunks from target components)")
        print(f"{'─'*80}")
        
        # Display detailed chunk information
        for chunk in result['chunks']:
            print(f"\n  🔍 [{chunk['component']}] Score: {chunk['score']:.3f}")
            print(f"     Match Reasons: {chunk.get('match_reasons', 'N/A')}")
            print(f"     Base Similarity: {chunk.get('base_similarity', 'N/A'):.3f}")
            print(f"     Domain Boost: {chunk.get('domain_boost', 'N/A'):.3f}")
            preview = chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']
            print(f"     Text Preview: {preview}")
            print(f"     Chunk ID: {chunk['chunk_id']}")
    
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
        "reasoning_summary": "Testing enhanced retrieval with LLM summaries..."
    }
    
    try:
        # Run retrieval
        results = retrieve_relevant_cases(sample_classifier_output, verbose=True)
        
        # Display results
        display_results(results)
        
        # Save to file
        save_option = input("\nSave results to JSON file? (y/n): ").strip().lower()
        if save_option == 'y':
            output_file = BASE_DIR / "retrieval_results_with_summaries.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)