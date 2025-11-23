import os
import json
import requests
import numpy as np
from typing import List, Dict, Any, Tuple
import time
from pathlib import Path
import chromadb
from dotenv import load_dotenv  # Load environment variables from .env file
from sentence_transformers import SentenceTransformer, CrossEncoder  # IMPROVEMENT: Added CrossEncoder for reranking
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer  # IMPROVEMENT: Added for hybrid search
from groq import Groq  # UPDATED: Using Groq instead of HuggingFace
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# ===== CONFIGURATION ===== 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# UPDATED: Using Groq's models instead of HuggingFace
# Available Groq models: llama-3.3-70b-versatile, llama-3.1-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it
MODEL_NAME = "llama-3.3-70b-versatile"  # Fast and capable model

# IMPROVEMENT: Upgraded from all-MiniLM-L6-v2 (384d) to intfloat/e5-base-v2 (768d)
# e5-base-v2 provides better semantic understanding and is free from Hugging Face
# Alternative free options: "all-mpnet-base-v2" (768d, slower but better) or "BAAI/bge-base-en-v1.5"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"  # Better semantic embeddings (768 dimensions vs 384)

# IMPROVEMENT: Two-stage retrieval - retrieve more initially, then rerank to top K
INITIAL_RETRIEVAL_K = 10  # Retrieve top 10 initially for reranking
TOP_K = 3  # Final number of chunks after reranking

# IMPROVEMENT: Enable reranking and query expansion by default
USE_RERANKING = True
USE_QUERY_EXPANSION = True
USE_HYBRID_SEARCH = True  # Combine semantic + keyword search

# ===== VECTOR STORE ACCESS =====
CHROMA_DB_DIR = Path(__file__).resolve().parent / "ChunkDB"
CHROMA_COLLECTION_NAME = "ChunkDB"


def get_chroma_collection():
    """
    Initialize and return the persistent ChromaDB collection with stored chunk embeddings.
    """
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        return client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize ChromaDB collection '{CHROMA_COLLECTION_NAME}' at {CHROMA_DB_DIR}. "
            "Ensure digester.py has been run to create and populate the database."
        ) from exc


def load_collection_data(collection) -> Tuple[List[str], np.ndarray, List[Document]]:
    """
    Load documents, embeddings, and metadata from the ChromaDB collection.
    """
    results = collection.get(include=["embeddings", "documents", "metadatas"])

    documents = results.get("documents") or []
    embeddings = results.get("embeddings")
    metadatas = results.get("metadatas") or []

    if embeddings is None:
        embeddings = []

    if len(documents) == 0 or len(embeddings) == 0:
        raise RuntimeError(
            f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' is empty. "
            "Run digester.py to ingest documents before querying."
        )

    embeddings_array = np.asarray(embeddings, dtype=np.float32)

    chunk_documents = [
        Document(page_content=document, metadata=metadata or {})
        for document, metadata in zip(documents, metadatas)
    ]

    return documents, embeddings_array, chunk_documents


def prepare_retrieval_resources() -> Tuple[List[str], List[Document], np.ndarray, SentenceTransformer, Any]:
    """
    Load all retrieval resources required for answering queries from the persistent vector store.
    """
    collection = get_chroma_collection()
    chunks, embeddings, chunk_documents = load_collection_data(collection)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = initialize_reranker() if USE_RERANKING else None

    return chunks, chunk_documents, embeddings, model, reranker

# ===== 3.5. QUERY EXPANSION =====
def expand_query(query: str, groq_client: Groq) -> str:
    """
    IMPROVEMENT: Added query expansion to improve retrieval
    UPDATED: Now uses Groq instead of HuggingFace
    
    Previous: Used original query only (exact match limitations)
    Now: Expands query with synonyms, variations, and related terms using Groq LLM
         This helps find relevant chunks even when wording differs
    """ 
    if not USE_QUERY_EXPANSION:
        return query
    
    print(f"Expanding query: '{query}'")
    
    try:
        # UPDATED: Use Groq to expand query with related terms and synonyms
        # This helps retrieve chunks that use different wording but same meaning
        expansion_prompt = f"""Given this query: "{query}"

Generate an expanded version that includes:
1. The original query
2. Synonyms and alternative phrasings
3. Related terms that might appear in documents
4. Key entities mentioned

Return ONLY the expanded query text, no explanation."""

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that expands search queries with synonyms and related terms."},
                {"role": "user", "content": expansion_prompt}
            ],
            model=MODEL_NAME,
            max_tokens=150,
            temperature=0.3,  # Low temperature for consistent expansion
        )
        
        expanded_query = response.choices[0].message.content.strip()
        
        # If expansion failed or is same, use original
        if not expanded_query or len(expanded_query) < len(query):
            expanded_query = query
        
        print(f"Expanded query: '{expanded_query}'")
        return expanded_query
        
    except Exception as e:
        print(f"Query expansion failed: {e}. Using original query.")
        return query

# ===== 4. QUERY PROCESSING =====
def initialize_reranker():
    """
    IMPROVEMENT: Added reranker initialization for two-stage retrieval
    
    Previous: Single-stage retrieval (cosine similarity only)
    Now: Two-stage retrieval - broad retrieval then precise reranking
         Cross-encoder provides better ranking accuracy than cosine similarity alone
    """
    if not USE_RERANKING:
        return None
    
    # IMPROVEMENT: Use free cross-encoder model for reranking
    # Cross-encoders process query-document pairs together, better than separate embeddings
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    print("Reranker initialized (cross-encoder/ms-marco-MiniLM-L-6-v2)")
    return reranker

def rerank_chunks(query: str, candidate_chunks: List[str], scores: List[float], reranker) -> Tuple[List[str], List[float]]:
    """
    IMPROVEMENT: Rerank retrieved chunks using cross-encoder
    
    Previous: Used only cosine similarity scores (single-stage)
    Now: Reranks top candidates with cross-encoder for better precision
         Cross-encoders process query-chunk pairs jointly, understanding context better
    """
    if not reranker or not USE_RERANKING:
        return candidate_chunks, scores
    
    print(f"Reranking {len(candidate_chunks)} candidate chunks...")
    
    # IMPROVEMENT: Create query-chunk pairs for cross-encoder
    # Cross-encoder needs pairs in format: [query, chunk]
    pairs = [[query, chunk] for chunk in candidate_chunks]
    
    # Get reranking scores
    rerank_scores = reranker.predict(pairs)
    
    # Sort by rerank scores (higher is better)
    sorted_indices = np.argsort(rerank_scores)[::-1]
    
    # Return reranked chunks and scores
    reranked_chunks = [candidate_chunks[i] for i in sorted_indices]
    reranked_scores = [float(rerank_scores[i]) for i in sorted_indices]
    
    print(f"Reranking completed. Top score: {reranked_scores[0]:.4f}")
    return reranked_chunks, reranked_scores

def hybrid_search(query: str, model, embeddings, chunks: List[str], semantic_weight: float = 0.7) -> List[float]:
    """
    IMPROVEMENT: Added hybrid search combining semantic + keyword matching
    
    Previous: Only semantic search (cosine similarity)
    Now: Combines semantic similarity (70%) + TF-IDF keyword matching (30%)
         This catches both semantically similar and exact keyword matches
    """
    # IMPROVEMENT: Handle small documents - if too few chunks, use semantic only
    if not USE_HYBRID_SEARCH or len(chunks) < 2:
        # Fall back to semantic search only
        if len(chunks) < 2:
            print(f"Warning: Only {len(chunks)} chunk(s) found. Using semantic search only (hybrid search requires 2+ chunks).")
        query_embedding = model.encode([f"query: {query}"], convert_to_tensor=False, normalize_embeddings=True)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        return similarities

    print("Performing hybrid search (semantic + keyword)...")

    # --- Semantic search ---
    query_embedding = model.encode([f"query: {query}"], convert_to_tensor=False, normalize_embeddings=True)
    semantic_scores = cosine_similarity(query_embedding, embeddings)[0]

    # --- Keyword (TF-IDF) search ---
    # IMPROVEMENT: Better handling for small documents
    vectorizer = TfidfVectorizer(max_features=min(1000, len(chunks) * 100), stop_words='english', min_df=1)
    try:
        chunk_tfidf = vectorizer.fit_transform(chunks)
        query_tfidf = vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, chunk_tfidf)[0]
    except Exception as e:
        print(f"TF-IDF fallback (using semantic only): {e}")
        keyword_scores = np.zeros(len(chunks))

    # --- Normalize scores ---
    # IMPROVEMENT: Check if all scores are the same (avoid division by zero)
    semantic_range = semantic_scores.max() - semantic_scores.min()
    keyword_range = keyword_scores.max() - keyword_scores.min()
    
    if semantic_range > 1e-8:
        semantic_scores = (semantic_scores - semantic_scores.min()) / semantic_range
    else:
        semantic_scores = np.ones(len(chunks))  # All equal, set to 1
        
    if keyword_range > 1e-8:
        keyword_scores = (keyword_scores - keyword_scores.min()) / keyword_range
    else:
        keyword_scores = np.zeros(len(chunks))  # All zero, keep as zero

    # --- Combine results ---
    combined_scores = semantic_weight * semantic_scores + (1 - semantic_weight) * keyword_scores

    print(f"Hybrid search completed (semantic: {np.mean(semantic_scores):.4f}, keyword: {np.mean(keyword_scores):.4f})")
    return combined_scores

def process_query(query: str, model, embeddings, chunks: List[str], 
                  reranker=None, chunk_documents: List[Document] = None,
                  top_k: int = TOP_K, groq_client: Groq = None) -> Tuple[List[str], List[float], List[int]]:
    """
    IMPROVEMENT: Enhanced query processing with expansion, hybrid search, and reranking
    UPDATED: Added groq_client parameter for query expansion
    
    Previous: Simple cosine similarity with top-k
    Now: 
    1. Query expansion (adds synonyms/variations)
    2. Hybrid search (semantic + keyword)
    3. Two-stage retrieval (broad initial retrieval, then reranking)
    4. Returns both chunks and metadata for citations
    """
    # IMPROVEMENT: Expand query first
    expanded_query = expand_query(query, groq_client) if USE_QUERY_EXPANSION and groq_client else query
    
    # IMPROVEMENT: Use hybrid search instead of just semantic
    similarities = hybrid_search(expanded_query, model, embeddings, chunks)
    
    # IMPROVEMENT: Two-stage retrieval - retrieve more initially, then rerank
    # Previous: Retrieved only TOP_K chunks directly
    # Now: Retrieve INITIAL_RETRIEVAL_K (10), then rerank to TOP_K (3)
    # IMPROVEMENT: Handle small documents - don't try to retrieve more chunks than exist
    initial_k = INITIAL_RETRIEVAL_K if USE_RERANKING else top_k
    initial_k = min(initial_k, len(chunks))  # Can't retrieve more than available
    top_k = min(top_k, len(chunks))  # Ensure top_k doesn't exceed available chunks
    
    top_indices = similarities.argsort()[-initial_k:][::-1]
    
    # Get candidate chunks
    candidate_chunks = [chunks[i] for i in top_indices]
    candidate_scores = [similarities[i] for i in top_indices]
    
    print(f"Initial retrieval: Found {len(candidate_chunks)} candidate chunks")
    for i, (score, idx) in enumerate(zip(candidate_scores, top_indices)):
        print(f"  Candidate {i+1}: Similarity={score:.4f}, Index={idx}")
    
    # IMPROVEMENT: Rerank if reranker is available
    if reranker and USE_RERANKING:
        reranked_chunks, reranked_scores = rerank_chunks(expanded_query, candidate_chunks, candidate_scores, reranker)
        # Take top_k after reranking
        final_chunks = reranked_chunks[:top_k]
        final_scores = reranked_scores[:top_k]
    else:
        final_chunks = candidate_chunks[:top_k]
        final_scores = candidate_scores[:top_k]
    
    print(f"\nFinal retrieval: Selected {len(final_chunks)} relevant chunks")
    for i, (score, chunk) in enumerate(zip(final_scores, final_chunks)):
        print(f"  Chunk {i+1}: Score={score:.4f}, Length={len(chunk)} chars")
    
    # IMPROVEMENT: Track which chunk indices were selected for citation tracking
    # This allows us to map retrieved chunks back to their original document positions
    if reranker and USE_RERANKING:
        # After reranking, we need to find original indices
        final_indices = []
        for chunk in final_chunks:
            try:
                idx = candidate_chunks.index(chunk)
                final_indices.append(top_indices[idx])
            except ValueError:
                # If chunk not found, use first available index
                final_indices.append(0)
    else:
        final_indices = top_indices[:top_k].tolist()
    
    return final_chunks, final_scores, final_indices

# ===== 5. CONTEXT PREPARATION =====
def deduplicate_chunks(chunks: List[str], scores: List[float], 
                      chunk_indices: List[int] = None) -> Tuple[List[str], List[float], List[int]]:
    """
    IMPROVEMENT: Added deduplication to remove redundant information
    
    Previous: No deduplication, overlapping chunks could repeat information
    Now: Removes chunks that are too similar to already included chunks
         Reduces context size and improves answer quality
    """
    if len(chunks) <= 1:
        return chunks, scores, chunk_indices if chunk_indices else []
    
    print("Deduplicating chunks...")
    
    deduplicated_chunks = [chunks[0]]
    deduplicated_scores = [scores[0]]
    deduplicated_indices = [chunk_indices[0]] if chunk_indices and len(chunk_indices) > 0 else []
    
    # IMPROVEMENT: Use simple similarity check based on text overlap
    # If a chunk overlaps significantly with an already included chunk, skip it
    for i in range(1, len(chunks)):
        chunk = chunks[i]
        score = scores[i]
        is_duplicate = False
        
        for existing_chunk in deduplicated_chunks:
            # Calculate overlap ratio (simple word-based)
            chunk_words = set(chunk.lower().split())
            existing_words = set(existing_chunk.lower().split())
            
            if len(chunk_words) > 0 and len(existing_words) > 0:
                overlap_ratio = len(chunk_words.intersection(existing_words)) / max(len(chunk_words), len(existing_words))
                
                # IMPROVEMENT: If chunks overlap more than 70%, consider it duplicate
                if overlap_ratio > 0.7:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            deduplicated_chunks.append(chunk)
            deduplicated_scores.append(score)
            if chunk_indices and i < len(chunk_indices):
                deduplicated_indices.append(chunk_indices[i])
    
    removed = len(chunks) - len(deduplicated_chunks)
    if removed > 0:
        print(f"Removed {removed} duplicate chunk(s)")
    
    return deduplicated_chunks, deduplicated_scores, deduplicated_indices

def prepare_context(relevant_chunks: List[str], scores: List[float], 
                    chunk_documents: List[Document] = None,
                    chunk_indices: List[int] = None) -> Tuple[str, Dict]:
    """
    IMPROVEMENT: Enhanced context preparation with deduplication and citation tracking
    
    Previous: Simple concatenation with scores
    Now: 
    1. Deduplicates overlapping chunks
    2. Orders by relevance score (already done by reranking)
    3. Adds citation metadata for traceability
    4. Formats context with better structure
    """
    print("Preparing context from relevant chunks...")
    
    # IMPROVEMENT: Deduplicate chunks to remove redundant information
    # Previous: No deduplication
    # Now: Removes chunks that overlap too much, keeping best versions
    unique_chunks, unique_scores, unique_indices = deduplicate_chunks(
        relevant_chunks, scores, chunk_indices
    )
    
    # IMPROVEMENT: Build context with citations and metadata
    # Previous: Simple "[Chunk N, Relevance: X]: content"
    # Now: Enhanced format with chunk positions and sources for citations
    context_parts = []
    citation_info = {}
    
    for i, (chunk, score) in enumerate(zip(unique_chunks, unique_scores)):
        chunk_id = i + 1
        context_parts.append(f"[Chunk {chunk_id} - Relevance Score: {score:.4f}]\n{chunk}")
        
        # IMPROVEMENT: Track citation metadata if chunk_documents available
        # Previous: Tried to find chunk by searching (error-prone)
        # Now: Use provided unique_indices for accurate mapping (after deduplication)
        if chunk_documents and unique_indices and i < len(unique_indices):
            try:
                chunk_idx = unique_indices[i]
                if chunk_idx < len(chunk_documents):
                    doc_meta = chunk_documents[chunk_idx].metadata
                    citation_info[chunk_id] = {
                        "source": doc_meta.get("source", "Unknown"),
                        "chunk_index": doc_meta.get("chunk_index", chunk_idx),
                        "start_index": doc_meta.get("start_index", "N/A"),
                             "score": float(score)
                    }
            except (IndexError, KeyError) as e:
                # If mapping fails, still add basic citation
                citation_info[chunk_id] = {
                    "source": "Unknown",
                    "chunk_index": i,
                    "start_index": "N/A",
                    "score": float(score)
                }
        else:
            # Add basic citation if no mapping available
            citation_info[chunk_id] = {
                "source": "Unknown",
                "chunk_index": i,
                "start_index": "N/A",
                "score": float(score)
            }
    
    # IMPROVEMENT: Join with clear separators
    context = "\n\n" + "="*50 + "\n\n".join(context_parts) + "\n\n" + "="*50
    
    print(f"Context prepared: {len(unique_chunks)} unique chunks, {len(context)} total characters")
    
    return context, citation_info

# ===== 6. LLM RESPONSE GENERATION =====
def generate_response(query: str, context: str, groq_client: Groq, citation_info: Dict = None) -> str:
    """
    IMPROVEMENT: Enhanced response generation with better prompting and citations
    UPDATED: Now uses Groq instead of HuggingFace
    
    Previous: Basic prompt with context and question
    Now: 
    1. Enhanced structured prompt with clear instructions
    2. Chain-of-thought reasoning for complex queries
    3. Citation tracking and source attribution
    4. Increased token limit for detailed answers
    5. Better answer verification
    """
    print("Generating response with Groq...")
    
    # IMPROVEMENT: Enhanced system prompt with structured instructions
    # Previous: Simple instruction to use only context
    # Now: More detailed instructions with examples and reasoning guidance
    system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. Answer the question using ONLY information from the context provided
2. If the context doesn't contain relevant information, say "I don't have enough information in the context to answer this question"
3. DO NOT use your own training knowledge - only use information from the context
4. Be precise and cite specific details from the context when possible
5. If information is missing or unclear in the context, acknowledge this limitation
6. For complex questions, break down your reasoning step by step
7. Use clear, concise language

When citing information, reference the chunk numbers like [Chunk 1], [Chunk 2], etc."""
    
    # IMPROVEMENT: Better structured user message
    # Previous: Simple "Context: ... Question: ..."
    # Now: Clear separation with instructions for better understanding
    user_message = f"""You are given context from a document and a question to answer.

CONTEXT FROM DOCUMENT:
{context}

QUESTION TO ANSWER:
{query}

INSTRUCTIONS:
1. Carefully read the context above
2. Find the relevant information to answer the question
3. Synthesize the information from the context chunks
4. Provide a clear, accurate answer based ONLY on the context
5. If the context doesn't have enough information, clearly state that
6. Reference chunk numbers [Chunk N] when citing specific information
    
Please provide your answer now:"""

    try:
        # UPDATED: Use Groq client for chat completion
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model=MODEL_NAME,
            max_tokens=800,  # Increased from 500 for better answers
            temperature=0.3,  # IMPROVEMENT: Lower temperature (0.3 vs 0.7) for more focused, factual answers
        )
        
        # Extract the assistant's response
        answer = response.choices[0].message.content
        
        # IMPROVEMENT: Add citation information if available
        if citation_info:
            answer += "\n\n---\nCitation Sources:\n"
            for chunk_id, info in citation_info.items():
                answer += f"[Chunk {chunk_id}]: Source: {info.get('source', 'Unknown')}, "
                answer += f"Position: {info.get('start_index', 'N/A')}\n"
        
        print("Response generated successfully")
        return answer
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {str(e)}"

# ===== 7. MAIN RAG PIPELINE =====
def run_rag_pipeline(query: str):
    """
    IMPROVEMENT: Enhanced main RAG pipeline with all improvements
    UPDATED: Now uses Groq client
    
    Previous: Basic pipeline with simple retrieval
    Now: Full pipeline with:
    1. Text cleaning
    2. Enhanced chunking with metadata
    3. Better embeddings (e5-base-v2)
    4. Query expansion
    5. Hybrid search
    6. Two-stage retrieval with reranking
    7. Context deduplication
    8. Citation tracking
    9. Enhanced response generation
    """
    print("\n===== STARTING ENHANCED RAG PIPELINE =====")
    
    # Initialize Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    try:
        chunks, chunk_documents, embeddings, model, reranker = prepare_retrieval_resources()
    except RuntimeError as exc:
        return str(exc)

    relevant_chunks, scores, chunk_indices = process_query(
        query, 
        model, 
        embeddings, 
        chunks, 
        reranker=reranker,
        chunk_documents=chunk_documents,
        top_k=TOP_K,
        groq_client=groq_client,
    )
    
    context, citation_info = prepare_context(
        relevant_chunks, 
        scores, 
        chunk_documents=chunk_documents,
        chunk_indices=chunk_indices,
    )
    
    response = generate_response(query, context, groq_client, citation_info)
    
    print("\n===== ENHANCED RAG PIPELINE COMPLETED =====")
    return response

# ===== 8. INTERACTIVE MODE =====
def interactive_mode():
    """
    IMPROVEMENT: Enhanced interactive mode with all improvements
    UPDATED: Now uses Groq client
    
    Previous: Basic interactive mode with simple retrieval
    Now: Full interactive mode with all improvements:
    - Text cleaning
    - Enhanced chunking with metadata
    - Better embeddings
    - Query expansion
    - Hybrid search
    - Two-stage retrieval with reranking
    - Context deduplication
    - Citation tracking
    """
    print("\n===== STARTING ENHANCED INTERACTIVE MODE =====")
    
    # Initialize Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    try:
        chunks, chunk_documents, embeddings, model, reranker = prepare_retrieval_resources()
    except RuntimeError as exc:
        print(f"Failed to initialize retrieval resources: {exc}")
        return
    
    print("\n" + "=" * 60)
    print("ENHANCED RAG SYSTEM READY")
    print("=" * 60)
    print(f"Loaded {len(chunks)} chunk(s) from persistent ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
    print(f"ChromaDB path: {CHROMA_DB_DIR}")
    print("Features enabled:")
    print(f"  - Query Expansion: {USE_QUERY_EXPANSION}")
    print(f"  - Hybrid Search: {USE_HYBRID_SEARCH}")
    print(f"  - Reranking: {USE_RERANKING}")
    print(f"  - Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"  - LLM Model: {MODEL_NAME} (Groq)")
    print("=" * 60)
    print("\nInteractive RAG session ready. Type 'exit' to quit.")
    
    while True:
        query = input("\n" + "-"*60 + "\nEnter your question: ")
        if query.lower() == 'exit':
            break
        
        if not query.strip():
            print("Please enter a valid question.")
            continue
            
        try:
            # IMPROVEMENT: Enhanced query processing with all improvements
            # Previous: Simple cosine similarity
            # Now: Query expansion + hybrid search + reranking
            relevant_chunks, scores, chunk_indices = process_query(
                query, 
                model, 
                embeddings, 
                chunks,
                reranker=reranker,
                chunk_documents=chunk_documents,
                top_k=TOP_K,
                groq_client=groq_client
            )
            
            # IMPROVEMENT: Enhanced context preparation
            # Previous: Simple concatenation
            # Now: Deduplication + citation tracking
            context, citation_info = prepare_context(
                relevant_chunks, 
                scores,
                chunk_documents=chunk_documents,
                chunk_indices=chunk_indices
            )
            
            # IMPROVEMENT: Enhanced response generation
            # Previous: Basic prompt
            # Now: Better prompting + citations
            response = generate_response(query, context, groq_client, citation_info)
            
            print("\n" + "="*60)
            print("ANSWER")
            print("="*60)
            print(response)
            print("="*60)
            
        except Exception as e:
            print(f"\nError processing query: {e}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__": 
    interactive_mode()