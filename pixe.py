import os
import json
import requests
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import re
from dotenv import load_dotenv  # Load environment variables from .env file
from sentence_transformers import SentenceTransformer, CrossEncoder  # IMPROVEMENT: Added CrossEncoder for reranking
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer  # IMPROVEMENT: Added for hybrid search
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# ===== CONFIGURATION ===== 
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("HF_API_TOKEN not found in environment variables. Please set it in your .env file.")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" 

# IMPROVEMENT: Upgraded from all-MiniLM-L6-v2 (384d) to intfloat/e5-base-v2 (768d)
# e5-base-v2 provides better semantic understanding and is free from Hugging Face
# Alternative free options: "all-mpnet-base-v2" (768d, slower but better) or "BAAI/bge-base-en-v1.5"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"  # Better semantic embeddings (768 dimensions vs 384)

# IMPROVEMENT: Increased chunk overlap for better context continuity
# Previous: CHUNK_SIZE = 1000, CHUNK_OVERLAP = 200
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300  # Increased from 200 to 300 for better context preservation

# IMPROVEMENT: Two-stage retrieval - retrieve more initially, then rerank to top K
INITIAL_RETRIEVAL_K = 10  # Retrieve top 10 initially for reranking
TOP_K = 3  # Final number of chunks after reranking

# IMPROVEMENT: Enable reranking and query expansion by default
USE_RERANKING = True
USE_QUERY_EXPANSION = True
USE_HYBRID_SEARCH = True  # Combine semantic + keyword search

# ===== 1. TEXT LOADING =====
def clean_text(text: str) -> str:
    """
    IMPROVEMENT: Added text cleaning function to remove noise and improve semantic understanding
    
    Previous version: Loaded raw text without cleaning
    Now: Removes navigation elements, normalizes whitespace, preserves semantic content
    """
    # Remove common navigation/UI elements (like "Search", "Sign In", "Menu", etc.)
    # These appear frequently in scraped wiki content and don't add semantic value
    navigation_words = ['Search', 'Sign In', 'Register', 'Menu', 'Explore', 'Media', 
                       'Seasons', 'Targets', 'Community', 'Skip to content']
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and common navigation patterns
        if not line:
            continue
        # Skip lines that are just navigation words
        if line in navigation_words or line.replace(' ', '').lower() in [w.lower() for w in navigation_words]:
            continue
        # Skip lines that are just page numbers or single characters
        if len(line) <= 2 and (line.isdigit() or line in ['|', '-', '_']):
            continue
        cleaned_lines.append(line)
    
    # Join back and normalize whitespace
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Normalize multiple spaces/newlines to single spaces/newlines
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def load_document(file_path: str) -> List[Document]:
    """
    IMPROVEMENT: Enhanced document loading with text cleaning
    
    Previous: Loaded raw text with all navigation/UI elements
    Now: Cleans text before creating document, improving semantic search quality
    """
    print(f"Loading document from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_content = file.read()
        
        # IMPROVEMENT: Clean the text before processing
        cleaned_content = clean_text(raw_content)
        original_size = len(raw_content)
        cleaned_size = len(cleaned_content)
        
        # Avoid division by zero for empty files
        if original_size > 0:
            percent_removed = ((original_size - cleaned_size) / original_size * 100)
            print(f"Original size: {original_size} chars, After cleaning: {cleaned_size} chars "
                  f"({percent_removed:.1f}% removed)")
        else:
            print(f"Warning: Empty file loaded")
        
        # Create LangChain Document with enhanced metadata
        document = Document(
            page_content=cleaned_content,
            metadata={
                "source": file_path,
                "file_size": cleaned_size,
                "original_size": original_size,
                "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                # IMPROVEMENT: Added metadata for tracking
                "cleaned": True
            }
        )
        
        print(f"Successfully loaded document: {cleaned_size} characters")
        return [document]
    except Exception as e:
        print(f"Error loading document: {e}")
        return []

# ===== 2. TEXT CHUNKING =====
def chunk_text(documents: List[Document]) -> Tuple[List[str], List[Document]]:
    """
    IMPROVEMENT: Enhanced chunking with metadata preservation
    
    Previous: Returned only chunk strings, lost metadata
    Now: Returns both chunk strings AND Document objects with metadata
         This enables better tracking, citations, and context preservation
    """
    print(f"Chunking text with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")
    
    if not documents:
        print("No documents to chunk")
        return [], []
    
    # IMPROVEMENT: Check if document is very small
    total_length = sum(len(doc.page_content) for doc in documents)
    if total_length == 0:
        print("Error: Document is empty after cleaning")
        return [], []
    
    # IMPROVEMENT: For very small documents, adjust chunk size
    effective_chunk_size = CHUNK_SIZE
    effective_overlap = CHUNK_OVERLAP
    
    if total_length < CHUNK_SIZE:
        print(f"Warning: Document is small ({total_length} chars). Adjusting chunk size to fit.")
        effective_chunk_size = max(500, total_length // 2)  # At least 500 chars or half the doc
        effective_overlap = min(CHUNK_OVERLAP, effective_chunk_size // 3)  # Reduce overlap proportionally
    
    # IMPROVEMENT: Enhanced text splitter with better separators
    # Previous: Basic RecursiveCharacterTextSplitter
    # Now: Added separators that respect semantic boundaries (paragraphs, sentences, etc.)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_overlap,  # IMPROVEMENT: Increased overlap from 200 to 300
        length_function=len,
        add_start_index=True,
        # IMPROVEMENT: Preserve separators for better semantic boundaries
        separators=["\n\n", "\n", ". ", " ", ""],  # Split by paragraphs first, then sentences
        is_separator_regex=False,
    )
    
    # Split the documents
    chunk_documents = text_splitter.split_documents(documents)
    
    # IMPROVEMENT: Ensure at least one chunk was created
    if len(chunk_documents) == 0:
        print("Warning: No chunks created. Creating single chunk from document.")
        # Create a single chunk from the document
        chunk_documents = documents
    
    # IMPROVEMENT: Add chunk metadata for better tracking
    for i, chunk_doc in enumerate(chunk_documents):
        chunk_doc.metadata.update({
            "chunk_id": i,
            "chunk_index": i,
            "total_chunks": len(chunk_documents),
            "chunk_size": len(chunk_doc.page_content),
        })
    
    print(f"Created {len(chunk_documents)} chunks")
    
    # Print chunk details for verification
    print("\n=== CHUNK DETAILS ===")
    for i, chunk in enumerate(chunk_documents):
        print(f"Chunk {i+1}: {len(chunk.page_content)} chars, "
              f"Start index: {chunk.metadata.get('start_index', 'N/A')}, "
              f"Source: {chunk.metadata.get('source', 'Unknown')}")
    
    # Extract chunk contents for embeddings
    chunk_contents = [chunk.page_content for chunk in chunk_documents]
    
    # IMPROVEMENT: Return both contents AND document objects
    # This allows us to track which chunk came from where, enabling citations
    return chunk_contents, chunk_documents

# ===== 3. EMBEDDING CREATION =====
def create_embeddings(chunks: List[str]):
    """
    IMPROVEMENT: Enhanced embedding creation with better model
    
    Previous: Used all-MiniLM-L6-v2 (384 dimensions, fast but limited)
    Now: Uses intfloat/e5-base-v2 (768 dimensions, better semantic understanding)
         Better captures nuanced meanings and relationships between concepts
    """
    print(f"Creating semantic embeddings with {EMBEDDING_MODEL_NAME}...")
    start_time = time.time()
    
    # IMPROVEMENT: Better embedding model with more dimensions
    # Previous: all-MiniLM-L6-v2 (384d, ~22M parameters)
    # Now: intfloat/e5-base-v2 (768d, better quality)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # IMPROVEMENT: Better encoding options for e5-base-v2
    # e5-base-v2 uses a specific format: "query: {text}" or "passage: {text}"
    # For document chunks, we use "passage:" prefix
    formatted_chunks = [f"passage: {chunk}" for chunk in chunks]
    
    # Create embeddings for each chunk
    # IMPROVEMENT: Added normalize_embeddings=True for better similarity computation
    embeddings = model.encode(
        formatted_chunks, 
        convert_to_tensor=False,
        normalize_embeddings=True,  # Normalize for cosine similarity
        show_progress_bar=True  # Show progress for large documents
    )
    
    duration = time.time() - start_time
    print(f"Created embeddings with {embeddings.shape[1]} dimensions in {duration:.2f} seconds")
    print(f"Average embedding norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
    
    return embeddings, model

# ===== 3.5. QUERY EXPANSION =====
def expand_query(query: str) -> str:
    """
    IMPROVEMENT: Added query expansion to improve retrieval
    
    Previous: Used original query only (exact match limitations)
    Now: Expands query with synonyms, variations, and related terms using free LLM
         This helps find relevant chunks even when wording differs
    """
    if not USE_QUERY_EXPANSION:
        return query
    
    print(f"Expanding query: '{query}'")
    
    try:
        client = InferenceClient(token=HF_API_TOKEN)
        
        # IMPROVEMENT: Use LLM to expand query with related terms and synonyms
        # This helps retrieve chunks that use different wording but same meaning
        expansion_prompt = f"""Given this query: "{query}"

Generate an expanded version that includes:
1. The original query
2. Synonyms and alternative phrasings
3. Related terms that might appear in documents
4. Key entities mentioned

Return ONLY the expanded query text, no explanation."""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that expands search queries with synonyms and related terms."},
            {"role": "user", "content": expansion_prompt}
        ]
        
        response = client.chat_completion(
            messages=messages,
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
                  top_k: int = TOP_K) -> Tuple[List[str], List[float], List[int]]:
    """
    IMPROVEMENT: Enhanced query processing with expansion, hybrid search, and reranking
    
    Previous: Simple cosine similarity with top-k
    Now: 
    1. Query expansion (adds synonyms/variations)
    2. Hybrid search (semantic + keyword)
    3. Two-stage retrieval (broad initial retrieval, then reranking)
    4. Returns both chunks and metadata for citations
    """
    # IMPROVEMENT: Expand query first
    expanded_query = expand_query(query) if USE_QUERY_EXPANSION else query
    
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
def generate_response(query: str, context: str, citation_info: Dict = None) -> str:
    """
    IMPROVEMENT: Enhanced response generation with better prompting and citations
    
    Previous: Basic prompt with context and question
    Now: 
    1. Enhanced structured prompt with clear instructions
    2. Chain-of-thought reasoning for complex queries
    3. Citation tracking and source attribution
    4. Increased token limit for detailed answers
    5. Better answer verification
    """
    print("Generating response with Mistral from Hugging Face...")
    
    # Check if token is set
    if not HF_API_TOKEN:
        return "Error: HF_API_TOKEN environment variable not set"
    
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
        # Initialize Hugging Face Inference Client
        client = InferenceClient(token=HF_API_TOKEN)
        
        # Use chat_completion for Mistral Instruct models
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # IMPROVEMENT: Increased max_tokens for more detailed answers
        # Previous: max_tokens=500 (often too short)
        # Now: max_tokens=800 (allows for detailed, well-reasoned answers)
        response = client.chat_completion(
            messages=messages,
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
def run_rag_pipeline(file_path: str, query: str):
    """
    IMPROVEMENT: Enhanced main RAG pipeline with all improvements
    
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
    
    # IMPROVEMENT: Step 1 - Load and clean document
    # Previous: Basic loading
    # Now: Text cleaning removes noise, improves semantic search
    documents = load_document(file_path)
    if not documents:
        return "Failed to load document"
    
    # IMPROVEMENT: Step 2 - Enhanced chunking with metadata preservation
    # Previous: Returned only chunk strings
    # Now: Returns both chunks and Document objects for citation tracking
    chunks, chunk_documents = chunk_text(documents)
    
    if not chunks:
        return "Failed to chunk document"
    
    # IMPROVEMENT: Step 3 - Better embeddings with e5-base-v2
    # Previous: all-MiniLM-L6-v2 (384d)
    # Now: intfloat/e5-base-v2 (768d, better semantic understanding)
    embeddings, model = create_embeddings(chunks)
    
    # IMPROVEMENT: Step 4 - Initialize reranker for two-stage retrieval
    # Previous: No reranking
    # Now: Two-stage retrieval improves precision
    reranker = initialize_reranker() if USE_RERANKING else None
    
    # IMPROVEMENT: Step 5 - Enhanced query processing
    # Previous: Simple cosine similarity
    # Now: Query expansion + hybrid search + reranking
    relevant_chunks, scores, chunk_indices = process_query(
        query, 
        model, 
        embeddings, 
        chunks, 
        reranker=reranker,
        chunk_documents=chunk_documents,
        top_k=TOP_K
    )
    
    # IMPROVEMENT: Step 6 - Enhanced context preparation with deduplication
    # Previous: Simple concatenation
    # Now: Deduplication + citation tracking
    context, citation_info = prepare_context(
        relevant_chunks, 
        scores, 
        chunk_documents=chunk_documents,
        chunk_indices=chunk_indices
    )
    
    # IMPROVEMENT: Step 7 - Enhanced response generation
    # Previous: Basic prompt
    # Now: Better prompting + citations + increased token limit
    response = generate_response(query, context, citation_info)
    
    print("\n===== ENHANCED RAG PIPELINE COMPLETED =====")
    return response

# ===== 8. INTERACTIVE MODE =====
def interactive_mode(file_path: str):
    """
    IMPROVEMENT: Enhanced interactive mode with all improvements
    
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
    
    # IMPROVEMENT: Load and prepare document once with all improvements
    # Previous: Basic loading and chunking
    # Now: Cleaned text, enhanced chunking with metadata preservation
    documents = load_document(file_path)
    if not documents:
        print("Failed to load document. Exiting...")
        return
    
    # IMPROVEMENT: Enhanced chunking returns both chunks and document objects
    # Previous: Only chunks
    # Now: Chunks + metadata for citation tracking
    chunks, chunk_documents = chunk_text(documents)
    
    if not chunks:
        print("Failed to chunk document. Exiting...")
        return
    
    # IMPROVEMENT: Better embeddings with e5-base-v2
    # Previous: all-MiniLM-L6-v2 (384d)
    # Now: intfloat/e5-base-v2 (768d)
    embeddings, model = create_embeddings(chunks)
    
    # IMPROVEMENT: Initialize reranker once for reuse across queries
    # Previous: No reranking
    # Now: Two-stage retrieval for better precision
    reranker = initialize_reranker() if USE_RERANKING else None
    
    print("\n" + "="*60)
    print("ENHANCED RAG SYSTEM READY")
    print("="*60)
    print("Features enabled:")
    print(f"  - Query Expansion: {USE_QUERY_EXPANSION}")
    print(f"  - Hybrid Search: {USE_HYBRID_SEARCH}")
    print(f"  - Reranking: {USE_RERANKING}")
    print(f"  - Embedding Model: {EMBEDDING_MODEL_NAME}")
    print("="*60)
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
                top_k=TOP_K
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
            response = generate_response(query, context, citation_info)
            
            print("\n" + "="*60)
            print("ANSWER")
            print("="*60)
            print(response)
            print("="*60)
            
        except Exception as e:
            print(f"\nError processing query: {e}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__": 
    file_path = 'tmnt.txt'
    interactive_mode(file_path)