import os
import json
import requests
import numpy as np
from typing import List, Dict, Any
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient


# ===== CONFIGURATION ===== 
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Set your token as environment variable
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" 
CHUNK_SIZE = 1000       # Size of text chunks (approximate)
CHUNK_OVERLAP = 200     # Overlap between chunks (approximate)
TOP_K = 3               # Number of chunks to retrieve

# ===== 1. TEXT LOADING =====
def load_document(file_path: str) -> str:
    """Load the document content from file"""
    print(f"Loading document from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"Successfully loaded document: {len(content)} characters")
        return content
    except Exception as e:
        print(f"Error loading document: {e}")
        return ""

# ===== 2. TEXT CHUNKING =====
def chunk_text(text: str) -> List[str]:
    """Split text into manageable chunks using a simple approach"""
    print(f"Chunking text with approximate size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")
    
    # Simple approach: split by paragraphs first, then combine
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the chunk size and we already have content
        if len(current_chunk) + len(paragraph) > CHUNK_SIZE and current_chunk:
            # Save the current chunk
            chunks.append(current_chunk)
            
            # Start a new chunk with overlap
            last_part = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else current_chunk
            current_chunk = last_part
        
        # Add paragraph with a newline
        if current_chunk:
            current_chunk += "\n\n" + paragraph
        else:
            current_chunk = paragraph
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Created {len(chunks)} chunks")
    return chunks

# ===== 3. EMBEDDING CREATION =====
def create_embeddings(chunks: List[str]):
    """Create TF-IDF embeddings for text chunks"""
    print("Creating TF-IDF embeddings...")
    start_time = time.time()
    
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Create embeddings for each chunk
    embeddings = vectorizer.fit_transform(chunks)
    
    duration = time.time() - start_time
    print(f"Created embeddings with {embeddings.shape[1]} features in {duration:.2f} seconds")
    
    return embeddings, vectorizer

# ===== 4. QUERY PROCESSING =====
def process_query(query: str, vectorizer, embeddings, chunks: List[str], top_k: int = TOP_K):
    """Process a query to find relevant chunks"""
    print(f"Processing query: '{query}'")
    
    # Create embedding for the query
    query_embedding = vectorizer.transform([query])
    
    # Compute similarity between query and all chunks
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get indices of top-k similar chunks
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Get the relevant chunks and their similarity scores
    relevant_chunks = [chunks[i] for i in top_indices]
    relevant_scores = [similarities[i] for i in top_indices]
    
    print(f"Found {len(relevant_chunks)} relevant chunks")
    for i, (score, idx) in enumerate(zip(relevant_scores, top_indices)):
        print(f"Chunk {i+1}: Similarity={score:.4f}, Index={idx}")
    
    return relevant_chunks, relevant_scores

# ===== 5. CONTEXT PREPARATION =====
def prepare_context(relevant_chunks: List[str], scores: List[float]) -> str:
    """Prepare context from relevant chunks"""
    print("Preparing context from relevant chunks...")
    
    context = "\n\n".join([f"[Chunk {i+1}, Relevance: {score:.2f}]: {chunk}" 
                          for i, (chunk, score) in enumerate(zip(relevant_chunks, scores))])
    
    return context

# ===== 6. LLM RESPONSE GENERATION =====
def generate_response(query: str, context: str) -> str:
    """Generate response using Mistral through Hugging Face Inference API"""
    print("Generating response with Mistral from Hugging Face...")
    
    # Check if token is set
    if not HF_API_TOKEN:
        return "Error: HF_API_TOKEN environment variable not set"
    
    # Prepare the system prompt and messages
    system_prompt = ("You are a helpful assistant. Answer the question based ONLY on the context provided. "
                    "If the context doesn't contain relevant information to answer the question say that you dont know, "
                    "dont use your own trained knowledge.")
    
    user_message = f"""Context:
{context}

Question: {query}"""

    try:
        # Initialize Hugging Face Inference Client
        client = InferenceClient(token=HF_API_TOKEN)
        
        # Use chat_completion for Mistral Instruct models
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = client.chat_completion(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=500,
            temperature=0.7,
        )
        
        # Extract the assistant's response
        answer = response.choices[0].message.content
        
        print("Response generated successfully")
        return answer
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {str(e)}"

# ===== 7. MAIN RAG PIPELINE =====
def run_rag_pipeline(file_path: str, query: str):
    """Run the complete RAG pipeline"""
    print("\n===== STARTING RAG PIPELINE =====")
    
    # 1. Load the document
    document = load_document(file_path)
    if not document:
        return "Failed to load document"
    
    # 2. Chunk the document
    chunks = chunk_text(document)
    
    # 3. Create embeddings
    embeddings, vectorizer = create_embeddings(chunks)
    
    # 4. Process query
    relevant_chunks, scores = process_query(query, vectorizer, embeddings, chunks)
    
    # 5. Prepare context
    context = prepare_context(relevant_chunks, scores)
    
    # 6. Generate response
    response = generate_response(query, context)
    
    print("\n===== RAG PIPELINE COMPLETED =====")
    return response

# ===== 8. INTERACTIVE MODE =====
def interactive_mode(file_path: str):
    """Run RAG in interactive mode"""
    print("\n===== STARTING INTERACTIVE MODE =====")
    
    # Load and prepare document once
    document = load_document(file_path)
    if not document:
        print("Failed to load document. Exiting...")
        return
    
    chunks = chunk_text(document)
    embeddings, vectorizer = create_embeddings(chunks)
    
    print("\nInteractive RAG session ready. Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break
            
        relevant_chunks, scores = process_query(query, vectorizer, embeddings, chunks)
        context = prepare_context(relevant_chunks, scores)
        response = generate_response(query, context)
        
        print("\n=== ANSWER ===")
        print(response)

# Example usage
if __name__ == "__main__": 
    file_path = 'essay.txt'
    
    # Option 1: Single query
    # query = "What is the main argument in the essay?"
    # answer = run_rag_pipeline(file_path, query)
    # print("\n=== ANSWER ===")
    # print(answer)
    
    # Option 2: Interactive mode
    interactive_mode(file_path)