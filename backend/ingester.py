import hashlib
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "LegalSummariesDB"
CHROMA_COLLECTION_NAME = "legal_summaries"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

# FIRAC component names
FIRAC_COMPONENTS = ["facts", "issues", "rules", "application", "conclusion"]


def parse_metadata(metadata_text: Union[str, Dict]) -> Dict[str, str]:
    """
    Parse metadata text from firac.py into a structured dictionary.
    
    Expected format (if string):
    FILE NAME: ...
    PARTIES: ...
    COURT LEVEL: ...
    JUDGE: ...
    YEAR: ...
    LEGAL DOMAIN: ...
    WINNING PARTY: ...
    
    Args:
        metadata_text: Raw metadata text from FIRAC extraction (str) or 
                      already parsed dict
        
    Returns:
        dict: Parsed metadata with keys: file_name, parties, court_level, 
              judge, year, legal_domain, winning_party
    """
    metadata = {
        "file_name": "",
        "parties": "",
        "court_level": "",
        "judge": "",
        "year": "",
        "legal_domain": "",
        "winning_party": "",
    }
    
    # If already a dict, return it (with defaults for missing keys)
    if isinstance(metadata_text, dict):
        metadata.update(metadata_text)
        return metadata
    
    if not metadata_text or not isinstance(metadata_text, str):
        return metadata
    
    # Parse each line
    lines = metadata_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match pattern: "KEY: value"
        match = re.match(r"^([^:]+):\s*(.+)$", line)
        if match:
            key = match.group(1).strip().lower().replace(" ", "_")
            value = match.group(2).strip()
            
            # Map keys to metadata dict
            if key == "file_name":
                metadata["file_name"] = value
            elif key == "parties":
                metadata["parties"] = value
            elif key == "court_level":
                metadata["court_level"] = value
            elif key == "judge":
                metadata["judge"] = value
            elif key == "year":
                metadata["year"] = value
            elif key == "legal_domain":
                metadata["legal_domain"] = value
            elif key == "winning_party":
                metadata["winning_party"] = value
    
    return metadata


def create_embeddings(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Create semantic embeddings for each chunk using the provided model.
    Uses the same format as digester.py (prepending "passage: ").
    
    Args:
        chunks: List of text chunks to embed
        model: SentenceTransformer model instance
        
    Returns:
        np.ndarray: Array of embeddings with shape (num_chunks, embedding_dim)
    """
    if not chunks:
        return np.array([])
    
    print(f"Creating semantic embeddings with {EMBEDDING_MODEL_NAME}...")
    start_time = time.time()
    
    # Format chunks with "passage: " prefix (same as digester.py)
    formatted_chunks = [f"passage: {chunk}" for chunk in chunks]
    embeddings = model.encode(
        formatted_chunks,
        convert_to_tensor=False,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    
    duration = time.time() - start_time
    print(f"Created embeddings with dimension {embeddings.shape[1]} in {duration:.2f} seconds")
    print(f"Average embedding norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
    
    return embeddings


def compute_document_hash(document_text: str) -> str:
    """
    Compute a SHA256 hash of the document text for change detection and traceability.
    
    Args:
        document_text: The original document text
        
    Returns:
        str: SHA256 hash of the document
    """
    if not document_text:
        return ""
    return hashlib.sha256(document_text.encode('utf-8')).hexdigest()


def ingest_firac_data(
    firac_data: Dict, 
    case_identifier: Optional[str] = None,
    source_file_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Ingest FIRAC data into ChromaDB, treating each FIRAC component as its own chunk.
    
    Each FIRAC element (facts, issues, rules, application, conclusion) becomes
    a separate chunk with its metadata attached and is stored in the vector database.
    
    Each chunk includes complete traceability metadata so you can trace any chunk
    back to its original document, including:
    - Source file path (if provided)
    - Document hash (for change detection)
    - Case identifier and file name
    - All case metadata (parties, court, judge, year, etc.)
    - FIRAC component type
    
    Args:
        firac_data: Dictionary from firac.py's run_firac() function containing:
            - 'document': str (original document)
            - 'metadata': str (metadata text)
            - 'facts': str
            - 'issues': str
            - 'rules': str
            - 'application': str
            - 'conclusion': str
            - 'facts_metadata': str
            - 'issues_metadata': str
            - 'rules_metadata': str
            - 'application_metadata': str
            - 'conclusion_metadata': str
            - 'error': str or None
        case_identifier: Optional identifier for this case (e.g., case name or ID).
                        If None, will try to extract from metadata.
        source_file_path: Optional path to the original source file (PDF, etc.)
                         This will be stored in metadata for traceability.
    """
    print("\n" + "=" * 80)
    print("📚 STARTING FIRAC DATA INGESTION")
    print("=" * 80)
    
    # Check for errors
    if firac_data.get("error"):
        error_msg = f"Cannot ingest FIRAC data: {firac_data['error']}"
        print(f"❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Initialize ChromaDB
    print(f"\nInitializing ChromaDB at {CHROMA_DB_DIR} (collection: {CHROMA_COLLECTION_NAME})")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Parse metadata
    metadata_text = firac_data.get("metadata", "")
    parsed_metadata = parse_metadata(metadata_text)
    
    # Use case identifier from metadata if not provided
    if not case_identifier:
        case_identifier = parsed_metadata.get("file_name", "unknown_case")
        if not case_identifier or case_identifier == "":
            case_identifier = "unknown_case"
    
    # Compute document hash for traceability and change detection
    original_document = firac_data.get("document", "")
    document_hash = compute_document_hash(original_document) if original_document else ""
    
    # Normalize source file path
    source_path_str = None
    if source_file_path:
        source_path = Path(source_file_path) if isinstance(source_file_path, str) else source_file_path
        source_path_str = str(source_path.resolve()) if source_path.exists() else str(source_path)
    
    print(f"\nCase identifier: {case_identifier}")
    print(f"Parsed metadata: {parsed_metadata}")
    if source_path_str:
        print(f"Source file: {source_path_str}")
    if document_hash:
        print(f"Document hash: {document_hash[:16]}...")
    
    # Prepare chunks: each FIRAC component becomes its own chunk
    chunks = []
    chunk_metadatas = []
    chunk_ids = []
    
    for component in FIRAC_COMPONENTS:
        content = firac_data.get(component, "").strip()
        
        # Skip empty components
        if not content:
            print(f"⚠ Skipping empty {component} component")
            continue
        
        # Get component-specific metadata (if available)
        component_metadata_text = firac_data.get(f"{component}_metadata", metadata_text)
        component_parsed_metadata = parse_metadata(component_metadata_text)
        
        # Use component-specific metadata if available, otherwise fall back to main metadata
        if not component_parsed_metadata.get("file_name"):
            component_parsed_metadata = parsed_metadata
        
        # Create chunk metadata with complete traceability information
        chunk_metadata = {
            **component_parsed_metadata,
            # FIRAC component identification
            "firac_component": component,
            "case_identifier": case_identifier,
            # Traceability to original document
            "document_hash": document_hash,
            "source_file_path": source_path_str if source_path_str else "",
            # Content information
            "content_length": len(content),
            # Processing information
            "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_model": EMBEDDING_MODEL_NAME,
        }
        
        # Generate unique chunk ID
        # Sanitize case_identifier for use in ID
        safe_case_id = re.sub(r'[^a-zA-Z0-9_-]', '_', case_identifier)[:50]
        chunk_id = f"{safe_case_id}_{component}_{int(time.time() * 1000)}"
        
        chunks.append(content)
        chunk_metadatas.append(chunk_metadata)
        chunk_ids.append(chunk_id)
        
        print(f"✓ Prepared {component} chunk ({len(content)} chars)")
    
    if not chunks:
        print("❌ ERROR: No valid FIRAC components to ingest")
        return
    
    # Create embeddings
    print(f"\nCreating embeddings for {len(chunks)} chunk(s)...")
    embeddings = create_embeddings(chunks, model)
    
    if embeddings.size == 0:
        print("❌ ERROR: Failed to create embeddings")
        return
    
    # Check for existing chunks from this case and remove them if re-ingesting
    existing = collection.get(
        where={"case_identifier": case_identifier},
        include=["metadatas"]
    )
    
    if existing.get("ids"):
        print(f"⚠ Found {len(existing['ids'])} existing chunk(s) for this case. Removing before re-ingestion...")
        collection.delete(where={"case_identifier": case_identifier})
    
    # Store in ChromaDB
    print(f"\nStoring {len(chunks)} chunk(s) in ChromaDB...")
    collection.add(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=chunk_metadatas,
    )
    
    print("\n" + "=" * 80)
    print("✅ FIRAC DATA INGESTION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Ingestion Summary:")
    print(f"   • Case: {case_identifier}")
    print(f"   • Chunks stored: {len(chunks)}")
    for i, component in enumerate(FIRAC_COMPONENTS):
        if component in [meta.get("firac_component") for meta in chunk_metadatas]:
            chunk_meta = next(
                (m for m in chunk_metadatas if m.get("firac_component") == component),
                None
            )
            if chunk_meta:
                print(f"   • {component.capitalize()}: {chunk_meta['content_length']} chars")
    print(f"   • Collection: {CHROMA_COLLECTION_NAME}")
    print(f"   • Database path: {CHROMA_DB_DIR}")


def get_legal_summaries_collection():
    """
    Initialize and return the ChromaDB collection for legal summaries.
    
    Returns:
        ChromaDB collection object
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        return collection
    except Exception:
        # Collection doesn't exist yet, create it
        return client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )


# Example usage
if __name__ == "__main__":
    # This would typically be called with data from firac.py
    # Example:
    # from firac import run_firac
    # from pathlib import Path
    # 
    # firac_result = run_firac()
    # source_file = Path("path/to/original/document.pdf")
    # ingest_firac_data(firac_result, source_file_path=source_file)
    
    print("This module is designed to be imported and used with firac.py")
    print("Example usage:")
    print("  from firac import run_firac")
    print("  from ingester import ingest_firac_data")
    print("  from pathlib import Path")
    print("  ")
    print("  firac_result = run_firac()")
    print("  source_file = Path('path/to/original/document.pdf')")
    print("  ingest_firac_data(firac_result, source_file_path=source_file)")
    print("  ")
    print("Each chunk stored will have complete traceability metadata:")
    print("  - file_name, parties, court_level, judge, year, legal_domain, winning_party")
    print("  - firac_component (facts/issues/rules/application/conclusion)")
    print("  - case_identifier")
    print("  - source_file_path (original document location)")
    print("  - document_hash (for change detection)")
    print("  - All other metadata fields")

