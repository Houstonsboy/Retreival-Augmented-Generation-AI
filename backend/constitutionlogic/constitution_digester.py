"""
Constitution Digester - Embeds constitution article chunks into ChromaDB
Processes embeddable chunks from constitution_embeddable_chunks.json
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "MasterRulesDB"
CHROMA_COLLECTION_NAME = "master_rules"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

# Path to the embeddable chunks JSON
EMBEDDABLE_CHUNKS_JSON = BASE_DIR / "Constsection" / "constitution_embeddable_chunks.json"


class ConstitutionDigester:
    """Handles embedding of constitution articles into ChromaDB."""
    
    def __init__(self):
        """Initialize the digester with embedding model and ChromaDB client."""
        print(f"\n{'='*80}")
        print("INITIALIZING CONSTITUTION DIGESTER")
        print(f"{'='*80}\n")
        
        # Initialize embedding model
        print(f"📦 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"✓ Model loaded successfully\n")
        
        # Initialize ChromaDB
        print(f"🗄️  Connecting to ChromaDB at: {CHROMA_DB_DIR}")
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Connected to collection: {CHROMA_COLLECTION_NAME}\n")
    
    def generate_chunk_id(self, chunk_data: Dict) -> str:
        """
        Generate a unique, deterministic ID for a constitution chunk.
        
        Args:
            chunk_data: Dictionary containing chunk information
            
        Returns:
            Unique hash-based ID
        """
        # Create a unique identifier from article number, chunk index, and source
        identifier = (
            f"const_{chunk_data['source_document']}_"
            f"art{chunk_data['article_number']}_"
            f"chunk{chunk_data['chunk_index']}"
        )
        
        # Add hash of content for additional uniqueness
        content_hash = hashlib.sha256(
            chunk_data['chunk_text'].encode('utf-8')
        ).hexdigest()[:8]
        
        return f"{identifier}_{content_hash}"
    
    def prepare_metadata(self, chunk_data: Dict) -> Dict:
        """
        Prepare metadata for ChromaDB storage.
        
        Args:
            chunk_data: Dictionary containing chunk information
            
        Returns:
            Cleaned metadata dictionary (ChromaDB compatible)
        """
        metadata = {
            'article_number': str(chunk_data['article_number']),
            'article_header': chunk_data['article_header'],
            'chapter': chunk_data.get('chapter', '') or '',
            'chapter_title': chunk_data.get('chapter_title', '') or '',
            'part': chunk_data.get('part', '') or '',
            'part_title': chunk_data.get('part_title', '') or '',
            'source_document': chunk_data['source_document'],
            'article_length': chunk_data['article_length'],
            'total_chunks': chunk_data['total_chunks'],
            'chunk_index': chunk_data['chunk_index'],
            'chunk_length': chunk_data['chunk_length'],
            'is_complete_article': chunk_data['is_complete_article'],
            'document_type': 'constitution',
            'ingestion_timestamp': int(time.time())
        }
        
        # ChromaDB requires all metadata values to be strings, ints, floats, or bools
        # Convert None values to empty strings
        for key, value in metadata.items():
            if value is None:
                metadata[key] = ''
        
        return metadata
    
    def embed_chunk(self, chunk_text: str) -> List[float]:
        """
        Generate embedding for a chunk of text.
        
        Args:
            chunk_text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        # Add instruction prefix for e5 models (improves retrieval quality)
        prefixed_text = f"passage: {chunk_text}"
        embedding = self.model.encode(prefixed_text, normalize_embeddings=True)
        return embedding.tolist()
    
    def check_existing_chunk(self, chunk_id: str) -> bool:
        """
        Check if a chunk already exists in the database.
        
        Args:
            chunk_id: Unique chunk identifier
            
        Returns:
            True if chunk exists, False otherwise
        """
        try:
            result = self.collection.get(ids=[chunk_id])
            return len(result['ids']) > 0
        except Exception:
            return False
    
    def ingest_chunks(
        self, 
        chunks: List[Dict],
        skip_existing: bool = True,
        batch_size: int = 10
    ) -> Dict:
        """
        Ingest multiple constitution chunks into ChromaDB.
        
        Args:
            chunks: List of chunk dictionaries from JSON
            skip_existing: If True, skip chunks that already exist
            batch_size: Number of chunks to process in each batch
            
        Returns:
            Dictionary with ingestion statistics
        """
        print(f"\n{'='*80}")
        print("STARTING CHUNK INGESTION")
        print(f"{'='*80}\n")
        print(f"Total chunks to process: {len(chunks)}")
        print(f"Batch size: {batch_size}")
        print(f"Skip existing: {skip_existing}\n")
        
        stats = {
            'total_chunks': len(chunks),
            'ingested': 0,
            'skipped': 0,
            'failed': 0,
            'failed_chunks': []
        }
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            print(f"{'─'*60}")
            print(f"BATCH {batch_num}/{total_batches} (Chunks {i+1}-{min(i+batch_size, len(chunks))})")
            print(f"{'─'*60}")
            
            batch_ids = []
            batch_embeddings = []
            batch_documents = []
            batch_metadatas = []
            
            for chunk_data in batch:
                try:
                    # Generate unique ID
                    chunk_id = self.generate_chunk_id(chunk_data)
                    
                    # Check if already exists
                    if skip_existing and self.check_existing_chunk(chunk_id):
                        article_num = chunk_data['article_number']
                        chunk_idx = chunk_data['chunk_index']
                        print(f"  ⏭️  Article {article_num}, Chunk {chunk_idx}: Already exists, skipping")
                        stats['skipped'] += 1
                        continue
                    
                    # Generate embedding
                    chunk_text = chunk_data['chunk_text']
                    embedding = self.embed_chunk(chunk_text)
                    
                    # Prepare metadata
                    metadata = self.prepare_metadata(chunk_data)
                    
                    # Add to batch
                    batch_ids.append(chunk_id)
                    batch_embeddings.append(embedding)
                    batch_documents.append(chunk_text)
                    batch_metadatas.append(metadata)
                    
                    article_num = chunk_data['article_number']
                    chunk_idx = chunk_data['chunk_index']
                    total_chunks = chunk_data['total_chunks']
                    print(f"  ✓ Article {article_num}, Chunk {chunk_idx}/{total_chunks}: Ready for ingestion")
                    
                except Exception as e:
                    article_num = chunk_data.get('article_number', 'unknown')
                    chunk_idx = chunk_data.get('chunk_index', 'unknown')
                    print(f"  ❌ Article {article_num}, Chunk {chunk_idx}: Error - {str(e)}")
                    stats['failed'] += 1
                    stats['failed_chunks'].append({
                        'article': article_num,
                        'chunk': chunk_idx,
                        'error': str(e)
                    })
                    continue
            
            # Ingest batch into ChromaDB
            if batch_ids:
                try:
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                    stats['ingested'] += len(batch_ids)
                    print(f"\n  💾 Ingested {len(batch_ids)} chunks into ChromaDB")
                except Exception as e:
                    print(f"\n  ❌ Batch ingestion failed: {str(e)}")
                    stats['failed'] += len(batch_ids)
                    for chunk_id in batch_ids:
                        stats['failed_chunks'].append({
                            'id': chunk_id,
                            'error': 'Batch ingestion failed'
                        })
            
            print()  # Blank line between batches
        
        # Final summary
        print(f"{'='*80}")
        print("INGESTION COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successfully ingested: {stats['ingested']}")
        print(f"⏭️  Skipped (existing):    {stats['skipped']}")
        print(f"❌ Failed:                {stats['failed']}")
        print(f"{'='*80}\n")
        
        return stats
    
    def load_and_ingest_from_json(
        self, 
        json_path: Path = EMBEDDABLE_CHUNKS_JSON,
        skip_existing: bool = True
    ) -> Dict:
        """
        Load embeddable chunks from JSON and ingest into ChromaDB.
        
        Args:
            json_path: Path to the embeddable chunks JSON file
            skip_existing: If True, skip chunks that already exist
            
        Returns:
            Dictionary with ingestion statistics
        """
        print(f"\n{'='*80}")
        print("LOADING CONSTITUTION CHUNKS FROM JSON")
        print(f"{'='*80}\n")
        print(f"📂 JSON file: {json_path}")
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        chunks = data.get('embeddable_chunks', [])
        
        print(f"✓ Loaded {len(chunks)} chunks")
        print(f"  - Total articles: {metadata.get('total_articles', 'N/A')}")
        print(f"  - Source document: {metadata.get('source_document', 'N/A')}")
        print(f"  - Chunk size: {metadata.get('chunk_size', 'N/A')}")
        print(f"  - Chunk overlap: {metadata.get('chunk_overlap', 'N/A')}\n")
        
        if not chunks:
            print("⚠️  No chunks found in JSON file")
            return {
                'total_chunks': 0,
                'ingested': 0,
                'skipped': 0,
                'failed': 0,
                'failed_chunks': []
            }
        
        # Ingest chunks
        return self.ingest_chunks(chunks, skip_existing=skip_existing)


def ingest_constitution_chunks(
    json_path: Optional[Path] = None,
    skip_existing: bool = True
) -> Dict:
    """
    Main function to ingest constitution chunks into ChromaDB.
    
    Args:
        json_path: Optional path to embeddable chunks JSON. Uses default if None.
        skip_existing: If True, skip chunks that already exist in the database
        
    Returns:
        Dictionary with ingestion statistics
    """
    digester = ConstitutionDigester()
    
    if json_path is None:
        json_path = EMBEDDABLE_CHUNKS_JSON
    
    return digester.load_and_ingest_from_json(
        json_path=json_path,
        skip_existing=skip_existing
    )


if __name__ == "__main__":
    """Run ingestion when script is executed directly."""
    try:
        stats = ingest_constitution_chunks()
        
        # Exit with appropriate code
        if stats['failed'] > 0:
            print(f"\n⚠️  Completed with {stats['failed']} failures")
            exit(1)
        else:
            print(f"\n✅ All chunks processed successfully!")
            exit(0)
            
    except Exception as e:
        print(f"\n🚨 CRITICAL ERROR: {str(e)}")
        exit(1)