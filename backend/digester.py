import re
import time
from pathlib import Path
from typing import List, Tuple

import chromadb
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import hashlib


# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "Repo"
CHROMA_DB_DIR = BASE_DIR / "ChunkDB"
CHROMA_COLLECTION_NAME = "ChunkDB"

EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300


# ===== 1. TEXT LOADING =====
def compute_file_hash(file_path: Path, block_size: int = 65536) -> str:
    """
    Compute a SHA256 hash of the file contents for change detection.
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(block_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file page by page.
    """
    print(f"Extracting text from PDF: {file_path}...")
    try:
        reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)
        print(f"PDF has {total_pages} pages")

        extracted_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                extracted_text.append(text)
                print(f"  Page {page_num}/{total_pages}: {len(text)} characters extracted")
            else:
                print(f"  Page {page_num}/{total_pages}: No text found (might be image-based)")

        combined_text = "\n\n".join(extracted_text)
        print(f"Total extracted: {len(combined_text)} characters from {len(extracted_text)} pages with text")

        if not combined_text.strip():
            print("Warning: No text could be extracted from the PDF. It might be image-based or encrypted.")
            return ""

        return combined_text

    except Exception as exc:
        print(f"Error extracting text from PDF: {exc}")
        return ""


def clean_text(text: str) -> str:
    """
    Clean raw text by removing navigation noise and normalizing whitespace.
    """
    navigation_words = [
        "Search",
        "Sign In",
        "Register",
        "Menu",
        "Explore",
        "Media",
        "Seasons",
        "Targets",
        "Community",
        "Skip to content",
    ]

    lines = text.split("\n")
    cleaned_lines = []

    lower_nav_words = [w.lower().replace(" ", "") for w in navigation_words]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        condensed = line.replace(" ", "").lower()
        if line in navigation_words or condensed in lower_nav_words:
            continue
        if len(line) <= 2 and (line.isdigit() or line in {"|", "-", "_"}):
            continue
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r" +", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def load_document(file_path: Path, file_hash: str | None = None) -> List[Document]:
    """
    Load a document from disk (supports .txt and .pdf) and return LangChain Document(s).
    """
    print(f"\n{'=' * 60}")
    print(f"Loading document from {file_path}...")
    print(f"{'=' * 60}")

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return []

    file_extension = file_path.suffix.lower()
    print(f"File type detected: {file_extension}")

    raw_content = ""

    if file_hash is None:
        file_hash = compute_file_hash(file_path)

    try:
        if file_extension == ".pdf":
            raw_content = extract_text_from_pdf(file_path)
            if not raw_content:
                print("Error: Could not extract text from PDF")
                return []
        elif file_extension == ".txt":
            print("Loading text file...")
            raw_content = file_path.read_text(encoding="utf-8")
        else:
            print(f"Warning: Unsupported file format '{file_extension}'. Attempting to read as text...")
            raw_content = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Error: Could not read file: {exc}")
        return []

    if not raw_content.strip():
        print("Error: No content extracted from file")
        return []

    original_size = len(raw_content)
    print(f"Raw content extracted: {original_size} characters")

    print("\nCleaning text...")
    cleaned_content = clean_text(raw_content)
    cleaned_size = len(cleaned_content)

    if not cleaned_content.strip():
        print("Warning: Text cleaning removed all content. Using raw content instead.")
        cleaned_content = raw_content
        cleaned_size = original_size

    if original_size > 0:
        percent_removed = ((original_size - cleaned_size) / original_size) * 100
        print(f"Original size: {original_size} chars")
        print(f"After cleaning: {cleaned_size} chars ({percent_removed:.1f}% removed)")
    else:
        print("Warning: Empty file loaded")

    document = Document(
        page_content=cleaned_content,
        metadata={
            "source": str(file_path),
            "file_type": file_extension,
            "file_size": cleaned_size,
            "original_size": original_size,
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cleaned": True,
            "file_name": file_path.name,
            "file_hash": file_hash,
        },
    )

    print(f"\n✓ Successfully loaded document: {cleaned_size} characters")
    print(f"{'=' * 60}\n")
    return [document]


# ===== 2. TEXT CHUNKING =====
def chunk_text(documents: List[Document]) -> Tuple[List[str], List[Document]]:
    """
    Split documents into chunks while preserving metadata.
    """
    print(f"Chunking text with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")

    if not documents:
        print("No documents to chunk")
        return [], []

    total_length = sum(len(doc.page_content) for doc in documents)
    if total_length == 0:
        print("Error: Document is empty after cleaning")
        return [], []

    effective_chunk_size = CHUNK_SIZE
    effective_overlap = CHUNK_OVERLAP

    if total_length < CHUNK_SIZE:
        print(f"Warning: Document is small ({total_length} chars). Adjusting chunk size to fit.")
        effective_chunk_size = max(500, total_length // 2 or total_length)
        effective_overlap = min(CHUNK_OVERLAP, max(0, effective_chunk_size // 3))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunk_documents = text_splitter.split_documents(documents)

    if len(chunk_documents) == 0:
        print("Warning: No chunks created. Creating single chunk from document.")
        chunk_documents = documents

    for index, chunk_doc in enumerate(chunk_documents):
        chunk_doc.metadata.update(
            {
                "chunk_id": index,
                "chunk_index": index,
                "total_chunks": len(chunk_documents),
                "chunk_size": len(chunk_doc.page_content),
            }
        )

    print(f"Created {len(chunk_documents)} chunks")

    chunk_contents = [chunk.page_content for chunk in chunk_documents]
    return chunk_contents, chunk_documents


# ===== 3. EMBEDDING CREATION =====
def create_embeddings(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Create semantic embeddings for each chunk using the provided model.
    """
    if not chunks:
        return np.array([])

    print(f"Creating semantic embeddings with {EMBEDDING_MODEL_NAME}...")
    start_time = time.time()

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


# ===== 4. INGESTION PIPELINE =====
def ingest_file(file_path: Path, collection, model: SentenceTransformer) -> None:
    """
    Process a single file and store its chunks in the ChromaDB collection.
    """
    file_hash = compute_file_hash(file_path)

    existing = collection.get(
        where={"source": str(file_path)},
        include=["metadatas"],
    )

    existing_metadatas = existing.get("metadatas") or []
    if existing_metadatas:
        for metadata in existing_metadatas:
            if metadata and metadata.get("file_hash") == file_hash:
                print(f"No changes detected for {file_path.name}; skipping re-ingestion.")
                return

        print(f"Detected updates to {file_path.name}; removing previous embeddings before re-ingestion.")
        collection.delete(where={"source": str(file_path)})

    documents = load_document(file_path, file_hash=file_hash)
    if not documents:
        print(f"Skipping {file_path}: no documents loaded.")
        return

    chunks, chunk_documents = chunk_text(documents)
    if not chunks:
        print(f"Skipping {file_path}: no chunks created.")
        return

    embeddings = create_embeddings(chunks, model)
    if embeddings.size == 0:
        print(f"Skipping {file_path}: failed to create embeddings.")
        return

    print(f"Preparing to store {len(chunks)} chunk(s) in ChromaDB...")

    ids = []
    metadatas = []
    documents_payload = []

    for chunk_doc, embedding in zip(chunk_documents, embeddings):
        chunk_index = chunk_doc.metadata.get("chunk_index", len(ids))
        chunk_id = f"{file_path.stem}_chunk_{chunk_index}"

        ids.append(chunk_id)
        documents_payload.append(chunk_doc.page_content)

        metadata = {
            **chunk_doc.metadata,
            "source": str(file_path),
            "file_name": file_path.name,
            "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_model": EMBEDDING_MODEL_NAME,
        }
        metadatas.append(metadata)

    collection.add(
        ids=ids,
        documents=documents_payload,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(f"✓ Stored {len(ids)} chunk(s) for {file_path.name} in ChromaDB.")


def digest_repo(repo_dir: Path = REPO_DIR) -> None:
    """
    Walk through the repository folder, ingest supported files, and persist embeddings to ChromaDB.
    """
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo directory not found: {repo_dir}")

    print(f"Initializing ChromaDB at {CHROMA_DB_DIR} (collection: {CHROMA_COLLECTION_NAME})")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    supported_extensions = {".txt", ".pdf"}
    files_processed = 0

    for file_path in sorted(repo_dir.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            print(f"Skipping unsupported file type: {file_path.name}")
            continue

        print(f"\n{'#' * 60}")
        print(f"Processing file: {file_path.name}")
        print(f"{'#' * 60}")
        ingest_file(file_path, collection, model)
        files_processed += 1

    print(f"\nDigest complete. Processed {files_processed} file(s).")


if __name__ == "__main__":
    digest_repo()

