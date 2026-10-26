#!/usr/bin/env python3
"""
Black's Law Dictionary Direct Embedder
Extracts text from PDF and directly embeds into ChromaDB without intermediate JSON file.
Processes page-by-page for memory efficiency and progress tracking.
"""

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
import os
import re
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "BlackLawDictionaryDB"
CHROMA_COLLECTION_NAME = "black_law_dictionary"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

# PDF Configuration
PDF_FILENAME = "BlackLaw7thEdition.pdf"
PDF_PATH = BASE_DIR / PDF_FILENAME

# Page range to process - CONFIGURE THESE
START_PAGE = 1200    # First page to extract (1-indexed)
END_PAGE = 1751    # Last page to extract (1-indexed)
DPI = 300          # OCR resolution


# ===== PDF EXTRACTION FUNCTIONS (UNCHANGED) =====

def detect_two_column_layout(img):
    """
    Detect if a page has a two-column layout by looking for the black vertical dividing line.
    If the black line exists = two columns. If not = single column.
    """
    gray = img.convert('L')
    width, height = gray.size
    
    center = width // 2
    strip_width = int(width * 0.02)
    center_start = center - strip_width
    center_end = center + strip_width
    
    center_strip = gray.crop((center_start, 0, center_end, height))
    pixels = np.array(center_strip)
    
    very_dark_pixels = np.sum(pixels < 30)
    total_pixels = pixels.size
    dark_ratio = very_dark_pixels / total_pixels
    
    has_black_line = dark_ratio > 0.01
    return has_black_line


def remove_running_header(text):
    """Remove running headers from column text."""
    lines = text.split('\n')
    if not lines:
        return text
    
    first_line_idx = 0
    for i, line in enumerate(lines):
        if line.strip():
            first_line_idx = i
            break
    
    first_line = lines[first_line_idx].strip()
    words = first_line.split()
    
    if len(words) <= 3:
        if first_line.endswith(('.', ',', ';', ')', ':', '—')):
            return text
        
        if len(words) == 1:
            return '\n'.join(lines[first_line_idx + 1:])
        
        if len(words) <= 3 and not any(first_line.endswith(p) for p in ['.', ',', ';', ')', ':', '—', '©']):
            return '\n'.join(lines[first_line_idx + 1:])
    
    return text


def is_continuation(left_text, right_text):
    """Check if right column text continues from left column."""
    left_stripped = left_text.strip()
    right_stripped = right_text.strip()
    
    if not left_stripped or not right_stripped:
        return False
    
    right_first_line = right_stripped.split('\n')[0].strip()
    
    headword_pattern = r'^([a-z][a-z\s-]+?)(?:\s*\([^)]+\))?,?\s+(?:n\.|v\.|adj\.|adv\.|prep\.|conj\.|See\s|[A-Z][a-z])'
    
    if re.match(headword_pattern, right_first_line):
        return False
    
    first_word = right_first_line.split()[0] if right_first_line else ""
    
    if first_word and first_word[0].islower():
        return True
    
    continuation_capitals = ['Thus', 'Therefore', 'Also', 'However', 'Moreover', 
                            'Furthermore', 'Additionally', 'Consequently', 'Hence',
                            'Accordingly', 'Nevertheless', 'Otherwise', 'Then',
                            'The', 'This', 'These', 'Those', 'Such', 'It']
    
    if first_word in continuation_capitals:
        return True
    
    continuation_lowercase = ['and', 'or', 'but', 'on', 'in', 'to', 'of', 'for', 'with', 
                             'from', 'by', 'at', 'as', 'if', 'when', 'while', 'which',
                             'that', 'than', 'so', 'yet', 'nor']
    
    if first_word.lower() in continuation_lowercase:
        return True
    
    return False


def extract_single_column(img):
    """Extract text from a single-column page."""
    text = pytesseract.image_to_string(img, lang='eng')
    return text


def extract_two_columns(img):
    """Extract text from a two-column page (left then right)."""
    width, height = img.size
    midpoint = width // 2
    overlap = 20
    
    left_box = (0, 0, midpoint + overlap, height)
    left_img = img.crop(left_box)
    left_text = pytesseract.image_to_string(left_img, lang='eng')
    left_text = remove_running_header(left_text)
    
    right_box = (midpoint - overlap, 0, width, height)
    right_img = img.crop(right_box)
    right_text = pytesseract.image_to_string(right_img, lang='eng')
    right_text = remove_running_header(right_text)
    
    if is_continuation(left_text, right_text):
        combined_text = left_text.strip() + " " + right_text.strip()
    else:
        combined_text = left_text.strip() + "\n\n" + right_text.strip()
    
    return combined_text


def extract_page_text(page, page_number, dpi=300):
    """Extract text from a page, auto-detecting layout type."""
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    is_two_column = detect_two_column_layout(img)
    
    if is_two_column:
        text = extract_two_columns(img)
        layout_type = "TWO_COLUMN"
    else:
        text = extract_single_column(img)
        text = remove_running_header(text)
        layout_type = "SINGLE_COLUMN"
    
    return text, layout_type


def normalize_metadata_ref(ref):
    """Normalize a cross-reference string."""
    if not ref or not isinstance(ref, str):
        return ref
    ref = re.sub(r'-\s*\r?\n\s*', '', ref)
    ref = ref.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    ref = re.sub(r'\s+', ' ', ref).strip()
    return ref


def extract_cross_references(definition_text):
    """Extract cross-reference metadata from definition text."""
    metadata = {
        'see': [],
        'cf': [],
        'also_termed': []
    }

    see_pattern = r'\bSee\s+(?:also\s+)?([A-Z][A-Z\s-]+?)(?:\s*[.;,]|\s*$|\s+\()'
    see_matches = re.finditer(see_pattern, definition_text)
    for match in see_matches:
        ref = normalize_metadata_ref(match.group(1).strip())
        if ref and ref not in metadata['see']:
            metadata['see'].append(ref)

    cf_pattern = r'\b[Cc]f\.\s+([A-Z][A-Z\s-]+?)(?:\s*[.;,]|\s*$|\s+\()'
    cf_matches = re.finditer(cf_pattern, definition_text)
    for match in cf_matches:
        ref = normalize_metadata_ref(match.group(1).strip())
        if ref and ref not in metadata['cf']:
            metadata['cf'].append(ref)

    also_pattern = r'\bAlso\s+(?:termed|spelled)\s+([a-z][a-z\s-]+?)(?:\s*[.;]|\s*$)'
    also_matches = re.finditer(also_pattern, definition_text, re.IGNORECASE)
    for match in also_matches:
        ref = normalize_metadata_ref(match.group(1).strip())
        if ref and ref not in metadata['also_termed']:
            metadata['also_termed'].append(ref)

    return metadata


def parse_dictionary_entry(entry_text, page_number):
    """Parse a dictionary entry into structured format."""
    entry_text = entry_text.strip()
    if not entry_text:
        return None
    
    entry_text = entry_text.replace('[CONTINUATION FROM PREVIOUS PAGE]', '').strip()
    
    lines = entry_text.split('\n')
    first_line = lines[0].strip()
    
    headword_match = re.match(r'^([a-z][a-z\s-]+?)(?:\s+\([^)]+\))?[,.]', first_line, re.IGNORECASE)
    
    if headword_match:
        headword = headword_match.group(1).strip()
    else:
        headword_fallback = re.match(r'^([^\s.,:]+(?:\s+[^\s.,:]+)*)', first_line)
        headword = headword_fallback.group(1).strip() if headword_fallback else first_line.split()[0]
    
    definition = entry_text
    cross_refs = extract_cross_references(definition)
    
    entry = {
        'page': page_number,
        'term': headword,
        'definition': definition,
        'metadata': {
            'see': cross_refs['see'],
            'cf': cross_refs['cf'],
            'also_termed': cross_refs['also_termed']
        }
    }
    
    return entry


def split_into_entries(text_with_breaks):
    """Split combined text into individual dictionary entries."""
    text_normalized = text_with_breaks.replace('__ENTRY_BREAK__', '\n\n')
    entries = re.split(r'\n\s*\n+', text_normalized)
    
    final_entries = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        if len(entry) > 10:
            final_entries.append(entry)
    
    return final_entries


# ===== EMBEDDING FUNCTIONS =====

class BlackLawDirectEmbedder:
    """Handles direct PDF extraction and embedding into ChromaDB."""
    
    def __init__(self):
        """Initialize the embedder with model and ChromaDB client."""
        print(f"\n{'='*80}")
        print("INITIALIZING BLACK'S LAW DICTIONARY DIRECT EMBEDDER")
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
        
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Connected to collection: {CHROMA_COLLECTION_NAME}\n")
    
    def generate_entry_id(self, entry_data: Dict) -> str:
        """Generate a unique, deterministic ID for a dictionary entry."""
        term = entry_data.get('term', '').lower().replace(' ', '_')
        page = entry_data.get('page', 0)
        
        identifier = f"bld_p{page}_{term}"
        
        definition = entry_data.get('definition', '')
        content_hash = hashlib.sha256(
            definition.encode('utf-8')
        ).hexdigest()[:8]
        
        return f"{identifier}_{content_hash}"
    
    def prepare_metadata(self, entry_data: Dict) -> Dict:
        """Prepare metadata for ChromaDB storage."""
        metadata = {
            'term': entry_data.get('term', ''),
            'page': entry_data.get('page', 0),
            'definition_length': len(entry_data.get('definition', '')),
            'document_type': 'legal_dictionary',
            'source': 'BlackLaw7thEdition',
            'ingestion_timestamp': int(time.time())
        }
        
        # Add cross-references
        meta = entry_data.get('metadata', {})
        if meta.get('see'):
            metadata['see'] = ', '.join(meta['see'])
        if meta.get('cf'):
            metadata['cf'] = ', '.join(meta['cf'])
        if meta.get('also_termed'):
            metadata['also_termed'] = ', '.join(meta['also_termed'])
        
        # Ensure all values are compatible with ChromaDB
        for key, value in metadata.items():
            if value is None:
                metadata[key] = ''
        
        return metadata
    
    def create_searchable_text(self, entry_data: Dict) -> str:
        """Create comprehensive searchable text from dictionary entry."""
        parts = []
        
        term = entry_data.get('term', '')
        if term:
            parts.append(f"Term: {term}")
        
        definition = entry_data.get('definition', '')
        if definition:
            parts.append(f"Definition: {definition}")
        
        meta = entry_data.get('metadata', {})
        
        if meta.get('see'):
            parts.append(f"See: {', '.join(meta['see'])}")
        
        if meta.get('cf'):
            parts.append(f"Compare: {', '.join(meta['cf'])}")
        
        if meta.get('also_termed'):
            parts.append(f"Also termed: {', '.join(meta['also_termed'])}")
        
        return '\n'.join(parts)
    
    def embed_entry(self, searchable_text: str) -> List[float]:
        """Generate embedding for a dictionary entry."""
        prefixed_text = f"passage: {searchable_text}"
        embedding = self.model.encode(prefixed_text, normalize_embeddings=True)
        return embedding.tolist()
    
    def check_existing_entry(self, entry_id: str) -> bool:
        """Check if an entry already exists in the database."""
        try:
            result = self.collection.get(ids=[entry_id])
            return len(result['ids']) > 0
        except Exception:
            return False
    
    def ingest_entry(self, entry_data: Dict, skip_existing: bool = True) -> bool:
        """
        Ingest a single dictionary entry into ChromaDB.
        
        Returns:
            bool: True if ingested, False if skipped or failed
        """
        try:
            entry_id = self.generate_entry_id(entry_data)
            
            if skip_existing and self.check_existing_entry(entry_id):
                return False
            
            searchable_text = self.create_searchable_text(entry_data)
            embedding = self.embed_entry(searchable_text)
            metadata = self.prepare_metadata(entry_data)
            
            self.collection.add(
                ids=[entry_id],
                embeddings=[embedding],
                documents=[searchable_text],
                metadatas=[metadata]
            )
            
            return True
            
        except Exception as e:
            print(f"\n      ❌ Error embedding entry: {str(e)}")
            return False
    
    def process_pdf(
        self, 
        pdf_path: Path,
        start_page: int = START_PAGE,
        end_page: int = END_PAGE,
        dpi: int = DPI,
        skip_existing: bool = True
    ) -> Dict:
        """
        Process PDF pages directly into ChromaDB.
        
        Args:
            pdf_path: Path to PDF file
            start_page: First page to process (1-indexed)
            end_page: Last page to process (1-indexed)
            dpi: OCR resolution
            skip_existing: Skip entries that already exist
        
        Returns:
            Dictionary with processing statistics
        """
        print(f"\n{'='*80}")
        print("STARTING DIRECT PDF PROCESSING")
        print(f"{'='*80}\n")
        print(f"📄 PDF: {pdf_path}")
        print(f"📑 Pages: {start_page} to {end_page}")
        print(f"🔍 DPI: {dpi}")
        print(f"⏭️  Skip existing: {skip_existing}\n")
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Open PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Convert to 0-indexed and validate
        start_idx = start_page - 1
        end_idx = min(end_page, total_pages)
        
        stats = {
            'total_pages': end_idx - start_idx,
            'pages_processed': 0,
            'total_entries_found': 0,
            'entries_ingested': 0,
            'entries_skipped': 0,
            'entries_failed': 0
        }
        
        print(f"{'='*80}")
        print("PROCESSING PAGES")
        print(f"{'='*80}\n")
        
        # Process each page
        for page_num in range(start_idx, end_idx):
            actual_page_number = page_num + 1
            
            # Extract text from page
            print(f"📖 Page {actual_page_number}/{end_idx}: Reading...", end=' ', flush=True)
            
            page = doc[page_num]
            page_text, layout_type = extract_page_text(page, actual_page_number, dpi=dpi)
            
            # Split into entries
            entries = split_into_entries(page_text)
            num_entries = len(entries)
            stats['total_entries_found'] += num_entries
            
            print(f"✓ [{layout_type}] {num_entries} entries found", flush=True)
            
            # Embed each entry
            if num_entries > 0:
                print(f"   ⚡ Embedding entries: ", end='', flush=True)
                
                for i, entry_text in enumerate(entries, 1):
                    parsed_entry = parse_dictionary_entry(entry_text, actual_page_number)
                    
                    if parsed_entry:
                        success = self.ingest_entry(parsed_entry, skip_existing=skip_existing)
                        
                        if success:
                            stats['entries_ingested'] += 1
                            print("✓", end='', flush=True)
                        else:
                            # Check if it was skipped or failed
                            entry_id = self.generate_entry_id(parsed_entry)
                            if self.check_existing_entry(entry_id):
                                stats['entries_skipped'] += 1
                                print("⏭", end='', flush=True)
                            else:
                                stats['entries_failed'] += 1
                                print("✗", end='', flush=True)
                    else:
                        stats['entries_failed'] += 1
                        print("✗", end='', flush=True)
                    
                    # Add space every 10 entries for readability
                    if i % 10 == 0:
                        print(" ", end='', flush=True)
                
                print(f" → {stats['entries_ingested']} embedded")
            
            stats['pages_processed'] += 1
            print()  # Blank line after each page
        
        doc.close()
        
        # Final summary
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"📄 Pages processed:       {stats['pages_processed']}")
        print(f"📝 Total entries found:   {stats['total_entries_found']}")
        print(f"✅ Entries embedded:      {stats['entries_ingested']}")
        print(f"⏭️  Entries skipped:       {stats['entries_skipped']}")
        print(f"❌ Entries failed:        {stats['entries_failed']}")
        print(f"{'='*80}\n")
        
        return stats


def main():
    """Main function to run direct embedding."""
    print(f"\n{'='*80}")
    print("BLACK'S LAW DICTIONARY - DIRECT PDF TO CHROMADB EMBEDDER")
    print(f"{'='*80}\n")
    
    # Check if PDF exists
    if not PDF_PATH.exists():
        print("❌ ERROR: PDF file not found!")
        print(f"Expected location: {PDF_PATH}")
        print()
        print(f"Please place '{PDF_FILENAME}' in the same folder as this script:")
        print(f"  {BASE_DIR}")
        return
    
    try:
        # Initialize embedder
        embedder = BlackLawDirectEmbedder()
        
        # Process PDF
        stats = embedder.process_pdf(
            pdf_path=PDF_PATH,
            start_page=START_PAGE,
            end_page=END_PAGE,
            dpi=DPI,
            skip_existing=True
        )
        
        # Exit with appropriate code
        if stats['entries_failed'] > 0:
            print(f"⚠️  Completed with {stats['entries_failed']} failures")
            exit(1)
        else:
            print(f"✅ All entries processed successfully!")
            exit(0)
            
    except Exception as e:
        print(f"\n🚨 CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\nMake sure you have installed the required packages:")
        print("  pip install PyMuPDF Pillow pytesseract numpy chromadb sentence-transformers")
        print("\nAnd that Tesseract OCR is installed on your system:")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("  macOS: brew install tesseract")
        print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        exit(1)


if __name__ == "__main__":
    main()