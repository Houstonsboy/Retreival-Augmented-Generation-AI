"""
Evidence Act Digester - Extracts chapters and articles from Evidence Act PDF
Outputs to JSON in the same format as constitution_embeddable_chunks.json
"""

import fitz  # PyMuPDF
import pytesseract
import io
import re
import json
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ===== CONFIGURATION =====
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "EvidenceSection"
EVIDENCE_PDF_PATH = SCRIPT_DIR.parent / "MasterRules" / "Evidence Act.pdf"

# Chunking configuration (same as constitutionbreaker.py)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300
MIN_CHUNK_SIZE = 150  # Minimum chunk size to prevent noise chunks

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output file
OUTPUT_JSON = OUTPUT_DIR / "evidence_embeddable_chunks.json"


def extract_text_from_pdf(file_path: Path, use_ocr_threshold: int = 50) -> str:
    """Extract text from PDF using PyMuPDF with OCR fallback for images."""
    print(f"\n{'='*80}")
    print(f"Extracting text from PDF: {file_path.name}")
    print(f"{'='*80}\n")
    
    try:
        doc = fitz.open(str(file_path))
        total_pages = len(doc)
        print(f"PDF has {total_pages} pages\n")
        
        extracted_text = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text().strip()
            needs_ocr = len(text) < use_ocr_threshold
            
            if needs_ocr:
                print(f"  Page {page_num + 1}/{total_pages}: Using OCR...")
                try:
                    zoom = 2
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    custom_config = r'--oem 3 --psm 6'
                    page_text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
                    print(f"    ✓ {len(page_text)} chars via OCR")
                except Exception as ocr_error:
                    print(f"    ✗ OCR failed: {ocr_error}")
                    page_text = text
            else:
                page_text = text
                print(f"  Page {page_num + 1}/{total_pages}: {len(page_text)} chars")
            
            if page_text.strip():
                extracted_text.append(page_text)
        
        doc.close()
        combined_text = "\n".join(extracted_text)
        print(f"\n✓ Total: {len(combined_text)} characters from {len(extracted_text)} pages")
        return combined_text
    
    except Exception as exc:
        print(f"✗ Error extracting text: {exc}")
        return ""


def clean_text(text: str) -> str:
    """Clean extracted text, removing headers, footers, and navigation elements."""
    lines = text.split("\n")
    cleaned_lines = []
    
    # Common headers/footers to remove
    skip_patterns = [
        r"^\d+$",  # Lone page numbers
        r"^Evidence Act$",
        r"^\[Rev\. \d{4}\]$",
        r"^CAP\. \d+",
    ]
    
    skip_patterns_compiled = [re.compile(pattern, re.IGNORECASE) for pattern in skip_patterns]
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip matched patterns
        should_skip = False
        for pattern in skip_patterns_compiled:
            if pattern.match(line):
                should_skip = True
                break
        
        if should_skip:
            continue
        
        # Keep the line
        cleaned_lines.append(line)
    
    # Join and normalize whitespace
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r" +", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    
    return cleaned_text.strip()


def chunk_article_text(article_text: str, chunk_size: int = CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP,
                       min_chunk_size: int = MIN_CHUNK_SIZE) -> List[str]:
    """
    Chunk article text with overlap, preventing short tail chunks.
    
    Uses hybrid approach:
    1. Predictive: Extends current chunk if remainder would be too small
    2. Post-processing: Merges any lingering short chunks with previous chunk
    
    Args:
        article_text: Full article body text
        chunk_size: Target chunk size (default 1000)
        overlap: Overlap between chunks (default 300)
        min_chunk_size: Minimum acceptable chunk size (default 150)
        
    Returns:
        List of chunks, with no chunk smaller than min_chunk_size
    """
    if len(article_text) <= chunk_size:
        return [article_text]
    
    chunks = []
    start = 0
    
    while start < len(article_text):
        end = start + chunk_size
        
        # DEFENSE 1: Predictive - Check if remainder would be too small
        remaining = len(article_text) - end
        if 0 < remaining < min_chunk_size:
            # Extend current chunk to include the small remainder
            end = len(article_text)
        
        # Word boundary detection (only if not at end)
        if end < len(article_text):
            last_space = article_text.rfind(' ', start, end)
            if last_space != -1 and last_space > start:
                end = last_space
        
        chunk = article_text[start:end].strip()
        chunks.append(chunk)
        
        # Exit if we've consumed all text
        if end >= len(article_text):
            break
        
        start = end - overlap
        
        # Ensure we're making progress
        if start <= chunks[-1].find(' ') + len(chunks) - 1:
            start = end
    
    # DEFENSE 2: Post-processing safety net - Merge any lingering short chunks
    if len(chunks) > 1 and len(chunks[-1]) < min_chunk_size:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()
    
    return chunks


def finalize_article_body(article_ref, body_lines: List[str]):
    """
    Finalize an article by adding its collected body text and chunking it.
    
    Args:
        article_ref: Reference to the article dictionary
        body_lines: List of body text lines collected
    """
    if article_ref is not None and body_lines:
        article_body_text = "\n".join(body_lines).strip()
        article_ref["article_length"] = len(article_body_text)
        
        # Chunk the article body
        chunks = chunk_article_text(article_body_text)
        
        # Store as numbered chunks in a dictionary
        chunked_body = {}
        for idx, chunk in enumerate(chunks, 1):
            chunked_body[f"chunk_{idx}"] = chunk
        
        article_ref["article_body"] = chunked_body
        article_ref["total_chunks"] = len(chunks)


def extract_chapters_only(text: str) -> List[Dict]:
    """
    Extract chapters, parts, articles (with chunked bodies), and sub-articles from the Evidence Act text.
    - Parts are identified as "Part [ROMAN] - TITLE" or "Part [ROMAN] – TITLE"
    - Articles are identified as "NUMBER." or "NUMBERLETTER." (e.g., "37.", "63A.")
    - Article bodies: Text between article header and next structural element (article/part/chapter)
    - Bodies are chunked using CHUNK_SIZE (1000) with CHUNK_OVERLAP (300) for large articles
    - Sub-articles: Articles with same base number + letter (e.g., "63A" after "63") are nested as sub-articles
    - Articles are assigned to the current part if in a part, otherwise to the chapter
    
    Returns:
        List of chapter dictionaries with chapter number, title, parts (with articles and sub-articles), and articles
    """
    print(f"\n{'='*80}")
    print("EXTRACTING CHAPTERS, PARTS, ARTICLES (WITH CHUNKED BODIES) FROM EVIDENCE ACT")
    print(f"{'='*80}\n")
    
    lines = text.split("\n")
    chapters = []
    
    # Pattern: matches lines with ONLY "CHAPTER" + number (Roman or Arabic numerals)
    chapter_pattern = re.compile(r'^CHAPTER\s+([IVXLCDM]+|\d+)$', re.IGNORECASE)
    # Pattern: matches "Part [ROMAN] - TITLE" or "Part [ROMAN] – TITLE" (both dash types)
    part_pattern = re.compile(r'^PART\s+([IVXLCDM]+)\s*[–-]\s*(.+)$', re.IGNORECASE)
    # Article pattern: matches "26.", "26A.", "37.", "63A.", etc.
    article_pattern = re.compile(r'^(\d+[A-Z]?)\.(?:\s+(.*))?$')
    
    current_chapter = None
    current_chapter_title = None
    current_parts = []
    current_part_index = -1  # Index of current part in current_parts list
    current_chapter_articles = []  # Articles directly in chapter (not in any part)
    last_article_base_number = None  # Track base number of last article (e.g., "63" from "63" or "63A")
    last_article_location = None  # Track where last article was added: ("part", index) or "chapter"
    
    # For collecting article bodies
    collecting_body = False
    current_article_body_lines = []
    current_article_ref = None  # Reference to the article object we're collecting body for
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for chapter
        chapter_match = chapter_pattern.match(line)
        if chapter_match:
            # Finalize any article body we were collecting
            if collecting_body and current_article_ref:
                finalize_article_body(current_article_ref, current_article_body_lines)
                collecting_body = False
                current_article_body_lines = []
                current_article_ref = None
            
            # Save previous chapter if exists
            if current_chapter is not None:
                # Convert empty article arrays to "none" in parts
                processed_parts = []
                for part in current_parts:
                    processed_part = part.copy()
                    if not part.get('articles'):
                        processed_part['articles'] = "none"
                    processed_parts.append(processed_part)
                
                chapter_data = {
                    "chapter_number": current_chapter,
                    "chapter_title": current_chapter_title,
                    "parts": processed_parts if processed_parts else "none",
                    "articles": current_chapter_articles if current_chapter_articles else "none"
                }
                chapters.append(chapter_data)
                print(f"📖 Chapter {current_chapter}: {current_chapter_title}")
                if processed_parts:
                    print(f"   Parts: {len(processed_parts)}")
                    for part in processed_parts:
                        articles_count = len(part.get('articles', [])) if isinstance(part.get('articles'), list) else 0
                        articles_info = f"{articles_count} articles" if articles_count > 0 else "none"
                        print(f"     - Part {part['part_number']}: {part['part_title']} ({articles_info})")
                else:
                    print(f"   Parts: none")
                if current_chapter_articles:
                    print(f"   Articles (direct): {len(current_chapter_articles)}")
                else:
                    print(f"   Articles (direct): none")
            
            # Start new chapter
            chapter_number = chapter_match.group(1)
            chapter_title = ""
            if i + 1 < len(lines):
                chapter_title = lines[i + 1].strip()
            
            current_chapter = chapter_number
            current_chapter_title = chapter_title
            current_parts = []
            current_part_index = -1
            current_chapter_articles = []
            last_article_base_number = None
            last_article_location = None
            collecting_body = False
            current_article_body_lines = []
            current_article_ref = None
            i += 1
        
        # Check for part (only if we're in a chapter)
        elif current_chapter is not None:
            part_match = part_pattern.match(line)
            if part_match:
                # Finalize any article body we were collecting
                if collecting_body and current_article_ref:
                    finalize_article_body(current_article_ref, current_article_body_lines)
                    collecting_body = False
                    current_article_body_lines = []
                    current_article_ref = None
                
                part_number = part_match.group(1)
                part_title = part_match.group(2).strip()
                
                # Add new part with empty articles list
                current_parts.append({
                    "part_number": part_number,
                    "part_title": part_title,
                    "articles": []
                })
                current_part_index = len(current_parts) - 1
                print(f"   📄 Part {part_number}: {part_title}")
        
        # Check for article (only if we're in a chapter)
        if current_chapter is not None:
            article_match = article_pattern.match(line)
            if article_match:
                # Finalize previous article body if we were collecting
                if collecting_body and current_article_ref:
                    finalize_article_body(current_article_ref, current_article_body_lines)
                    collecting_body = False
                    current_article_body_lines = []
                    current_article_ref = None
                
                article_number = article_match.group(1)
                article_header = article_match.group(2) if article_match.group(2) else ""
                
                # If header is on the same line after period, keep it
                # Otherwise, next line might be the header
                if not article_header and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # Check if next line is likely a header (not starting with article number or subsection)
                    if not article_pattern.match(next_line) and not re.match(r'^\(\d+\)', next_line):
                        i += 1
                        article_header = next_line
                
                # Extract base number (e.g., "63" from "63A" or "63" from "63")
                base_number_match = re.match(r'^(\d+)', article_number)
                current_base_number = base_number_match.group(1) if base_number_match else None
                
                # Check if this is a sub-article (e.g., "63A" following "63")
                is_sub_article = False
                if (current_base_number and 
                    last_article_base_number and 
                    current_base_number == last_article_base_number and 
                    len(article_number) > len(current_base_number)):  # Has letter suffix
                    is_sub_article = True
                
                article_data = {
                    "article_number": article_number,
                    "article_header": article_header,
                    "article_body": {},
                    "article_length": 0,
                    "total_chunks": 0,
                    "sub_articles": "none",
                    "source_document": EVIDENCE_PDF_PATH.name
                }
                
                # Handle sub-article
                if is_sub_article and last_article_location:
                    # Create sub-article data with body fields
                    sub_article_data = {
                        "article_number": article_number,
                        "article_header": article_header,
                        "article_body": {},
                        "article_length": 0,
                        "total_chunks": 0,
                        "source_document": EVIDENCE_PDF_PATH.name
                    }
                    
                    # Add as sub-article to the previous article
                    if last_article_location == "chapter":
                        parent_article = current_chapter_articles[-1]
                        if parent_article.get("sub_articles") == "none":
                            parent_article["sub_articles"] = []
                        parent_article["sub_articles"].append(sub_article_data)
                        current_article_ref = parent_article["sub_articles"][-1]
                        print(f"       ↳ Sub-article {article_number}: {article_header[:50] if article_header else '(no header)'} → Article {last_article_base_number}")
                    elif last_article_location[0] == "part":
                        part_idx = last_article_location[1]
                        parent_article = current_parts[part_idx]["articles"][-1]
                        if parent_article.get("sub_articles") == "none":
                            parent_article["sub_articles"] = []
                        parent_article["sub_articles"].append(sub_article_data)
                        current_article_ref = parent_article["sub_articles"][-1]
                        print(f"       ↳ Sub-article {article_number}: {article_header[:50] if article_header else '(no header)'} → Article {last_article_base_number}")
                    
                    # Start collecting body for this sub-article
                    collecting_body = True
                    current_article_body_lines = []
                    
                    # Don't update last_article_base_number for sub-articles
                else:
                    # Regular article - assign to current part if in a part, otherwise to chapter
                    if current_part_index >= 0 and current_part_index < len(current_parts):
                        current_parts[current_part_index]["articles"].append(article_data)
                        last_article_location = ("part", current_part_index)
                        current_article_ref = current_parts[current_part_index]["articles"][-1]
                        print(f"     📝 Article {article_number}: {article_header[:50] if article_header else '(no header)'} → Part {current_parts[current_part_index]['part_number']}")
                    else:
                        current_chapter_articles.append(article_data)
                        last_article_location = "chapter"
                        current_article_ref = current_chapter_articles[-1]
                        print(f"     📝 Article {article_number}: {article_header[:50] if article_header else '(no header)'} → Chapter")
                    
                    # Update last article base number
                    last_article_base_number = current_base_number
                    
                    # Start collecting body for this article
                    collecting_body = True
                    current_article_body_lines = []
        
        # If we're collecting body and this line is not a structure marker, add to body
        # This needs to be checked after all structure checks
        if collecting_body and current_article_ref and line:
            # Check if this line is NOT a new structural element
            if (not chapter_pattern.match(line) and 
                not part_pattern.match(line) and 
                not article_pattern.match(line)):
                # This line is part of the article body
                current_article_body_lines.append(line)
        
        i += 1
    
    # Finalize any last article body we were collecting
    if collecting_body and current_article_ref:
        finalize_article_body(current_article_ref, current_article_body_lines)
        collecting_body = False
        current_article_body_lines = []
        current_article_ref = None
    
    # Save last chapter
    if current_chapter is not None:
        # Convert empty article arrays to "none" in parts
        processed_parts = []
        for part in current_parts:
            processed_part = part.copy()
            if not part.get('articles'):
                processed_part['articles'] = "none"
            processed_parts.append(processed_part)
        
        chapter_data = {
            "chapter_number": current_chapter,
            "chapter_title": current_chapter_title,
            "parts": processed_parts if processed_parts else "none",
            "articles": current_chapter_articles if current_chapter_articles else "none"
        }
        chapters.append(chapter_data)
        print(f"📖 Chapter {current_chapter}: {current_chapter_title}")
        if processed_parts:
            print(f"   Parts: {len(processed_parts)}")
            for part in processed_parts:
                articles_count = len(part.get('articles', [])) if isinstance(part.get('articles'), list) else 0
                articles_info = f"{articles_count} articles" if articles_count > 0 else "none"
                print(f"     - Part {part['part_number']}: {part['part_title']} ({articles_info})")
        else:
            print(f"   Parts: none")
        if current_chapter_articles:
            print(f"   Articles (direct): {len(current_chapter_articles)}")
        else:
            print(f"   Articles (direct): none")
    
    print(f"\n✓ Found {len(chapters)} chapters")
    return chapters


def parse_evidence_act(text: str) -> List[Dict]:
    """
    Parse the Evidence Act text into structured chapters and articles.
    
    Article format: number + period (e.g., "26.", "27.", "65.")
    Article parts: number + letter + period (e.g., "26A.", "26B.")
    Chapters: "CHAPTER" keyword on its own line, with title below
    
    Returns:
        List of article dictionaries with metadata
    """
    print(f"\n{'='*80}")
    print("PARSING EVIDENCE ACT STRUCTURE")
    print(f"{'='*80}\n")
    
    lines = text.split("\n")
    articles = []
    
    current_chapter = None
    current_chapter_title = None
    current_article_number = None
    current_article_header = None
    current_article_body = []
    
    # Patterns
    # Chapter pattern: matches lines with ONLY "CHAPTER" + number (Roman or Arabic numerals)
    # Examples: "Chapter II", "CHAPTER 2", "Chapter I"
    chapter_pattern = re.compile(r'^CHAPTER\s+([IVXLCDM]+|\d+)$', re.IGNORECASE)
    # Article pattern: matches "26.", "26A.", "26B.", etc.
    article_pattern = re.compile(r'^(\d+[A-Z]?)\.(?:\s+(.*))?$')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for chapter
        chapter_match = chapter_pattern.match(line)
        if chapter_match:
            # Save previous article if exists
            if current_article_number:
                article_body_text = "\n".join(current_article_body).strip()
                if article_body_text:
                    articles.append({
                        "article_number": current_article_number,
                        "article_header": current_article_header or "",
                        "chapter": current_chapter,
                        "chapter_title": current_chapter_title,
                        "article_body": article_body_text,
                        "article_length": len(article_body_text)
                    })
                current_article_number = None
                current_article_header = None
                current_article_body = []
            
            # Extract chapter
            current_chapter = chapter_match.group(1)
            # Next line should be chapter title
            if i + 1 < len(lines):
                i += 1
                current_chapter_title = lines[i].strip()
                print(f"📖 Found Chapter {current_chapter}: {current_chapter_title}")
            
            i += 1
            continue
        
        # Check for article start
        article_match = article_pattern.match(line)
        if article_match:
            # Save previous article if exists
            if current_article_number:
                article_body_text = "\n".join(current_article_body).strip()
                if article_body_text:
                    articles.append({
                        "article_number": current_article_number,
                        "article_header": current_article_header or "",
                        "chapter": current_chapter,
                        "chapter_title": current_chapter_title,
                        "article_body": article_body_text,
                        "article_length": len(article_body_text)
                    })
            
            # Start new article
            current_article_number = article_match.group(1)
            current_article_header = article_match.group(2) if article_match.group(2) else ""
            current_article_body = []
            
            # If header is on the same line after period, keep it
            # Otherwise, next line might be the header
            if not current_article_header and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Check if next line is likely a header (not starting with article number or subsection)
                if not article_pattern.match(next_line) and not re.match(r'^\(\d+\)', next_line):
                    i += 1
                    current_article_header = next_line
            
            print(f"  📄 Article {current_article_number}: {current_article_header[:60]}")
            i += 1
            continue
        
        # If we're in an article, add to body
        if current_article_number and line:
            current_article_body.append(line)
        
        i += 1
    
    # Save last article
    if current_article_number:
        article_body_text = "\n".join(current_article_body).strip()
        if article_body_text:
            articles.append({
                "article_number": current_article_number,
                "article_header": current_article_header or "",
                "chapter": current_chapter,
                "chapter_title": current_chapter_title,
                "article_body": article_body_text,
                "article_length": len(article_body_text)
            })
    
    print(f"\n✓ Parsed {len(articles)} articles")
    return articles


def create_embeddable_chunks(articles: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 300) -> Dict:
    """
    Create embeddable chunks from articles.
    For short articles, the entire article is one chunk.
    For long articles, split into multiple overlapping chunks.
    
    Returns:
        Dictionary with metadata and embeddable_chunks list
    """
    print(f"\n{'='*80}")
    print("CREATING EMBEDDABLE CHUNKS")
    print(f"{'='*80}\n")
    print(f"Chunk size: {chunk_size} chars, Overlap: {chunk_overlap} chars\n")
    
    embeddable_chunks = []
    
    for article in articles:
        article_number = article["article_number"]
        article_header = article["article_header"]
        chapter = article.get("chapter", "")
        chapter_title = article.get("chapter_title", "")
        article_body = article["article_body"]
        article_length = article["article_length"]
        
        # Combine header and body for the full article text
        full_text = f"{article_number}. {article_header}\n{article_body}".strip()
        
        # If article fits in one chunk, create single chunk
        if len(full_text) <= chunk_size:
            chunk = {
                "article_number": article_number,
                "article_header": article_header,
                "chapter": chapter or "",
                "chapter_title": chapter_title or "",
                "source_document": "Evidence Act.pdf",
                "article_length": article_length,
                "total_chunks": 1,
                "chunk_index": 1,
                "chunk_text": full_text,
                "chunk_length": len(full_text),
                "is_complete_article": True
            }
            embeddable_chunks.append(chunk)
            print(f"  ✓ Article {article_number}: 1 chunk ({len(full_text)} chars)")
        else:
            # Split into multiple chunks with overlap
            chunks_text = []
            start = 0
            chunk_num = 1
            
            while start < len(full_text):
                end = start + chunk_size
                chunk_text = full_text[start:end]
                chunks_text.append(chunk_text)
                start = end - chunk_overlap
                chunk_num += 1
                
                # Avoid infinite loop
                if end >= len(full_text):
                    break
            
            # Create chunk objects
            total_chunks = len(chunks_text)
            for idx, chunk_text in enumerate(chunks_text, 1):
                chunk = {
                    "article_number": article_number,
                    "article_header": article_header,
                    "chapter": chapter or "",
                    "chapter_title": chapter_title or "",
                    "source_document": "Evidence Act.pdf",
                    "article_length": article_length,
                    "total_chunks": total_chunks,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "chunk_length": len(chunk_text),
                    "is_complete_article": False
                }
                embeddable_chunks.append(chunk)
            
            print(f"  ✓ Article {article_number}: {total_chunks} chunks (total {article_length} chars)")
    
    # Create final output structure
    output = {
        "metadata": {
            "total_articles": len(articles),
            "total_embeddable_chunks": len(embeddable_chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "source_document": "Evidence Act.pdf",
            "description": "Each chunk is an individual embeddable unit. 'chunk_text' contains either the full article (for short articles) or the specific chunk (for long articles). Embed 'chunk_text' and store all metadata."
        },
        "embeddable_chunks": embeddable_chunks
    }
    
    print(f"\n✓ Created {len(embeddable_chunks)} embeddable chunks from {len(articles)} articles")
    return output


def create_flat_embeddable_chunks(chapters: List[Dict]) -> Dict:
    """
    Convert the chapter-based structure to flat embeddable chunks.
    Each chunk becomes a top-level item with metadata about its source.
    
    Output format matches constitution_embeddable_chunks.json:
    {
        "article_number": "...",
        "article_header": "...",
        "chapter": "...",
        "chapter_title": "...",
        "part": "..." or null,
        "part_title": "..." or null,
        "source_document": "Evidence Act.pdf",
        "article_length": ...,
        "total_chunks": ...,
        "chunk_index": ...,
        "chunk_text": "...",
        "chunk_length": ...,
        "is_complete_article": true/false
    }
    
    Args:
        chapters: List of chapter dictionaries from extract_chapters_only()
        
    Returns:
        Dictionary with metadata and embeddable_chunks list
    """
    print(f"\n{'='*80}")
    print("CREATING FLAT EMBEDDABLE CHUNKS")
    print(f"{'='*80}\n")
    
    embeddable_chunks = []
    
    def process_article(article: Dict, chapter_number: str, chapter_title: str, 
                        part_number: str = None, part_title: str = None):
        """Process a single article and its sub-articles into flat chunks."""
        chunks_created = []
        
        article_number = article.get('article_number', '')
        article_header = article.get('article_header', '')
        article_body = article.get('article_body', {})
        article_length = article.get('article_length', 0)
        total_chunks = article.get('total_chunks', 0)
        source_document = article.get('source_document', 'Evidence Act.pdf')
        
        # Process each chunk in the article body
        if isinstance(article_body, dict) and article_body:
            for chunk_key in sorted(article_body.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0):
                chunk_text = article_body[chunk_key]
                # Extract chunk index from key (e.g., "chunk_1" -> 1)
                chunk_index = int(chunk_key.split('_')[1]) if '_' in chunk_key else 1
                
                chunk_data = {
                    "article_number": article_number,
                    "article_header": article_header,
                    "chapter": chapter_number,
                    "chapter_title": chapter_title,
                    "part": part_number,
                    "part_title": part_title,
                    "source_document": source_document,
                    "article_length": article_length,
                    "total_chunks": total_chunks,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "chunk_length": len(chunk_text),
                    "is_complete_article": total_chunks == 1
                }
                chunks_created.append(chunk_data)
        
        # Process sub-articles recursively
        sub_articles = article.get('sub_articles', 'none')
        if isinstance(sub_articles, list):
            for sub_article in sub_articles:
                sub_chunks = process_article(
                    sub_article, 
                    chapter_number, 
                    chapter_title, 
                    part_number, 
                    part_title
                )
                chunks_created.extend(sub_chunks)
        
        return chunks_created
    
    # Process all chapters
    for chapter in chapters:
        chapter_number = chapter.get('chapter_number', '')
        chapter_title = chapter.get('chapter_title', '')
        
        # Process articles in parts
        parts = chapter.get('parts', 'none')
        if isinstance(parts, list):
            for part in parts:
                part_number = part.get('part_number', '')
                part_title = part.get('part_title', '')
                
                articles = part.get('articles', 'none')
                if isinstance(articles, list):
                    for article in articles:
                        chunks = process_article(
                            article, 
                            chapter_number, 
                            chapter_title, 
                            part_number, 
                            part_title
                        )
                        embeddable_chunks.extend(chunks)
                        if chunks:
                            print(f"  ✓ Article {article.get('article_number', '?')}: {len(chunks)} chunks → Part {part_number}")
        
        # Process direct chapter articles (not in any part)
        direct_articles = chapter.get('articles', 'none')
        if isinstance(direct_articles, list):
            for article in direct_articles:
                chunks = process_article(
                    article, 
                    chapter_number, 
                    chapter_title, 
                    None,  # No part
                    None   # No part title
                )
                embeddable_chunks.extend(chunks)
                if chunks:
                    print(f"  ✓ Article {article.get('article_number', '?')}: {len(chunks)} chunks → Chapter {chapter_number} (direct)")
    
    # Count totals for metadata
    total_articles = len(set((c['article_number'], c['chapter']) for c in embeddable_chunks))
    
    # Create final output structure (matching constitution_embeddable_chunks.json format)
    output = {
        "metadata": {
            "total_articles": total_articles,
            "total_embeddable_chunks": len(embeddable_chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "source_document": "Evidence Act.pdf",
            "description": "Each chunk is an individual embeddable unit. 'chunk_text' contains either the full article (for short articles) or the specific chunk (for long articles). Embed 'chunk_text' and store all metadata."
        },
        "embeddable_chunks": embeddable_chunks
    }
    
    print(f"\n✓ Created {len(embeddable_chunks)} flat embeddable chunks from {total_articles} articles")
    return output


def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print("EVIDENCE ACT EXTRACTOR (CHAPTERS, PARTS, ARTICLES WITH CHUNKED BODIES, SUB-ARTICLES)")
    print(f"{'='*80}\n")
    print(f"Input: {EVIDENCE_PDF_PATH}")
    print(f"Output: {OUTPUT_DIR / 'evidence_chapters.json'}\n")
    print(f"Output (flat chunks): {OUTPUT_DIR / 'evidence_chapter3.json'}\n")
    
    # Step 1: Extract text from PDF
    if not EVIDENCE_PDF_PATH.exists():
        print(f"✗ Error: PDF not found at {EVIDENCE_PDF_PATH}")
        return
    
    raw_text = extract_text_from_pdf(EVIDENCE_PDF_PATH)
    if not raw_text:
        print("✗ Error: Failed to extract text from PDF")
        return
    
    # Step 2: Clean text
    print(f"\n{'='*80}")
    print("CLEANING TEXT")
    print(f"{'='*80}\n")
    print(f"Raw text length: {len(raw_text)} chars")
    cleaned_text = clean_text(raw_text)
    print(f"Cleaned text length: {len(cleaned_text)} chars")
    print(f"Removed: {len(raw_text) - len(cleaned_text)} chars ({((len(raw_text) - len(cleaned_text)) / len(raw_text) * 100):.1f}%)")
    
    # Step 3: Extract chapters only (not articles)
    chapters = extract_chapters_only(cleaned_text)
    
    if not chapters:
        print("\n✗ Error: No chapters found in the document")
        return
    
    # Step 4: Prepare output data
    total_parts = sum(len(ch['parts']) if isinstance(ch['parts'], list) else 0 for ch in chapters)
    # Count articles, sub-articles, and chunks: from parts and direct chapter articles
    total_articles = 0
    total_sub_articles = 0
    total_chunks = 0
    
    def count_articles_subs_and_chunks(articles_list):
        """Helper to count articles, their sub-articles, and total chunks"""
        article_count = len(articles_list)
        sub_count = 0
        chunk_count = 0
        for article in articles_list:
            # Count chunks for main article
            chunk_count += article.get('total_chunks', 0)
            # Count sub-articles and their chunks
            if isinstance(article.get('sub_articles'), list):
                sub_count += len(article['sub_articles'])
                for sub_article in article['sub_articles']:
                    chunk_count += sub_article.get('total_chunks', 0)
        return article_count, sub_count, chunk_count
    
    for ch in chapters:
        # Articles in parts
        if isinstance(ch['parts'], list):
            for part in ch['parts']:
                if isinstance(part.get('articles'), list):
                    art_count, sub_count, chunk_count = count_articles_subs_and_chunks(part['articles'])
                    total_articles += art_count
                    total_sub_articles += sub_count
                    total_chunks += chunk_count
        # Direct chapter articles
        if isinstance(ch.get('articles'), list):
            art_count, sub_count, chunk_count = count_articles_subs_and_chunks(ch['articles'])
            total_articles += art_count
            total_sub_articles += sub_count
            total_chunks += chunk_count
    
    output_data = {
        "metadata": {
            "total_chapters": len(chapters),
            "total_parts": total_parts,
            "total_articles": total_articles,
            "total_sub_articles": total_sub_articles,
            "total_chunks": total_chunks,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "source_document": "Evidence Act.pdf",
            "description": f"Chapters, parts, articles (with chunked bodies), and sub-articles extracted from Evidence Act PDF. Article bodies are chunked using size={CHUNK_SIZE} chars with overlap={CHUNK_OVERLAP} chars. Articles are listed within their respective parts (if in a part) or directly under the chapter (if not in a part). Sub-articles (e.g., 63A after 63) are nested within their parent article. Parts, articles, and sub-articles use 'none' if empty."
        },
        "chapters": chapters
    }
    
    # Step 5: Save to JSON
    chapters_output = OUTPUT_DIR / "evidence_chapters.json"
    print(f"\n{'='*80}")
    print("SAVING TO JSON")
    print(f"{'='*80}\n")
    
    with open(chapters_output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved to: {chapters_output}")
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Total chapters: {output_data['metadata']['total_chapters']}")
    print(f"  Total parts: {output_data['metadata']['total_parts']}")
    print(f"  Total articles: {output_data['metadata']['total_articles']}")
    print(f"  Total sub-articles: {output_data['metadata']['total_sub_articles']}")
    print(f"  Total chunks: {output_data['metadata']['total_chunks']}")
    print(f"  Chunk size: {output_data['metadata']['chunk_size']} chars")
    print(f"  Chunk overlap: {output_data['metadata']['chunk_overlap']} chars")
    print(f"  Output file: {chapters_output}")
    print(f"{'='*80}\n")
    
    # Show all chapters with their parts and articles
    print("Chapters found:")
    for i, chapter in enumerate(chapters, 1):
        parts_info = "none"
        if isinstance(chapter['parts'], list) and chapter['parts']:
            parts_info = f"{len(chapter['parts'])} part(s)"
        articles_info = "none"
        direct_articles = len(chapter['articles']) if isinstance(chapter.get('articles'), list) else 0
        part_articles = 0
        if isinstance(chapter['parts'], list):
            for part in chapter['parts']:
                if isinstance(part.get('articles'), list):
                    part_articles += len(part['articles'])
        total_chapter_articles = direct_articles + part_articles
        if total_chapter_articles > 0:
            articles_info = f"{total_chapter_articles} article(s)"
        
        print(f"  {i}. Chapter {chapter['chapter_number']}: {chapter['chapter_title']}")
        print(f"     Parts: {parts_info}, Articles: {articles_info}")
        
        if isinstance(chapter['parts'], list) and chapter['parts']:
            for part in chapter['parts']:
                part_articles_count = len(part.get('articles', [])) if isinstance(part.get('articles'), list) else 0
                print(f"       - Part {part['part_number']}: {part['part_title']} ({part_articles_count} articles)")
        
        if isinstance(chapter.get('articles'), list) and chapter['articles']:
            print(f"       - Direct articles: {len(chapter['articles'])}")
    
    print("\n✅ Chapter, part, article, and sub-article extraction complete!")
    
    # Step 6: Create flat embeddable chunks (like constitution_embeddable_chunks.json)
    flat_chunks_data = create_flat_embeddable_chunks(chapters)
    
    # Step 7: Save flat chunks to evidence_chapter3.json
    flat_chunks_output = OUTPUT_DIR / "evidence_chapter3.json"
    print(f"\n{'='*80}")
    print("SAVING FLAT EMBEDDABLE CHUNKS")
    print(f"{'='*80}\n")
    
    with open(flat_chunks_output, 'w', encoding='utf-8') as f:
        json.dump(flat_chunks_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved flat chunks to: {flat_chunks_output}")
    print(f"\n{'='*80}")
    print("FLAT CHUNKS SUMMARY")
    print(f"{'='*80}")
    print(f"  Total articles: {flat_chunks_data['metadata']['total_articles']}")
    print(f"  Total embeddable chunks: {flat_chunks_data['metadata']['total_embeddable_chunks']}")
    print(f"  Chunk size: {flat_chunks_data['metadata']['chunk_size']} chars")
    print(f"  Chunk overlap: {flat_chunks_data['metadata']['chunk_overlap']} chars")
    print(f"  Output file: {flat_chunks_output}")
    print(f"{'='*80}\n")
    
    print("\n✅ All extraction complete! Both hierarchical and flat chunk formats generated.")


if __name__ == "__main__":
    main()

