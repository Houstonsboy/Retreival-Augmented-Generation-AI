import fitz  # PyMuPDF
import pytesseract
import io
import re
import os
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====
SCRIPT_DIR = Path(__file__).parent

# 1. Output directory within the same folder as the script
OUTPUT_DIR = SCRIPT_DIR / "Constsection"

# 2. Constitution directory located one level up in MasterRules
CONSTITUTION_DIR = SCRIPT_DIR / "../MasterRules/TheConstitutionOfKenya.pdf"

# 3. Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300


def extract_text_from_pdf(file_path: Path, use_ocr_threshold: int = 50) -> str:
    """
    Extract text from PDF using PyMuPDF with OCR fallback for images.
    Handles both text-based and image-based PDFs.
    """
    print(f"Extracting text from PDF: {file_path}...")
    
    try:
        doc = fitz.open(str(file_path))
        total_pages = len(doc)
        print(f"PDF has {total_pages} pages")
        
        extracted_text = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            page_text = ""
            
            # Try direct text extraction first
            text = page.get_text().strip()
            
            # Determine if OCR is needed
            needs_ocr = len(text) < use_ocr_threshold
            
            if needs_ocr:
                print(f"  Page {page_num + 1}/{total_pages}: Image-based, using OCR (found only {len(text)} chars)...")
                
                try:
                    # Get page as high-quality image
                    zoom = 2  # Higher zoom = better OCR accuracy
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR with custom config for better accuracy
                    custom_config = r'--oem 3 --psm 6'  # LSTM engine, assume uniform block of text
                    page_text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
                    
                    print(f"  Page {page_num + 1}/{total_pages}: {len(page_text)} characters extracted via OCR")
                
                except Exception as ocr_error:
                    print(f"  Page {page_num + 1}/{total_pages}: OCR failed - {ocr_error}")
                    page_text = text  # Fallback to whatever text was found
            else:
                page_text = text
                print(f"  Page {page_num + 1}/{total_pages}: {len(page_text)} characters (direct extraction)")
            
            if page_text.strip():
                extracted_text.append(page_text)
        
        doc.close()
        combined_text = "\n\n".join(extracted_text)
        print(f"Total extracted: {len(combined_text)} characters from {len(extracted_text)} pages")
        
        if not combined_text.strip():
            print("Warning: No text could be extracted from the PDF.")
            return ""
        
        return combined_text
    
    except Exception as exc:
        print(f"Error extracting text from PDF: {exc}")
        return ""


def clean_text(text: str) -> str:
    """
    Enhanced cleaning for OCR text which often has extra noise.
    Removes navigation elements, page headers, and normalizes whitespace.
    """
    navigation_words = [
        "Search", "Sign In", "Register", "Menu", "Explore", 
        "Media", "Seasons", "Targets", "Community", "Skip to content"
    ]
    
    # Common page header patterns to remove (EXCLUDING CHAPTER and PART patterns)
    page_header_patterns = [
        r"^\d+$",  # Page numbers like "114" on their own line
        r"^Constitution of Kenya$",  # Document title
        r"^\[Rev\. \d{4}\]$",  # Revision year like "[Rev. 2022]"
        r"^The Constitution of Kenya$",  # Alternative title
        # REMOVED: r"^CHAPTER \w+$",  # DO NOT remove CHAPTER headings
        # REMOVED: r"^PART \d+$",  # DO NOT remove PART headings
    ]
    
    # Compile regex patterns
    header_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in page_header_patterns]
    
    lines = text.split("\n")
    cleaned_lines = []
    
    lower_nav_words = [w.lower().replace(" ", "") for w in navigation_words]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip navigation elements
        condensed = line.replace(" ", "").lower()
        if line in navigation_words or condensed in lower_nav_words:
            continue
        
        # Check for page header patterns (but NOT CHAPTER or PART)
        is_header = False
        for pattern in header_patterns:
            if pattern.match(line):
                is_header = True
                break
        
        if is_header:
            continue
        
        # IMPORTANT: Preserve CHAPTER and PART lines
        # Check if this is a CHAPTER line (keep it)
        if re.match(r'^CHAPTER\s+', line, re.IGNORECASE):
            cleaned_lines.append(line)
            continue
        
        # Check if this is a PART line (keep it)
        if re.match(r'^PART\s+\d+\s*[–-]', line, re.IGNORECASE):
            cleaned_lines.append(line)
            continue
        
        # Skip lines that are just page numbers (also catch Roman numerals)
        if re.match(r'^[IVXLCDM]+$', line) and len(line) <= 10:
            # Could be a Roman numeral page number or section number
            # We'll be conservative and only skip if it's short
            continue
        
        # Skip very short lines that are likely noise
        if len(line) <= 2 and (line.isdigit() or line in {"|", "-", "_", ".", ","}):
            continue
        
        # Skip lines with mostly special characters (OCR noise)
        if len(line) > 0:
            # Calculate ratio of alphanumeric or space characters
            alnum_ratio = sum(c.isalnum() or c.isspace() for c in line) / len(line)
            if alnum_ratio < 0.5:
                continue
        
        # Remove common header patterns that might be combined with other text
        # Example: "114 Constitution of Kenya [Rev. 2022]" on one line
        line = re.sub(r'^\d+\s+', '', line)  # Remove leading page numbers
        line = re.sub(r'\s*Constitution of Kenya\s*', ' ', line, flags=re.IGNORECASE)
        line = re.sub(r'\s*\[Rev\. \d{4}\]\s*', ' ', line, flags=re.IGNORECASE)
        line = re.sub(r'\s*The Constitution of Kenya\s*', ' ', line, flags=re.IGNORECASE)
        
        # Also clean up any remaining header-like text in the middle of lines
        line = re.sub(r'\bConstitution of Kenya\b', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\b\[Rev\. \d{4}\]\b', '', line, flags=re.IGNORECASE)
        
        # Clean up extra spaces from the removals
        line = re.sub(r'\s+', ' ', line).strip()
        
        if line:  # Only add non-empty lines after cleaning
            cleaned_lines.append(line)
    
    cleaned_text = "\n".join(cleaned_lines)
    
    # Normalize whitespace
    cleaned_text = re.sub(r" +", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    
    # Remove common OCR artifacts
    cleaned_text = re.sub(r"[|]{2,}", "", cleaned_text)  # Multiple pipes
    cleaned_text = re.sub(r"_{3,}", "", cleaned_text)     # Multiple underscores
    
    # Additional cleaning: Remove header sequences that span multiple lines
    # Pattern: number + "Constitution of Kenya" + "[Rev. 2022]" across lines
    cleaned_text = re.sub(
        r'\n?\d+\n?Constitution of Kenya\n?\[Rev\. \d{4}\]\n?', 
        '\n', 
        cleaned_text, 
        flags=re.IGNORECASE
    )
    
    # Remove any remaining isolated page numbers at line starts
    lines = cleaned_text.split('\n')
    final_lines = []
    
    for line in lines:
        # Remove leading page numbers followed by space
        line = re.sub(r'^\d+\s+', '', line)
        if line.strip():
            final_lines.append(line.strip())
    
    cleaned_text = '\n'.join(final_lines)
    
    return cleaned_text.strip()


def load_constitution_document(file_path: Path) -> str:
    """
    Load and process the Constitution PDF document with OCR support.
    Returns the cleaned text content.
    """
    print(f"\n{'=' * 60}")
    print(f"Loading Constitution: {file_path.name}")
    print(f"{'=' * 60}")
    
    if not file_path.exists():
        error_msg = f"Error: File not found: {file_path}"
        print(error_msg)
        return error_msg
    
    print(f"File found: {file_path.name}")
    
    try:
        # Extract text from PDF with OCR support
        raw_content = extract_text_from_pdf(file_path)
        
        if not raw_content:
            error_msg = "Error: Could not extract text from PDF"
            print(error_msg)
            return error_msg
        
        original_size = len(raw_content)
        print(f"Raw content extracted: {original_size} characters")
        
        # Clean the text
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
        
        print(f"\n✓ Successfully loaded Constitution: {cleaned_size} characters")
        print(f"{'=' * 60}\n")
        
        return cleaned_content
    
    except Exception as exc:
        error_msg = f"Error: Could not read file: {exc}"
        print(error_msg)
        return error_msg


def find_numbered_sections_with_headers(text: str, max_matches: int = 264) -> List[Dict[str, str]]:
    """
    Find all lines that start with a number followed by a period,
    and extract the line above it (the header).
    Also tracks which PART and CHAPTER each article belongs to.
    Extracts FULL ARTICLE TEXT from current article to just before next article.
    Stops after finding max_matches articles.
    
    Returns:
        List of dictionaries with: header_line, number, numbered_sentence, 
        full_article_text, part, part_title, chapter
    """
    lines = text.split('\n')
    results = []
    
    # Pattern to match lines starting with number followed by period
    numbered_pattern = re.compile(r'^(\d+)\.\s+(.+)', re.IGNORECASE)
    
    # Pattern to match PART lines
    part_pattern = re.compile(r'^PART\s+(\d+)\s*[–-]\s*(.+)', re.IGNORECASE)
    
    # Pattern to match CHAPTER lines
    chapter_pattern = re.compile(r'^CHAPTER\s+(.+)', re.IGNORECASE)
    
    # Track current PART
    current_part = None
    current_part_title = None
    
    # Track current CHAPTER
    current_chapter = None
    
    i = 0
    while i < len(lines) and len(results) < max_matches:
        line_stripped = lines[i].strip()
        
        # Check if this is a PART line
        part_match = part_pattern.match(line_stripped)
        if part_match:
            current_part = part_match.group(1)
            current_part_title = part_match.group(2).strip()
            print(f"Found PART {current_part}: {current_part_title}")
            i += 1
            continue
        
        # Check if this is a CHAPTER line
        chapter_match = chapter_pattern.match(line_stripped)
        if chapter_match:
            current_chapter = chapter_match.group(1).strip()
            print(f"Found CHAPTER: {current_chapter}")
            # CHAPTER ends PART tracking
            current_part = None
            current_part_title = None
            i += 1
            continue
        
        # Check if current line starts with number + period (start of an article)
        match = numbered_pattern.match(line_stripped)
        
        if match:
            number = match.group(1)
            first_sentence_part = match.group(2)
            
            # Get the line above (header)
            header = ""
            if i > 0:
                header = lines[i - 1].strip()
                # If the line above is empty, try to get the line before that
                if not header and i > 1:
                    header = lines[i - 2].strip()
            
            # Extract FULL ARTICLE TEXT
            # Start from current line (the article number)
            article_lines = []
            j = i
            
            # Collect lines until we hit next article or end of document
            while j < len(lines):
                current_line = lines[j].strip()
                
                # Check if this is the start of the NEXT article (different number)
                next_match = numbered_pattern.match(current_line)
                if next_match and next_match.group(1) != number and j > i:
                    # We've reached the next article, stop here
                    break
                
                # Check if we hit a new PART or CHAPTER
                if j > i:
                    if part_pattern.match(current_line) or chapter_pattern.match(current_line):
                        # Hit a new section, stop before it
                        break
                
                # Add non-empty lines to article
                if current_line:
                    article_lines.append(lines[j])  # Keep original line with formatting
                
                j += 1
            
            # Combine all lines of the article
            full_article_text = '\n'.join(article_lines)
            
            # Add to results
            results.append({
                'header': header,
                'number': number,
                'sentence': first_sentence_part,  # Keep first sentence for reference
                'full_article_text': full_article_text,
                'part': current_part,
                'part_title': current_part_title,
                'chapter': current_chapter
            })
            
            # Move index to the end of this article for next iteration
            i = j
        else:
            i += 1
    
    print(f"\nFound {len(results)} numbered articles (max {max_matches}).")
    return results


def chunk_article_text(article_text: str, chunk_size: int = CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chunk article text into smaller pieces with overlap.
    
    Args:
        article_text: The FULL text of the article
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # If text is shorter than chunk_size, return as single chunk
    if len(article_text) <= chunk_size:
        return [article_text]
    
    chunks = []
    start = 0
    
    while start < len(article_text):
        # Calculate end position for this chunk
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at a word boundary
        if end < len(article_text):
            # Look for the last space within the chunk to avoid breaking words
            last_space = article_text.rfind(' ', start, end)
            if last_space != -1 and last_space > start:
                end = last_space
        
        # Extract the chunk
        chunk = article_text[start:end].strip()
        chunks.append(chunk)
        
        # Move start position for next chunk (with overlap)
        start = end - overlap
        
        # Ensure we make progress even if overlap is large
        if start <= chunks[-1].find(' ') + len(chunks) - 1:
            start = end
    
    return chunks


def process_articles_with_chunks(sections: List[Dict[str, str]]) -> List[Dict]:
    """
    Process articles and create chunks for each one based on FULL ARTICLE TEXT.
    Adds chunk information to each article.
    
    Returns:
        List of article dictionaries with chunk information added
    """
    processed_articles = []
    
    for section in sections:
        # Use the FULL article text for chunking
        article_text = section['full_article_text']
        
        # Create chunks for this article
        chunks = chunk_article_text(article_text)
        
        # Add chunk information to the article
        article_with_chunks = section.copy()
        article_with_chunks['full_text'] = article_text
        article_with_chunks['text_length'] = len(article_text)
        article_with_chunks['num_chunks'] = len(chunks)
        article_with_chunks['chunks'] = chunks
        
        processed_articles.append(article_with_chunks)
    
    return processed_articles


def main():
    """
    Main function to extract and display section headers with chunking.
    """
    print("\n" + "=" * 80)
    print("CONSTITUTION SECTION HEADER EXTRACTOR")
    print("=" * 80 + "\n")
    
    # DEBUG: Print the path being used
    print(f"DEBUG: Looking for PDF at: {CONSTITUTION_DIR}")
    print(f"DEBUG: Absolute path: {CONSTITUTION_DIR.resolve()}")
    print(f"DEBUG: Does file exist? {CONSTITUTION_DIR.exists()}")
    
    # Load the constitution
    constitution_text = load_constitution_document(CONSTITUTION_DIR)
    
    if constitution_text.startswith("Error:"):
        print(f"\nDEBUG: Error loading document: {constitution_text}")
        print("Failed to load document. Exiting.")
        return
    
    # DEBUG: Check if we got any text
    print(f"\nDEBUG: Constitution text length: {len(constitution_text)}")
    if len(constitution_text) < 100:
        print(f"DEBUG: First 500 chars of text:\n{constitution_text[:500]}")
    
    # Find all numbered sections with their headers
    print("\n" + "=" * 80)
    print("EXTRACTING SECTION HEADERS (Lines above numbered items)")
    print("=" * 80 + "\n")
    
    sections = find_numbered_sections_with_headers(constitution_text)
    
    if not sections:
        print("No numbered sections found in the document.")
        print(f"DEBUG: Showing first 1000 chars of cleaned text for inspection:")
        print("-" * 80)
        print(constitution_text[:1000])
        print("-" * 80)
        return
    
    print(f"Found {len(sections)} numbered sections (max 264).\n")
    
    # Process articles and create chunks
    print("\n" + "=" * 80)
    print("CHUNKING ARTICLES")
    print("=" * 80 + "\n")
    print(f"Chunk Size: {CHUNK_SIZE} characters")
    print(f"Chunk Overlap: {CHUNK_OVERLAP} characters\n")
    
    processed_articles = process_articles_with_chunks(sections)
    
    # Calculate statistics
    total_chunks = sum(article['num_chunks'] for article in processed_articles)
    articles_with_multiple_chunks = sum(1 for article in processed_articles if article['num_chunks'] > 1)
    
    print(f"Total Articles: {len(processed_articles)}")
    print(f"Total Chunks: {total_chunks}")
    print(f"Articles with multiple chunks: {articles_with_multiple_chunks}")
    print(f"Articles with single chunk: {len(processed_articles) - articles_with_multiple_chunks}\n")
    
    # DEBUG: Check output directory
    print(f"DEBUG: Output directory: {OUTPUT_DIR}")
    print(f"DEBUG: Does output directory exist? {OUTPUT_DIR.exists()}")
    
    # Save to file
    output_file = OUTPUT_DIR / "constitutechecker.txt"
    print(f"DEBUG: Output file path: {output_file}")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"DEBUG: Created output directory if needed")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("CONSTITUTION SECTION HEADERS WITH CHUNKING ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Articles Found: {len(processed_articles)} (Maximum: 264)\n")
            f.write(f"Chunk Size: {CHUNK_SIZE} characters\n")
            f.write(f"Chunk Overlap: {CHUNK_OVERLAP} characters\n")
            f.write(f"Total Chunks Generated: {total_chunks}\n")
            f.write(f"Articles with Multiple Chunks: {articles_with_multiple_chunks}\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, article in enumerate(processed_articles, 1):
                f.write(f"MATCH #{idx}\n")
                f.write(f"Header: {article['header'] if article['header'] else '[No header found]'}\n")
                f.write(f"Number: {article['number']}\n")
                
                # Display first 100 characters of the sentence for readability in the file
                sentence_preview = article['sentence'][:100]
                if len(article['sentence']) > 100:
                    sentence_preview += "..."
                f.write(f"Sentence (first 100 chars): {article['number']}. {sentence_preview}\n")
                
                if article['chapter']:
                    f.write(f"CHAPTER: {article['chapter']}\n")
                else:
                    f.write(f"CHAPTER: [Not within a CHAPTER]\n")
                
                if article['part']:
                    f.write(f"PART: PART {article['part']} - {article['part_title']}\n")
                else:
                    f.write(f"PART: [Not within a PART section]\n")
                
                f.write(f"Full Article Length: {article['text_length']} characters\n")
                f.write(f"Number of Chunks: {article['num_chunks']}\n")
                
                if article['num_chunks'] > 1:
                    f.write(f"\nChunk Breakdown:\n")
                    for chunk_idx, chunk in enumerate(article['chunks'], 1):
                        f.write(f"  Chunk {chunk_idx}/{article['num_chunks']} ({len(chunk)} chars):\n")
                        # Show first 150 characters of each chunk for preview
                        chunk_preview = chunk[:150]
                        if len(chunk) > 150:
                            chunk_preview += "..."
                        f.write(f"  {chunk_preview}\n\n")
                
                f.write("-" * 80 + "\n\n")
        
        print(f"\n✓ Results saved to: {output_file}")
        print(f"DEBUG: File size: {output_file.stat().st_size} bytes")
        
    except Exception as e:
        print(f"ERROR: Failed to write to file: {e}")
        print(f"DEBUG: Error type: {type(e).__name__}")


if __name__ == "__main__":
    main()