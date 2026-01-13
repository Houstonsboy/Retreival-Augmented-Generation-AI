import fitz  # PyMuPDF
import pytesseract
import io
import re
import json
from PIL import Image
from pathlib import Path
from typing import List, Dict

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
    """Extract text from PDF using PyMuPDF with OCR fallback for images."""
    print(f"Extracting text from PDF: {file_path}...")
    
    try:
        doc = fitz.open(str(file_path))
        total_pages = len(doc)
        print(f"PDF has {total_pages} pages")
        
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
                    print(f"  Page {page_num + 1}/{total_pages}: {len(page_text)} chars via OCR")
                except Exception as ocr_error:
                    print(f"  OCR failed: {ocr_error}")
                    page_text = text
            else:
                page_text = text
                print(f"  Page {page_num + 1}/{total_pages}: {len(page_text)} chars")
            
            if page_text.strip():
                extracted_text.append(page_text)
        
        doc.close()
        combined_text = "\n\n".join(extracted_text)
        print(f"Total: {len(combined_text)} characters from {len(extracted_text)} pages")
        return combined_text
    
    except Exception as exc:
        print(f"Error extracting text: {exc}")
        return ""


def clean_text(text: str) -> str:
    """Enhanced cleaning for OCR text."""
    navigation_words = [
        "Search", "Sign In", "Register", "Menu", "Explore", 
        "Media", "Seasons", "Targets", "Community", "Skip to content"
    ]
    
    page_header_patterns = [
        r"^\d+$",
        r"^Constitution of Kenya$",
        r"^\[Rev\. \d{4}\]$",
        r"^The Constitution of Kenya$",
    ]
    
    header_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in page_header_patterns]
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
        
        is_header = False
        for pattern in header_patterns:
            if pattern.match(line):
                is_header = True
                break
        
        if is_header:
            continue
        
        if re.match(r'^CHAPTER\s+', line, re.IGNORECASE):
            cleaned_lines.append(line)
            continue
        
        if re.match(r'^PART\s+\d+\s*[–-]', line, re.IGNORECASE):
            cleaned_lines.append(line)
            continue
        
        if re.match(r'^[IVXLCDM]+$', line) and len(line) <= 10:
            continue
        
        if len(line) <= 2 and (line.isdigit() or line in {"|", "-", "_", ".", ","}):
            continue
        
        if len(line) > 0:
            alnum_ratio = sum(c.isalnum() or c.isspace() for c in line) / len(line)
            if alnum_ratio < 0.5:
                continue
        
        line = re.sub(r'^\d+\s+', '', line)
        line = re.sub(r'\s*Constitution of Kenya\s*', ' ', line, flags=re.IGNORECASE)
        line = re.sub(r'\s*\[Rev\. \d{4}\]\s*', ' ', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+', ' ', line).strip()
        
        if line:
            cleaned_lines.append(line)
    
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r" +", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    
    return cleaned_text.strip()


def load_constitution_document(file_path: Path) -> str:
    """Load and process the Constitution PDF."""
    print(f"\n{'=' * 60}")
    print(f"Loading Constitution: {file_path.name}")
    print(f"{'=' * 60}")
    
    if not file_path.exists():
        return f"Error: File not found: {file_path}"
    
    try:
        raw_content = extract_text_from_pdf(file_path)
        if not raw_content:
            return "Error: Could not extract text from PDF"
        
        cleaned_content = clean_text(raw_content)
        print(f"\n✓ Successfully loaded: {len(cleaned_content)} characters")
        return cleaned_content
    except Exception as exc:
        return f"Error: Could not read file: {exc}"


def find_numbered_sections_with_headers(text: str, max_matches: int = 264) -> List[Dict[str, str]]:
    """Find all constitutional articles with metadata including chapter titles."""
    lines = text.split('\n')
    results = []
    
    numbered_pattern = re.compile(r'^(\d+)\.\s+(.+)', re.IGNORECASE)
    part_pattern = re.compile(r'^PART\s+(\d+)\s*[–-]\s*(.+)', re.IGNORECASE)
    chapter_pattern = re.compile(r'^CHAPTER\s+(.+)', re.IGNORECASE)
    
    current_part = None
    current_part_title = None
    current_chapter = None
    current_chapter_title = None
    
    i = 0
    while i < len(lines) and len(results) < max_matches:
        line_stripped = lines[i].strip()
        
        part_match = part_pattern.match(line_stripped)
        if part_match:
            current_part = part_match.group(1)
            current_part_title = part_match.group(2).strip()
            i += 1
            continue
        
        chapter_match = chapter_pattern.match(line_stripped)
        if chapter_match:
            current_chapter = chapter_match.group(1).strip()
            current_part = None
            current_part_title = None
            
            # Extract chapter title from the next non-empty line
            current_chapter_title = None
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line:
                    # Check if it's not a PART or numbered article
                    if not part_pattern.match(next_line) and not numbered_pattern.match(next_line):
                        current_chapter_title = next_line
                    break
                j += 1
            
            i += 1
            continue
        
        match = numbered_pattern.match(line_stripped)
        if match:
            number = match.group(1)
            first_sentence_part = match.group(2)
            
            header = ""
            if i > 0:
                header = lines[i - 1].strip()
                if not header and i > 1:
                    header = lines[i - 2].strip()
            
            article_lines = []
            j = i
            
            while j < len(lines):
                current_line = lines[j].strip()
                next_match = numbered_pattern.match(current_line)
                if next_match and next_match.group(1) != number and j > i:
                    break
                if j > i:
                    if part_pattern.match(current_line) or chapter_pattern.match(current_line):
                        break
                if current_line:
                    article_lines.append(lines[j])
                j += 1
            
            full_article_text = '\n'.join(article_lines)
            
            results.append({
                'header': header,
                'number': number,
                'sentence': first_sentence_part,
                'full_article_text': full_article_text,
                'part': current_part,
                'part_title': current_part_title,
                'chapter': current_chapter,
                'chapter_title': current_chapter_title
            })
            
            i = j
        else:
            i += 1
    
    print(f"\nFound {len(results)} numbered articles.")
    return results


def chunk_article_text(article_text: str, chunk_size: int = CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Chunk article text with overlap."""
    if len(article_text) <= chunk_size:
        return [article_text]
    
    chunks = []
    start = 0
    
    while start < len(article_text):
        end = start + chunk_size
        if end < len(article_text):
            last_space = article_text.rfind(' ', start, end)
            if last_space != -1 and last_space > start:
                end = last_space
        
        chunk = article_text[start:end].strip()
        chunks.append(chunk)
        start = end - overlap
        
        if start <= chunks[-1].find(' ') + len(chunks) - 1:
            start = end
    
    return chunks


def process_articles_with_chunks(sections: List[Dict[str, str]], source_pdf_path: Path) -> List[Dict]:
    """Process articles and create chunks with source document metadata."""
    processed_articles = []
    
    for section in sections:
        article_text = section['full_article_text']
        chunks = chunk_article_text(article_text)
        
        article_with_chunks = section.copy()
        article_with_chunks['text_length'] = len(article_text)
        article_with_chunks['num_chunks'] = len(chunks)
        article_with_chunks['chunks'] = chunks
        article_with_chunks['source_document'] = source_pdf_path.name
        
        processed_articles.append(article_with_chunks)
    
    return processed_articles


def flatten_articles_to_embeddable_chunks(articles: List[Dict]) -> List[Dict]:
    """
    Convert articles with multiple chunks into individual embeddable units.
    Each chunk becomes its own record with metadata (WITHOUT full_article_text to save space).
    
    Args:
        articles: List of articles with nested chunks
        
    Returns:
        List of individual chunk records ready for embedding
    """
    embeddable_chunks = []
    
    for article in articles:
        num_chunks = article['num_chunks']
        
        # Create base metadata (shared across all chunks)
        base_metadata = {
            'article_number': article['number'],
            'article_header': article.get('header', ''),
            'chapter': article.get('chapter', ''),
            'chapter_title': article.get('chapter_title', ''),
            'part': article.get('part', ''),
            'part_title': article.get('part_title', ''),
            'source_document': article['source_document'],
            'article_length': article['text_length'],
            'total_chunks': num_chunks
        }
        
        # Create individual chunk records
        for chunk_idx, chunk_text in enumerate(article['chunks'], 1):
            chunk_record = base_metadata.copy()
            chunk_record['chunk_index'] = chunk_idx
            chunk_record['chunk_text'] = chunk_text  # This is EITHER full article OR the specific chunk
            chunk_record['chunk_length'] = len(chunk_text)
            chunk_record['is_complete_article'] = (num_chunks == 1)
            
            embeddable_chunks.append(chunk_record)
    
    return embeddable_chunks


def extract_constitution_articles(pdf_path: Path = None) -> List[Dict]:
    """
    Main extraction function that returns structured articles.
    This is the primary function to be called from other modules.
    
    Args:
        pdf_path: Path to Constitution PDF. If None, uses CONSTITUTION_DIR.
        
    Returns:
        List of article dictionaries with chunks and metadata
    """
    if pdf_path is None:
        pdf_path = CONSTITUTION_DIR
    
    print("\n" + "=" * 80)
    print("CONSTITUTION ARTICLE EXTRACTOR")
    print("=" * 80 + "\n")
    
    # Load constitution
    constitution_text = load_constitution_document(pdf_path)
    
    if constitution_text.startswith("Error:"):
        print(f"Failed to load document: {constitution_text}")
        return []
    
    # Extract articles
    sections = find_numbered_sections_with_headers(constitution_text)
    
    if not sections:
        print("No articles found.")
        return []
    
    # Process and chunk articles with source document
    processed_articles = process_articles_with_chunks(sections, pdf_path)
    
    return processed_articles


def main():
    """Main function - extracts articles and outputs to both TXT and JSON formats."""
    articles = extract_constitution_articles()
    
    if not articles:
        print("No articles to process. Exiting.")
        return
    
    # Calculate statistics
    total_chunks = sum(article['num_chunks'] for article in articles)
    articles_with_multiple_chunks = sum(1 for article in articles if article['num_chunks'] > 1)
    
    print(f"\n{'=' * 80}")
    print("EXTRACTION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Articles: {len(articles)}")
    print(f"Total Embeddable Chunks: {total_chunks}")
    print(f"Articles with multiple chunks: {articles_with_multiple_chunks}")
    print(f"Chunk Size: {CHUNK_SIZE} characters")
    print(f"Chunk Overlap: {CHUNK_OVERLAP} characters")
    print(f"Source Document: {articles[0]['source_document'] if articles else 'N/A'}\n")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # ===== OUTPUT 1: Human-readable TXT file (for inspection) =====
    txt_output_file = OUTPUT_DIR / "constitutionchecker.txt"
    
    try:
        with open(txt_output_file, 'w', encoding='utf-8') as f:
            f.write("CONSTITUTION ARTICLE EXTRACTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Articles: {len(articles)}\n")
            f.write(f"Chunk Size: {CHUNK_SIZE} characters\n")
            f.write(f"Chunk Overlap: {CHUNK_OVERLAP} characters\n")
            f.write(f"Total Chunks: {total_chunks}\n")
            f.write(f"Articles with multiple chunks: {articles_with_multiple_chunks}\n")
            f.write(f"Source Document: {articles[0]['source_document'] if articles else 'N/A'}\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, article in enumerate(articles, 1):
                f.write(f"{'=' * 80}\n")
                f.write(f"ARTICLE #{idx}\n")
                f.write(f"{'=' * 80}\n\n")
                
                f.write(f"Article Number: {article['number']}\n")
                f.write(f"Header: {article['header'] if article['header'] else '[No header]'}\n")
                f.write(f"Source Document: {article['source_document']}\n")
                
                if article['chapter']:
                    chapter_display = article['chapter']
                    if article.get('chapter_title'):
                        chapter_display += f" - {article['chapter_title']}"
                    f.write(f"Chapter: {chapter_display}\n")
                else:
                    f.write(f"Chapter: [Not within a CHAPTER]\n")
                
                if article['part']:
                    f.write(f"Part: PART {article['part']} - {article['part_title']}\n")
                else:
                    f.write(f"Part: [Not within a PART]\n")
                
                f.write(f"\nArticle Length: {article['text_length']} characters\n")
                f.write(f"Number of Chunks: {article['num_chunks']}\n")
                f.write(f"\n{'-' * 80}\n")
                f.write("FULL ARTICLE TEXT:\n")
                f.write(f"{'-' * 80}\n\n")
                f.write(f"{article['full_article_text']}\n\n")
                
                if article['num_chunks'] > 1:
                    f.write(f"{'-' * 80}\n")
                    f.write("CHUNK BREAKDOWN:\n")
                    f.write(f"{'-' * 80}\n\n")
                    for chunk_idx, chunk in enumerate(article['chunks'], 1):
                        f.write(f"--- Chunk {chunk_idx}/{article['num_chunks']} ({len(chunk)} chars) ---\n\n")
                        f.write(f"{chunk}\n\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"✓ Human-readable report saved to: {txt_output_file}")
        
    except Exception as e:
        print(f"ERROR: Failed to write TXT file: {e}")
    
    # ===== OUTPUT 2: Flat embeddable chunks (PRIMARY - for ingester) =====
    embeddable_json_file = OUTPUT_DIR / "constitution_embeddable_chunks.json"
    
    try:
        # Flatten articles into individual embeddable chunks
        embeddable_chunks = flatten_articles_to_embeddable_chunks(articles)
        
        # Calculate total size savings
        total_chunk_text_size = sum(len(chunk['chunk_text']) for chunk in embeddable_chunks)
        
        json_data = {
            "metadata": {
                "total_articles": len(articles),
                "total_embeddable_chunks": len(embeddable_chunks),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "source_document": articles[0]['source_document'] if articles else None,
                "description": "Each chunk is an individual embeddable unit. 'chunk_text' contains either the full article (for short articles) or the specific chunk (for long articles). Embed 'chunk_text' and store all metadata."
            },
            "embeddable_chunks": embeddable_chunks
        }
        
        with open(embeddable_json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        file_size_mb = embeddable_json_file.stat().st_size / (1024 * 1024)
        
        print(f"✓ Embeddable chunks saved to: {embeddable_json_file}")
        print(f"  → {len(embeddable_chunks)} individual chunks ready for embedding")
        print(f"  → File size: {file_size_mb:.2f} MB")
        
        # Show sample chunk structure
        if embeddable_chunks:
            print(f"\n📌 Sample chunk structure (short article):")
            # Find a single-chunk article
            single_chunk = next((c for c in embeddable_chunks if c['is_complete_article']), embeddable_chunks[0])
            print(f"   - article_number: {single_chunk['article_number']}")
            print(f"   - article_header: {single_chunk['article_header']}")
            print(f"   - chunk_index: {single_chunk['chunk_index']}/{single_chunk['total_chunks']}")
            print(f"   - chunk_length: {single_chunk['chunk_length']} chars")
            print(f"   - is_complete_article: {single_chunk['is_complete_article']}")
            print(f"   - chunk_text: {single_chunk['chunk_text'][:80]}...")
            
            # Find a multi-chunk article
            multi_chunk = next((c for c in embeddable_chunks if not c['is_complete_article']), None)
            if multi_chunk:
                print(f"\n📌 Sample chunk structure (long article - chunk {multi_chunk['chunk_index']}):")
                print(f"   - article_number: {multi_chunk['article_number']}")
                print(f"   - article_header: {multi_chunk['article_header']}")
                print(f"   - chunk_index: {multi_chunk['chunk_index']}/{multi_chunk['total_chunks']}")
                print(f"   - chunk_length: {multi_chunk['chunk_length']} chars")
                print(f"   - is_complete_article: {multi_chunk['is_complete_article']}")
                print(f"   - chunk_text: {multi_chunk['chunk_text'][:80]}...")
        
    except Exception as e:
        print(f"ERROR: Failed to write embeddable chunks file: {e}")
    
    # ===== OUTPUT 3: Original nested structure (backward compatibility) =====
    original_json_file = OUTPUT_DIR / "constitution_articles.json"
    
    try:
        json_data_original = {
            "metadata": {
                "total_articles": len(articles),
                "total_chunks": total_chunks,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "source_document": articles[0]['source_document'] if articles else None,
                "note": "This is the nested structure with full article text. Use 'constitution_embeddable_chunks.json' for embedding."
            },
            "articles": articles
        }
        
        with open(original_json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data_original, f, indent=2, ensure_ascii=False)
        
        original_size_mb = original_json_file.stat().st_size / (1024 * 1024)
        
        print(f"✓ Original nested structure saved to: {original_json_file}")
        print(f"  → File size: {original_size_mb:.2f} MB\n")
        
    except Exception as e:
        print(f"ERROR: Failed to write original JSON file: {e}")
    
    print(f"{'=' * 80}")
    print("✅ EXTRACTION COMPLETE - Ready for embedding!")
    print(f"{'=' * 80}\n")
    print("📌 FILES GENERATED:")
    print(f"   1. {txt_output_file.name} - Human-readable inspection")
    print(f"   2. {embeddable_json_file.name} - PRIMARY file for ingester ⭐")
    print(f"   3. {original_json_file.name} - Nested structure (backup)\n")
    print("🎯 KEY FEATURE:")
    print("   - 'chunk_text' is the ONLY text field in embeddable chunks")
    print("   - For short articles: chunk_text = full article")
    print("   - For long articles: chunk_text = specific chunk portion")
    print("   - No duplication of full_article_text = smaller file size!\n")
    print("🚀 Next step: Use 'constitution_embeddable_chunks.json' with the ingester\n")


if __name__ == "__main__":
    main()