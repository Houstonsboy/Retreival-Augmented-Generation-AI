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
    Removes navigation elements and normalizes whitespace.
    """
    navigation_words = [
        "Search", "Sign In", "Register", "Menu", "Explore", 
        "Media", "Seasons", "Targets", "Community", "Skip to content"
    ]
    
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
        
        # Skip very short lines that are likely noise
        if len(line) <= 2 and (line.isdigit() or line in {"|", "-", "_", ".", ","}):
            continue
        
        # Skip lines with mostly special characters (OCR noise)
        if len(line) > 0:
            alnum_ratio = sum(c.isalnum() or c.isspace() for c in line) / len(line)
            if alnum_ratio < 0.5:
                continue
        
        cleaned_lines.append(line)
    
    cleaned_text = "\n".join(cleaned_lines)
    
    # Normalize whitespace
    cleaned_text = re.sub(r" +", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    
    # Remove common OCR artifacts
    cleaned_text = re.sub(r"[|]{2,}", "", cleaned_text)  # Multiple pipes
    cleaned_text = re.sub(r"_{3,}", "", cleaned_text)     # Multiple underscores
    
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
    Stops after finding max_matches articles.
    
    Returns:
        List of dictionaries with: header_line, number, numbered_sentence, part, part_title, chapter
    """
    lines = text.split("\n")
    results = []
    
    # Pattern to match lines starting with number followed by period
    # Matches: "1. ", "23. ", "456. ", etc.
    numbered_pattern = re.compile(r'^(\d+)\.\s+(.+)', re.IGNORECASE)
    
    # Pattern to match PART lines: "PART 2 – OTHER PUBLIC FUNDS"
    part_pattern = re.compile(r'^PART\s+(\d+)\s*[–-]\s*(.+)', re.IGNORECASE)
    
    # Pattern to match CHAPTER lines: "CHAPTER THIRTEEN"
    chapter_pattern = re.compile(r'^CHAPTER\s+(.+)', re.IGNORECASE)
    
    # Track current PART
    current_part = None
    current_part_title = None
    
    # Track current CHAPTER
    current_chapter = None
    
    for i, line in enumerate(lines):
        # Stop if we've reached the maximum number of matches
        if len(results) >= max_matches:
            print(f"\nReached maximum of {max_matches} articles. Stopping search.")
            break
            
        line_stripped = line.strip()
        
        # Check if this is a PART line
        part_match = part_pattern.match(line_stripped)
        if part_match:
            current_part = part_match.group(1)
            current_part_title = part_match.group(2).strip()
            print(f"Found PART {current_part}: {current_part_title}")
            continue
        
        # Check if this is a CHAPTER line
        chapter_match = chapter_pattern.match(line_stripped)
        if chapter_match:
            current_chapter = chapter_match.group(1).strip()
            print(f"Found CHAPTER: {current_chapter}")
            # CHAPTER ends PART tracking
            current_part = None
            current_part_title = None
            continue
        
        # Check if current line starts with number + period
        match = numbered_pattern.match(line_stripped)
        
        if match:
            number = match.group(1)
            sentence = match.group(2)
            
            # Get the line above (header)
            header = ""
            if i > 0:
                header = lines[i - 1].strip()
                
                # If the line above is empty, try to get the line before that
                if not header and i > 1:
                    header = lines[i - 2].strip()
            
            results.append({
                'header': header,
                'number': number,
                'sentence': sentence,
                'part': current_part,
                'part_title': current_part_title,
                'chapter': current_chapter
            })
    
    return results


def main():
    """
    Main function to extract and display section headers.
    """
    print("\n" + "=" * 80)
    print("CONSTITUTION SECTION HEADER EXTRACTOR")
    print("=" * 80 + "\n")
    
    # Load the constitution
    constitution_text = load_constitution_document(CONSTITUTION_DIR)
    
    if constitution_text.startswith("Error:"):
        print("\nFailed to load document. Exiting.")
        return
    
    # Find all numbered sections with their headers
    print("\n" + "=" * 80)
    print("EXTRACTING SECTION HEADERS (Lines above numbered items)")
    print("=" * 80 + "\n")
    
    sections = find_numbered_sections_with_headers(constitution_text)
    
    if not sections:
        print("No numbered sections found in the document.")
        return
    
    print(f"Found {len(sections)} numbered sections (max 264).\n")
    print("-" * 80 + "\n")
    
    # Display results
    for idx, section in enumerate(sections, 1):
        print(f"MATCH #{idx}")
        print(f"Header (line above): {section['header'] if section['header'] else '[No header found]'}")
        print(f"Number: {section['number']}")
        print(f"Sentence: {section['number']}. {section['sentence']}")
        if section['chapter']:
            print(f"CHAPTER: {section['chapter']}")
        else:
            print(f"CHAPTER: [Not within a CHAPTER]")
        if section['part']:
            print(f"PART: PART {section['part']} - {section['part_title']}")
        else:
            print(f"PART: [Not within a PART section]")
        print("-" * 80 + "\n")
    
    # Save to file
    output_file = OUTPUT_DIR / "constitutechecker.txt"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CONSTITUTION SECTION HEADERS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Articles Found: {len(sections)} (Maximum: 264)\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, section in enumerate(sections, 1):
            f.write(f"MATCH #{idx}\n")
            f.write(f"Header: {section['header'] if section['header'] else '[No header found]'}\n")
            f.write(f"Number: {section['number']}\n")
            f.write(f"Sentence: {section['number']}. {section['sentence']}\n")
            if section['chapter']:
                f.write(f"CHAPTER: {section['chapter']}\n")
            else:
                f.write(f"CHAPTER: [Not within a CHAPTER]\n")
            if section['part']:
                f.write(f"PART: PART {section['part']} - {section['part_title']}\n")
            else:
                f.write(f"PART: [Not within a PART section]\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"\n✓ Results also saved to: {output_file}")


if __name__ == "__main__":
    main()