#!/usr/bin/env python3
"""
BlackLaw PDF Text Extractor with Auto Column Detection - JSON Output Version
Extracts text from first 20 pages, automatically detecting single vs two-column layouts
Outputs JSON format with metadata including:
- Page number
- Phrase/term
- Definition
- Cross-references (See, Cf., Also termed)
"""

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
import os
import json
from pathlib import Path
import re


def detect_two_column_layout(img):
    """
    Detect if a page has a two-column layout by looking for the black vertical dividing line.
    If the black line exists = two columns. If not = single column.
    
    Args:
        img: PIL Image object
    
    Returns:
        bool: True if black dividing line found (two columns), False otherwise (single column)
    """
    # Convert to grayscale
    gray = img.convert('L')
    width, height = gray.size
    
    # Look at the exact center vertical strip (middle 2-3% of width)
    # This is where the black dividing line should be
    center = width // 2
    strip_width = int(width * 0.02)  # Very narrow strip at center
    center_start = center - strip_width
    center_end = center + strip_width
    
    # Crop the center vertical strip
    center_strip = gray.crop((center_start, 0, center_end, height))
    
    # Convert to numpy array
    pixels = np.array(center_strip)
    
    # Count very dark pixels (black line should be very dark, < 30 out of 255)
    very_dark_pixels = np.sum(pixels < 30)
    total_pixels = pixels.size
    
    dark_ratio = very_dark_pixels / total_pixels
    
    # If more than 1% of the center strip is very dark, we found the black line
    # This threshold is low because the line is thin but should run vertically
    has_black_line = dark_ratio > 0.01
    
    return has_black_line


def remove_running_header(text):
    """
    Remove running headers from column text.
    Running headers appear at the very top of columns - usually single words or short phrases.
    
    Args:
        text: Extracted text from a column
    
    Returns:
        str: Text with running header removed
    """
    lines = text.split('\n')
    if not lines:
        return text
    
    # Check first non-empty line
    first_line_idx = 0
    for i, line in enumerate(lines):
        if line.strip():
            first_line_idx = i
            break
    
    first_line = lines[first_line_idx].strip()
    
    # Running headers are typically:
    # - Very short (1-3 words)
    # - At the very top
    # - Not punctuated definitions
    # - Often bold/isolated
    
    words = first_line.split()
    
    # If first line is 1-3 words and doesn't end with definition markers, it's likely a header
    if len(words) <= 3:
        # Check if it looks like a header vs. actual content
        # Headers don't typically end with: period, comma, semicolon, parenthesis
        # Headers are usually: single word, or short phrase
        
        # If it ends with punctuation that suggests it's content, keep it
        if first_line.endswith(('.', ',', ';', ')', ':', '—')):
            # This looks like actual content, keep it
            return text
        
        # If all uppercase or title case single word, likely header
        if len(words) == 1:
            # Single word without punctuation at top = likely header
            return '\n'.join(lines[first_line_idx + 1:])
        
        # If 2-3 words without punctuation, likely header
        if len(words) <= 3 and not any(first_line.endswith(p) for p in ['.', ',', ';', ')', ':', '—', '©']):
            return '\n'.join(lines[first_line_idx + 1:])
    
    return text


def is_continuation(left_text, right_text):
    """
    Check if right column text continues from left column.
    
    For dictionary content:
    - Definitions can span columns even if left ends with period
    - Need to check if right column starts a NEW entry (headword) or continues definition
    
    Args:
        left_text: Text from left column
        right_text: Text from right column
    
    Returns:
        bool: True if continuation, False if new section
    """
    left_stripped = left_text.strip()
    right_stripped = right_text.strip()
    
    if not left_stripped or not right_stripped:
        return False
    
    # Get the first line of right column
    right_first_line = right_stripped.split('\n')[0].strip()
    
    # Check if right column starts with a dictionary headword
    # Dictionary headwords match pattern: word(s) followed by pronunciation or part of speech
    headword_pattern = r'^([a-z][a-z\s-]+?)(?:\s*\([^)]+\))?,?\s+(?:n\.|v\.|adj\.|adv\.|prep\.|conj\.|See\s|[A-Z][a-z])'
    
    if re.match(headword_pattern, right_first_line):
        # Right column starts with a new dictionary entry
        return False
    
    # Check if right column starts with a sentence continuation
    # Continuations typically start with:
    # - Lowercase word (mid-sentence)
    # - Capital word that's clearly continuing (Thus, Therefore, Also, etc.)
    # - List continuation (and, or, but)
    
    first_word = right_first_line.split()[0] if right_first_line else ""
    
    # If starts with lowercase, it's definitely continuation
    if first_word and first_word[0].islower():
        return True
    
    # Capital words that indicate continuation (not new entries)
    continuation_capitals = ['Thus', 'Therefore', 'Also', 'However', 'Moreover', 
                            'Furthermore', 'Additionally', 'Consequently', 'Hence',
                            'Accordingly', 'Nevertheless', 'Otherwise', 'Then',
                            'The', 'This', 'These', 'Those', 'Such', 'It']
    
    if first_word in continuation_capitals:
        return True
    
    # Common lowercase continuation starters
    continuation_lowercase = ['and', 'or', 'but', 'on', 'in', 'to', 'of', 'for', 'with', 
                             'from', 'by', 'at', 'as', 'if', 'when', 'while', 'which',
                             'that', 'than', 'so', 'yet', 'nor']
    
    if first_word.lower() in continuation_lowercase:
        return True
    
    # If we get here, right column likely starts a new section
    # (Capital letter that's not a known continuation word = new entry)
    return False


def extract_single_column(img):
    """Extract text from a single-column page."""
    text = pytesseract.image_to_string(img, lang='eng')
    return text


def extract_two_columns(img):
    """
    Extract text from a two-column page (left then right).
    Removes running headers from each column, then intelligently combines them.
    Preserves blank lines that separate dictionary entries.
    """
    width, height = img.size
    midpoint = width // 2
    
    # Add small overlap to avoid cutting words
    overlap = 20
    
    # Crop left column
    left_box = (0, 0, midpoint + overlap, height)
    left_img = img.crop(left_box)
    left_text = pytesseract.image_to_string(left_img, lang='eng')
    
    # Remove running header from left column
    left_text = remove_running_header(left_text)
    
    # Crop right column
    right_box = (midpoint - overlap, 0, width, height)
    right_img = img.crop(right_box)
    right_text = pytesseract.image_to_string(right_img, lang='eng')
    
    # Remove running header from right column
    right_text = remove_running_header(right_text)
    
    # Check if right column continues from left column
    if is_continuation(left_text, right_text):
        # Continuation: join with space (same paragraph)
        combined_text = left_text.strip() + " " + right_text.strip()
    else:
        # New section: separate with blank line to preserve entry boundaries
        combined_text = left_text.strip() + "\n\n" + right_text.strip()
    
    return combined_text


def extract_page_text(page, page_number, dpi=300):
    """
    Extract text from a page, auto-detecting layout type.
    Running headers are removed at the column level during extraction.
    
    Args:
        page: PyMuPDF page object
        page_number: Actual page number (for reference)
        dpi: Resolution for rendering
    
    Returns:
        tuple: (extracted_text, layout_type)
    """
    # Render page to high-resolution image
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Detect layout type
    is_two_column = detect_two_column_layout(img)
    
    if is_two_column:
        text = extract_two_columns(img)  # Headers removed inside this function
        layout_type = "TWO_COLUMN"
    else:
        text = extract_single_column(img)
        text = remove_running_header(text)  # Remove header for single column too
        layout_type = "SINGLE_COLUMN"
    
    return text, layout_type


def extract_cross_references(definition_text):
    """
    Extract cross-reference metadata from definition text.
    Looks for: See [word], Cf. [word], Also termed [phrase]
    
    Args:
        definition_text: The definition text to analyze
    
    Returns:
        dict: Dictionary with 'see', 'cf', and 'also_termed' lists
    """
    metadata = {
        'see': [],
        'cf': [],
        'also_termed': []
    }
    
    # Pattern for "See WORD" or "See also WORD"
    see_pattern = r'\bSee\s+(?:also\s+)?([A-Z][A-Z\s-]+?)(?:\s*[.;,]|\s*$|\s+\()'
    see_matches = re.finditer(see_pattern, definition_text)
    for match in see_matches:
        ref = match.group(1).strip()
        if ref and ref not in metadata['see']:
            metadata['see'].append(ref)
    
    # Pattern for "Cf." or "cf."
    cf_pattern = r'\b[Cc]f\.\s+([A-Z][A-Z\s-]+?)(?:\s*[.;,]|\s*$|\s+\()'
    cf_matches = re.finditer(cf_pattern, definition_text)
    for match in cf_matches:
        ref = match.group(1).strip()
        if ref and ref not in metadata['cf']:
            metadata['cf'].append(ref)
    
    # Pattern for "Also termed" or "Also spelled"
    also_pattern = r'\bAlso\s+(?:termed|spelled)\s+([a-z][a-z\s-]+?)(?:\s*[.;]|\s*$)'
    also_matches = re.finditer(also_pattern, definition_text, re.IGNORECASE)
    for match in also_matches:
        ref = match.group(1).strip()
        if ref and ref not in metadata['also_termed']:
            metadata['also_termed'].append(ref)
    
    return metadata


def parse_dictionary_entry(entry_text, page_number):
    """
    Parse a dictionary entry into structured JSON format.
    
    Args:
        entry_text: Raw text of the dictionary entry
        page_number: Page number where this entry appears
    
    Returns:
        dict: Structured entry with metadata
    """
    entry_text = entry_text.strip()
    if not entry_text:
        return None
    
    # Remove continuation markers
    entry_text = entry_text.replace('[CONTINUATION FROM PREVIOUS PAGE]', '').strip()
    
    # Pattern to extract headword and definition
    # Format: "headword (pronunciation), part_of_speech. Definition..."
    # Or: "headword. Definition..."
    
    lines = entry_text.split('\n')
    first_line = lines[0].strip()
    
    # Try to extract headword (the term being defined)
    # The headword appears at the start, often followed by pronunciation in parens
    # Examples:
    # "absolution (ab-sa-loo-shan). 1. Release..."
    # "abstain, vb. 1. To refrain..."
    # "Burford abstention. A federal court's..."
    
    # Match pattern: term at start, possibly with pronunciation, then period or comma
    headword_match = re.match(r'^([a-z][a-z\s-]+?)(?:\s+\([^)]+\))?[,.]', first_line, re.IGNORECASE)
    
    if headword_match:
        headword = headword_match.group(1).strip()
    else:
        # Fallback: take first word(s) before punctuation
        headword_fallback = re.match(r'^([^\s.,:]+(?:\s+[^\s.,:]+)*)', first_line)
        headword = headword_fallback.group(1).strip() if headword_fallback else first_line.split()[0]
    
    # The full definition is the entire entry text (including the headword line)
    # We keep everything as-is to preserve the structure
    definition = entry_text
    
    # Extract cross-references from definition
    cross_refs = extract_cross_references(definition)
    
    # Build the entry object
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
    """
    Split combined text into individual dictionary entries.
    Each entry is separated by blank lines (double newlines) in the original text.
    
    Args:
        text_with_breaks: Text with __ENTRY_BREAK__ markers and blank line separators
    
    Returns:
        list: List of individual entry texts
    """
    # First, replace entry break markers with double newlines
    text_normalized = text_with_breaks.replace('__ENTRY_BREAK__', '\n\n')
    
    # Split by blank lines (double or triple newlines indicate entry boundaries)
    # This matches the original script's behavior of using whitespace to separate entries
    entries = re.split(r'\n\s*\n+', text_normalized)
    
    final_entries = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        # Filter out very short fragments (likely OCR artifacts or headers)
        if len(entry) > 10:  # At minimum, an entry should have a headword and some definition
            final_entries.append(entry)
    
    return final_entries


def extract_pdf_pages_to_json(pdf_path, output_path, start_page=18, end_page=30, dpi=300):
    """
    Extract text from a range of pages in a PDF and output as JSON.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the JSON output
        start_page: First page to extract (1-indexed, default: 18)
        end_page: Last page to extract (1-indexed, default: 30)
        dpi: Resolution for OCR (default: 300)
    """
    print(f"Opening PDF: {pdf_path}")
    print("="*80)
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Convert to 0-indexed and validate range
    start_idx = start_page - 1
    end_idx = min(end_page, total_pages)
    
    print(f"Total pages in PDF: {total_pages}")
    print(f"Extracting pages {start_page} to {end_idx}...")
    print(f"DPI: {dpi}")
    print("="*80)
    print()
    
    all_entries = []
    
    for page_num in range(start_idx, end_idx):
        actual_page_number = page_num + 1
        print(f"Page {actual_page_number}/{end_idx}...", end=' ', flush=True)
        
        page = doc[page_num]
        
        # Extract text with auto layout detection
        page_text, layout_type = extract_page_text(page, actual_page_number, dpi=dpi)
        
        # Split into individual entries
        entries = split_into_entries(page_text)
        
        # Parse each entry
        for entry_text in entries:
            parsed_entry = parse_dictionary_entry(entry_text, actual_page_number)
            if parsed_entry:
                all_entries.append(parsed_entry)
        
        print(f"✓ [{layout_type}] - {len(entries)} entries")
    
    doc.close()
    
    # Create output structure
    output_data = {
        'source': os.path.basename(pdf_path),
        'pages_extracted': f"{start_page}-{end_idx}",
        'total_entries': len(all_entries),
        'entries': all_entries
    }
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print()
    print("="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)
    print(f"Output saved to: {output_path}")
    print(f"Total entries extracted: {len(all_entries):,}")
    print(f"File size: {os.path.getsize(output_path):,} bytes")
    
    # Print sample entry
    if all_entries:
        print()
        print("Sample entry:")
        print(json.dumps(all_entries[0], indent=2))


def main():
    """Main function to run the extraction."""
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Define file paths
    pdf_filename = "BlackLaw7thEdition.pdf"
    pdf_path = script_dir / pdf_filename
    output_path = script_dir / "extracted_dictionary_pages_18_to_30.json"
    
    # Check if PDF exists
    if not pdf_path.exists():
        print("ERROR: PDF file not found!")
        print(f"Expected location: {pdf_path}")
        print()
        print(f"Please place '{pdf_filename}' in the same folder as this script:")
        print(f"  {script_dir}")
        return
    
    # Extract text from pages 18-30 and save as JSON
    try:
        extract_pdf_pages_to_json(
            pdf_path=str(pdf_path),
            output_path=str(output_path),
            start_page=18,
            end_page=30,
            dpi=300  # Adjust if needed: higher = better quality but slower
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you have installed the required packages:")
        print("  pip install PyMuPDF Pillow pytesseract numpy")
        print("\nAnd that Tesseract OCR is installed on your system:")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("  macOS: brew install tesseract")
        print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")


if __name__ == "__main__":
    main()