import fitz  # PyMuPDF
import pytesseract
import io
import re
import os
import requests
import json
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

# 4. LLM Classification Limit - MODIFIED TO 10
MAX_ARTICLES_TO_CLASSIFY = 10

# 5. Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ===== LEGAL DOMAIN DEFINITIONS =====
LEGAL_DOMAINS = """
Constitutional Article Map by Legal Domain:

1. Constitutional Supremacy & Sovereignty
   * Articles: 1 – 11
   * Focus: Power of the people, supremacy of the law, and national values.

2. Citizenship & Immigration Law
   * Articles: 12 – 18
   * Focus: Acquisition, dual citizenship, and revocation.

3. Human Rights & Bill of Rights
   * Articles: 19 – 59
   * Focus: Civil liberties, socio-economic rights, and enforcement of rights.

4. Land & Property Law
   * Articles: 60 – 68
   * Focus: Land ownership, classification, and the National Land Commission.

5. Environmental & Natural Resources Law
   * Articles: 69 – 72 (also Art 42)
   * Focus: Conservation, sustainable development, and resource management.

6. Ethics, Integrity & Anti-Corruption
   * Articles: 73 – 80
   * Focus: Leadership standards and conduct for State Officers (Chapter Six).

7. Electoral & Political Party Law
   * Articles: 81 – 92
   * Focus: IEBC, voting, and regulation of political parties.

8. Parliamentary & Legislative Law
   * Articles: 93 – 128
   * Focus: The National Assembly, Senate, and the legislative process.

9. Executive & State Office Law
   * Articles: 129 – 158
   * Focus: Powers of the President, Cabinet, Attorney-General, and DPP.

10. Judiciary & Procedural Law
   * Articles: 159 – 173
   * Focus: Court hierarchy, judicial independence, and the JSC.

11. Devolution & County Government Law
   * Articles: 174 – 200
   * Focus: Structure and functions of the 47 County Governments.

12. Public Finance Management (PFM)
   * Articles: 201 – 231
   * Focus: Taxation, revenue sharing, the Auditor-General, and the Central Bank.

13. Public Service & Labour Law
   * Articles: 232 – 237 (also Art 41)
   * Focus: Values of public service, the PSC, and the TSC.

14. National Security Law
   * Articles: 238 – 247
   * Focus: KDF, National Intelligence Service, and the National Police Service.

15. Constitutional Commissions & Watchdogs
   * Articles: 248 – 254
   * Focus: Independent commissions and their oversight functions.

16. Constitutional Litigation & Amendments
   * Articles: 255 – 257
   * Focus: Referendum and amendment procedures.

17. Interpretation & General Provisions
   * Articles: 258 – 260
   * Focus: Legal definitions and rules for interpreting the Constitution.

18. Transitional & Implementation Provisions
   * Articles: 261 – 264
   * Focus: Enactment of consequential legislation and transition mechanisms.
"""


def call_groq_api(prompt: str, system_message: str, max_tokens: int = 500) -> str:
    """
    Make a request to Groq API to get LLM response.
    
    Args:
        prompt: The user prompt to send
        system_message: The system instruction
        max_tokens: Maximum tokens in response
        
    Returns:
        str: The model's response
    """
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1  # Low temperature for consistent classification
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        response = requests.post(
            GROQ_API_URL,
            json=payload,
            headers=headers,
            timeout=120
        )
        
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Unexpected response format: {result}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Groq API request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Error calling Groq API: {str(e)}")


def classify_article_domain(article_number: str, article_header: str, 
                           article_text: str, chapter: str, part: str) -> str:
    """
    Use LLM to classify the legal domain of a constitutional article.
    
    Args:
        article_number: The article number (e.g., "261")
        article_header: The article header/title
        article_text: Full text of the article
        chapter: Chapter information
        part: Part information
        
    Returns:
        str: The classified legal domain
    """
    system_message = f"""You are a legal expert specializing in constitutional law of Kenya. 
Your task is to classify constitutional articles into their correct legal domain based on their content.

{LEGAL_DOMAINS}

INSTRUCTIONS:
1. Read the article number, header, chapter, part, and content carefully
2. Determine which legal domain (1-18) best fits this article
3. Respond with ONLY the domain number and name in this exact format:
   "Domain X: Domain Name"
   
For example:
   "Domain 12: Public Finance Management (PFM)"

Be precise and consistent. Consider the article's primary focus and content, not just its number.
If an article clearly fits multiple domains, choose the PRIMARY domain based on the main subject matter.
"""

    # Truncate article text if too long (keep first 2000 chars for context)
    article_preview = article_text[:2000]
    if len(article_text) > 2000:
        article_preview += "\n... [truncated]"
    
    user_prompt = f"""Classify this constitutional article:

Article Number: {article_number}
Header: {article_header}
Chapter: {chapter if chapter else 'Not specified'}
Part: {part if part else 'Not specified'}

Article Content:
{article_preview}

Which legal domain does this article belong to? Respond with only the domain number and name."""

    try:
        print(f"  📊 Classifying Article {article_number}...", end=" ")
        response = call_groq_api(user_prompt, system_message, max_tokens=100)
        
        # Extract domain from response
        # Expected format: "Domain X: Domain Name"
        domain_match = re.search(r'Domain\s+(\d+):\s*(.+)', response, re.IGNORECASE)
        
        if domain_match:
            domain_num = domain_match.group(1)
            domain_name = domain_match.group(2).strip()
            classified_domain = f"Domain {domain_num}: {domain_name}"
            print(f"✓ {classified_domain}")
            return classified_domain
        else:
            # Fallback: return the raw response if format doesn't match
            print(f"⚠️ Unusual format: {response}")
            return response.strip()
            
    except Exception as e:
        error_msg = f"Classification Error: {str(e)}"
        print(f"✗ {error_msg}")
        return error_msg


def classify_all_articles(articles: List[Dict]) -> List[Dict]:
    """
    Classify only the first MAX_ARTICLES_TO_CLASSIFY articles using the LLM.
    Adds 'legal_domain' field to each article dictionary.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        List of articles with legal_domain field added (only first 10 classified)
    """
    print("\n" + "=" * 80)
    print(f"CLASSIFYING FIRST {MAX_ARTICLES_TO_CLASSIFY} ARTICLES INTO LEGAL DOMAINS")
    print("=" * 80 + "\n")
    print(f"Classifying first {MAX_ARTICLES_TO_CLASSIFY} of {len(articles)} articles using {MODEL_NAME}...\n")
    
    classified_articles = []
    
    for idx, article in enumerate(articles, 1):
        # Only classify the first MAX_ARTICLES_TO_CLASSIFY articles
        if idx <= MAX_ARTICLES_TO_CLASSIFY:
            print(f"[{idx}/{MAX_ARTICLES_TO_CLASSIFY}] ", end="")
            
            # Get article metadata
            article_num = article['number']
            header = article.get('header', 'No header')
            full_text = article.get('full_article_text', '')
            chapter = article.get('chapter', 'Not specified')
            part_info = ""
            if article.get('part'):
                part_info = f"Part {article['part']} - {article.get('part_title', '')}"
            else:
                part_info = "Not specified"
            
            # Classify the article
            legal_domain = classify_article_domain(
                article_num, 
                header, 
                full_text, 
                chapter, 
                part_info
            )
            
            # Add legal_domain to article
            article_with_domain = article.copy()
            article_with_domain['legal_domain'] = legal_domain
            classified_articles.append(article_with_domain)
        else:
            # For articles beyond the limit, add them without classification
            article_with_domain = article.copy()
            article_with_domain['legal_domain'] = "Not classified (beyond limit)"
            classified_articles.append(article_with_domain)
    
    print(f"\n✓ Classification complete! {MAX_ARTICLES_TO_CLASSIFY} articles classified with LLM.\n")
    print(f"✓ Remaining {len(articles) - MAX_ARTICLES_TO_CLASSIFY} articles marked as 'Not classified'.\n")
    return classified_articles


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
    """Find all constitutional articles with metadata."""
    lines = text.split('\n')
    results = []
    
    numbered_pattern = re.compile(r'^(\d+)\.\s+(.+)', re.IGNORECASE)
    part_pattern = re.compile(r'^PART\s+(\d+)\s*[–-]\s*(.+)', re.IGNORECASE)
    chapter_pattern = re.compile(r'^CHAPTER\s+(.+)', re.IGNORECASE)
    
    current_part = None
    current_part_title = None
    current_chapter = None
    
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
                'chapter': current_chapter
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


def process_articles_with_chunks(sections: List[Dict[str, str]]) -> List[Dict]:
    """Process articles and create chunks."""
    processed_articles = []
    
    for section in sections:
        article_text = section['full_article_text']
        chunks = chunk_article_text(article_text)
        
        article_with_chunks = section.copy()
        article_with_chunks['full_text'] = article_text
        article_with_chunks['text_length'] = len(article_text)
        article_with_chunks['num_chunks'] = len(chunks)
        article_with_chunks['chunks'] = chunks
        
        processed_articles.append(article_with_chunks)
    
    return processed_articles


def main():
    """Main function with legal domain classification for first 10 articles only."""
    print("\n" + "=" * 80)
    print("CONSTITUTION EXTRACTOR WITH LEGAL DOMAIN CLASSIFICATION (FIRST 10 ARTICLES)")
    print("=" * 80 + "\n")
    
    # Load constitution
    constitution_text = load_constitution_document(CONSTITUTION_DIR)
    
    if constitution_text.startswith("Error:"):
        print("Failed to load document. Exiting.")
        return
    
    # Extract articles
    sections = find_numbered_sections_with_headers(constitution_text)
    
    if not sections:
        print("No articles found.")
        return
    
    # Process and chunk articles
    processed_articles = process_articles_with_chunks(sections)
    
    # NEW: Classify ONLY first 10 articles using LLM
    classified_articles = classify_all_articles(processed_articles)
    
    # Calculate statistics
    total_chunks = sum(article['num_chunks'] for article in classified_articles)
    articles_with_multiple_chunks = sum(1 for article in classified_articles if article['num_chunks'] > 1)
    articles_classified = sum(1 for article in classified_articles if article.get('legal_domain') != "Not classified (beyond limit)")
    
    print(f"Total Articles: {len(classified_articles)}")
    print(f"Articles Classified with LLM: {articles_classified}")
    print(f"Total Chunks: {total_chunks}")
    print(f"Articles with multiple chunks: {articles_with_multiple_chunks}\n")
    
    # Save to file
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "constitutechecker.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("CONSTITUTION ANALYSIS WITH LEGAL DOMAIN CLASSIFICATION\n")
            f.write(f"(First {MAX_ARTICLES_TO_CLASSIFY} articles classified with LLM)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Articles: {len(classified_articles)}\n")
            f.write(f"Articles Classified: {articles_classified}\n")
            f.write(f"Chunk Size: {CHUNK_SIZE} characters\n")
            f.write(f"Chunk Overlap: {CHUNK_OVERLAP} characters\n")
            f.write(f"Total Chunks: {total_chunks}\n")
            f.write(f"Model Used: {MODEL_NAME}\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, article in enumerate(classified_articles, 1):
                f.write(f"MATCH #{idx}\n")
                f.write(f"Header: {article['header'] if article['header'] else '[No header]'}\n")
                f.write(f"Number: {article['number']}\n")
                
                sentence_preview = article['sentence'][:100]
                if len(article['sentence']) > 100:
                    sentence_preview += "..."
                f.write(f"Sentence: {article['number']}. {sentence_preview}\n")
                
                # Add legal domain (classified or not)
                f.write(f"Legal Domain: {article.get('legal_domain', 'Not classified')}\n")
                
                if article['chapter']:
                    f.write(f"Chapter: {article['chapter']}\n")
                else:
                    f.write(f"Chapter: [Not within a CHAPTER]\n")
                
                if article['part']:
                    f.write(f"Part: PART {article['part']} - {article['part_title']}\n")
                else:
                    f.write(f"Part: [Not within a PART]\n")
                
                f.write(f"Article Length: {article['text_length']} characters\n")
                f.write(f"Number of Chunks: {article['num_chunks']}\n")
                
                if article['num_chunks'] > 1:
                    f.write(f"\nChunk Breakdown:\n")
                    for chunk_idx, chunk in enumerate(article['chunks'], 1):
                        f.write(f"  Chunk {chunk_idx}/{article['num_chunks']} ({len(chunk)} chars):\n")
                        chunk_preview = chunk[:150]
                        if len(chunk) > 150:
                            chunk_preview += "..."
                        f.write(f"  {chunk_preview}\n\n")
                
                f.write("-" * 80 + "\n\n")
        
        print(f"\n✓ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"ERROR: Failed to write file: {e}")


if __name__ == "__main__":
    main()