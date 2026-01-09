import fitz  # PyMuPDF
import pytesseract
import io
import re
import os
import json
import requests
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "MasterRules"
CONSTITUTION_PDF = "TheConstitutionOfKenya.pdf"  # Adjust filename as needed

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# System prompt for LLM
SYSTEM_PROMPT = """System Prompt: Kenyan Constitutional Data Architect

Role
You are a Legal Data Architect specializing in the Constitution of Kenya 2010. Your task is to transform raw, noisy text into a structured JSON array suitable for a high-precision Vector Database (RAG).

Processing Rules
1. Hierarchy Tracking:
   * Chapters: Identify "CHAPTER [NUMBER]" (e.g., CHAPTER TWO). Convert the number to an integer (e.g., 2). The line immediately below is the Chapter Title.
   * Parts: If a line starts with "PART [NO]", capture the number and the title. If no Part is present, these fields must be `null`.

2. The "Title-Above-Number" Rule:
   * You must pair the Article Title (the line of text) with the Article Number found on the line immediately below it.
   * Example: If "Culture." is followed by "11. (1)...", the title is "Culture" and the number is 11.

3. Content Cleaning (Noise Removal):
   * Strictly strip the following strings from the `text_content`: "Constitution of Kenya", "[Rev. 2022]", and standalone page numbers.
   * Ensure the `text_content` starts directly with the Article Number (e.g., "4. (1) Kenya is...") and preserves all internal subsections/clauses.

4. Legal Domain Mapping:
   * Assign exactly one of these 10 domains to each article based on its content:
      1. Foundational & Constitutional Order: Sovereignty, Supremacy, Symbols, Territory.
      2. Citizenship & National Membership: Birth, Registration, Dual Citizenship, Revocation.
      3. Fundamental Rights & Freedoms: Bill of Rights, Application of rights.
      4. Political Rights & Democratic Governance: Elections, Political parties.
      5. Public Power, Leadership & Accountability: Integrity, Leadership (Ch. 6), Public Service.
      6. Separation of Powers & State Institutions: Parliament, Judiciary, Commissions.
      7. Devolution & Intergovernmental Relations: Counties, County/National relations.
      8. Public Finance, Resources & Economic Governance: Land, Taxation, Revenue, Audit.
      9. Justice, Rule of Law & Dispute Resolution: Courts, Legal System.
      10. Constitutional Amendment, Interpretation & Enforcement: Amendment procedures, Definitions.

Output Format
Return only a valid JSON array of objects. Do not include conversational text, explanations, or markdown formatting. The output must be pure JSON that can be parsed directly.

Each object must have this exact structure:
{
  "source": "constitution2010",
  "chapter_no": integer,
  "chapter_title": "string",
  "part_no": integer_or_null,
  "part_title": "string_or_null",
  "article_no": integer,
  "article_title": "string",
  "legal_domain": "string (name of the selected domain)",
  "summary": "1-sentence legal summary",
  "keywords": ["list", "of", "5", "keywords"],
  "text_content": "Full cleaned text of the article including clauses"
}"""


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


def call_groq_api(prompt: str, system_message: str, max_tokens: int = 4000) -> str:
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
            "temperature": 0.1
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        print(f"📡 Sending request to Groq API (model: {MODEL_NAME})...")
        
        response = requests.post(
            GROQ_API_URL,
            json=payload,
            headers=headers,
            timeout=120  # 2 minute timeout for long documents
        )
        
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the response text from Groq's format
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Unexpected response format: {result}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Groq API request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Error calling Groq API: {str(e)}")


def chunk_text(text: str, max_chars: int = 15000) -> List[str]:
    """
    Split text into chunks that fit within the LLM's context window.
    Attempts to split on chapter boundaries when possible.
    """
    chunks = []
    
    # Try to split on chapter boundaries
    chapter_pattern = r'^CHAPTER\s+[A-Z]+\s*\n'
    splits = re.split(f'({chapter_pattern})', text)
    
    current_chunk = ""
    
    for i, part in enumerate(splits):
        if len(current_chunk) + len(part) < max_chars:
            current_chunk += part
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = part
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def process_constitution_with_llm(constitution_text: str) -> List[Dict]:
    """
    Process the constitution text through the LLM to extract structured data.
    Handles chunking if the text is too long.
    """
    print("\n" + "=" * 60)
    print("PROCESSING CONSTITUTION WITH LLM")
    print("=" * 60)
    
    # Check if text needs chunking
    text_length = len(constitution_text)
    print(f"Constitution text length: {text_length} characters")
    
    all_articles = []
    
    # If text is very long, chunk it
    if text_length > 15000:
        print("Text is long, processing in chunks...")
        chunks = chunk_text(constitution_text)
        print(f"Split into {len(chunks)} chunks")
        
        for idx, chunk in enumerate(chunks):
            print(f"\n--- Processing chunk {idx + 1}/{len(chunks)} ---")
            prompt = f"Extract the constitutional articles from this text section:\n\n{chunk}"
            
            try:
                response = call_groq_api(prompt, SYSTEM_PROMPT, max_tokens=6000)
                
                # Parse JSON response
                # Remove markdown code blocks if present
                response_clean = re.sub(r'```json\s*|\s*```', '', response).strip()
                
                chunk_articles = json.loads(response_clean)
                
                if isinstance(chunk_articles, list):
                    all_articles.extend(chunk_articles)
                    print(f"✓ Extracted {len(chunk_articles)} articles from chunk {idx + 1}")
                else:
                    print(f"⚠ Unexpected response format for chunk {idx + 1}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON from chunk {idx + 1}: {e}")
                print(f"Response preview: {response[:500]}...")
            except Exception as e:
                print(f"❌ Error processing chunk {idx + 1}: {e}")
    else:
        # Process entire text at once
        prompt = f"Extract all constitutional articles from this text:\n\n{constitution_text}"
        
        try:
            response = call_groq_api(prompt, SYSTEM_PROMPT, max_tokens=8000)
            
            # Parse JSON response
            response_clean = re.sub(r'```json\s*|\s*```', '', response).strip()
            all_articles = json.loads(response_clean)
            
            print(f"✓ Extracted {len(all_articles)} articles")
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            print(f"Response preview: {response[:500]}...")
        except Exception as e:
            print(f"❌ Error processing constitution: {e}")
    
    return all_articles


def save_structured_data(articles: List[Dict], output_path: Path):
    """
    Save the structured article data to a JSON file.
    """
    print(f"\n💾 Saving {len(articles)} articles to {output_path}...")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully saved to {output_path}")
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total articles extracted: {len(articles)}")
        
        # Count by chapter
        chapters = {}
        for article in articles:
            chapter = article.get('chapter_no')
            chapters[chapter] = chapters.get(chapter, 0) + 1
        
        print(f"\nArticles by chapter:")
        for chapter in sorted(chapters.keys()):
            print(f"  Chapter {chapter}: {chapters[chapter]} articles")
            
    except Exception as e:
        print(f"❌ Error saving file: {e}")


def main():
    """
    Main execution function.
    """
    print("\n" + "🇰🇪" * 30)
    print("KENYAN CONSTITUTION EXTRACTION TOOL")
    print("🇰🇪" * 30 + "\n")
    
    # Set the path to your constitution PDF
    constitution_path = REPO_DIR / CONSTITUTION_PDF
    
    # You can also specify a direct path:
    # constitution_path = Path("/path/to/your/constitution.pdf")
    
    # Step 1: Load and extract text from PDF
    constitution_text = load_constitution_document(constitution_path)
    
    if constitution_text.startswith("Error"):
        print("❌ Failed to load constitution. Exiting.")
        return
    
    # Step 2: Process with LLM
    articles = process_constitution_with_llm(constitution_text)
    
    if not articles:
        print("❌ No articles were extracted. Exiting.")
        return
    
    # Step 3: Save structured data
    output_path = BASE_DIR / "constitution_structured.json"
    save_structured_data(articles, output_path)
    
    print("\n✅ Constitution extraction complete!")


if __name__ == "__main__":
    main()