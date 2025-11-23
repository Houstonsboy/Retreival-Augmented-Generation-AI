import re
import time
import os
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from langchain_core.documents import Document
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "Repo"
WILSON_PDF_NAME = "Joseph Ngunguru Wanjohi v Republic 2014KEHC5356(KLR).pdf"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

MODEL_NAME = "llama-3.3-70b-versatile"


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file page by page.
    Similar to digester.py implementation.
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
    Similar to digester.py implementation.
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


def load_wilson_document() -> str:
    """
    Load and process the Wilson Wanjala PDF document.
    Returns the cleaned text content.
    """
    print(f"\n{'=' * 60}")
    print(f"Loading Wilson Wanjala document...")
    print(f"{'=' * 60}")

    file_path = REPO_DIR / WILSON_PDF_NAME

    if not file_path.exists():
        error_msg = f"Error: File not found: {file_path}"
        print(error_msg)
        return error_msg

    print(f"File found: {file_path.name}")

    try:
        # Extract text from PDF
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

        print(f"\n✓ Successfully loaded document: {cleaned_size} characters")
        print(f"{'=' * 60}\n")

        return cleaned_content

    except Exception as exc:
        error_msg = f"Error: Could not read file: {exc}"
        print(error_msg)
        return error_msg


FACTS_EXTRACTION_PROMPT = """You are a legal document analyst specializing in Kenyan court judgments.
Think carefully before classifying each piece of content.
Ask yourself: Is this describing WHAT HAPPENED, or is this 
the COURT'S OPINION about what happened?

Only include content that describes events, testimony, or 
evidence - not the court's evaluation of that evidence
TASK: Extract ALL factual content from this court judgment.

═══════════════════════════════════════════════════════════════
STEP 1: UNDERSTAND THE DOCUMENT
═══════════════════════════════════════════════════════════════

First, identify:
- Type of case (criminal/civil, trial/appeal)
- Main charge or issue
- Key parties involved

═══════════════════════════════════════════════════════════════
STEP 2: WHAT TO EXTRACT
═══════════════════════════════════════════════════════════════

INCLUDE these as facts:

✓ BACKGROUND: Parties, relationships, occupations, context
✓ EVENTS: What happened, when, where (chronological narrative)
✓ WITNESSES: What each PW/DW testified (attribute clearly)
✓ EVIDENCE: Physical items, forensic findings, medical evidence
✓ PROSECUTION VERSION: "The prosecution case was that..."
✓ DEFENCE VERSION: "The defence/appellant stated that..."
✓ PROCEDURAL: Arrest, investigation, body discovery, lower court proceedings

EXCLUDE these (NOT facts):

✗ Court's analysis: "We find that...", "The court holds..."
✗ Credibility assessment: "The witness was believable..."
✗ Legal citations: "In Rex v. Kipkering..."
✗ Legal principles: "The burden of proof requires..."
✗ Court's reasoning about the law
✗ Final orders, sentence, or judgment
✗ Grounds of appeal (unless extracting procedural facts)

═══════════════════════════════════════════════════════════════
STEP 3: OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Structure your extraction as:

**CASE INFORMATION**
- Case: [Name and citation]
- Court: [Which court]
- Type: [Trial/Appeal, Criminal/Civil]
- Charge: [Main offence charged]

**PARTIES**
- Accused/Appellant: [Name]
- Victim/Deceased/Complainant: [Name]
- Relationship: [How they knew each other]

**BACKGROUND FACTS**
[Context about the parties, their lives, occupations, etc.]

**CHRONOLOGICAL EVENTS**
[The incident narrative - what happened, in order]
- Include specific dates and times
- Include specific locations
- Include who did what

**WITNESS TESTIMONIES**
For each witness:
- [Designation] ([Name if given]) - [Role/Relationship]:
  [What they testified]

**PHYSICAL/FORENSIC EVIDENCE**
[Items recovered, medical findings, post-mortem results]

**PROSECUTION CASE**
[The prosecution's theory of what happened]

**DEFENCE CASE**  
[The accused's version of events]

**PROCEDURAL FACTS**
[Arrest, investigation steps, court proceedings]

═══════════════════════════════════════════════════════════════
IMPORTANT RULES
═══════════════════════════════════════════════════════════════

1. Be COMPREHENSIVE - facts may be scattered throughout; find them all
2. Be SPECIFIC - exact dates, times, locations, names
3. Be NEUTRAL - report all versions without evaluating truth
4. ATTRIBUTE clearly - "PW3 testified that..." not just stating as fact
5. PRESERVE details - do not summarize away important specifics
6. When in doubt, include it - better to over-extract than miss facts

═══════════════════════════════════════════════════════════════
JUDGMENT TO ANALYZE
═══════════════════════════════════════════════════════════════

{document_text}

═══════════════════════════════════════════════════════════════
EXTRACTED FACTS
═══════════════════════════════════════════════════════════════
"""


def extract_facts_with_llm(document_text: str) -> str:
    """
    Use Groq LLM to extract facts from the court judgment.
    """
    print("\n===== EXTRACTING FACTS USING LLM =====")
    print("Initializing Groq client...")
    
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Format the prompt with the document text
        prompt = FACTS_EXTRACTION_PROMPT.format(document_text=document_text)
        
        print("Sending request to Groq LLM (Llama 3.3 70B)...")
        print(f"Document length: {len(document_text)} characters")
        
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal document analyst specializing in extracting factual content from Kenyan court judgments. Extract only facts, not legal analysis or court opinions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=MODEL_NAME,
            temperature=0.1,  # More deterministic output
            max_tokens=4000,  # Large token limit for comprehensive extraction
        )
        
        extracted_facts = response.choices[0].message.content.strip()
        print("✓ Facts extraction completed")
        print(f"Extracted facts length: {len(extracted_facts)} characters")
        
        return extracted_facts
        
    except Exception as e:
        error_msg = f"Error extracting facts with LLM: {str(e)}"
        print(error_msg)
        return error_msg


def run_firac() -> Dict[str, str]:
    """
    Main function to run FIRAC analysis on Wilson Wanjala document.
    Returns a dictionary with both the full document and extracted facts.
    """
    print("\n===== STARTING FIRAC PROCESSING =====")
    
    try:
        # Load the document
        document_content = load_wilson_document()
        
        if document_content.startswith("Error:"):
            return {
                "document": document_content,
                "facts": "",
                "error": document_content
            }
        
        print("\n" + "=" * 60)
        print("FULL DOCUMENT CONTENT")
        print("=" * 60)
        print(document_content)
        print("\n" + "=" * 60)
        print("END OF FULL DOCUMENT")
        print("=" * 60)
        
        # Extract facts using LLM
        extracted_facts = extract_facts_with_llm(document_content)
        
        print("\n" + "=" * 60)
        print("EXTRACTED FACTS")
        print("=" * 60)
        print(extracted_facts)
        print("=" * 60)
        
        print("\n===== FIRAC PROCESSING COMPLETED =====")
        
        return {
            "document": document_content,
            "facts": extracted_facts,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"FIRAC processing failed: {str(e)}"
        print(error_msg)
        return {
            "document": "",
            "facts": "",
            "error": error_msg
        }


if __name__ == "__main__":
    result = run_firac()
    print("\n" + "=" * 60)
    print("DOCUMENT CONTENT:")
    print("=" * 60)
    print(result[:1000] + "..." if len(result) > 1000 else result)

