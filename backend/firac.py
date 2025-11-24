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

# ===== FACTS EXTRACTION PROMPT =====
FACTS_EXTRACTION_PROMPT = """You are a precision legal fact extractor for court judgments.

═══════════════════════════════════════════════════════════════
FUNDAMENTAL PRINCIPLE
═══════════════════════════════════════════════════════════════

FACTS = What happened in the real world (events, evidence, testimony)
NOT FACTS = What people say ABOUT what happened (arguments, analysis, reasoning)

═══════════════════════════════════════════════════════════════
CRITICAL DISTINCTION: TRIAL FACTS vs APPEAL ARGUMENTS
═══════════════════════════════════════════════════════════════

Many documents are APPEALS or APPLICATIONS that reference trial facts.
You must distinguish:

✓ TRIAL FACTS (Extract these):
  - What witnesses testified AT THE TRIAL
  - What evidence was presented AT THE TRIAL
  - What the accused said AT THE TRIAL
  - What happened during the incident/crime

✗ APPEAL ARGUMENTS (Do NOT extract these):
  - What lawyers argue IN THE APPEAL about the trial
  - Grounds of appeal
  - Submissions by counsel in the current proceeding
  - Challenges to evidence or procedure

DETECTION RULE:
- Phrases like "the trial court found..." or "evidence at trial showed..." 
  → Extract the underlying fact
- Phrases like "counsel submitted..." or "the appellant argues..." 
  → Skip entirely (this is an argument)

═══════════════════════════════════════════════════════════════
SECTION 1: CASE BACKGROUND
═══════════════════════════════════════════════════════════════

ALWAYS include:
✓ Case name, number, citation
✓ Court and judge(s)
✓ Parties (accused/appellant, complainant/respondent, etc.)
✓ Type of proceeding (trial, appeal, bail application, review)
✓ Charges or claims
✓ Counsel names (if mentioned)

═══════════════════════════════════════════════════════════════
SECTION 2: CHRONOLOGICAL EVENTS
═══════════════════════════════════════════════════════════════

Extract events that happened in the real world:
✓ What occurred on specific dates
✓ Actions by individuals (accused, victim, police, witnesses)
✓ Sequence of events leading to charges
✓ Physical actions and movements

Format: "On [date], [person] [action]"

═══════════════════════════════════════════════════════════════
SECTION 3: WITNESS TESTIMONY
═══════════════════════════════════════════════════════════════

⚠️ CRITICAL RULES - READ CAREFULLY:

ONLY include in this section:
✓ Direct testimony from trial witnesses (PW1, PW2, DW1, DW2, etc.)
✓ Must use phrases: "PW1 testified that...", "DW2 stated that..."
✓ What investigating officers reported observing
✓ Statements by the accused to police or in court

NEVER include in this section:
✗ Lawyer statements in appeal/application ("Counsel stated...")
✗ Submissions or arguments ("Mr./Ms. [Name] submitted...")
✗ References to testimony without the testimony itself

⚠️ MANDATORY SELF-CHECK:
Before adding anything to TESTIMONY section, ask:
- Is this a witness speaking at trial? YES → Include
- Is this a lawyer arguing in appeal? YES → EXCLUDE
- Does it start with "Counsel/Mr./Ms. [Name] submitted"? YES → EXCLUDE

⚠️ IF NO TESTIMONY AVAILABLE:
If the document is an appeal/application that doesn't contain actual 
trial testimony, you MUST write:
"Not available - this document does not contain detailed witness testimony 
from the trial. It only references that testimony occurred."

═══════════════════════════════════════════════════════════════
SECTION 4: PHYSICAL/FORENSIC EVIDENCE
═══════════════════════════════════════════════════════════════

✓ Items recovered (weapons, money, documents, etc.)
✓ Medical/autopsy findings
✓ Forensic reports (DNA, fingerprints, ballistics)
✓ Digital evidence (CCTV, phone records, messages)
✓ Scientific test results

Format: State what was found, not arguments about its validity
Example: "Police recovered 100 US dollars from the house"
NOT: "Counsel argued the money was illegally seized"

═══════════════════════════════════════════════════════════════
SECTION 5: PROSECUTION CASE
═══════════════════════════════════════════════════════════════

Include the prosecution's VERSION of events:
✓ "The prosecution alleged that..."
✓ "The prosecution's theory was that..."
✓ What they claimed happened

This is their story/narrative, stated neutrally.

═══════════════════════════════════════════════════════════════
SECTION 6: DEFENCE/ACCUSED'S VERSION
═══════════════════════════════════════════════════════════════

⚠️ CRITICAL RULES - READ CAREFULLY:

ONLY include what the ACCUSED PERSON said happened:
✓ "The accused testified that..."
✓ "The accused told police that..."
✓ "The defence case was that..." (when describing their factual narrative)

NEVER include lawyer arguments:
✗ "Counsel submitted the conviction was wrong" (lawyer argument)
✗ "The appellant argues there was no evidence" (lawyer argument)
✗ "Mr. [Name] contended the search was illegal" (lawyer argument)

⚠️ DETECTION TEST:
- Does this describe what the accused CLAIMS HAPPENED? → Include
- Does this describe what the lawyer ARGUES ABOUT what happened? → Exclude

⚠️ COMMON TRAP IN APPEAL DOCUMENTS:
Appeals often state: "The appellant claims he was convicted on suspicion"
This is a GROUND OF APPEAL (lawyer argument), NOT the accused's version.

The accused's version would be: "The accused testified he was at home 
during the incident" or "The accused stated he found the money on the street"

⚠️ IF NO DEFENCE VERSION AVAILABLE:
If the document doesn't contain what the accused actually said, write:
"Not available - this document does not contain the accused's testimony 
or statement about what happened."

═══════════════════════════════════════════════════════════════
SECTION 7: PROCEDURAL HISTORY
═══════════════════════════════════════════════════════════════

✓ Arrest and investigation steps
✓ Charges filed (when, what charges, where)
✓ Trial proceedings (dates, pleas, proceedings)
✓ Conviction and sentence details
✓ Appeals filed
✓ Applications made (bail, stay, review)
✓ Key procedural dates

═══════════════════════════════════════════════════════════════
SECTION 8: TRIAL COURT'S FACTUAL FINDINGS (Appeals/Reviews Only)
═══════════════════════════════════════════════════════════════

In appeal documents, include what the TRIAL COURT found as fact:
✓ "The trial court found that [factual finding]"
✓ "The magistrate noted [factual observation]"
✓ "The trial court relied on [type of evidence]"

Examples of what to INCLUDE:
✓ "The trial court convicted based on circumstantial evidence"
✓ "The magistrate noted contradictions in PW1's testimony"
✓ "The trial court found the accused had no explanation for the money"
✓ "The judge observed the charge sheet was defective"

Examples of what to EXCLUDE:
✗ "The trial court correctly applied the law" (legal analysis)
✗ "The magistrate held that the burden was discharged" (legal reasoning)
✗ "The court found the accused guilty because..." (reasoning)

DISTINCTION: Include WHAT they found, not WHY they found it.

═══════════════════════════════════════════════════════════════
SECTION 9: COURT ORDERS (Outcome Only)
═══════════════════════════════════════════════════════════════

Include ONLY the final orders as pure facts:
✓ Sentence imposed: "Sentenced to X years imprisonment"
✓ Bail granted: "Released on bail of Kshs X"
✓ Appeal outcome: "Appeal allowed/dismissed"
✓ Acquittal/conviction: "Accused acquitted/convicted"

DO NOT include:
✗ Reasoning behind the orders
✗ Analysis of why the decision was made
✗ Legal basis for the decision

═══════════════════════════════════════════════════════════════
ABSOLUTE EXCLUSIONS - NEVER EXTRACT THESE
═══════════════════════════════════════════════════════════════

✗ Phrases starting with:
  - "Counsel submitted..."
  - "Mr./Ms. [Name] argued..."
  - "The appellant/applicant contends..."
  - "It is submitted that..."
  - "The defence/prosecution urges..."

✗ Legal analysis:
  - "The court finds that..."
  - "The court holds that..."
  - "In my view..."
  - "I am satisfied that..."
  - "The law requires..."

✗ Legal principles and citations:
  - "It is trite law that..."
  - "In the case of [Case Name]..."
  - "Section X provides..."
  - "The principle established in..."

✗ Credibility assessments:
  - "The witness was credible"
  - "I believe PW1"
  - "The testimony was reliable"

✗ Grounds of appeal:
  - List of errors alleged by appellant
  - Challenges to conviction
  - Legal arguments about procedure

═══════════════════════════════════════════════════════════════
SELF-CHECK BEFORE OUTPUTTING
═══════════════════════════════════════════════════════════════

Before finalizing your extraction, verify:

1. TESTIMONY section:
   □ Does it contain "PW1 testified..." or "DW1 stated..."? 
   □ Does it contain "Counsel submitted..." or "Mr./Ms. X argued..."?
   → If YES to second question, REMOVE those entries

2. DEFENCE VERSION section:
   □ Does it describe what accused SAID HAPPENED?
   □ Does it describe what lawyer ARGUED about the case?
   → If YES to second question, REMOVE and write "Not available"

3. TRIAL COURT FINDINGS (if appeals):
   □ Does it state WHAT the court found (facts)?
   □ Does it state WHY the court decided (reasoning)?
   → If YES to second question, REMOVE the reasoning parts

4. Overall check:
   □ Would someone reading this understand what actually happened?
   □ Have I included any arguments or analysis by mistake?
   □ Have I acknowledged when information is not available?

═══════════════════════════════════════════════════════════════
EXAMPLES - CORRECT vs INCORRECT EXTRACTION
═══════════════════════════════════════════════════════════════

SCENARIO 1: Lawyer Argument vs Trial Fact
─────────────────────────────────────────
SOURCE: "Mr. Makumi submitted that none of the prosecution witnesses 
gave direct evidence connecting the applicant to the offence."

❌ WRONG: "None of the prosecution witnesses gave direct evidence"
✓ CORRECT: [Do not extract - this is a lawyer's argument]

If you want to extract the underlying fact, look for: "The prosecution 
case was based on circumstantial evidence" (if stated as fact elsewhere)

─────────────────────────────────────────
SCENARIO 2: Appeal Ground vs Accused's Version
─────────────────────────────────────────
SOURCE: "The appellant claims he was convicted purely on suspicion when 
there was no direct evidence connecting him to the crime."

❌ WRONG: "The accused claimed he was convicted on suspicion"
✓ CORRECT: [Do not extract - this is a ground of appeal, not the 
accused's version of events]

The accused's version would be what he said happened during the incident,
not what he argues about his conviction.

─────────────────────────────────────────
SCENARIO 3: Actual Testimony vs Reference to Testimony
─────────────────────────────────────────
SOURCE: "Counsel for the applicant argued that the prosecution witnesses 
were not credible."

❌ WRONG: "The prosecution witnesses were not credible"
✓ CORRECT: [Do not extract - this is a lawyer's argument about credibility]

─────────────────────────────────────────
SOURCE: "PW1 John Kamau, a police officer, testified that he arrived at 
the scene at 2pm and found the accused standing near the body."

✓ CORRECT: "PW1 John Kamau (police officer) testified that he arrived 
at the scene at 2pm and found the accused standing near the body."

─────────────────────────────────────────
SCENARIO 4: Trial Court Finding vs Appeal Court Analysis
─────────────────────────────────────────
SOURCE: "From the record, the trial magistrate convicted the appellant 
on the basis of circumstantial evidence. There is a strong indication 
that suspicion underlined the conviction."

✓ CORRECT for "Trial Court Findings": 
"The trial magistrate convicted based on circumstantial evidence. 
The conviction was underlined by suspicion."

❌ WRONG: Do not add the appeal judge's opinion: "I find the conviction 
was unsafe" (that's analysis, not a fact)

─────────────────────────────────────────
SCENARIO 5: Document Without Trial Details
─────────────────────────────────────────
DOCUMENT TYPE: Bail application ruling

SITUATION: Document mentions the trial happened but doesn't detail testimony

✓ CORRECT OUTPUT:
**WITNESS TESTIMONY**
Not available - this is a bail application ruling that does not contain 
detailed witness testimony from the trial. The document references that 
prosecution witnesses testified, but their testimony is not detailed here.

**DEFENCE VERSION**
Not available - this document does not contain the accused's testimony 
from the trial.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Structure your extraction as:

**CASE BACKGROUND**
[Include all case details, parties, charges]

**CHRONOLOGICAL EVENTS**  
[Events in order with dates]

**WITNESS TESTIMONY**
[Actual testimony from trial, or "Not available" with explanation]

**PHYSICAL/FORENSIC EVIDENCE**
[Items, reports, findings]

**PROSECUTION CASE**
[Their version of events]

**DEFENCE/ACCUSED'S VERSION**
[What accused said happened, or "Not available" with explanation]

**PROCEDURAL HISTORY**
[Arrest, charges, trial, appeals]

**TRIAL COURT'S FACTUAL FINDINGS** (For appeals/reviews only)
[What lower court found as fact]

**FINAL ORDERS**
[Outcome only - no reasoning]

═══════════════════════════════════════════════════════════════
DOCUMENT TO ANALYZE
═══════════════════════════════════════════════════════════════

{document_text}

═══════════════════════════════════════════════════════════════
BEGIN EXTRACTION
═══════════════════════════════════════════════════════════════

Remember:
- Facts = What happened (events, evidence, testimony)
- Not facts = What people argue about what happened
- When in doubt: Is this describing reality, or arguing about reality?
- Acknowledge when information is not available in the document
"""

SYSTEM_PROMPT = "You are a legal document analyst specializing in extracting factual content from Kenyan court judgments. Extract only facts, not legal analysis or court opinions."


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





def extract_facts_with_llm(document_text: str) -> str:
    """
    Use Groq LLM to extract facts from the court judgment.
    Returns extracted facts or error message.
    """
    print("\n===== EXTRACTING FACTS USING LLM =====")
    print(f"Document length: {len(document_text)} characters")
    
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        prompt = FACTS_EXTRACTION_PROMPT.format(document_text=document_text)
        
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
            temperature=0.1,
            max_tokens=4000,
        )
        
        extracted_facts = response.choices[0].message.content.strip()
        print(f"✓ Facts extracted: {len(extracted_facts)} characters")
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

