import re
import fitz  # PyMuPDF
import pytesseract
import io
import time
import os
from PIL import Image
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from dotenv import load_dotenv
import  requests

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "Repo"
PDF_PATH = "Muruatetu  another v Republic Katiba Institute  5 others (Amicus Curiae) (Petition 15  16of2015) 2021KESC31(KLR) (6July2021) (Directions).pdf"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def extract_text_from_pdf(file_path: Path, use_ocr_threshold: int = 50) -> str:
    """
    Extract text from PDF using PyMuPDF with OCR fallback for images.
    Handles both text-based and image-based PDFs.
    
    Args:
        file_path: Path to PDF file
        use_ocr_threshold: Minimum text length to skip OCR (default: 50 chars)
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


def load_wilson_document() -> str:
    """
    Load and process the Wilson Wanjala PDF document with OCR support.
    Returns the cleaned text content.
    """
    print(f"\n{'=' * 60}")
    print(f"Loading Wilson Wanjala document...")
    print(f"{'=' * 60}")

    file_path = REPO_DIR / PDF_PATH

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

        print(f"\n✓ Successfully loaded document: {cleaned_size} characters")
        print(f"{'=' * 60}\n")

        return cleaned_content

    except Exception as exc:
        error_msg = f"Error: Could not read file: {exc}"
        print(error_msg)
        return error_msg




"""
Legal Document Extraction System - Facts and Issues Extraction for Kenyan Court Judgments
"""

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

ISSUES_EXTRACTION_PROMPT = """You are a legal document analyst specializing in Kenyan court judgments.
Think carefully about what the court is being asked to decide.
Ask yourself: What LEGAL QUESTIONS does the court need to answer 
to reach its decision?

TASK: Identify ALL legal issues that the court addressed in this judgment.

═══════════════════════════════════════════════════════════════
STEP 1: UNDERSTAND WHAT "ISSUES" MEANS
═══════════════════════════════════════════════════════════════

Issues are the KEY LEGAL QUESTIONS that the court must answer to decide the case.

Think of issues as questions beginning with:
- "Whether..." (most common in Kenyan judgments)
- "Did the [party] prove..."
- "Was there..."
- "Is the [accused/appellant/party]..."

═══════════════════════════════════════════════════════════════
STEP 2: WHERE TO FIND ISSUES
═══════════════════════════════════════════════════════════════

Kenyan judgments don't always state issues explicitly. Look in:

✓ EXPLICIT STATEMENTS:
  - "The issues for determination are..."
  - "The questions for this court are..."
  - "This appeal raises the following issues..."
  - "The court must decide whether..."

✓ GROUNDS OF APPEAL (in appellate cases):
  Each ground usually points to an issue:
  - Ground: "The learned magistrate erred in finding..."
    → Issue: Whether the magistrate erred in finding...

✓ INTRODUCTORY PARAGRAPHS:
  - Often frames what the case is about
  - May say "This is an appeal against..." or "The accused is charged with..."

✓ LEGAL ANALYSIS SECTIONS:
  - Look for transitional phrases like:
    - "The first question is..."
    - "We now turn to..."
    - "The next matter concerns..."
    - "It must be determined whether..."

✓ CHARGES/CLAIMS:
  - In trials: Each charge raises an issue (guilt/liability)
  - In civil cases: Each claim or prayer raises an issue

✓ PROCEDURAL CHALLENGES:
  - Admissibility of evidence
  - Jurisdiction questions
  - Proper procedure followed

═══════════════════════════════════════════════════════════════
STEP 3: TYPES OF ISSUES IN KENYAN CASES
═══════════════════════════════════════════════════════════════

CRIMINAL TRIAL ISSUES:
- Whether the prosecution proved the charge beyond reasonable doubt
- Whether the accused's defence raises reasonable doubt
- Whether identification was credible
- Whether a dying declaration was properly admitted
- Whether circumstantial evidence proves guilt

CRIMINAL APPEAL ISSUES:
- Whether the trial court properly evaluated evidence
- Whether the conviction is safe
- Whether the sentence is manifestly excessive/lenient
- Whether proper procedure was followed
- Whether the magistrate/judge erred in law or fact

CIVIL TRIAL ISSUES:
- Whether the plaintiff proved their case on balance of probabilities
- Whether the defendant is liable
- Whether specific elements of a claim are satisfied
- Quantum of damages

CIVIL APPEAL ISSUES:
- Whether the trial judge erred in findings of fact
- Whether the trial judge erred in law
- Whether the decision is supported by evidence
- Whether damages awarded are appropriate

PROCEDURAL ISSUES (any case type):
- Whether the court has jurisdiction
- Whether evidence was properly admitted/excluded
- Whether proper notice was given
- Whether statutory timelines were met

═══════════════════════════════════════════════════════════════
STEP 4: HOW TO FRAME ISSUES
═══════════════════════════════════════════════════════════════

Convert statements into legal questions:

EXAMPLE 1:
Court says: "The appellant challenges the credibility of PW1's testimony"
Issue: "Whether PW1's testimony was credible and reliable"

EXAMPLE 2:
Ground of Appeal: "The learned trial judge erred in finding that the appellant was properly identified"
Issue: "Whether the identification evidence was sufficient to prove the appellant committed the offence"

EXAMPLE 3:
Court says: "The prosecution must prove malice aforethought"
Issue: "Whether the prosecution proved malice aforethought beyond reasonable doubt"

EXAMPLE 4:
In a contract case: "The defendant argues the contract was void"
Issue: "Whether the contract between the parties was valid and enforceable"

═══════════════════════════════════════════════════════════════
STEP 5: WHAT TO EXCLUDE
═══════════════════════════════════════════════════════════════

DO NOT include:

✗ Facts (what happened) - those are separate
✗ Evidence details - describe those under facts
✗ The court's answer to the issue - that's the holding/conclusion
✗ Legal principles cited - those are rules
✗ Procedural history (unless jurisdiction itself is in issue)
✗ Overly specific sub-points - combine into main issues
✗ Rhetorical questions used in reasoning

═══════════════════════════════════════════════════════════════
STEP 6: OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Present issues in this format:

**MAIN ISSUES FOR DETERMINATION**

1. [First major issue - frame as a question]
   - [Sub-issue if applicable]
   - [Sub-issue if applicable]

2. [Second major issue]
   - [Sub-issue if applicable]

3. [Third major issue]

**SUBSIDIARY/PROCEDURAL ISSUES** (if any)

- [Any procedural or preliminary issues]
- [E.g., admissibility, jurisdiction]

═══════════════════════════════════════════════════════════════
IMPORTANT RULES
═══════════════════════════════════════════════════════════════

1. Be COMPREHENSIVE - identify all issues the court addressed
2. Frame as QUESTIONS - use "Whether..." or "Did..." format
3. Be SPECIFIC - relate to the particular case facts
4. Distinguish MAIN issues from subsidiary issues
5. In appeals: Convert grounds of appeal into issues
6. Stay NEUTRAL - don't indicate which side should win
7. Each issue should be ANSWERABLE with the court's reasoning
8. Group related questions under main issues
9. Prioritize in order of importance when possible
10. When in doubt, include it as a potential issue

═══════════════════════════════════════════════════════════════
JUDGMENT TO ANALYZE
═══════════════════════════════════════════════════════════════

{document_text}

═══════════════════════════════════════════════════════════════
EXTRACTED ISSUES
═══════════════════════════════════════════════════════════════
"""


APPLICATION_ANALYSIS_EXTRACTION_PROMPT = """You are a legal document analyst specializing in Kenyan court judgments.
Your job is to extract the COURT'S APPLICATION/ANALYSIS section of IRAC.

Think carefully before including any passage.
Ask yourself: Is this the court EXPLAINING HOW THE LAW APPLIES TO THE FACTS,
or is it just a summary of facts or a bare statement of the final result?

TASK: Extract the COURT'S LEGAL REASONING (APPLICATION/ANALYSIS) from this judgment.

═══════════════════════════════════════════════════════════════
STEP 1: UNDERSTAND WHAT "APPLICATION/ANALYSIS" MEANS
═══════════════════════════════════════════════════════════════

The Application/Analysis is the court's EXPLANATION of:
- How it applies legal rules and principles to the established facts
- Why it accepts or rejects particular evidence
- How it evaluates credibility, weight of evidence, and arguments
- How it reasons from statutes, case law, and legal principles to reach a result

It is the "A" in IRAC: the COURT'S REASONING.

═══════════════════════════════════════════════════════════════
STEP 2: WHERE TO FIND APPLICATION/ANALYSIS IN KENYAN JUDGMENTS
═══════════════════════════════════════════════════════════════

Look for sections where the judge:
✓ Discusses evidence and then says what it PROVES or DOES NOT PROVE
✓ Weighs competing versions of events
✓ Evaluates credibility of witnesses
✓ Applies legal tests or elements to the facts
✓ Interprets statutes or previous decisions and then applies them

Common phrases:
- "I have carefully considered..."
- "The question then is..."
- "From the evidence on record..."
- "In my view..."
- "The court therefore finds that..."
- "Applying the above principles to the facts..."

These sections usually appear AFTER the facts and issues and BEFORE the final orders.

═══════════════════════════════════════════════════════════════
STEP 3: WHAT TO INCLUDE
═══════════════════════════════════════════════════════════════

INCLUDE as Application/Analysis:

✓ The court's evaluation of the evidence
  - Why it believes or disbelieves certain witnesses
  - How it resolves contradictions

✓ The court's application of legal tests/elements
  - E.g., how the court decides if malice aforethought, negligence, consent, identification, etc. are proved

✓ The court's use of legal principles/case law/statutes
  - Where the court quotes or summarizes the law and then LINKS it to the case facts

✓ The reasoning that connects ISSUES → RULES → FACTS → RESULT
  - For each issue, how the court reasons towards an answer

═══════════════════════════════════════════════════════════════
STEP 4: WHAT TO EXCLUDE
═══════════════════════════════════════════════════════════════

DO NOT include:

✗ Pure factual narration (what happened, witness testimony) – that belongs to FACTS
✗ Simple statements of the issues – those belong to ISSUES
✗ Bare legal rules without application
  - E.g., "The law is that the prosecution must prove its case beyond reasonable doubt"
✗ The final result or orders only (e.g., "The appeal is dismissed") – these belong to CONCLUSION
✗ Headnotes or summaries written by editors (only use the judge's reasoning)

═══════════════════════════════════════════════════════════════
STEP 5: OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Present the Application/Analysis in an ISSUE-BY-ISSUE structure:

═══════════════════════════════════════════════════════════════
SECTION 3: APPLICATION / ANALYSIS
═══════════════════════════════════════════════════════════════

**ISSUE-BY-ISSUE APPLICATION/ANALYSIS**

1. Issue 1: [State the issue in question form, matching Section 2]

   **Rule(s) Applied:**
   - [Summarize the main legal rules, statutes, or cases the court relied on for this issue]

   **Court's Application / Reasoning:**
   - [Explain, in your own words, how the court applied those rules to the facts]
   - [Include how the court evaluated evidence and credibility]
   - [Explain why the court accepted or rejected each party's argument]

2. Issue 2: [State the issue]

   **Rule(s) Applied:**
   - [...]

   **Court's Application / Reasoning:**
   - [...]

3. [Continue for all issues identified in Section 2]

IMPORTANT:
- Paraphrase the court's reasoning in CLEAR LANGUAGE
- Preserve the logical steps in the reasoning
- Explicitly link RULES → FACTS → INTERMEDIATE FINDINGS

═══════════════════════════════════════════════════════════════
IMPORTANT RULES
═══════════════════════════════════════════════════════════════

1. Do NOT repeat the entire facts section – focus on reasoning.
2. Make the connection between law and facts explicit.
3. Keep the structure ISSUE → RULE(S) APPLIED → REASONING.
4. Be faithful to the judgment – do not invent reasoning.
5. If the court does not clearly separate issues, infer reasonable groupings based on the discussion.

═══════════════════════════════════════════════════════════════
JUDGMENT TO ANALYZE
═══════════════════════════════════════════════════════════════

{document_text}

═══════════════════════════════════════════════════════════════
EXTRACTED APPLICATION / ANALYSIS
═══════════════════════════════════════════════════════════════
"""

CONCLUSION_EXTRACTION_PROMPT = """You are a legal document analyst specializing in Kenyan court judgments.
Your job is to extract the CONCLUSION/HOLDING section of IRAC.

Think carefully:
- The CONCLUSION answers the issues.
- It states the FINAL OUTCOME and ORDERS of the court.

TASK: Extract the COURT'S CONCLUSIONS and FINAL ORDERS from this judgment.

═══════════════════════════════════════════════════════════════
STEP 1: UNDERSTAND WHAT "CONCLUSION/HOLDING" MEANS
═══════════════════════════════════════════════════════════════

The Conclusion/Holding is:

- How the court ANSWERED each legal issue
- The final outcome between the parties
- The ultimate orders of the court

Examples:
- "The appeal is dismissed/allowed."
- "The accused is acquitted/convicted."
- "Judgment is entered for the plaintiff."
- "The suit is struck out for want of jurisdiction."
- "The sentence is set aside/varied/upheld."

═══════════════════════════════════════════════════════════════
STEP 2: WHERE TO FIND CONCLUSION/HOLDING
═══════════════════════════════════════════════════════════════

Look at the END of the judgment for:

✓ Phrases like:
  - "In the result..."
  - "In the upshot..."
  - "For the foregoing reasons..."
  - "I therefore find..."
  - "The final orders of this court are as follows..."
  - "I accordingly..."

✓ Clear statements on:
  - Whether each main issue is answered YES or NO
  - Whether the appeal is allowed/dismissed
  - Whether the accused is guilty/not guilty
  - Damages awarded, sentences imposed, costs, declarations, etc.

═══════════════════════════════════════════════════════════════
STEP 3: WHAT TO INCLUDE
═══════════════════════════════════════════════════════════════

INCLUDE:

✓ For EACH main issue:
  - A concise statement of how the court resolved it
  - E.g., "Whether identification was positive" → "The court held that identification was not positive."

✓ Overall case outcome:
  - Appeal allowed/dismissed
  - Suit succeeds/fails
  - Conviction upheld/overturned
  - Sentence varied/confirmed

✓ Final orders:
  - Conviction(s) and count(s)
  - Sentence details (imprisonment, fines, probation, etc.)
  - Damages awarded (general/special damages)
  - Injunctions, declarations, specific performance
  - Orders as to costs

═══════════════════════════════════════════════════════════════
STEP 4: WHAT TO EXCLUDE
═══════════════════════════════════════════════════════════════

DO NOT include:

✗ Detailed reasoning – that belongs in APPLICATION/ANALYSIS.
✗ Full discussion of evidence.
✗ Repetition of the facts.
✗ Long quotations from statutes or other cases.

You are only extracting the FINAL ANSWERS and ORDERS, not the reasoning.

═══════════════════════════════════════════════════════════════
STEP 5: OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Present the conclusion/holding in a structured way:

═══════════════════════════════════════════════════════════════
SECTION 4: CONCLUSION / HOLDING
═══════════════════════════════════════════════════════════════

**HOLDING ON EACH MAIN ISSUE**

1. [Restate Issue 1 as a question]
   - [Court's short answer, e.g., "Yes, the prosecution proved malice aforethought" or "No, the identification was not free from error"]

2. [Issue 2]
   - [Court's short answer]

3. [Continue for all main issues]

**OVERALL OUTCOME**

- [Clear statement of overall result, e.g., "The appeal is dismissed", "The appeal is allowed", "The suit is dismissed", "Judgment is entered for the plaintiff", etc.]

**FINAL ORDERS**

- [Order 1: e.g., "The conviction on count I is upheld/set aside"]
- [Order 2: e.g., "The sentence is reduced to..."]
- [Order 3: e.g., "The respondent shall pay costs of the appeal/trial"]
- [Any other orders: declarations, injunctions, directions]

═══════════════════════════════════════════════════════════════
IMPORTANT RULES
═══════════════════════════════════════════════════════════════

1. Be concise – conclusions should be short and direct.
2. Ensure every MAIN ISSUE from the issues section has a corresponding holding.
3. Make the overall outcome unambiguous.
4. Include all material final orders affecting the parties.
5. Do NOT add your own opinions – just restate the court's holdings.

═══════════════════════════════════════════════════════════════
JUDGMENT TO ANALYZE
═══════════════════════════════════════════════════════════════

{document_text}

═══════════════════════════════════════════════════════════════
EXTRACTED CONCLUSION / HOLDING
═══════════════════════════════════════════════════════════════
"""

COMBINED_EXTRACTION_PROMPT = """You are a legal document analyst specializing in Kenyan court judgments.

Your task is to extract a FULL IRAC STRUCTURE from the judgment below:
- FACTS (what happened)
- ISSUES (legal questions)
- APPLICATION/ANALYSIS (court's reasoning)
- CONCLUSION/HOLDING (final answers and orders)

CRITICAL: You MUST provide FOUR separate sections in your response:
1. FACTS section
2. ISSUES section
3. APPLICATION / ANALYSIS section
4. CONCLUSION / HOLDING section

═══════════════════════════════════════════════════════════════
PART A: FACTS EXTRACTION
═══════════════════════════════════════════════════════════════

Extract ALL factual content - the events, testimony, evidence, and what parties claim happened.

INCLUDE:
✓ Background of parties and relationships
✓ Chronological events (what happened, when, where)
✓ Witness testimonies (attributed clearly)
✓ Physical/forensic evidence
✓ Prosecution/Plaintiff's case theory
✓ Defence/Defendant's version of events
✓ Procedural facts (arrest, investigation, lower court proceedings)

EXCLUDE:
✗ Court's analysis or opinions
✗ Legal reasoning
✗ Citations of other cases
✗ Legal principles
✗ Court's final decision or orders

═══════════════════════════════════════════════════════════════
PART B: ISSUES EXTRACTION
═══════════════════════════════════════════════════════════════

Identify the KEY LEGAL QUESTIONS the court needed to answer.

Look for issues in:
✓ Explicit statements: "The issues for determination are..."
✓ Grounds of appeal in appellate cases
✓ Charges in criminal trials
✓ Claims in civil cases
✓ Procedural challenges (jurisdiction, admissibility, etc.)

Frame each issue as a question starting with "Whether..."

═══════════════════════════════════════════════════════════════
PART C: APPLICATION / ANALYSIS EXTRACTION
═══════════════════════════════════════════════════════════════

Extract the court's reasoning showing HOW it applied the law to the facts.

INCLUDE:
✓ Evaluation of evidence and credibility
✓ Application of legal tests/elements to the established facts
✓ Discussion of relevant statutes and case law and their application
✓ The logical steps that move from issues → rules → facts → intermediate findings

EXCLUDE:
✗ Pure restatement of facts
✗ Mere statements of issues without reasoning
✗ Bare legal rules without application
✗ Final orders or outcome (those go in CONCLUSION / HOLDING)

═══════════════════════════════════════════════════════════════
PART D: CONCLUSION / HOLDING EXTRACTION
═══════════════════════════════════════════════════════════════

Extract how the court ANSWERED each main issue and the FINAL OUTCOME.

INCLUDE:
✓ For each main issue: a clear answer (Yes/No or equivalent)
✓ Overall case result (appeal allowed/dismissed; suit succeeds/fails; conviction upheld/set aside)
✓ Final orders (conviction, sentence, damages, costs, declarations, etc.)

EXCLUDE:
✗ Detailed reasoning (belongs in APPLICATION / ANALYSIS)
✗ Repetition of full facts

═══════════════════════════════════════════════════════════════
REQUIRED OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

You MUST structure your response EXACTLY as follows:

═════════════════════════════════════════════════════════════════════
SECTION 1: FACTS
═════════════════════════════════════════════════════════════════════

**CASE INFORMATION**
- Case: [Name and citation]
- Court: [Which court]
- Type: [Trial/Appeal, Criminal/Civil]
- Charge/Claim: [Main charge or cause of action]

**PARTIES**
- Accused/Appellant/Plaintiff: [Name]
- Victim/Respondent/Defendant: [Name]
- Relationship: [How they knew each other]

**BACKGROUND FACTS**
[Context about the parties]

**CHRONOLOGICAL EVENTS**
[What happened, in chronological order with dates/times/locations]

**WITNESS TESTIMONIES**
[For each witness: what they testified, clearly attributed]

**PHYSICAL/FORENSIC EVIDENCE**
[Items recovered, medical findings, etc.]

**PROSECUTION/PLAINTIFF'S CASE**
[Their theory of what happened]

**DEFENCE/DEFENDANT'S CASE**
[Their version of events]

**PROCEDURAL FACTS**
[Arrest, investigation, lower court proceedings]

═════════════════════════════════════════════════════════════════════
SECTION 2: ISSUES
═════════════════════════════════════════════════════════════════════

**MAIN ISSUES FOR DETERMINATION**

1. [First major issue - framed as a "Whether..." question]
   - [Sub-issue if applicable]

2. [Second major issue]
   - [Sub-issue if applicable]

3. [Continue numbering all main issues]

**SUBSIDIARY/PROCEDURAL ISSUES** (if any)

- [Any procedural or preliminary issues]

═════════════════════════════════════════════════════════════════════
SECTION 3: APPLICATION / ANALYSIS
═════════════════════════════════════════════════════════════════════

**ISSUE-BY-ISSUE APPLICATION/ANALYSIS**

1. Issue 1: [Restate Issue 1]

   **Rule(s) Applied:**
   - [Key statutes/cases/principles the court relied on]

   **Court's Application / Reasoning:**
   - [How the court applied those rules to the established facts]
   - [How the court evaluated credibility and weight of evidence]
   - [Why the court accepted/rejected each party's arguments]

2. Issue 2: [Restate Issue 2]

   **Rule(s) Applied:**
   - [...]

   **Court's Application / Reasoning:**
   - [...]

3. [Continue for all issues identified in Section 2]

═════════════════════════════════════════════════════════════════════
SECTION 4: CONCLUSION / HOLDING
═════════════════════════════════════════════════════════════════════

**HOLDING ON EACH MAIN ISSUE**

1. [Issue 1 as a question]
   - [Court's short answer]

2. [Issue 2]
   - [Court's short answer]

3. [Continue for all main issues]

**OVERALL OUTCOME**

- [Overall result: appeal allowed/dismissed, suit succeeds/fails, conviction upheld/set aside, etc.]

**FINAL ORDERS**

- [Order 1]
- [Order 2]
- [Order 3]
- [Any other material orders]

═════════════════════════════════════════════════════════════════════

IMPORTANT REMINDERS:
- Keep FACTS, ISSUES, APPLICATION/ANALYSIS, and CONCLUSION/HOLDING clearly separated.
- FACTS describe WHAT HAPPENED (neutral, all versions).
- ISSUES are LEGAL QUESTIONS.
- APPLICATION/ANALYSIS is the COURT'S REASONING.
- CONCLUSION/HOLDING is the COURT'S FINAL ANSWERS AND ORDERS.
- Be comprehensive but clear.
- Use the exact headings and section divider lines shown above.

═══════════════════════════════════════════════════════════════
JUDGMENT TO ANALYZE
═══════════════════════════════════════════════════════════════

{document_text}

═══════════════════════════════════════════════════════════════
BEGIN YOUR EXTRACTION BELOW
═══════════════════════════════════════════════════════════════
"""


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


def extract_all_firac_combined(document_text: str) -> dict:
    """
    Use Groq LLM to extract ALL FIRAC components in a single call.
    This is more efficient than making separate API calls.
    
    Returns a dictionary with all FIRAC components.
    """
    print("\n===== EXTRACTING ALL FIRAC COMPONENTS (COMBINED) =====")
    print("Connecting to Groq API...")
    
    try:
        prompt = f"""You are a legal document analyst specializing in Kenyan court judgments.

Extract ALL FIRAC components from this judgment and present them in clearly separated sections:

SECTION 1: FACTS
[Extract the factual background, events, and circumstances of the case]

SECTION 2: ISSUES
[Extract the key legal questions the court had to decide]

SECTION 3: APPLICATION
[Extract how the court applied legal principles, statutes, precedents, and reasoning]

SECTION 4: CONCLUSION
[Extract the court's final ruling, decision, orders, and any directions]

Here is the court judgment:

{document_text}

Remember to clearly label each section and be comprehensive in your extraction."""
        
        print("Sending combined extraction request to Groq LLM...")
        print(f"Document length: {len(document_text)} characters")
        
        system_message = "You are a legal document analyst specializing in extracting structured information from Kenyan court judgments. You must provide clearly separated sections for FACTS, ISSUES, APPLICATION, and CONCLUSION."
        
        extracted_content = call_groq_api(prompt, system_message, max_tokens=10000)
        
        print("✓ Combined extraction completed")
        print(f"Total extracted content length: {len(extracted_content)} characters")
        
        # Parse the response to separate all sections
        result = {
            'full_response': extracted_content,
            'facts': '',
            'issues': '',
            'application': '',
            'conclusion': ''
        }
        
        # Try to split the response into all sections
        sections_found = 0
        
        if "SECTION 1: FACTS" in extracted_content:
            sections_found += 1
            if "SECTION 2: ISSUES" in extracted_content:
                facts_section = extracted_content.split("SECTION 2: ISSUES")[0].replace("SECTION 1: FACTS", "").strip()
                result['facts'] = facts_section
                sections_found += 1
        
        if "SECTION 2: ISSUES" in extracted_content:
            if "SECTION 3: APPLICATION" in extracted_content:
                issues_section = extracted_content.split("SECTION 3: APPLICATION")[0].split("SECTION 2: ISSUES")[1].strip()
                result['issues'] = issues_section
                sections_found += 1
        
        if "SECTION 3: APPLICATION" in extracted_content:
            if "SECTION 4: CONCLUSION" in extracted_content:
                application_section = extracted_content.split("SECTION 4: CONCLUSION")[0].split("SECTION 3: APPLICATION")[1].strip()
                result['application'] = application_section
                sections_found += 1
        
        if "SECTION 4: CONCLUSION" in extracted_content:
            conclusion_section = extracted_content.split("SECTION 4: CONCLUSION")[1].strip()
            result['conclusion'] = conclusion_section
            sections_found += 1
        
        if sections_found == 5:
            print(f"✓ Successfully separated all sections:")
            print(f"   • Facts: {len(result['facts'])} chars")
            print(f"   • Issues: {len(result['issues'])} chars")
            print(f"   • Application: {len(result['application'])} chars")
            print(f"   • Conclusion: {len(result['conclusion'])} chars")
        else:
            print(f"⚠ Only found {sections_found}/4 sections. Partial extraction.")
        
        return result
        
    except Exception as e:
        error_msg = f"Error in combined extraction: {str(e)}"
        print(error_msg)
        return {
            'full_response': error_msg,
            'facts': error_msg,
            'issues': error_msg,
            'application': error_msg,
            'conclusion': error_msg
        }


def extract_facts_only(document_text: str) -> str:
    """
    Use Groq LLM to extract ONLY facts from the court judgment.
    """
    print("\n===== EXTRACTING FACTS ONLY =====")
    
    try:
        prompt = FACTS_EXTRACTION_PROMPT.format(document_text=document_text)
        system_message = "You are a legal document analyst specializing in extracting factual content from Kenyan court judgments. Extract only facts, not legal analysis or court opinions."
        
        extracted_facts = call_groq_api(prompt, system_message, max_tokens=4000)
        
        print("✓ Facts extraction completed")
        print(f"Extracted content length: {len(extracted_facts)} characters")
        
        return extracted_facts
        
    except Exception as e:
        error_msg = f"Error extracting facts: {str(e)}"
        print(error_msg)
        return error_msg


def extract_issues_only(document_text: str) -> str:
    """
    Use Groq LLM to extract ONLY issues from the court judgment.
    """
    print("\n===== EXTRACTING ISSUES ONLY =====")
    
    try:
        prompt = ISSUES_EXTRACTION_PROMPT.format(document_text=document_text)
        system_message = "You are a legal document analyst specializing in identifying legal issues from Kenyan court judgments. Extract only the key legal questions the court addressed."
        
        extracted_issues = call_groq_api(prompt, system_message, max_tokens=2000)
        
        print("✓ Issues extraction completed")
        print(f"Extracted content length: {len(extracted_issues)} characters")
        
        return extracted_issues
        
    except Exception as e:
        error_msg = f"Error extracting issues: {str(e)}"
        print(error_msg)
        return error_msg


def extract_application_only(document_text: str) -> str:
    """
    Use Groq LLM to extract ONLY the application/analysis from the court judgment.
    
    This function extracts how the court applied legal principles, statutes, precedents,
    and reasoning to the facts of the case. It focuses on the court's legal analysis
    and interpretation process.
    
    Args:
        document_text (str): The full text of the court judgment
        
    Returns:
        str: Extracted application/analysis section
    """
    print("\n===== EXTRACTING APPLICATION/ANALYSIS ONLY =====")
    
    try:
        prompt = APPLICATION_EXTRACTION_PROMPT.format(document_text=document_text)
        system_message = "You are a legal document analyst specializing in extracting legal application and analysis from Kenyan court judgments. Extract how the court applied legal principles, precedents, and reasoning to resolve the issues."
        
        extracted_application = call_groq_api(prompt, system_message, max_tokens=6000)
        
        print("✓ Application/Analysis extraction completed")
        print(f"Extracted content length: {len(extracted_application)} characters")
        
        return extracted_application
        
    except Exception as e:
        error_msg = f"Error extracting application: {str(e)}"
        print(error_msg)
        return error_msg


def extract_conclusion_only(document_text: str) -> str:
    """
    Use Groq LLM to extract ONLY the conclusion from the court judgment.
    
    This function extracts the court's final ruling, decision, orders, and any
    directions given. It focuses on the outcome and what the court ultimately
    decided, including any costs, remedies, or further actions required.
    
    Args:
        document_text (str): The full text of the court judgment
        
    Returns:
        str: Extracted conclusion section
    """
    print("\n===== EXTRACTING CONCLUSION ONLY =====")
    
    try:
        prompt = CONCLUSION_EXTRACTION_PROMPT.format(document_text=document_text)
        system_message = "You are a legal document analyst specializing in extracting conclusions and rulings from Kenyan court judgments. Extract the court's final decision, orders, and any directions given."
        
        extracted_conclusion = call_groq_api(prompt, system_message, max_tokens=2000)
        
        print("✓ Conclusion extraction completed")
        print(f"Extracted content length: {len(extracted_conclusion)} characters")
        
        return extracted_conclusion
        
    except Exception as e:
        error_msg = f"Error extracting conclusion: {str(e)}"
        print(error_msg)
        return error_msg


def run_firac_individual() -> dict:
    """
    Run FIRAC extraction with INDIVIDUAL API calls for each component.
    This makes 4 separate API calls but allows for more focused extraction.
    
    Returns:
        dict: {
            'document': str,           # Original cleaned document text
            'facts': str,              # Extracted facts
            'issues': str,             # Extracted issues
            'application': str,        # Extracted application/analysis
            'conclusion': str,         # Extracted conclusion
            'error': str or None       # Error message if any
        }
    """
    print("\n" + "=" * 80)
    print("🏛️  STARTING FIRAC EXTRACTION (INDIVIDUAL API CALLS)")
    print(f"🤖 Using Groq API with model: {MODEL_NAME}")
    print("=" * 80)
    
    result = {
        'document': '',
        'facts': '',
        'issues': '',
        'application': '',
        'conclusion': '',
        'error': None
    }
    
    try:
        # Step 1: Read PDF
        print("\n[STEP 1/5] Reading PDF document...")
        document_text = load_wilson_document()
        result['document'] = document_text
        
        # Step 2: Extract facts
        print("\n[STEP 2/5] Extracting facts...")
        result['facts'] = extract_facts_only(document_text)
        
        # Step 3: Extract issues
        print("\n[STEP 3/5] Extracting issues...")
        result['issues'] = extract_issues_only(document_text)
        
        # Step 4: Extract application/analysis
        print("\n[STEP 4/5] Extracting application and analysis...")
        result['application'] = extract_application_only(document_text)
        
        # Step 5: Extract conclusion
        print("\n[STEP 5/5] Extracting conclusion...")
        result['conclusion'] = extract_conclusion_only(document_text)
        
        print("\n" + "=" * 80)
        print("✅ FIRAC EXTRACTION FINISHED SUCCESSFULLY (INDIVIDUAL CALLS)")
        print("=" * 80)
        print(f"\n📊 Extraction Summary:")
        print(f"   • Facts: {len(result['facts'])} characters")
        print(f"   • Issues: {len(result['issues'])} characters")
        print(f"   • Application: {len(result['application'])} characters")
        print(f"   • Conclusion: {len(result['conclusion'])} characters")
        
        return result
        
    except FileNotFoundError as e:
        error_msg = f"PDF file not found: {str(e)}"
        print(f"\n❌ ERROR: {error_msg}")
        result['error'] = error_msg
        return result
        
    except Exception as e:
        error_msg = f"Extraction failed: {str(e)}"
        print(f"\n❌ ERROR: {error_msg}")
        result['error'] = error_msg
        return result


def run_firac() -> dict:
    """
    Run FIRAC extraction with a SINGLE combined API call.

    Returns:
        dict: {
            'document': str,           # Original cleaned document text
            'facts': str,              # Extracted facts
            'issues': str,             # Extracted issues
            'application': str,        # Extracted application/analysis
            'conclusion': str,         # Extracted conclusion
            'full_response': str,      # Complete LLM response
            'error': str or None       # Error message if any
        }
    """
    print("\n" + "=" * 80)
    print("🏛️  STARTING FIRAC EXTRACTION (COMBINED API CALL)")
    print(f"🤖 Using Groq API with model: {MODEL_NAME}")
    print("=" * 80)

    result = {
        'document': '',
        'facts': '',
        'issues': '',
        'application': '',
        'conclusion': '',
        'full_response': '',
        'error': None
    }

    try:
        # Step 1: Read PDF
        print("\n[STEP 1/2] Reading PDF document...")
        document_text = load_wilson_document()
        result['document'] = document_text

        # Step 2: Extract all components in one call
        print("\n[STEP 2/2] Extracting all FIRAC components in single API call...")
        extraction_result = extract_all_firac_combined(document_text)

        result['facts'] = extraction_result.get('facts', '')
        result['issues'] = extraction_result.get('issues', '')
        result['application'] = extraction_result.get('application', '')
        result['conclusion'] = extraction_result.get('conclusion', '')
        result['full_response'] = extraction_result.get('full_response', '')

        print("\n" + "=" * 80)
        print("✅ FIRAC EXTRACTION FINISHED SUCCESSFULLY (COMBINED CALL)")
        print("=" * 80)
        print(f"\n📊 Extraction Summary:")
        print(f"   • Facts: {len(result['facts'])} characters")
        print(f"   • Issues: {len(result['issues'])} characters")
        print(f"   • Application: {len(result['application'])} characters")
        print(f"   • Conclusion: {len(result['conclusion'])} characters")

        return result

    except FileNotFoundError as e:
        error_msg = f"PDF file not found: {str(e)}"
        print(f"\n❌ ERROR: {error_msg}")
        result['error'] = error_msg
        return result

    except Exception as e:
        error_msg = f"Extraction failed: {str(e)}"
        print(f"\n❌ ERROR: {error_msg}")
        result['error'] = error_msg
        return result


# Example usage
if __name__ == "__main__":
    result = run_firac()
    
    print("\n" + "=" * 80)
    print("FINAL EXTRACTION RESULT")
    print("=" * 80)
    
    if result.get("error"):
        print(f"ERROR: {result['error']}")
    else:
        print(f"\n📄 Document length: {len(result['document'])} characters")
        
        if 'extraction' in result and isinstance(result['extraction'], dict):
            extraction = result['extraction']
            
            # Display Facts Section
            print("\n" + "=" * 80)
            print("📋 SECTION 1: FACTS EXTRACTED")
            print("=" * 80)
            
            if extraction.get('facts'):
                print(extraction['facts'])
                print(f"\n✓ Facts length: {len(extraction['facts'])} characters")
            else:
                print("⚠ No facts extracted")
            
            # Display Issues Section
            print("\n" + "=" * 80)
            print("⚖️  SECTION 2: ISSUES EXTRACTED")
            print("=" * 80)
            
            if extraction.get('issues'):
                print(extraction['issues'])
                print(f"\n✓ Issues length: {len(extraction['issues'])} characters")
            else:
                print("⚠ No issues extracted")
            
            # Summary
            print("\n" + "=" * 80)
            print("📊 EXTRACTION SUMMARY")
            print("=" * 80)
            print(f"Total content extracted: {len(extraction.get('full_response', ''))} characters")
            print(f"Facts section: {len(extraction.get('facts', ''))} characters")
            print(f"Issues section: {len(extraction.get('issues', ''))} characters")
            
        elif 'facts' in result:
            # Backward compatibility if result structure is different
            print("\n" + "=" * 80)
            print("📋 EXTRACTED FACTS")
            print("=" * 80)
            print(result['facts'])
            print(f"\n✓ Facts length: {len(result['facts'])} characters")
            
        # Show document preview
        print("\n" + "=" * 80)
        print("📄 DOCUMENT PREVIEW (First 1000 characters)")
        print("=" * 80)
        doc_preview = result['document'][:1000] + "..." if len(result['document']) > 1000 else result['document']
        print(doc_preview)