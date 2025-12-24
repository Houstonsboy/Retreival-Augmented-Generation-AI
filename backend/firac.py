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
import json

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "Repo"
PDF_PATH = "Wilson Wanjala Mkendeshwo v Republic (Criminal Appeal 97of2002) 2002KECA166(KLR) (18October2002) (Judgment).pdf"

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




   
METADATA_EXTRACTION_PROMPT = """You are a legal document analyst specializing in Kenyan court judgments.

Your task is to extract the CORE METADATA of this case. Think of this as the "case header" information.

You must identify, from the judgment text itself, the following items:

1. File name or case title as it appears in the judgment
2. Parties (Appellant/Respondent or Plaintiff/Defendant, etc.)
3. Court level (e.g., Supreme Court of Kenya, Court of Appeal, High Court, Magistrates Court, Environment and Land Court, Employment and Labour Relations Court, etc.)
4. Judge(s) who wrote or delivered the judgment
5. Year of the judgment (from the judgment date, not necessarily the year of filing)
6. Legal domain (the area of law this case falls under, e.g., Criminal Law, Civil Law, Constitutional Law, Family Law, Commercial Law, Employment Law, Land Law, Tort Law, Contract Law, Property Law, Administrative Law, etc.)
7. Winning party (who won the case: either "plaintiff" or "defendant". For criminal cases, if the accused is convicted, the prosecution wins; if acquitted, the accused wins. For appeals, determine who prevailed: if appeal is dismissed, the respondent wins; if allowed, the appellant wins.)

When reading the document, pay special attention to:
- The cover page / first lines
- The heading block (often in ALL CAPS, e.g. "IN THE HIGH COURT OF KENYA AT NAIROBI")
- The line with the judge's name (often ending with "J.", "JA", "JJ.A", "SCJ", etc.)
- The date line (e.g., "Dated and delivered at Nairobi this 18th day of October 2002")
- The nature of the case, charges, claims, or issues to determine the legal domain
- The final orders, judgment outcome, and conclusion to determine who won (plaintiff or defendant)

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (PLAIN TEXT)
═══════════════════════════════════════════════════════════════

Return the result as a plain text block with EXACTLY these seven lines
(using these labels in ALL CAPS, followed by a colon):

FILE NAME: ...
PARTIES: ...
COURT LEVEL: ...
JUDGE: ...
YEAR: ...
LEGAL DOMAIN: ...
WINNING PARTY: ...

Examples:
FILE NAME: Julius Muriithi & 11 others v Stephen Musyoka Ivai [2020] eKLR
PARTIES: Julius Muriithi & 11 others v Stephen Musyoka Ivai
COURT LEVEL: High Court of Kenya (at Nairobi)
JUDGE: Lesiit J.
YEAR: 2020
LEGAL DOMAIN: Civil Law
WINNING PARTY: plaintiff

FILE NAME: Wilson Wanjala Mkendeshwo v Republic [2002] KECA 166
PARTIES: Wilson Wanjala Mkendeshwo v Republic
COURT LEVEL: Court of Appeal of Kenya
JUDGE: [Judge names]
YEAR: 2002
LEGAL DOMAIN: Criminal Law
WINNING PARTY: defendant

If you are unsure about any field, make your best reasonable inference from the text
and optionally note the inference in parentheses, e.g.
COURT LEVEL: High Court of Kenya (inferred from heading "IN THE HIGH COURT OF KENYA AT NAIROBI").
LEGAL DOMAIN: Criminal Law (inferred from charge of murder under Penal Code).
WINNING PARTY: plaintiff (inferred from judgment entered for the plaintiff and appeal dismissed).

Do NOT include any other commentary before or after these seven lines.

═══════════════════════════════════════════════════════════════
JUDGMENT TO ANALYZE
═══════════════════════════════════════════════════════════════

{document_text}

═══════════════════════════════════════════════════════════════
BEGIN METADATA LINES ONLY
═══════════════════════════════════════════════════════════════
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
RULE_EXTRACTION_PROMPT = """You are a legal document analyst specializing in Kenyan court judgments.
Your job is to extract the binding LEGAL RULE(S) (the "R" in IRAC) from the judgment.

Think carefully about each legal statement.
Ask yourself for every candidate rule:
- Was this principle NECESSARY for the court to reach its final decision on the issues?
- If I removed this principle, would the outcome logically change?

TASK: Identify and clearly state the LEGAL RULE(S) (Ratio Decidendi), not mere commentary or dicta.

═══════════════════════════════════════════════════════════════
STEP 1: UNDERSTAND WHAT "RULE" MEANS
═══════════════════════════════════════════════════════════════

For purposes of this task, the RULE is:

- The binding legal principle (Ratio Decidendi) that the court APPLIES to resolve a specific issue
- The legal test, elements, or standard the court uses to decide the case
- In statutory cases: the court's authoritative INTERPRETATION of ambiguous statutory language

The Rule is NOT:
- A restatement of raw facts
- A general comment on justice, fairness, or policy
- Background description of the law that is not actually applied
- The final outcome itself (that belongs to CONCLUSION)

═══════════════════════════════════════════════════════════════
STEP 2: WHERE TO FIND THE RULE IN KENYAN JUDGMENTS
═══════════════════════════════════════════════════════════════

Look in the legal analysis sections where the judge:

✓ States "the law is that...", "the correct position in law is..."
✓ Sets out ELEMENTS or TESTS (e.g., the elements of murder, negligence, adverse possession)
✓ Summarises case law and then adopts a PRINCIPLE from it
✓ Interprets statutory provisions and explains what certain phrases MEAN in practice
✓ Connects earlier authority with the current case, often just before applying it

Common signals:
- "The applicable law is..."
- "The principles to be applied are..."
- "In the case of X v Y, the court held that..."
- "From the foregoing authorities, the test is as follows..."
- "Section [X] of the [Act] provides that..."

Focus on the MAJORITY reasoning or the single judgment that decides the case.
Ignore concurring or dissenting opinions unless the judgment expressly adopts them as the court's reasoning.

═══════════════════════════════════════════════════════════════
STEP 3: PAIN POINT 1 – DISTINGUISH HOLDING FROM DICTA
═══════════════════════════════════════════════════════════════

Dicta includes:
- Hypothetical scenarios ("If the facts had been different...")
- General comments not necessary to the decision
- Broad statements about public policy that are not tied to the issues

To test if a legal statement is part of the RULE:
- Ask: Was this legal principle NECESSARY for deciding the issues on the actual facts?
- If removing that statement would NOT change the result, it is likely dicta, not the Rule.

ONLY include the principle(s) that MUST have been true for the court's conclusion to make sense.

═══════════════════════════════════════════════════════════════
STEP 4: PAIN POINT 2 – SCOPE OF THE RULE
═══════════════════════════════════════════════════════════════

The Rule must:
- Be general enough to apply to future similar cases
- Be specific enough to be anchored in the MATERIAL FACTS of this case

Avoid:
- Rules that are TOO NARROW (e.g., tied to trivial or unique facts that do not matter legally)
- Rules that are TOO BROAD (e.g., "All contracts must be honoured" without linking to the particular legal issue)

When stating the Rule:
- Capture the general legal test
- If necessary, mention the KEY material conditions or limits (e.g., "where identification is by a single witness at night...")

═══════════════════════════════════════════════════════════════
STEP 5: PAIN POINT 3 – MAJORITY VS CONCURRING / DISSENTING
═══════════════════════════════════════════════════════════════

ONLY treat as Rule:
- The reasoning adopted by the court that actually forms the DECISION (majority or single-judge decision)

DO NOT treat as Rule:
- Concurring opinions that suggest a different route to the same outcome
- Dissenting opinions that disagree with the outcome

If multiple judges write, focus on:
- "The judgment of the court is as follows..."
- The opinion that clearly commands the majority and provides the operative reasoning.

═══════════════════════════════════════════════════════════════
STEP 6: PAIN POINT 4 – SYNTHESIZING COMPLEX RULES
═══════════════════════════════════════════════════════════════

In many cases, the Rule may be a MULTI-ELEMENT TEST, e.g.:
- Elements of an offence (actus reus + mens rea)
- Elements of negligence, contractual validity, etc.
- Conditions for granting an injunction, judicial review, etc.

The court may:
- Summarise principles from several older cases
- Then restate them as a combined test

Your job is to:
- Pull out the FINAL test or set of elements the court adopts
- State it clearly as a numbered or bulleted list of requirements, if appropriate

═══════════════════════════════════════════════════════════════
STEP 7: PAIN POINT 5 – STATUTORY INTERPRETATION CASES
═══════════════════════════════════════════════════════════════

Where the case turns on a statute, regulation, or constitutional provision:

The Rule is NOT:
- A copy-paste of the statutory text alone

The Rule IS:
- The court's binding INTERPRETATION of ambiguous words or provisions
- Any TEST the court creates to decide when the statutory provision applies
- Clarification of legislative intent that is applied to the facts

Look for:
- "The proper interpretation of section X is that..."
- "These words must be understood to mean..."
- "For a person to fall under this section, the following must be shown..."

═══════════════════════════════════════════════════════════════
STEP 8: WHAT TO EXCLUDE
═══════════════════════════════════════════════════════════════

DO NOT include:
✗ Pure factual narration
✗ Procedural history
✗ Mere summary of other cases without stating the principle adopted
✗ Policy discussions that are not tied to the outcome
✗ Final outcome or orders (that belongs to CONCLUSION)
✗ Rhetorical statements not used as a legal test

═══════════════════════════════════════════════════════════════
STEP 9: OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Present the rules in a structure that aligns with the issues:

═══════════════════════════════════════════════════════════════
SECTION 3: RULE(S)
═══════════════════════════════════════════════════════════════

**CORE LEGAL RULES BY ISSUE**

1. Issue 1: [Restate the issue in question form]

   **Core Rule / Principle (Ratio Decidendi):**
   - [Clear statement of the binding legal principle the court applied to this issue]

   **Source / Authority:**
   - [Key statute sections, constitutional provisions, or cases relied on]

   **Scope and Conditions:**
   - [Any important qualifications, material fact conditions, or limits on the Rule]

2. Issue 2: [Restate the issue]

   **Core Rule / Principle (Ratio Decidendi):**
   - [...]

   **Source / Authority:**
   - [...]

   **Scope and Conditions:**
   - [...]

3. [Continue for all main issues]

**STATUTORY INTERPRETATION RULES (if applicable)**

- [Summarise any specific interpretation of statutory text adopted by the court]

═══════════════════════════════════════════════════════════════
IMPORTANT REMINDERS
═══════════════════════════════════════════════════════════════

1. Focus on LEGALLY NECESSARY principles – not background commentary.
2. State the Rule at an appropriate level of generality (not too broad, not too narrow).
3. Separate RULE (legal test or principle) from APPLICATION (how it is applied to facts).
4. Tie each Rule to a specific Issue wherever possible.
5. Ignore concurring and dissenting reasoning unless expressly adopted by the court.

═══════════════════════════════════════════════════════════════
JUDGMENT TO ANALYZE
═══════════════════════════════════════════════════════════════

{document_text}

═══════════════════════════════════════════════════════════════
EXTRACTED RULE(S)
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

    1. Do NOT repeat the entire facts section - focus on reasoning.
    2. Make the connection between law and facts explicit.
    3. Keep the structure ISSUE → RULE(S) APPLIED → REASONING.
    4. Be faithful to the judgment - do not invent reasoning.
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

    ✗ Detailed reasoning - that belongs in APPLICATION/ANALYSIS.
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

    1. Be concise -conclusions should be short and direct.
    2. Ensure every MAIN ISSUE from the issues section has a corresponding holding.
    3. Make the overall outcome unambiguous.
    4. Include all material final orders affecting the parties.
    5. Do NOT add your own opinions - just restate the court's holdings.

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
- **RULES (governing legal principles)**
- APPLICATION/ANALYSIS (court's reasoning)
- CONCLUSION/HOLDING (final answers and orders)

CRITICAL: You MUST provide FIVE separate sections in your response:
1. FACTS section
2. ISSUES section
**2A. RULES section**
3. APPLICATION / ANALYSIS section
4. CONCLUSION / HOLDING section

═══════════════════════════════════════════════════════════════
PART A: FACTS EXTRACTION (I. FACTS)
═══════════════════════════════════════════════════════════════

**GUIDING PRINCIPLE: What Happened?** Extract ALL factual content – the events, testimony, evidence, and what parties claim happened. This is the neutral narrative.

INCLUDE:
✓ Background of parties and relationships
✓ Chronological events (what happened, when, where)
✓ Witness testimonies (attributed clearly: "PW1 testified that...")
✓ Physical/forensic evidence
✓ Prosecution/Plaintiff's case theory
✓ Defence/Defendant's version of events
✓ Procedural facts (arrest, investigation, lower court proceedings)

EXCLUDE:
✗ **Court's Opinion/Legal Reasoning (Rule/Application):** Do not include "The court finds that...", "The law holds that...", or any legal analysis.
✗ Legal principles or citations.
✗ Final orders or judgment.

═══════════════════════════════════════════════════════════════
PART B: ISSUES EXTRACTION (I. ISSUES)
═══════════════════════════════════════════════════════════════

**GUIDING PRINCIPLE: What Legal Question Must Be Answered?** Identify the KEY LEGAL QUESTIONS the court needed to answer to resolve the dispute.

INCLUDE:
✓ Explicit statements: "The issues for determination are..."
✓ Grounds of appeal (converted to questions)
✓ Each element of the charge or claim being contested.

Frame each issue as a question starting with "Whether..."

EXCLUDE:
✗ The court's answer to the question (that belongs in CONCLUSION).
✗ Statements of the law/legal principles (that belongs in RULES).

═══════════════════════════════════════════════════════════════
**PART C: RULES EXTRACTION (R. RULES)**
═══════════════════════════════════════════════════════════════

**GUIDING PRINCIPLE: What is the BINDING LAW (Ratio Decidendi)?** Identify and extract the authoritative legal standard (test, statute, common law principle) the court relied upon for each Issue.

**CRITICAL GUIDANCE (Addressing Pain Points):**
1.  **Holding vs. Dicta:** Extract only the legal principle **necessary** to resolve the issue; EXCLUDE extraneous, non-binding comments (Obiter Dictum).
2.  **Case Fragmentation:** ONLY use the rules explicitly adopted and relied upon in the **Majority Opinion**. Ignore principles from Concurring or Dissenting opinions.
3.  **Synthesize Elements:** If the Rule is a multi-part legal test (e.g., elements of negligence), the extracted Rule MUST be a complete list of those elements/requirements.
4.  **Scope:** The Rule must be stated as a **general principle of law**, not tied to the specific names/facts of the current case (that link is made in Application).

WHAT TO EXTRACT:
✓ Specific Statutes/Sections cited.
✓ Common Law legal tests (elements).
✓ Key precedent cases cited and the generalized principle derived from them.

EXCLUDE:
✗ Discussion of how the law *applies* to the facts (that belongs in APPLICATION).
✗ The court's final judgment.

═══════════════════════════════════════════════════════════════
PART D: APPLICATION / ANALYSIS EXTRACTION (A. APPLICATION)
═══════════════════════════════════════════════════════════════

**GUIDING PRINCIPLE: How Does the Law Apply to the Facts?** Extract the court's reasoning showing HOW it applied the **extracted RULES** to the established facts.

INCLUDE:
✓ Evaluation of evidence and credibility (e.g., "PW1's testimony was contradictory, therefore the identification Rule is not met...").
✓ The logical steps that move from issues → **rules** → facts → intermediate findings (e.g., "Applying the three-part test for fraud, the element of reliance is met because...").
✓ Discussion of *why* the court accepted or rejected each party's arguments based on the law.

EXCLUDE:
✗ **Bare Rules:** Do not repeat the full legal principles from the RULES section without tying them directly to the facts.
✗ Pure restatement of facts without analysis.
✗ Final orders or outcome.

═══════════════════════════════════════════════════════════════
PART E: CONCLUSION / HOLDING EXTRACTION (C. CONCLUSION)
═══════════════════════════════════════════════════════════════

**GUIDING PRINCIPLE: What is the Final Answer and Order?** Extract how the court ANSWERED each main issue and the FINAL OUTCOME.

INCLUDE:
✓ For each main issue: a concise, clear answer (Yes/No or equivalent).
✓ Overall case result (appeal allowed/dismissed; suit succeeds/fails; conviction upheld/set aside).
✓ Final orders (sentence, damages, costs, declarations, etc.).

EXCLUDE:
✗ Detailed reasoning (belongs in APPLICATION/ANALYSIS).
✗ Long quotations or repetition of facts.

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
**SECTION 2A: RULES**
═════════════════════════════════════════════════════════════════════

**GOVERNING LEGAL PRINCIPLES (Ratio Decidendi)**

1. Rule for Issue 1: [Restate the issue for context]
   - **Source:** [Specific Statute/Section or Case Name (e.g., *Rex v Kipkering*)]
   - **Principle/Elements:** [The binding legal test or element list for this issue, stated generally]

2. Rule for Issue 2: [Restate the issue]
   - **Source:** [...]
   - **Principle/Elements:** [...]

3. [Continue for all main issues]

═════════════════════════════════════════════════════════════════════
SECTION 3: APPLICATION / ANALYSIS
═════════════════════════════════════════════════════════════════════

**ISSUE-BY-ISSUE APPLICATION/ANALYSIS**

1. Issue 1: [Restate Issue 1]

**Rules Referenced:**
- [List the rules from SECTION 2A used for this issue]

**Court's Application / Reasoning:**
- [How the court applied those general rules to the established facts, evaluating evidence/credibility against each rule element]

2. Issue 2: [Restate Issue 2]

**Rules Referenced:**
- [...]

**Court's Application / Reasoning:**
- [...]

3. [Continue for all issues identified in Section 2]

═════════════════════════════════════════════════════════════════════
SECTION 4: CONCLUSION / HOLDING
═════════════════════════════════════════════════════════════════════

**HOLDING ON EACH MAIN ISSUE**

1. [Issue 1 as a question]
- [Court's short answer (e.g., "Yes, the prosecution proved malice aforethought")]

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
- Keep FACTS, ISSUES, **RULES**, APPLICATION/ANALYSIS, and CONCLUSION/HOLDING clearly separated.
- **The RULES section (2A) MUST contain the binding legal principles (Ratio Decidendi) only, stated generally.**
- The APPLICATION section (3) MUST show the link between the RULES (2A) and the FACTS (1).

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

def extract_metadata_only(document_text: str) -> str:
    """
    Use Groq LLM to extract ONLY the core metadata of the case:
    - file name / case title
    - parties
    - court level
    - judge
    - year
    - legal domain of the case
    - winning party (plaintiff or defendant)

    Returns:
        str: A plain-text block with 7 lines:
             FILE NAME: ...
             PARTIES: ...
             COURT LEVEL: ...
             JUDGE: ...
             YEAR: ...
             LEGAL DOMAIN: ...
             WINNING PARTY: ...
             If extraction fails, returns an error message string.
    """
    print("\n===== EXTRACTING METADATA ONLY =====")
    
    try:
        prompt = METADATA_EXTRACTION_PROMPT.format(document_text=document_text)
        system_message = (
            "You are a legal document analyst specializing in Kenyan court judgments. "
            "Extract ONLY the core case metadata and return it as SEVEN plain-text lines: "
            "FILE NAME, PARTIES, COURT LEVEL, JUDGE, YEAR, LEGAL DOMAIN, WINNING PARTY."
        )
        
        raw_metadata = call_groq_api(prompt, system_message, max_tokens=800)
        
        print("✓ Metadata extraction completed")
        print(f"Extracted content length: {len(raw_metadata)} characters")
        print("Metadata preview:")
        print(raw_metadata[:500])
        
        # Just return the string as-is (frontend will render it)
        return raw_metadata.strip()
        
    except Exception as e:
        error_msg = f"Error extracting metadata: {str(e)}"
        print(error_msg)
        # Still return a string, not a dict
        return error_msg


def extract_all_firac_combined(document_text: str) -> dict:
    """
    Use Groq LLM to extract ALL FIRAC components (including RULES)
    AND METADATA in a single call.

    Returns a dictionary with all components as plain strings.
    """
    print("\n===== EXTRACTING ALL FIRAC COMPONENTS (COMBINED) =====")
    print("Connecting to Groq API...")
    
    try:
        prompt = f"""You are a legal document analyst specializing in Kenyan court judgments.

Extract ALL key components from this judgment and present them in clearly separated sections:

SECTION 0: METADATA
[Extract the core case metadata as seven lines:
 FILE NAME: ...
 PARTIES: ...
 COURT LEVEL: ...
 JUDGE: ...
 YEAR: ...
 LEGAL DOMAIN: ...
 WINNING PARTY: ... (either "plaintiff" or "defendant" - analyze the final judgment outcome to determine who won)]

SECTION 1: FACTS
[Extract the factual background, events, and circumstances of the case]

SECTION 2: ISSUES
[Extract the key legal questions the court had to decide]

SECTION 3: RULES
[Extract the binding legal rules / principles / tests (ratio decidendi) the court applied to resolve the issues. 
Focus on legal principles that were necessary for the decision, not dicta or background commentary.]

SECTION 4: APPLICATION
[Extract how the court applied those legal rules to the established facts, including evaluation of evidence and reasoning.]

SECTION 5: CONCLUSION
[Extract the court's final ruling, answers to the issues, final outcome, and any orders or directions.]

Here is the court judgment:

{document_text}

Remember to clearly label each section exactly as:
- SECTION 0: METADATA
- SECTION 1: FACTS
- SECTION 2: ISSUES
- SECTION 3: RULES
- SECTION 4: APPLICATION
- SECTION 5: CONCLUSION

Be comprehensive and keep each section clearly separated.
"""
        
        print("Sending combined extraction request to Groq LLM...")
        print(f"Document length: {len(document_text)} characters")
        
        system_message = (
            "You are a legal document analyst specializing in extracting structured FIRAC information "
            "and core metadata from Kenyan court judgments. You must provide clearly separated sections "
            "for METADATA, FACTS, ISSUES, RULES, APPLICATION, and CONCLUSION using the exact section headings."
        )
        
        extracted_content = call_groq_api(prompt, system_message, max_tokens=10000)
        
        print("✓ Combined extraction completed")
        print(f"Total extracted content length: {len(extracted_content)} characters")
        
        # Parse the response to separate all sections
        result = {
            'full_response': extracted_content,
            'metadata': '',
            'facts': '',
            'issues': '',
            'rules': '',
            'application': '',
            'conclusion': ''
        }
        
        sections_found = 0

        # METADATA
        if "SECTION 0: METADATA" in extracted_content and "SECTION 1: FACTS" in extracted_content:
            metadata_section = extracted_content.split("SECTION 1: FACTS")[0]
            metadata_section = metadata_section.replace("SECTION 0: METADATA", "").strip()
            result['metadata'] = metadata_section
            sections_found += 1

        # FACTS
        if "SECTION 1: FACTS" in extracted_content and "SECTION 2: ISSUES" in extracted_content:
            facts_section = extracted_content.split("SECTION 2: ISSUES")[0]
            facts_section = facts_section.replace("SECTION 1: FACTS", "").strip()
            result['facts'] = facts_section
            sections_found += 1

        # ISSUES
        if "SECTION 2: ISSUES" in extracted_content and "SECTION 3: RULES" in extracted_content:
            issues_section = extracted_content.split("SECTION 3: RULES")[0].split("SECTION 2: ISSUES", 1)[1]
            result['issues'] = issues_section.strip()
            sections_found += 1

        # RULES
        if "SECTION 3: RULES" in extracted_content and "SECTION 4: APPLICATION" in extracted_content:
            rules_section = extracted_content.split("SECTION 4: APPLICATION")[0].split("SECTION 3: RULES", 1)[1]
            result['rules'] = rules_section.strip()
            sections_found += 1

        # APPLICATION
        if "SECTION 4: APPLICATION" in extracted_content and "SECTION 5: CONCLUSION" in extracted_content:
            application_section = extracted_content.split("SECTION 5: CONCLUSION")[0].split("SECTION 4: APPLICATION", 1)[1]
            result['application'] = application_section.strip()
            sections_found += 1

        # CONCLUSION
        if "SECTION 5: CONCLUSION" in extracted_content:
            conclusion_section = extracted_content.split("SECTION 5: CONCLUSION", 1)[1]
            result['conclusion'] = conclusion_section.strip()
            sections_found += 1
        
        if sections_found == 6:
            print(f"✓ Successfully separated all sections:")
            print(f"   • Metadata: {len(result['metadata'])} chars")
            print(f"   • Facts: {len(result['facts'])} chars")
            print(f"   • Issues: {len(result['issues'])} chars")
            print(f"   • Rules: {len(result['rules'])} chars")
            print(f"   • Application: {len(result['application'])} chars")
            print(f"   • Conclusion: {len(result['conclusion'])} chars")
        else:
            print(f"⚠ Only found {sections_found}/6 sections. Partial extraction.")
        
        return result
        
    except Exception as e:
        error_msg = f"Error in combined extraction: {str(e)}"
        print(error_msg)
        return {
            'full_response': error_msg,
            'metadata': error_msg,
            'facts': error_msg,
            'issues': error_msg,
            'rules': error_msg,
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

def extract_rules_only(document_text: str) -> str:
    """
    Use Groq LLM to extract ONLY the binding legal rule(s) (Ratio Decidendi)
    from the court judgment.

    This function focuses on identifying the core legal principles, tests,
    or standards that the court applied to resolve the issues, distinguishing
    them from dicta, policy commentary, and mere factual narration.

    Args:
        document_text (str): The full text of the court judgment

    Returns:
        str: Extracted rule(s) section
    """
    print("\n===== EXTRACTING RULE(S) ONLY =====")
    
    try:
        prompt = RULE_EXTRACTION_PROMPT.format(document_text=document_text)
        system_message = (
            "You are a legal document analyst specializing in extracting binding legal rules "
            "(ratio decidendi) from Kenyan court judgments. Identify the core legal principles, "
            "tests, or standards the court applied to resolve the issues, distinguishing them "
            "from dicta and general commentary."
        )
        
        extracted_rules = call_groq_api(prompt, system_message, max_tokens=4000)
        
        print("✓ Rule(s) extraction completed")
        print(f"Extracted content length: {len(extracted_rules)} characters")
        
        return extracted_rules
        
    except Exception as e:
        error_msg = f"Error extracting rule(s): {str(e)}"
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
        prompt = APPLICATION_ANALYSIS_EXTRACTION_PROMPT.format(document_text=document_text)
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


# def run_firac_individual() -> dict:
#     """
#     Run FIRAC extraction with INDIVIDUAL API calls for each component.
#     This makes 4 separate API calls but allows for more focused extraction.
    
#     Returns:
#         dict: {
#             'document': str,           # Original cleaned document text
#             'facts': str,              # Extracted facts
#             'issues': str,             # Extracted issues
#             'application': str,        # Extracted application/analysis
#             'conclusion': str,         # Extracted conclusion
#             'error': str or None       # Error message if any
#         }
#     """
#     print("\n" + "=" * 80)
#     print("🏛️  STARTING FIRAC EXTRACTION (INDIVIDUAL API CALLS)")
#     print(f"🤖 Using Groq API with model: {MODEL_NAME}")
#     print("=" * 80)
    
#     result = {
#         'document': '',
#         'facts': '',
#         'issues': '',
#         'application': '',
#         'conclusion': '',
#         'error': None
#     }
    
#     try:
#         # Step 1: Read PDF
#         print("\n[STEP 1/5] Reading PDF document...")
#         document_text = load_wilson_document()
#         result['document'] = document_text
        
#         # Step 2: Extract facts
#         print("\n[STEP 2/5] Extracting facts...")
#         result['facts'] = extract_facts_only(document_text)
        
#         # Step 3: Extract issues
#         print("\n[STEP 3/5] Extracting issues...")
#         result['issues'] = extract_issues_only(document_text)
        
#         # Step 4: Extract application/analysis
#         print("\n[STEP 4/5] Extracting application and analysis...")
#         result['application'] = extract_application_only(document_text)
        
#         # Step 5: Extract conclusion
#         print("\n[STEP 5/5] Extracting conclusion...")
#         result['conclusion'] = extract_conclusion_only(document_text)
        
#         print("\n" + "=" * 80)
#         print("✅ FIRAC EXTRACTION FINISHED SUCCESSFULLY (INDIVIDUAL CALLS)")
#         print("=" * 80)
#         print(f"\n📊 Extraction Summary:")
#         print(f"   • Facts: {len(result['facts'])} characters")
#         print(f"   • Issues: {len(result['issues'])} characters")
#         print(f"   • Application: {len(result['application'])} characters")
#         print(f"   • Conclusion: {len(result['conclusion'])} characters")
        
#         return result
        
#     except FileNotFoundError as e:
#         error_msg = f"PDF file not found: {str(e)}"
#         print(f"\n❌ ERROR: {error_msg}")
#         result['error'] = error_msg
#         return result
        
#     except Exception as e:
#         error_msg = f"Extraction failed: {str(e)}"
#         print(f"\n❌ ERROR: {error_msg}")
#         result['error'] = error_msg
#         return result

def run_firac() -> dict:
    """
    Run FIRAC extraction with a SINGLE combined API call (for FIRAC components),
    plus a separate metadata extraction.

    Returns:
        dict: {
            'document': str,
            'metadata': str,          # Extracted metadata (file_name, parties, court_level, judge, year, legal_domain, winning_party)
            'facts': str,
            'issues': str,
            'rules': str,
            'application': str,
            'conclusion': str,
            'facts_metadata': str,    # Metadata attached to facts section
            'issues_metadata': str,   # Metadata attached to issues section
            'rules_metadata': str,    # Metadata attached to rules section
            'application_metadata': str,  # Metadata attached to application section
            'conclusion_metadata': str,    # Metadata attached to conclusion section
            'full_response': str,
            'error': str or None
        }
    """
    print("\n" + "=" * 80)
    print("🏛️  STARTING FIRAC + METADATA EXTRACTION")
    print(f"🤖 Using Groq API with model: {MODEL_NAME}")
    print("=" * 80)

    result = {
        'document': '',
        'metadata': {},
        'facts': '',
        'issues': '',
        'rules': '',
        'application': '',
        'conclusion': '',
        'full_response': '',
        'error': None
    }

    try:
        # Step 1: Read PDF
        print("\n[STEP 1/3] Reading PDF document...")
        document_text = load_wilson_document()
        result['document'] = document_text

        # Step 2: Extract all FIRAC components in one call
        print("\n[STEP 2/3] Extracting all FIRAC components in single API call...")
        extraction_result = extract_all_firac_combined(document_text)

        result['facts'] = extraction_result.get('facts', '')
        result['issues'] = extraction_result.get('issues', '')
        result['rules'] = extraction_result.get('rules', '')
        result['application'] = extraction_result.get('application', '')
        result['conclusion'] = extraction_result.get('conclusion', '')
        result['full_response'] = extraction_result.get('full_response', '')

        # Step 3: Extract metadata
        print("\n[STEP 3/3] Extracting metadata...")
        metadata_text = extract_metadata_only(document_text)
        result['metadata'] = metadata_text
        
        # Step 4: Attach metadata to each FIRAC section
        print("\n[STEP 4/4] Attaching metadata to each FIRAC section...")
        result['facts_metadata'] = metadata_text
        result['issues_metadata'] = metadata_text
        result['rules_metadata'] = metadata_text
        result['application_metadata'] = metadata_text
        result['conclusion_metadata'] = metadata_text
        
        print("\n" + "=" * 80)
        print("✅ FIRAC + METADATA EXTRACTION FINISHED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n📊 Extraction Summary:")
        print(f"   • Metadata: {len(metadata_text)} characters")
        print(f"   • Facts: {len(result['facts'])} characters")
        print(f"   • Issues: {len(result['issues'])} characters")
        print(f"   • Rules: {len(result['rules'])} characters")
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
    print (result)
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