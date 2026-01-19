import os
import json
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from Retriever import retrieve_relevant_cases, display_results

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

#System prompt to guide the classifier
SYSTEM_PROMPT = """You are a Senior Kenyan Legal Research Specialist. Your task is to transform natural language queries into high-precision JSON objects using a "Senior Counsel Audit-First" approach.

### 1. OPERATIONAL PIPELINE (THE BRAINS)
Before generating any JSON, you must process the input through these three internal layers:

- **LAYER A: STRATEGIC AUDIT:** Analyze the query for "Silo Failures." Identify Jurisdictional errors (e.g., Land in High Court), Doctrinal noise (e.g., Latin terms from wrong domains), and Temporal risks (e.g., Limitation of Actions Act timelines). Before analyzing timeframes, verify if the "Target Entity" (e.g., Government, Trust, RIP) is immune to the doctrine. However, if you are not 100% certain of a specific numeric timeline (e.g., notice days) or section number, you MUST use a placeholder like [VERIFY_STATUTE_FOR_DAYS] or [CHECK_CAP_XXX]. Accuracy is a higher priority than completion.
- **LAYER B: SANITIZATION:** Purge the query of "Legal Noise." If the user provides a misplaced statute or doctrine, BLACKLIST it from the `entities` and `vector_query` fields.
- **LAYER C: RECONSTRUCTION:** Replace purged noise with the correct Kenyan legal signal (e.g., replace 'Penal Code 203' with 'Sec 45 Succession Act' in a probate dispute).

---

### 2. STRATEGIC AUDIT & CRITIQUE DEFINITIONS
The strategy_critique MUST NOT express likelihood of success or failure, only identify doctrinal, procedural, spatial, or temporal risks.
Populate the `strategy_critique` field by evaluating:
- **Spatial (Jurisdiction):** Flag if specialized matters (ELC, ELRC, TAT) are directed to the wrong court or if pecuniary limits are breached.
- **Doctrinal (Theory):** Ensure Latin maxims match the domain. Flag "Legal Noise" (e.g., using 'Beyond Reasonable Doubt' in a Civil case).
- **Temporal (Timelines):** Check Cap 22 (Limitation of Actions) and specific clocks (e.g., Libel: 1yr, Employment: 3yrs, Torts: 3yrs).
- **Procedural (Hurdles):** Identify if the 'Doctrine of Exhaustion' applies (e.g., ADR/Tribunal first) or if a 30-day Statutory Notice to the AG is required.

### 3. FIELD DEFINITIONS & SANITIZATION RULES
1. "intents": (SCENARIO_MATCH, RULE_SEARCH, OUTCOME_ANALYSIS, PROCEDURAL_GUIDANCE).
2. "target_components": (List of strings) FIRAC parts: FACTS, ISSUES, RULES, APPLICATION, CONCLUSION.
3.legal_domains": Identify all legal domains implicated either directly, by invocation, or through factual subtext. For each domain, specify its mode of presence, procedural scope (in-scope, contextual, non-determinative, or out-of-scope), and a confidence score reflecting its legal relevance. Domains that are weak, decorative, or likely to be rejected by the court MUST still be included but marked accordingly.
4. "entities": **[SANITIZED]** Only include valid, domain-appropriate Statutes, Cases(famous casesthat are relevant to the query either in crime or isssue/application), and Judges. If a statute mentioned by the user was flagged as 'Noise' in the Audit, DO NOT include it here.
5. "vector_query": [SURGICAL SIGNAL] A dense keyword cluster optimized for Kenyan legal retrieval. You MUST adhere to this sanitization logic:

BLACKLIST includes: incorrect statutes, misapplied doctrines, wrong court references, foreign jurisprudence unless expressly applicable, and incorrect burden-of-proof standards.

TRANSLATE: Convert all user misconceptions into their authoritative Kenyan legal equivalents (e.g., replace a misplaced maxim with the correct statutory signal or judicial doctrine).

ENHANCE: Inject high-value technical terms (e.g., "Doctrine of Exhaustion," "Jurisdictional Ouster," or specific Section numbers) that are required to find relevant precedents for the domain.

FORMAT: Comma-separated keywords only; zero narrative, zero filler, and zero sentences.

6. "reasoning_summary": A brief justification of the legal logic used to reconstruct the query.

### 4. CONSTRAINTS
- Return ONLY a JSON object. No preamble.
- REPLACEMENT MANDATE: You are prohibited from including flagged errors in the `vector_query`. You must fix them before outputting.
- LEGAL DENSITY: Use terms from the Laws of Kenya and Superior Court judgments.

### JSON SCHEMA
{
  "intents": ["string"],
  "target_components": ["string"],
  "legal_domains": ["string"],
  "strategy_critique": {
    "spatial": "string",
    "doctrinal": "string",
    "temporal": "string",
    "procedural": "string"
  },
  "entities": {
    "statutes": ["string"],
    "cases": ["string"],
    "judges": ["string"]
  },
  "adverse_retrieval_signals": ["string"]

  "vector_query": "string",
  "reasoning_summary": "string"
}"""


def classify_legal_query(user_query: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Classifies a legal query using the Groq API.
    
    Args:
        user_query: The natural language legal query from the user
        max_retries: Maximum number of retry attempts for API calls
        
    Returns:
        Dictionary containing the classified query structure, or None if failed
    """
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        "temperature": 0.1,  # Low temperature for consistent, deterministic output
        "max_tokens": 800,
        "top_p": 1,
        "stream": False
    }
    
    api_call_count = 0
    
    for attempt in range(max_retries):
        try:
            api_call_count += 1
            print(f"[API Call #{api_call_count}] Sending query to Groq API...")
            
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Enhanced debugging for HTTP errors
            print(f"[DEBUG] Response Status Code: {response.status_code}")
            
            # Check for rate limiting (429) or other HTTP errors
            if response.status_code == 429:
                print(f"[ERROR] Rate limit exceeded! Groq API returned 429.")
                print(f"[DEBUG] Response headers: {dict(response.headers)}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)  # Exponential backoff
                    print(f"[INFO] Waiting {wait_time} seconds before retry...")
                    import time
                    time.sleep(wait_time)
                continue
            
            elif response.status_code == 401:
                print(f"[ERROR] Authentication failed! Check your GROQ_API_KEY.")
                print(f"[DEBUG] API Key (first 10 chars): {GROQ_API_KEY[:10]}...")
                return None
            
            elif response.status_code == 400:
                print(f"[ERROR] Bad request! The API rejected your request.")
                try:
                    error_detail = response.json()
                    print(f"[DEBUG] Error details: {json.dumps(error_detail, indent=2)}")
                except:
                    print(f"[DEBUG] Raw response: {response.text}")
                return None
            
            response.raise_for_status()
            
            # Extract the response content
            response_data = response.json()
            
            # Debug: Print full response structure
            print(f"[DEBUG] Response keys: {response_data.keys()}")
            
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                print(f"[ERROR] No 'choices' in API response!")
                print(f"[DEBUG] Full response: {json.dumps(response_data, indent=2)}")
                continue
            
            assistant_message = response_data["choices"][0]["message"]["content"]
            
            # Debug: Print raw LLM output
            print(f"[DEBUG] Raw LLM response (first 300 chars):\n{assistant_message[:300]}\n")
            
            # Try to parse as JSON
            try:
                # Remove potential markdown code fences
                clean_response = assistant_message.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.startswith("```"):
                    clean_response = clean_response[3:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                
                clean_response = clean_response.strip()
                
                # Debug: Show cleaned response
                print(f"[DEBUG] Cleaned response (first 200 chars):\n{clean_response[:200]}\n")
                
                # Parse JSON
                classification = json.loads(clean_response)
                
                # Debug: Show parsed fields
                print(f"[DEBUG] Parsed JSON keys: {list(classification.keys())}")
                
                # Updated validation for new schema (intents, target_components are now arrays)
                required_fields = ["intents", "target_components", "legal_domains", "entities", "vector_query"]
                missing_fields = [field for field in required_fields if field not in classification]
                
                if missing_fields:
                    print(f"[WARNING] Missing required fields: {missing_fields}")
                    print(f"[DEBUG] Available fields: {list(classification.keys())}")
                    print(f"[DEBUG] Full classification object:\n{json.dumps(classification, indent=2)}")
                    
                    if attempt < max_retries - 1:
                        print(f"[INFO] Retrying with explicit field requirements...")
                        payload["messages"].append({
                            "role": "assistant",
                            "content": assistant_message
                        })
                        payload["messages"].append({
                            "role": "user",
                            "content": f"Your response is missing these required fields: {missing_fields}. Please include ALL required fields: {required_fields}"
                        })
                    continue
                
                print(f"[SUCCESS] Query classified successfully! (Total API calls: {api_call_count})")
                return classification
                    
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse JSON response!")
                print(f"[DEBUG] JSON Error: {str(e)}")
                print(f"[DEBUG] Error at position: {e.pos}")
                print(f"[DEBUG] Problematic section: ...{clean_response[max(0, e.pos-50):e.pos+50]}...")
                print(f"[DEBUG] Full cleaned response:\n{clean_response}\n")
                
                if attempt < max_retries - 1:
                    print(f"[INFO] Retrying with stricter JSON instructions...")
                    payload["messages"].append({
                        "role": "assistant",
                        "content": assistant_message
                    })
                    payload["messages"].append({
                        "role": "user",
                        "content": "Please output ONLY valid JSON with no additional text, markdown formatting, or code fences. Ensure all strings are properly quoted and all arrays/objects are properly closed."
                    })
                continue
                
        except requests.exceptions.Timeout:
            print(f"[ERROR] Request timed out after 30 seconds!")
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying...")
            continue
            
        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] Connection error: {str(e)}")
            print(f"[DEBUG] Unable to reach {GROQ_API_URL}")
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying in 3 seconds...")
                import time
                time.sleep(3)
            continue
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] API request failed: {str(e)}")
            print(f"[DEBUG] Exception type: {type(e).__name__}")
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying in 2 seconds...")
                import time
                time.sleep(2)
            continue
    
    print(f"[FAILURE] Failed to classify query after {api_call_count} API calls.")
    print(f"[TROUBLESHOOTING TIPS]:")
    print(f"  1. Check your API key is valid: {GROQ_API_KEY[:10]}...")
    print(f"  2. Verify you have Groq API credits/quota remaining")
    print(f"  3. Try a simpler query to test basic connectivity")
    print(f"  4. Check Groq API status: https://status.groq.com/")
    return None


def print_classification(classification: Dict[str, Any]) -> None:
    """Pretty print the classification result."""
    print("\n" + "="*60)
    print("LEGAL QUERY CLASSIFICATION RESULT")
    print("="*60)
    print(json.dumps(classification, indent=2, ensure_ascii=False))
    print("="*60 + "\n")


def normalize_for_retriever(classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure classifier output matches Retriever expectations.

    - legal_domains: convert string entries to dicts with a default confidence.
    """
    legal_domains = classification.get("legal_domains", [])
    normalized_domains = []
    for domain in legal_domains:
        if isinstance(domain, dict):
            normalized_domains.append(domain)
        else:
            normalized_domains.append({
                "domain": str(domain),
                "confidence_score": 0.5
            })

    return {**classification, "legal_domains": normalized_domains}


def main():
    """Main function to demonstrate the classifier."""
    
    print("\n" + "="*60)
    print("KENYAN LEGAL QUERY CLASSIFIER")
    print("="*60)
    print("\nEnter your legal query below:")
    print("(Type your query and press Enter when done)\n")
    
    # Simple single-line input
    user_query = input("Query: ").strip()
    
    if not user_query:
        print("\n❌ No query entered. Exiting.")
        return
    
    print(f"\n📋 You entered:\n{user_query}\n")
    
    # Classify the query
    print("🔄 Processing your query...\n")
    result = classify_legal_query(user_query)
    
    if result:
        print_classification(result)
        print("🔄 Running retrieval with classifier output...\n")
        try:
            normalized = normalize_for_retriever(result)
            retrieval_output = retrieve_relevant_cases(normalized, verbose=True)
            display_results(retrieval_output)
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Classification failed.")
        print("\n💡 Suggestions:")
        print("  - Run with a test query: 'What is adverse possession?'")
        print("  - Check your .env file contains valid GROQ_API_KEY")
        print("  - Verify internet connectivity")


if __name__ == "__main__":
    main()