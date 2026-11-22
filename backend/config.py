"""
Central configuration for the knowledge-switch (Steps 1-8).
Import only from the files edited in this switch; leave other modules as they are.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

# --- New case law collection (replaces LegalSummariesDB/legal_summaries) ---
CASELAW_DB_DIR = BASE_DIR / "CaseLawDB"
CASELAW_COLLECTION_NAME = "case_law"

# --- New statutes collection (replaces master_rules for non-Constitution acts) ---
STATUTES_DB_DIR = BASE_DIR / "StatutesDB"
STATUTES_COLLECTION_NAME = "statutes"

# --- Kept for Constitution fallback (document_type == "constitution") ---
MASTER_RULES_DB_DIR = BASE_DIR / "constitutionlogic" / "MasterRulesDB"
MASTER_RULES_COLLECTION_NAME = "master_rules"

# --- SCCCases directory (added to path-security allowlist in app.py) ---
SCC_CASES_DIR = BASE_DIR / "SCCCases"

# --- Shared embedding model ---
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

# --- Statute manifest (act slugs, titles, and aliases for STATUTE_MAP) ---
STATUTE_MANIFEST_PATH = DATA_DIR / "processed" / "statutes" / "manifest.json"

# --- SCCVision analyst timezone (used as "today" for date checks) ---
SCC_TIMEZONE = "Africa/Nairobi"
