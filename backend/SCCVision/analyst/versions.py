"""Version constants for the SCC analyst pipeline.

Bump RULES_VERSION whenever rules.py changes behaviour (derive_statuses,
compute_confidence, deterministic_verdict, cert_eligibility, etc.).
Bump PROMPT_VERSION when inspect.md changes.
Bump VERIFY_PROMPT_VERSION when verify.md changes.

input_hash = sha256(canonical_json + PROMPT_VERSION + VERIFY_PROMPT_VERSION
                    + RULES_VERSION + MODEL)

A change to any of these constants invalidates all cached analyses.
"""

PROMPT_VERSION = "2.2.0"
VERIFY_PROMPT_VERSION = "2.2.0"
RULES_VERSION = "2.2.0"
MODEL = "qwen/qwen3.8-27b"
MAX_CALLS = 2
