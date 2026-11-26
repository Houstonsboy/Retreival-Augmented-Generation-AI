"""Seed a synthetic analysis snapshot that passes all s.106B gate conditions.

Writes directly to Firestore via Firebase Admin SDK so that the Playwright
screenshot script can call POST /api/evidence/certificates with a real
analysis_id and get certificate_ready back.

Usage:
    cd /home/gift/Retreival-Augmented-Generation-AI
    backend/venv/bin/python scripts/seed_eligible_analysis.py
"""

import os
import sys
import uuid
import json

# Add backend to path so Firebase Admin is usable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import firebase_admin
from firebase_admin import credentials, firestore as fb_firestore

KEY_FILE = os.path.join(
    os.path.dirname(__file__), "..",
    "backend", "legalai-scc-firebase-adminsdk-fbsvc-73bde27853.json",
)

cred = credentials.Certificate(KEY_FILE)
firebase_admin.initialize_app(cred)
db = fb_firestore.client()

# UID of the e2e-advocate test user (confirmed from backend logs)
UID = "0LKQvIUcmHMsrVhPZ15t4kon6p82"

analysis_id = uuid.uuid4().hex

# Synthetic snapshot: every gate condition passes
#   4a  — file identified, hash verified
#   4b  — make/model provided via device_facts
#   2a  — lawful_control + regular_use (device_facts)
#   2b  — regular_feed (device_facts)
#   2c  — operating_properly (device_facts)
#   2d  — messages_on_this_device + screenshot_on_device + no M-Pesa items → skips code check
#   4d  — owner_is_declarant + declaration_confirmed (device_facts)
#   INTEGRITY — refs_grounded=True, mpesa_cross_check consistent=True, rating=ADEQUATE
snapshot = {
    "analysis_id": analysis_id,
    "input_hash": "synthetic-eligible-" + analysis_id,
    "prompt_version": "v2.2.0",
    "verify_prompt_version": "v2.2.0",
    "rules_version": "s106b-v1",
    "model": "llama-3.3-70b-versatile",
    "calls_used": 2,
    "trigger_fired": False,
    "derived_statuses": {
        "A": "FAVOURABLE",
        "B": "FAVOURABLE",
        "C": "FAVOURABLE",
    },
    "cert_eligibility": {
        "eligible": True,
        "reasons": [],
    },
    "confidence": 0.92,
    "verdict": "FAVOURABLE",
    "integrity_rating": "ADEQUATE",
    "unstable_fields": [],
    "model_disagreements": {},
    "timestamp_info": {},
    "confidence_breakdown": {},
    "guided_follow_ups": [],
    "case_id": "",
    "round": 1,
    "parent_analysis_id": None,
    # --- Gate-critical fields ---
    "evidence": [
        {
            "evidence_id": "ev-synth-001",
            "type": "whatsapp_screenshot",        # NOT mpesa_screenshot → skips M-Pesa code check
            "file_name": "conversation.jpg",
            "sha256": "aabbcc" * 10 + "aabb",
            "server_sha256": "aabbcc" * 10 + "aabb",
            "hash_match": True,                   # 4a: hash integrity OK
            "added_at": None,
        }
    ],
    "result": {
        "elements": {
            "A": {"status": "FAVOURABLE", "summary": "Service contract identified."},
            "B": {"status": "FAVOURABLE", "summary": "Payment confirmed."},
            "C": {"status": "FAVOURABLE", "summary": "Communication record present."},
        },
        "mpesa_payments": [],                     # no M-Pesa → 2d skips code check
        "conversation_integrity": {
            "rating": "ADEQUATE",                 # INTEGRITY: not WEAK
            "issues": [],
        },
        "plain_summary": "Synthetic eligible analysis for dissertation screenshot.",
    },
    "verification": {
        "refs_grounded": True,                    # INTEGRITY: refs grounded
        "mpesa_cross_check": {
            "consistent": True,                   # INTEGRITY: M-Pesa check passes
            "invented_fields": False,
            "codes_llm_only": [],
        },
    },
    "schema_version": 3,
    "truncated": False,
}

ref = (
    db.collection("users")
    .document(UID)
    .collection("analyses")
    .document(analysis_id)
)
ref.set(snapshot)
print(json.dumps({"analysis_id": analysis_id, "uid": UID}))
