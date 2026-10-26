#!/usr/bin/env python3
"""Set the role (type field) of a user by email.

Usage (from repo root with backend venv active):
    python backend/scripts/set_role.py <email> <admin|advocate>

Requires FIREBASE_SERVICE_ACCOUNT_PATH set in backend/.env (or environment).
"""
import sys
import os
from pathlib import Path

# Allow importing from backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import firebase_admin
from firebase_admin import credentials, auth, firestore

VALID_ROLES = ("admin", "advocate")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <email> <admin|advocate>")
        sys.exit(1)

    email = sys.argv[1].strip()
    role = sys.argv[2].strip()

    if role not in VALID_ROLES:
        print(f"Error: role must be one of {VALID_ROLES}, got '{role}'")
        sys.exit(1)

    sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
    if not sa_path:
        print("Error: FIREBASE_SERVICE_ACCOUNT_PATH not set in backend/.env")
        sys.exit(1)
    if not os.path.exists(sa_path):
        print(f"Error: service account file not found at '{sa_path}'")
        sys.exit(1)

    if not firebase_admin._apps:
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    try:
        user_record = auth.get_user_by_email(email)
    except auth.UserNotFoundError:
        print(f"Error: no Firebase Auth user with email '{email}'")
        sys.exit(1)

    uid = user_record.uid
    doc_ref = db.collection("users").document(uid)

    if not doc_ref.get().exists:
        print(f"Error: no Firestore profile for uid '{uid}' (email: {email})")
        print("The user must complete sign-up before you can set their role.")
        sys.exit(1)

    doc_ref.update({"type": role, "updatedAt": firestore.SERVER_TIMESTAMP})
    updated = doc_ref.get().to_dict()
    print(f"Set role for {email} (uid: {uid})")
    print(f"  username : {updated.get('username', 'N/A')}")
    print(f"  type     : {updated.get('type')}")


if __name__ == "__main__":
    main()
