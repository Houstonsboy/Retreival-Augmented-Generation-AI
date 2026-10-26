#!/usr/bin/env python3
"""One-time bootstrap: create an admin user in Firebase Auth + Firestore.

Usage (from repo root with backend venv active):
    python backend/scripts/bootstrap_admin.py <email> <password> <username>

Creates the Firebase Auth account if it does not exist, then writes
users/{uid} and usernames/{username} via Admin SDK (bypasses client rules).
"""
import sys
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import firebase_admin
from firebase_admin import credentials, auth, firestore

VALID_ROLES = ("admin", "advocate")


def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <email> <password> <username>")
        sys.exit(1)

    email, password, username = sys.argv[1], sys.argv[2], sys.argv[3]

    if len(username) < 3 or len(username) > 30:
        print("Error: username must be 3-30 characters")
        sys.exit(1)
    import re
    if not re.match(r'^[a-z0-9_]+$', username):
        print("Error: username must be lowercase letters, numbers, underscore only")
        sys.exit(1)

    sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
    if not sa_path:
        print("Error: FIREBASE_SERVICE_ACCOUNT_PATH not set in backend/.env")
        sys.exit(1)
    if not os.path.isabs(sa_path):
        sa_path = str(BACKEND_DIR / sa_path)
    if not os.path.exists(sa_path):
        print(f"Error: service account not found at '{sa_path}'")
        sys.exit(1)

    if not firebase_admin._apps:
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    # Create or retrieve the Auth user
    try:
        user_record = auth.get_user_by_email(email)
        print(f"Auth user already exists (uid: {user_record.uid})")
    except auth.UserNotFoundError:
        user_record = auth.create_user(email=email, password=password, email_verified=True)
        print(f"Created Auth user (uid: {user_record.uid})")

    uid = user_record.uid

    # Check username isn't taken by someone else
    uname_doc = db.collection("usernames").document(username).get()
    if uname_doc.exists and uname_doc.to_dict().get("uid") != uid:
        print(f"Error: username '{username}' is already taken by a different user")
        sys.exit(1)

    # Write Firestore profile (Admin SDK — bypasses client rules)
    batch = db.batch()
    batch.set(db.collection("users").document(uid), {
        "username": username,
        "email": email,
        "type": "admin",
        "authProvider": "password",
        "advocateVerified": False,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })
    batch.set(db.collection("usernames").document(username), {"uid": uid})
    batch.commit()

    print(f"Bootstrap complete:")
    print(f"  uid      : {uid}")
    print(f"  email    : {email}")
    print(f"  username : {username}")
    print(f"  type     : admin")
    print()
    print("You can now sign in at http://localhost:3000/auth/sign-in")


if __name__ == "__main__":
    main()
