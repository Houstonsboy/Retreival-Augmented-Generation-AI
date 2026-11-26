"""
Seed the Firestore CaseLaws collection from the existing backend/SCCCases/ folder.

Run once from the backend directory:
    source venv/bin/activate
    python scripts/seed_caselaws.py

Each file gets a Firestore document in the CaseLaws collection.
Files are served at runtime via /api/case-laws/file/<filename> (backend auth endpoint).
Files already present in Firestore (matched by fileName) are skipped.
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import firebase_admin
from firebase_admin import credentials, firestore

SERVICE_ACCOUNT = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
if not SERVICE_ACCOUNT:
    sys.exit("ERROR: FIREBASE_SERVICE_ACCOUNT_PATH not set in .env")

sa_path = SERVICE_ACCOUNT if os.path.isabs(SERVICE_ACCOUNT) else BACKEND_DIR / SERVICE_ACCOUNT
if not Path(sa_path).exists():
    sys.exit(f"ERROR: service account not found at {sa_path}")

cred = credentials.Certificate(str(sa_path))
firebase_admin.initialize_app(cred)

db = firestore.client()

SCC_CASES_DIR = BACKEND_DIR / "SCCCases"
ALLOWED_EXTS = {'.pdf', '.docx'}


def find_existing(filename: str):
    """Return the existing doc snapshot if found, else None."""
    docs = list(db.collection('CaseLaws').where('fileName', '==', filename).limit(1).get())
    return docs[0] if docs else None


def seed():
    if not SCC_CASES_DIR.exists():
        sys.exit(f"ERROR: {SCC_CASES_DIR} does not exist")

    files = sorted(
        [f for f in SCC_CASES_DIR.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTS],
        key=lambda f: f.name,
    )

    print(f"Found {len(files)} file(s) in {SCC_CASES_DIR.name}/")
    seeded = 0
    skipped = 0

    for f in files:
        existing = find_existing(f.name)
        local_path = str(f.resolve())

        if existing:
            # Patch localPath if missing from earlier seed run
            if not existing.to_dict().get('localPath'):
                existing.reference.update({'localPath': local_path})
                print(f"  PATCH {f.name}")
            else:
                print(f"  SKIP  {f.name}")
            skipped += 1
            continue

        ext = f.suffix.lower().lstrip('.')
        title = f.stem

        db.collection('CaseLaws').add({
            'title': title,
            'fileName': f.name,
            'fileType': ext,
            'localPath': local_path,
            'uploadedBy': 'system',
            'uploadedByUid': 'system',
            'uploadedAt': firestore.SERVER_TIMESTAMP,
            'fileSizeBytes': f.stat().st_size,
        })
        print(f"  OK    {f.name}")
        seeded += 1

    print(f"\nDone — seeded {seeded}, skipped {skipped}")


if __name__ == '__main__':
    seed()
