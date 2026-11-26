# Patch 2: backend/auth.py ADC fallback (no key file needed on Cloud Run)

Today, if `FIREBASE_SERVICE_ACCOUNT_PATH` is unset, init returns early and every protected
route gives 503. This change falls back to Cloud Run's own service identity.

Replace the init block (around lines 14–33) with:

```python
import os
import firebase_admin
from firebase_admin import credentials

_BUCKET = "legalai-scc.firebasestorage.app"
_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "legalai-scc")

def _init_firebase():
    if firebase_admin._apps:
        return
    sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
    if sa_path and not os.path.isabs(sa_path):
        sa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), sa_path)

    try:
        if sa_path and os.path.exists(sa_path):
            cred = credentials.Certificate(sa_path)          # local dev
            src = f"key file {sa_path}"
        else:
            cred = credentials.ApplicationDefault()           # Cloud Run service account
            src = "Application Default Credentials"
        firebase_admin.initialize_app(cred, {"projectId": _PROJECT, "storageBucket": _BUCKET})
        print(f"Firebase Admin initialised via {src}")
    except Exception as e:
        print(f"WARNING: Firebase Admin init failed ({e}) — protected routes will return 503")
```

Keep whatever `_get_db()` / 503 handling you already have. They work the same way after this change.

Grant the Cloud Run service account the roles it needs (once):

```bash
gcloud config set project legalai-scc   # your gcloud is currently on jukebox-996de
PN=$(gcloud projects describe legalai-scc --format='value(projectNumber)')
SA="${PN}-compute@developer.gserviceaccount.com"
for ROLE in roles/datastore.user roles/firebaseauth.viewer roles/storage.objectAdmin; do
  gcloud projects add-iam-policy-binding legalai-scc --member="serviceAccount:$SA" --role="$ROLE"
done
```

After that, you can drop the `--add-volume`, `--add-volume-mount` and
`FIREBASE_SERVICE_ACCOUNT_PATH` lines from the deploy command.
