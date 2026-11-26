# Patch 1: backend/app.py CORS (multiple origins + Vercel previews)

Replace lines 39–40:

```python
_CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "http://localhost:3000")
CORS(app, origins=_CORS_ORIGIN)
```

with:

```python
import re

# Comma-separated exact origins, e.g. "https://legalai.vercel.app,http://localhost:3000"
_CORS_ORIGINS = [
    o.strip().rstrip("/")
    for o in os.environ.get("CORS_ORIGIN", "http://localhost:3000").split(",")
    if o.strip()
]

# Optional regex for Vercel preview URLs, e.g. r"^https://legalai-[a-z0-9-]+\.vercel\.app$"
_preview = os.environ.get("CORS_ORIGIN_REGEX", "").strip()
if _preview:
    _CORS_ORIGINS.append(re.compile(_preview))

CORS(
    app,
    origins=_CORS_ORIGINS,
    allow_headers=["Authorization", "Content-Type"],
    methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    max_age=600,  # browsers cache the preflight for 10 minutes
)
```

Flask-CORS accepts both strings and compiled regexes in `origins`.

Set it on Cloud Run. The `^@^` prefix tells gcloud to split variables on `@`, so the commas in the value are kept:

```bash
gcloud run services update legalai-backend --region africa-south1 \
  --update-env-vars "^@^CORS_ORIGIN=https://YOUR_APP.vercel.app,http://localhost:3000@CORS_ORIGIN_REGEX=^https://YOUR_PROJECT-[a-z0-9-]+\.vercel\.app$"
```
