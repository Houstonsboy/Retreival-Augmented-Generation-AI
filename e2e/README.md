# LegalAI e2e (Playwright)

Put this folder at the repo root as `e2e/`, next to `backend/` and `frontend/`.

## Setup (once)

```bash
cd e2e
npm install
npm install firebase-admin          # only for the create-user script
npx playwright install chromium     # you already have chromium-1248
cp .env.example .env                # fill in URLs + test credentials
npm run create-user                 # makes the advocate test account in legalai-scc
echo -e ".env\n.auth/\ntest-results/\nplaywright-report/" >> .gitignore
```

## Run

| Command | What it does |
|---|---|
| `npm run smoke` | Backend only: health, CORS preflight, 401 without a token. No browser. Run it right after each Cloud Run deploy. |
| `npm run test:local` | Full suite against localhost:3000 / :5000 |
| `npm run test:prod` | Full suite against Vercel + Cloud Run |
| `npm run test:headed` | Same, with a visible Chromium window (you have DISPLAY=:1) |
| `npm run report` | Open the HTML report with traces, screenshots and video of failures |

## What a failure usually means

| Symptom | Cause |
|---|---|
| health times out | Cloud Run cold start or a crash. Check `gcloud run services logs read legalai-backend --region africa-south1` |
| `access-control-allow-origin` mismatch | `CORS_ORIGIN` on Cloud Run ≠ frontend URL (check for a trailing slash or http/https) |
| 503 on protected route | Firebase Admin didn't initialise (see patches/02) |
| setup fails with unauthorized-domain | Vercel domain missing from Firebase Authorized domains |
| 403 PROFILE_REQUIRED | No `users/{uid}` doc. Rerun `npm run create-user` |
| 403 role | Profile `type` isn't advocate/admin |
| no `Authorization` header | A call bypasses `apiFetch` |

## Patches

`patches/` holds three small source changes the deploy and tests rely on: multi-origin CORS,
the ADC fallback in auth.py, and data-testid hooks in ChatView.
