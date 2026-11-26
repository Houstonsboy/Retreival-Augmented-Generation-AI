#!/usr/bin/env bash
# Latency measurement script for the Flask backend.
# Run manually: bash eval/measure_latency.sh
# Prerequisites: GROQ_API_KEY and FIREBASE_SERVICE_ACCOUNT_PATH must be set.

set -euo pipefail
BACKEND_DIR="$(cd "$(dirname "$0")/../backend" && pwd)"
VENV_PYTHON="$BACKEND_DIR/venv/bin/python"
FLASK_APP="$BACKEND_DIR/app.py"
BASE_URL="http://localhost:5000"
N=10  # repetitions per endpoint

echo "=== Starting Flask server ==="
cd "$BACKEND_DIR"
FLASK_APP=app.py "$VENV_PYTHON" -m flask run --port 5000 &
SERVER_PID=$!
echo "PID: $SERVER_PID"
sleep 10

echo "=== Waiting for health check ==="
curl -s "$BASE_URL/api/health" | grep -q "ok" || { echo "Server not ready"; kill $SERVER_PID; exit 1; }

measure() {
  local LABEL="$1"
  local URL="$2"
  local DATA="$3"
  local METHOD="${4:-POST}"
  local AUTH_HEADER="${5:-}"

  local times=()
  for i in $(seq 1 $N); do
    local t
    if [ "$METHOD" = "GET" ]; then
      t=$(curl -s -o /dev/null -w "%{time_total}" -H "$AUTH_HEADER" "$URL")
    else
      t=$(curl -s -o /dev/null -w "%{time_total}" -X POST \
        -H "Content-Type: application/json" \
        -H "$AUTH_HEADER" \
        -d "$DATA" "$URL")
    fi
    times+=("$t")
    echo "  $LABEL run $i: ${t}s"
  done

  # Compute median and p95 (requires python3)
  python3 - <<EOF
times = [float(x) for x in "${times[*]}".split()]
times.sort()
n = len(times)
median = times[n // 2]
p95_idx = int(0.95 * n)
p95 = times[min(p95_idx, n - 1)]
print(f"  $LABEL MEDIAN={median:.3f}s  P95={p95:.3f}s")
EOF
}

echo ""
echo "=== Endpoint latency (N=$N) ==="
echo "NOTE: These require a valid Firebase ID token in AUTH_TOKEN env var."
AUTH_HEADER="Authorization: Bearer ${AUTH_TOKEN:-dummy_token}"

measure "GET  /api/health" "$BASE_URL/api/health" "" "GET"

measure "POST /api/qretrieve" "$BASE_URL/api/qretrieve" \
  '{"query":"What is the limitation period for a contract claim in Small Claims Court?"}' \
  "POST" "$AUTH_HEADER"

measure "POST /api/generate-legal-answer" "$BASE_URL/api/generate-legal-answer" \
  '{"query":"Is oral evidence of an M-Pesa payment admissible?","retrieval":{"rules":[],"results":[]},"classification":{}}' \
  "POST" "$AUTH_HEADER"

measure "POST /sccvision/extract (health proxy)" "$BASE_URL/sccvision/health" "" "GET"

echo ""
echo "=== Done. Kill server? ==="
read -p "Kill server PID $SERVER_PID? [y/N] " yn
if [[ "$yn" =~ ^[Yy]$ ]]; then
  kill "$SERVER_PID"
  echo "Server stopped."
fi
