#!/usr/bin/env bash
# Start the whole app inside a Codespace / dev container:
# Ollama (if needed) -> FastAPI on :8000 (background) -> Streamlit on :8501.
# Press Ctrl+C to stop both.
set -uo pipefail
cd "$(dirname "$0")/.."

bash .devcontainer/start-ollama.sh || echo "WARNING: Ollama is not running; the app will show how to fix it."

uvicorn localmind.api.main:app --host 0.0.0.0 --port 8000 >/tmp/localmind-api.log 2>&1 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT

echo "Starting API (log: /tmp/localmind-api.log)..."
for _ in $(seq 1 60); do
  curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "API failed to start:" >&2
    tail -20 /tmp/localmind-api.log >&2
    exit 1
  fi
  sleep 1
done
echo "API ready: port 8000 (/docs)."

echo "Starting UI: open the 'Ports' tab and click the globe icon next to port 8501."
# Codespaces serves forwarded ports through its own HTTPS proxy, which breaks
# Streamlit's CORS/XSRF checks (file uploads fail with 403). The port is private
# to your GitHub account, so turning these checks off here is safe.
streamlit run src/localmind/ui/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
