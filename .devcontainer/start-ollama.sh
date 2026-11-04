#!/usr/bin/env bash
# Start the Ollama server in the background if it isn't running.
# Containers have no systemd, so Ollama doesn't start on its own after a restart.
set -uo pipefail

echo "Checking Ollama..."
# -m 3: never wait more than 3 s for an answer, so this can't hang silently.
if curl -sf -m 3 http://localhost:11434/api/version >/dev/null 2>&1; then
  echo "Ollama is running."
  exit 0
fi
echo "Ollama is not running; starting it..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed yet (run: bash .devcontainer/setup.sh)" >&2
  exit 1
fi
nohup ollama serve >/tmp/ollama.log 2>&1 &
for _ in $(seq 1 30); do
  if curl -sf -m 3 http://localhost:11434/api/version >/dev/null 2>&1; then
    echo "Ollama is running."
    exit 0
  fi
  sleep 1
done
echo "Ollama did not start; see /tmp/ollama.log" >&2
exit 1
