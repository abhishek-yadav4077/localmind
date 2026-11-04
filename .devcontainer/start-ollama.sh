#!/usr/bin/env bash
# Start the Ollama server in the background if it isn't running.
# Containers have no systemd, so Ollama doesn't start on its own after a restart.
set -uo pipefail

if curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
  exit 0
fi
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed yet (run: bash .devcontainer/setup.sh)" >&2
  exit 1
fi
nohup ollama serve >/tmp/ollama.log 2>&1 &
for _ in $(seq 1 30); do
  if curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
    echo "Ollama is running."
    exit 0
  fi
  sleep 1
done
echo "Ollama did not start; see /tmp/ollama.log" >&2
exit 1
