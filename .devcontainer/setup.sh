#!/usr/bin/env bash
# Runs once when the Codespace / dev container is created.
# Installs the app, installs Ollama and downloads the models (~5.5 GB).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/4] Python packages"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo "==> [2/4] Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  # The installer unpacks a .tar.zst archive, so zstd must be present.
  sudo apt-get update -qq
  sudo apt-get install -y -qq zstd
  curl -fsSL https://ollama.com/install.sh | sh
fi
bash .devcontainer/start-ollama.sh

echo "==> [3/4] Models (small ones that fit in 8 GB RAM)"
for model in ${LOCALMIND_MODELS:-llama3.2 qwen2.5:3b llama3.2:1b nomic-embed-text}; do
  ollama pull "$model"
done

echo "==> [4/4] Settings file"
# Environment variables from devcontainer.json and Codespaces secrets take
# priority over .env, so this copy only provides defaults.
[ -f .env ] || cp .env.example .env

echo
echo "Setup complete. Next: python scripts/check_setup.py"
