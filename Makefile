# Convenience targets. Run `make help` to list them.
.PHONY: help install api ui dev test lint format check

help:  ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install the app with all extras and dev tools
	pip install -e ".[dev]"

api:  ## Run the FastAPI backend on :8000 (auto-reload)
	uvicorn localmind.api.main:app --reload --port 8000

ui:  ## Run the Streamlit frontend on :8501
	streamlit run src/localmind/ui/app.py --browser.gatherUsageStats false

dev:  ## Run API and UI together (Ctrl+C stops both)
	trap 'kill 0' INT; $(MAKE) api & $(MAKE) ui & wait

test:  ## Run the test suite (no Ollama/LangSmith needed)
	pytest

lint:  ## Lint and check formatting
	ruff check . && ruff format --check .

format:  ## Auto-fix lint issues and format
	ruff check --fix . && ruff format .

check: lint test  ## Everything CI runs
