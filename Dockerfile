# One image for both services: the API (default command) and the Streamlit UI
# (compose overrides the command). Ollama runs in its own container.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Dependencies first, in their own layer: code edits don't reinstall them.
#    Only the package skeleton is needed for pip to resolve dependencies.
COPY pyproject.toml README.md ./
COPY src/localmind/__init__.py src/localmind/__init__.py
RUN pip install ".[rag,web]"

# 2) The actual code (fast layer).
COPY src ./src
RUN pip install --no-deps --force-reinstall .
COPY scripts ./scripts
COPY eval/dataset.json ./eval/dataset.json
COPY .streamlit ./.streamlit

# Run as an unprivileged user; data lives in a volume.
RUN useradd --create-home --uid 1000 localmind \
    && mkdir -p /app/data /app/eval/results \
    && chown -R localmind /app/data /app/eval
USER localmind

ENV LOCALMIND_DATABASE_URL=sqlite:////app/data/localmind.db \
    LOCALMIND_CHROMA_DIR=/app/data/chroma \
    LOCALMIND_OLLAMA_BASE_URL=http://ollama:11434

EXPOSE 8000 8501
CMD ["uvicorn", "localmind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
