# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1

WORKDIR /app

# Dependencies only: cached until uv.lock / pyproject.toml change
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

# Embedder + reranker ONNX weights and ara/fra/eng traineddata go here
RUN mkdir -p /app/models/tessdata
RUN /app/.venv/bin/python scripts/download_models.py


FROM python:3.12-slim

RUN useradd --uid 1000 --create-home app \
    && mkdir -p /data && chown app:app /data

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/models /app/models

ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/app/models \
    HF_HUB_OFFLINE=1 \
    TESSDATA_PREFIX=/app/models/tessdata \
    DATA_DIR=/data

USER app
WORKDIR /app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "mcclub_rag.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
