FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PRODUCT_SEARCH_ARTIFACT_DIR=/app/artifacts/smoke \
    PRODUCT_SEARCH_VERIFY_ARTIFACTS=1 \
    PRODUCT_SEARCH_STRICT_ENV=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY constraints ./constraints
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install -c constraints/validated.txt .

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser artifacts/smoke ./artifacts/smoke

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import json,urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3))" || exit 1

CMD ["uvicorn", "product_search.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
