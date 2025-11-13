FROM pytorch/pytorch:2.8.0-cuda12.1-cudnn8-runtime as builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev git && rm -rf /var/lib/apt/lists/*

COPY services/ml/requirements.txt ml-requirements.txt
COPY services/api/requirements.txt api-requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels \
    -r ml-requirements.txt -r api-requirements.txt || true

FROM pytorch/pytorch:2.8.0-cuda12.1-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* || true && \
    rm -rf /wheels

COPY services/api/app ./app
COPY services/ml ./ml
COPY services/api/config.py ./config.py

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512 PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
