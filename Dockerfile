# syntax=docker/dockerfile:1

# ===== Base builder image =====
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

# ===== CRITICAL: Multi-layer requirements installation =====
# Сначала устанавливаем стабильные Django зависимости
COPY backend/requirements.txt ./backend/requirements.txt
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip==24.2 && \
    pip install -r backend/requirements.txt

# Наконец, проектные зависимости (меняются часто)
COPY backend/requirements-dev.txt ./backend/requirements-dev.txt
RUN . /opt/venv/bin/activate && \
    pip install -r backend/requirements-dev.txt

# ===== Final runtime image =====
FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (runtime) + dos2unix for Windows compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl dos2unix && \
    rm -rf /var/lib/apt/lists/*

# Copy venv from builder (all layers cached!)
COPY --from=builder /opt/venv /opt/venv

# Copy project (изменения в коде не затрагивают зависимости)
COPY backend /app/backend

# Static files dir
RUN mkdir -p /app/backend/staticfiles /app/backend/logs

# Expose port
EXPOSE 8000

# Copy entrypoint script and fix Windows line endings
COPY docker/entrypoint.sh /entrypoint.sh
RUN dos2unix /entrypoint.sh && chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
