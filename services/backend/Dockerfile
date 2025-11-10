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

RUN mkdir -p /pip-cache

ARG REQUIREMENTS_PROFILE=base
ARG REQUIREMENTS_PATH=backend/requirements/${REQUIREMENTS_PROFILE}.txt

COPY backend/requirements/ ./backend/requirements/

RUN --mount=type=cache,target=/pip-cache \
    python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip==24.2 && \
    pip install --cache-dir=/pip-cache -r $REQUIREMENTS_PATH

# ===== Final runtime image =====
FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl dos2unix postgresql-client && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY backend /app/backend
RUN mkdir -p /app/backend/staticfiles /app/backend/logs

EXPOSE 8000

COPY backend/docker/entrypoint.sh /entrypoint.sh
RUN dos2unix /entrypoint.sh && chmod +x /entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
