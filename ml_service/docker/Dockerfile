# Multi-stage Docker build for ML Inference Service
# Production-ready контейнер с оптимизациями

FROM python:3.11-slim as builder

# Метаданные
LABEL maintainer="Hydraulic Diagnostics Team"
LABEL version="1.0.0"
LABEL description="Enterprise ML Inference Service"

# Оптимизации Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование и установка зависимостей
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt

# Production stage
FROM python:3.11-slim

# Непривилегированный пользователь
RUN groupadd -r mlservice && useradd -r -g mlservice mlservice

# Рабочая директория
WORKDIR /app

# Копирование установленных пакетов
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Копирование кода приложения
COPY --chown=mlservice:mlservice . .

# Создание необходимых директорий
RUN mkdir -p models logs && chown -R mlservice:mlservice .

# Переключение на непривилегированного пользователя
USER mlservice

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Порты
EXPOSE 8001 9090

# Запуск
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]