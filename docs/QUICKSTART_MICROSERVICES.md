# Быстрый старт — Microservices Development

Руководство для локальной разработки и тестирования **backend + ml_service + rag_service**.

---

## Предварительные требования

- Docker и Docker Compose
- Python 3.11+
- Git

---

## Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas
git checkout refactor/requirements-docker-structure
```

---

## Шаг 2: Настройка окружения

### 2.1 Скопировать .env.example

```bash
cp .env.example .env
```

### 2.2 Генерировать безопасные ключи

```bash
# Генерировать ML_INTERNAL_API_KEY
python3 -c "import secrets; print('ML_INTERNAL_API_KEY=' + secrets.token_urlsafe(32))"

# Генерировать RAG_INTERNAL_API_KEY
python3 -c "import secrets; print('RAG_INTERNAL_API_KEY=' + secrets.token_urlsafe(32))"

# Генерировать DJANGO_SECRET_KEY
python3 -c "from django.core.management.utils import get_random_secret_key; print('DJANGO_SECRET_KEY=' + get_random_secret_key())"
```

### 2.3 Обновить .env файл

Вставьте сгенерированные ключи в `.env`:

```bash
# .env
DJANGO_SECRET_KEY=<your-generated-key>
ML_INTERNAL_API_KEY=<your-generated-ml-key>
RAG_INTERNAL_API_KEY=<your-generated-rag-key>
```

---

## Шаг 3: Запуск сервисов

### 3.1 Запустить Docker Compose

```bash
# Собрать и запустить все сервисы
docker-compose up --build -d

# Проверить статус
docker-compose ps
```

**Ожидаемый результат:**
```
NAME                IMAGE                         STATUS    PORTS
hdx-backend         ...                          Up        0.0.0.0:8000->8000/tcp
hdx-celery          ...                          Up
hdx-celery-beat     ...                          Up
hdx-ml-service      ...                          Up (healthy)
hdx-postgres        timescale/timescaledb:...    Up        0.0.0.0:5432->5432/tcp
hdx-rag-service     ...                          Up (healthy)
hdx-redis           redis:7-alpine               Up        0.0.0.0:6379->6379/tcp
```

### 3.2 Проверить логи

```bash
# Все сервисы
docker-compose logs -f

# Только backend
docker-compose logs -f backend

# Только ml_service
docker-compose logs -f ml_service

# Только rag_service
docker-compose logs -f rag_service
```

---

## Шаг 4: Health Checks

### 4.1 Проверка Backend

```bash
curl http://localhost:8000/health/
```

**Ожидаемый результат:**
```json
{
  "status": "healthy",
  "timestamp": 1699548271.234,
  "checks": {
    "database": "ok",
    "redis": "ok",
    "ml_service": "ok",
    "rag_service": "ok"
  }
}
```

### 4.2 Проверка ML Service (через internal network)

```bash
docker exec hdx-backend curl http://ml_service:8001/health
```

**Ожидаемый результат:**
```json
{
  "status": "healthy",
  "models_loaded": ["catboost", "xgboost", "random_forest", "adaptive"],
  "cache_status": "connected",
  "timestamp": 1699548271.234
}
```

### 4.3 Проверка RAG Service (через internal network)

```bash
docker exec hdx-backend curl http://rag_service:8002/health
```

**Ожидаемый результат:**
```json
{
  "status": "healthy",
  "timestamp": 1699548271.234,
  "service": "rag-service",
  "version": "0.1.0"
}
```

---

## Шаг 5: E2E Тестирование

### 5.1 Создать тестового пользователя

```bash
# Зайти в backend контейнер
docker exec -it hdx-backend bash

# Создать superuser
python manage.py createsuperuser
# Username: admin
# Email: admin@example.com
# Password: admin123 (dev only!)

exit
```

### 5.2 Получить JWT Token

```bash
curl -X POST http://localhost:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

**Сохраните access token:**
```bash
export TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."
```

### 5.3 Тест ML Prediction

```bash
curl -X POST http://localhost:8000/api/diagnostics/anomaly/detect/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "system_id": 1,
    "sensor_data": {
      "pressure": [100.5, 101.2, 99.8, 100.1, 100.3],
      "temperature": [45.3, 45.1, 45.5, 45.2, 45.4],
      "flow": [25.0, 24.8, 25.2, 25.1, 24.9],
      "vibration": [0.5, 0.6, 0.5, 0.5, 0.6]
    }
  }'
```

**Ожидаемый ответ:**
```json
{
  "system_id": 1,
  "prediction": {
    "is_anomaly": false,
    "anomaly_score": 0.23,
    "severity": "normal",
    "confidence": 0.88
  },
  "ensemble_score": 0.23,
  "total_processing_time_ms": 18.5,
  "trace_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 5.4 Тест RAG Query

```bash
curl -X POST http://localhost:8000/api/rag/query/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to fix hydraulic pressure drop?",
    "system_id": 1,
    "max_results": 3
  }'
```

**Ожидаемый ответ:**
```json
{
  "response": "To fix hydraulic pressure drop, check the following components...",
  "sources": [
    {
      "document_id": 1,
      "title": "Hydraulic Troubleshooting Guide",
      "snippet": "Pressure drop can be caused by...",
      "score": 0.92
    }
  ],
  "metadata": {
    "model": "llama3.2:latest",
    "processing_time_ms": 340,
    "tokens_used": 180
  }
}
```

---

## Шаг 6: Проверка Internal Network Isolation

### 6.1 Попытка прямого доступа к ml_service (должно провалиться)

```bash
# Попытка без API key
curl -X POST http://localhost:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{}'

# Ожидаемый результат: Connection refused (порт не exposed)
```

### 6.2 Проверка через internal network

```bash
# Зайти в backend контейнер
docker exec -it hdx-backend bash

# Попытка без API key (должно вернуть 403)
curl -X POST http://ml_service:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{}'

# Ожидаемый результат: 403 Forbidden

# С правильным API key (должно работать)
curl -X POST http://ml_service:8001/api/v1/predict \
  -H "X-Internal-API-Key: $ML_INTERNAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": {
      "system_id": 1,
      "pressure": [100.5, 101.2, 99.8]
    },
    "use_cache": false
  }'

exit
```

---

## Шаг 7: Мониторинг

### 7.1 Прометей метрики

```bash
# ML Service metrics (через backend container)
docker exec hdx-backend curl http://ml_service:8001/metrics
```

### 7.2 Structured Logs

```bash
# JSON логи с jq
docker-compose logs ml_service | grep -v "INFO:" | jq .
```

---

## Шаг 8: Разработка

### Hot Reload (Development)

**Backend:**
```bash
# Volume mount включён в docker-compose.yml
# Изменения в ./backend/ применяются автоматически
```

**ML Service:**
```bash
# Перезапустить после изменений
docker-compose restart ml_service
```

**RAG Service:**
```bash
# Перезапустить после изменений
docker-compose restart rag_service
```

---

## Troubleshooting

### Проблема: ml_service не запускается

```bash
# Проверьте логи
docker-compose logs ml_service

# Проверьте, что ML_INTERNAL_API_KEY установлен
docker exec hdx-ml-service env | grep ML_INTERNAL_API_KEY

# Проверьте модели
docker exec hdx-ml-service ls -la /app/models/
```

### Проблема: 403 Forbidden от ml_service

```bash
# Проверьте, что ключи совпадают
docker exec hdx-backend env | grep ML_INTERNAL_API_KEY
docker exec hdx-ml-service env | grep ML_INTERNAL_API_KEY

# Если не совпадают — обновите .env и перезапустите
docker-compose restart backend ml_service
```

### Проблема: Backend не может подключиться к ml_service

```bash
# Проверьте Docker network
docker network inspect hydraulic-diagnostic-saas_internal

# Проверьте DNS resolution
docker exec hdx-backend ping -c 2 ml_service
```

---

## Следующие шаги

1. ✅ **Проверить E2E сценарий**
2. ✅ **Загрузить тестовые данные**
3. ✅ **Настроить frontend интеграцию**
4. 🚧 **Запустить unit/integration тесты**
5. 🚧 **Performance benchmarks**

---

## Полезные команды

```bash
# Остановить все сервисы
docker-compose down

# Остановить и удалить volumes
docker-compose down -v

# Пересобрать один сервис
docker-compose up --build -d ml_service

# Просмотр логов за последние 5 минут
docker-compose logs --since 5m ml_service

# Зайти в контейнер
docker exec -it hdx-ml-service bash
```

---

## Документация

- [Microservices Integration Guide](./MICROSERVICES_INTEGRATION.md)
- [ML Service README](../ml_service/README.md)
- [RAG Service README](../rag_service/README.md)

---

## Поддержка

Для вопросов и проблем создайте issue в GitHub репозитории.
