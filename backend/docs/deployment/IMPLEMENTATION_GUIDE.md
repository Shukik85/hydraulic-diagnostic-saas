# Implementation Guide (Backend)

Практические рекомендации по работе с backend Hydraulic Diagnostic SaaS.

---

## 1. Окружение
- Python 3.10+
- PostgreSQL 16+ / TimescaleDB 2.17+
- Redis 7+
- Ollama (локально) для LLM: Qwen3:8b, nomic-embed-text

`.env` ключи (пример):
```
DJANGO_SETTINGS_MODULE=core.settings
DATABASE_URL=postgres://user:pass@db:5432/app
REDIS_URL=redis://redis:6379/0

# AI
OLLAMA_BASE_URL=http://host.docker.internal:11434
DEFAULT_LLM_MODEL=qwen3:8b
DEFAULT_EMBEDDING_MODEL=nomic-embed-text
```

---

## 2. Команды разработки
```bash
# Запуск dev-среды
make dev

# Миграции
make migrate

# Создание суперпользователя
make superuser

# Тестовые данные и инициализация RAG
make init-data

# Тесты
make test

# Качество кода
make lint
make format
```

---

## 3. Celery
```bash
celery -A core worker --loglevel=info
celery -A core beat --loglevel=info
```
Очереди: `ai_tasks`, `diagnostics`, `users`. Логи задач включают task_prerun/postrun/failure сигналы.

---

## 4. TimescaleDB задачи
- ensure_partitions_for_range(table, start, end, chunk_interval)
- cleanup_old_partitions(table, retention_period)
- compress_old_chunks(table, compression_age)
- get_hypertable_stats(table)
- timescale_health_check()

Рекомендации:
- Планировать с помощью beat/cron (дневные/недельные окна)
- Использовать параметризованные SQL-запросы

---

## 5. RAG
- RAGOrchestrator/VectorIndex/LocalStorageBackend находятся в `apps/rag_assistant/rag_core.py`
- Документы индексируются через задачи `apps/rag_assistant/tasks.py`
- Тест RAG `backend/test_rag.py` — пример работы пайплайна

Советы:
- Для больших коллекций — батчевать документы
- Контролировать размер FAISS индекса и версионирование (v_<version>)

---

## 6. Кэширование
- Redis: результаты поиска, FAQ, метрики
- TTL для частых запросов; инвалидация при изменениях документов

---

## 7. Безопасность и стиль
- Всегда параметризовать SQL (`cursor.execute(sql, [params])`)
- Поддерживать стиль: isort/black/flake8
- Допускается `# noqa: E402` после `django.setup()` в исполняемых скриптах
- Сложные функции — рефакторить (допустимо временно `# noqa: C901`)

---

## 8. Диагностика проблем
- TimescaleDB: проверка расширения и hypertables через `timescaledb_information`
- Celery: `celery -A core inspect active`, Flower
- Ollama: `curl /api/version`, `ollama list`

---

## 9. Продакшен
```bash
docker-compose -f docker-compose.prod.yml up -d

docker-compose -f docker-compose.prod.yml exec backend \
  python manage.py migrate && \
  python manage.py collectstatic --noinput
```

Проверить readiness/health endpoints и доступ к Redis/DB/Ollama.
