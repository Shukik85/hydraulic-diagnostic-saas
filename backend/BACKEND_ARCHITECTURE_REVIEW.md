# Backend Architecture Review

Документ описывает актуальную архитектуру backend-части Hydraulic Diagnostic SaaS.

---

## 1. Обзор
- Django 5.2 (DRF для API)
- TimescaleDB (PostgreSQL) для временных рядов
- Celery + Redis для фоновых задач
- RAG Assistant: FAISS + LangChain, локальные LLM через Ollama (Qwen3), Embeddings (nomic-embed-text)
- Кэширование: Redis

Структура:
```
backend/
├── core/                  # настройки, celery, urls
├── apps/
│   ├── diagnostics/       # домен диагностики и сенсоров
│   ├── rag_assistant/     # RAG ассистент и индексация
│   └── users/             # пользователи/аутентификация
└── tests/                 # тесты
```

---

## 2. Данные и TimescaleDB
- SensorData хранится в TimescaleDB (hypertable)
- Автоматическое управление партициями и политиками:
  - ensure_partitions_for_range — создание чанков в диапазоне
  - cleanup_old_partitions — удаление старых чанков (retention)
  - compress_old_chunks — сжатие старых чанков
  - get_hypertable_stats — сводная статистика
  - timescale_health_check — проверка состояния расширения и фоновых задач

Ключевые принципы:
- Только параметризованные SQL-запросы (без f-strings)
- Чёткие интервалы chunk’ов (например, 7 дней)
- Регулярные задачи по сжатию/ретенции данных

---

## 3. RAG Assistant
Компоненты:
- EmbeddingsProvider (SentenceTransformer или внешние эмбеддинги)
- VectorIndex (FAISS, IP/L2)
- LocalStorageBackend (хранение байт индекса + metadata.json — версии v_*)
- RAGOrchestrator (сборка/сохранение/загрузка индексов)

Пайплайн:
1) encode(docs) → build FAISS index → serialize → save_index(version, bytes, meta)
2) load_index(version) → deserialize → search(k)

Особенности:
- Нормализация эмбеддингов (L2)
- IP метрика соответствует косинусному сходству при нормализации
- Версионирование путей: v_<version>/index.faiss и metadata.json

---

## 4. Celery
- Конфигурация в `backend/core/celery.py`
- Worker + Beat; сигналы task_prerun/task_postrun/task_failure
- Маршрутизация очередей: `ai_tasks`, `diagnostics`, `users`
- Таймауты/ограничения по памяти из env через decouple

Запуск:
```
celery -A core worker --loglevel=info
celery -A core beat --loglevel=info
```

---

## 5. Кэширование
- Redis для кэша RAG (поисковые результаты, FAQ) и метрик
- Ключи строятся детерминированно, TTL на часто меняемые сущности
- Инвалидация кэша при изменении документов

---

## 6. Безопасность и качество
- Bandit: запрет f-string в SQL; параметризация `cursor.execute(sql, [params])`
- Pre-commit: trailing whitespace/eof, isort, black, flake8, mypy
- E402: разрешенный паттерн импортов после `django.setup()` с `# noqa: E402`

---

## 7. Тестирование
- pytest + pytest-django; smoke_diagnostics.py для быстрой проверки моделей
- test_rag.py — интеграционный тест RAG пайплайна (FAISS + Ollama)
- TimescaleDB: проверка hypertables через timescaledb_information

---

## 8. Развёртывание и окружение
- Docker Compose (dev/prod YAML) + .env(.production)
- Makefile цели: dev, migrate, superuser, init-data, test, lint, format
- Настройки LLM (OLLAMA_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL)

---

Документ синхронизирован с текущей веткой и кодовой базой.
