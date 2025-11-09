# Test Running Instructions (Backend)

Краткая памятка по запуску тестов backend.

## Базовые команды
```bash
cd backend

# Все тесты
pytest

# Покрытие
pytest --cov=apps --cov-report=html

# Конкретный файл
pytest tests/test_models.py -v

# Параллельно (если установлен xdist)
pytest -n auto
```

## Специальные тесты
```bash
# Smoke для диагностики моделей/связей
python tests/smoke/smoke_diagnostics.py

# RAG pipeline (FAISS + Ollama)
python tests/unit/test_rag.py
```

## Проверки качества
```bash
flake8 .
isort --check-only .
black --check .
bandit -c ../.bandit -r .
```

## Окружение
- DJANGO_SETTINGS_MODULE=core.settings
- База данных (TimescaleDB/PostgreSQL) должна быть доступна
- Для RAG pipeline — запущен Ollama с моделями: qwen3:8b, nomic-embed-text

## Подсказки
- E402 в исполняемых скриптах после `django.setup()` допустим с `# noqa: E402`
- Для нестабильных тестов используйте `-k` для отбора и `--lf` для перезапуска последних упавших
