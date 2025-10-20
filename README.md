# 🔧 Hydraulic Diagnostic SaaS

**Интеллектуальная SaaS-платформа для диагностики и мониторинга гидравлических систем**

[![CI Status](https://github.com/Shukik85/hydraulic-diagnostic-saas/actions/workflows/ci.yml/badge.svg)](https://github.com/Shukik85/hydraulic-diagnostic-saas/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2-green.svg)](https://djangoproject.com/)
[![TimescaleDB](https://img.shields.io/badge/TimescaleDB-2.17-orange.svg)](https://www.timescale.com/)

> Передовое решение для мониторинга гидравлических систем с AI-поддержкой, RAG-ассистентом на базе Qwen3 + LangChain и масштабируемой архитектурой на TimescaleDB.

---

## 🎯 Основные возможности

📈 **Мониторинг в реальном времени** - сбор и анализ данных датчиков  
🤖 **AI-диагностика** - автоматическое выявление аномалий  
💬 **RAG Assistant** - интеллектуальный помощник на базе LLM  
📀 **TimescaleDB** - высокопроизводительное хранение временных рядов  
⚡ **Celery** - асинхронная обработка данных  
📱 **Современный UI** - Nuxt 3 + Vue 3 + Chart.js

---

## 🛠️ Технологический стек

| Компонент | Технология | Версия |
|----------|-------------|--------|
| **Backend** | Django + DRF | 5.2+ |
| **Database** | TimescaleDB + PostgreSQL | 2.17 + 16+ |
| **Cache** | Redis | 7.0+ |
| **Task Queue** | Celery + Redis | 5.4+ |
| **AI/LLM** | Ollama + Qwen3 | Локально |
| **RAG** | LangChain + FAISS | 0.3+ |
| **WebSockets** | Django Channels | 4.1+ |
| **Frontend** | Nuxt 3 + Vue 3 | 3.0+ |
| **Стили** | Tailwind CSS | 3.4+ |
| **CI/CD** | GitHub Actions | - |

---

## 🚀 Быстрый старт

### 1. Клонирование и настройка
```bash
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas
cp .env.example .env  # Отредактируйте под свои нужды
```

### 2. Запуск через Docker Compose
```bash
# Режим разработки
make dev

# Альтернативно
docker-compose -f docker-compose.dev.yml up -d
```

### 3. Инициализация
```bash
# Миграции БД
make migrate

# Создание админа
make superuser

# Тестовые данные + RAG система
make init-data
```

### 4. Доступк приложению
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000/api/
- **Admin Panel**: http://localhost:8000/admin/
- **API Docs**: http://localhost:8000/api/docs/

---

## 🏗️ Архитектура

Проект построен по **модульной архитектуре** с разделением ответственности:

```
hydraulic-diagnostic-saas/
├── backend/           # Django бэкэнд
│   ├── apps/
│   │   ├── diagnostics/    # Система диагностики
│   │   ├── rag_assistant/  # RAG-ассистент
│   │   └── users/          # Пользователи
│   └── core/           # Настройки Django
├── nuxt_frontend/    # Nuxt 3 фронтэнд
├── docker/           # Docker конфигурация
└── data/             # Локальные данные (FAISS индексы)
```

### 🔍 Ключевые компоненты

- **`apps.diagnostics`** - Мониторинг гидросистем, AI-анализ аномалий
- **`apps.rag_assistant`** - Интеллектуальный помощник на базе Qwen3
- **TimescaleDB** - Оптимизированное хранение и анализ временных рядов
- **FAISS** - Векторная база знаний для семантического поиска

---

## 🔧 Makefile команды

```bash
# 🚀 Разработка
make dev              # Запустить в режиме dev
make logs             # Посмотреть логи
make shell            # Django shell
make migrate          # Миграции
make superuser        # Создать админа
make init-data        # Тестовые данные + RAG

# 🧪 Тестирование
make test             # Все тесты
make test-backend     # Только backend
make test-rag         # Тест RAG системы
make smoke-test       # Smoke тесты

# 🎨 Качество кода
make lint             # Проверка линтером
make format           # Автоформатирование
make check            # Pre-commit проверка

# 🚢 Production
make prod             # Запустить production
make prod-logs        # Production логи
make backup-db        # Бекап БД
```

---

## 🐍 Пример использования API

### 1. Добавить новую гидросистему
```python
import requests

response = requests.post('http://localhost:8000/api/systems/', {
    'name': 'Промышленный пресс #1',
    'system_type': 'industrial',
    'max_pressure': 250.0,
    'location': 'Цех 1'
})
print(response.json())
```

### 2. Отправить данные датчика
```python
sensor_data = {
    'system_id': 1,
    'sensor_type': 'pressure',
    'value': 185.5,
    'unit': 'bar',
    'timestamp': '2025-10-21T00:00:00Z'
}
response = requests.post('http://localhost:8000/api/sensor-data/', sensor_data)
```

### 3. Использовать RAG-ассистента
```python
query = {
    'question': 'Как диагностировать проблемы с давлением?',
    'system_id': 1
}
response = requests.post('http://localhost:8000/api/rag/query/', query)
print(response.json()['answer'])
```

---

## 🧪 Тестирование

Проект использует **pytest** для backend и **Vitest** для frontend.

```bash
# Запуск всех тестов
make test

# Тесты с покрытием
pytest --cov=apps --cov-report=html

# Специальные тесты
python smoke_diagnostics.py  # Smoke тесты
python test_rag.py           # Тест RAG
```

Подробности в [TESTING.md](TESTING.md)

---

## 🛡️ Безопасность и качество

Проект следует **лучшим практикам** разработки:

✅ **SQL Injection защита** - параметризованные запросы  
✅ **Pre-commit хуки** - автоматическая проверка кода  
✅ **GitHub Actions CI** - автоматические тесты  
✅ **Type Hints** - поддержка mypy  
✅ **Code Coverage** - покрытие тестами > 80%

```bash
# Проверка безопасности
bandit -r backend/ -c .bandit

# Линтинг
flake8 backend/
isort --check backend/
black --check backend/

# Pre-commit проверка
pre-commit run --all-files
```

---

## 📊 Особенности TimescaleDB

Проект оптимизирован для работы с **большими объемами временных данных**:

- **Автоматические гипертаблицы** для `SensorData`
- **Chunk management** - автоматические партиции по 7 дней
- **Compression** - сжатие старых данных
- **Retention policies** - автоочистка по расписанию

Подробности в [backend/BACKEND_ARCHITECTURE_REVIEW.md](backend/BACKEND_ARCHITECTURE_REVIEW.md)

---

## 🤖 AI и RAG возможности

Платформа интегрирована с **локальными LLM**:

- **Qwen3:8b** (через Ollama) для генерации ответов
- **nomic-embed-text** для создания embeddings
- **FAISS** для векторного поиска
- **LangChain** для оркестрации RAG-pipeline

### RAG Пример:
```python
# Создание базы знаний
from apps.rag_assistant.models import RagSystem, Document

rag_system = RagSystem.objects.create(name="Гидросистемы")
Document.objects.create(
    rag_system=rag_system,
    title="Руководство по эксплуатации",
    content="При снижении давления проверьте насос..."
)

# Запрос к ассистенту
response = requests.post('http://localhost:8000/api/rag/query/', {
    'question': 'Почему упало давление?'
})
```

---

## 📝 Документация

- [TESTING.md](TESTING.md) - Инструкции по тестированию
- [backend/BACKEND_ARCHITECTURE_REVIEW.md](backend/BACKEND_ARCHITECTURE_REVIEW.md) - Обзор архитектуры
- [backend/IMPLEMENTATION_GUIDE.md](backend/IMPLEMENTATION_GUIDE.md) - Гайд по разработке
- [RUN_PRECOMMIT_TESTS.md](RUN_PRECOMMIT_TESTS.md) - Pre-commit тестирование

---

## 🤝 Contributing

1. Fork репозитория
2. Создайте feature branch: `git checkout -b feature/new-feature`
3. Коммитьте изменения: `git commit -m 'Add new feature'`
4. Push в branch: `git push origin feature/new-feature`
5. Открывайте Pull Request

### Требования:
- Покрытие тестами > 80%
- Прохождение pre-commit хуков
- Обновление документации

---

## 📄 Лицензия

Проект распространяется под лицензией MIT.

---

## 👨‍💻 Автор

**Плотников Александр**  
📧 shukik85@ya.ru  
🐙 [@Shukik85](https://github.com/Shukik85)

---

**⭐ Star на GitHub, если проект полезен!**