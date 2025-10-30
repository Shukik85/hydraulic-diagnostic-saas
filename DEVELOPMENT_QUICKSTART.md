# 🚀 Hydraulic Diagnostic SaaS - Quick Start Guide

## Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Node.js 18+ (для фронтенда)
- Git

## 🏃‍♂️ Быстрый старт (1 команда)

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas

# 2. Скопируйте переменные окружения
cp .env.example .env

# 3. Запустите всё одной командой
docker compose up --build
```

**Готово!** Проект будет доступен:
- Backend API: http://localhost:8000
- Admin: http://localhost:8000/admin
- API Docs: http://localhost:8000/api/docs/
- Health Check: http://localhost:8000/health/

## 🔧 Development Setup

### Backend разработка

```bash
# Создать виртуальное окружение
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установить зависимости
pip install -r requirements.txt -r requirements-dev.txt

# Запустить миграции
python manage.py migrate

# Создать суперпользователя
python manage.py createsuperuser

# Запустить dev сервер
python manage.py runserver
```

### Frontend разработка

```bash
cd nuxt_frontend
npm install
npm run dev
```

### Запуск Celery (для фоновых задач)

```bash
# В отдельном терминале
cd backend
celery -A core worker -l info

# В другом терминале для periodic tasks
celery -A core beat -l info
```

## 🧪 Тестирование

```bash
# Все тесты
cd backend
pytest

# Smoke тесты
python smoke_diagnostics.py

# Тесты с coverage
pytest --cov=apps --cov-report=html

# Конкретные тесты
pytest apps/users/tests/
```

## 📋 Pre-commit хуки

```bash
# Установить pre-commit хуки
pre-commit install

# Запустить проверки вручную
pre-commit run --all-files

# Обновить хуки
pre-commit autoupdate
```

## 🐛 Troubleshooting

### Проблема с Docker
```bash
# Очистить Docker кеш
docker system prune -a

# Пересобрать контейнеры
docker compose down -v
docker compose up --build
```

### Проблема с правами доступа (Linux)
```bash
# Исправить права на файлы
sudo chown -R $USER:$USER .
find . -name "*.py" -exec chmod 644 {} \;
find . -name "manage.py" -exec chmod 755 {} \;
```

### Проблема с базой данных
```bash
# Сбросить базу данных
docker compose down -v
docker volume rm hydraulic-diagnostic-saas_pgdata
docker compose up --build
```

### Отладка Celery
```bash
# Проверить статус Redis
docker exec -it hdx-redis redis-cli ping

# Проверить очереди Celery
docker exec -it hdx-celery celery -A core inspect active
```

## 📊 Мониторинг и логи

```bash
# Логи всех сервисов
docker compose logs -f

# Логи конкретного сервиса
docker compose logs -f backend
docker compose logs -f celery

# Проверка health checks
curl http://localhost:8000/health/
```

## 🔐 Безопасность (Dev)

**⚠️ Важно для разработки:**

1. **.env** файл **НЕ коммитится** в git
2. Используйте strong паролях в продакшене
3. Настройте HTTPS для прода
4. Обновляйте зависимости регулярно

```bash
# Проверка безопасности зависимостей
pip-audit -r backend/requirements.txt

# Статический анализ безопасности
bandit -r backend/apps/
```

## 📁 Структура проекта

```
.
├── backend/                 # Django API
│   ├── apps/               # Django приложения
│   │   ├── users/         # Аутентификация
│   │   ├── diagnostics/   # Основная логика
│   │   └── rag_assistant/ # AI ассистент
│   ├── core/              # Настройки Django
│   └── tests/             # Тесты
├── nuxt_frontend/          # Nuxt.js фронтенд
├── docker-compose.yml      # Dev окружение
├── .env.example           # Пример переменных
└── ROADMAP_INCREMENTAL.md # План разработки
```

## 🚀 Следующие шаги

1. Изучите [ROADMAP_INCREMENTAL.md](./ROADMAP_INCREMENTAL.md)
2. Проверьте [DoD_CHECKLISTS.md](./DoD_CHECKLISTS.md)
3. Ознакомьтесь с [backend/BACKEND_IMPLEMENTATION_PLAN.md](./backend/BACKEND_IMPLEMENTATION_PLAN.md)

## 📞 Поддержка

Если возникли проблемы:

1. Проверьте логи: `docker compose logs -f`
2. Убедитесь что все переменные в `.env` заданы
3. Проверьте что Docker и Docker Compose установлены
4. Создайте issue в репозитории с полным описанием проблемы
