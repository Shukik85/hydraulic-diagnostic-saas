# 🛠️ Hydraulic Diagnostic SaaS

<div align="center">

**Интеллектуальная SaaS-платформа для диагностики и мониторинга гидравлических систем**

_Передовое решение с AI-поддержкой, RAG-ассистентом на базе Qwen3 + LangChain и TimescaleDB_

[![Django](https://img.shields.io/badge/Django-5.2+-darkgreen.svg)](https://docs.djangoproject.com/)
[![TimescaleDB](https://img.shields.io/badge/TimescaleDB-2.17+-orange.svg)](https://www.timescale.com/)
[![Nuxt](https://img.shields.io/badge/Nuxt-4-darkgreen.svg)](https://nuxt.com/)
[![Vue](https://img.shields.io/badge/Vue-3-4FC08D.svg)](https://vuejs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-06B6D4.svg)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com/)

</div>

---

## 📚 Содержание

- [Возможности](#-возможности)
- [Технологический стек](#-технологический-стек)
- [Быстрый старт](#-быстрый-старт)
- [Архитектура проекта](#-архитектура-проекта)
- [API](#-api)
- [Разработка](#-разработка)
- [Контакты](#-контакты)

---

## 🎯 Возможности

### 📈 Мониторинг в реальном времени

- **Сбор данных датчиков** давления, температуры, вибрации
- **Система мониторинга** с иерархией: Системы → Оборудование / Датчики
- **Дашборд** с интерактивными графиками и метриками

### 🤖 AI-диагностика и RAG

- **Автоматическое выявление аномалий** с помощью ML
- **RAG-ассистент** на базе Qwen3 + LangChain
- **Интеллектуальный поиск** в базе знаний с FAISS
- **Предсказание отказов** за недели до проблем

### 💬 Интеграция LLM

- **Ollama + Qwen3:8b** локально без облака
- **Embeddings** через nomic-embed-text
- **Семантический поиск** документов
- **Контекстные ответы** на вопросы о гидросистемах

### ⚡ Высокопроизводительность

- **TimescaleDB** для временных рядов (2.17+)
- **Celery + Redis** асинхронная обработка
- **Django Channels** для WebSocket
- **Кэширование** на Redis
- **Гипертаблицы** с автоматическим сжатием

### 📊 Аналитика и отчёты

- **Отчёты** с аналитикой тенденций
- **Диагностические сессии** с результатами
- **История событий** датчиков
- **Экспорт данных** в формате

### ⚙️ Управление пользователями

- **Профиль** с настройками
- **Уведомления** email/push/in-app
- **Двухфакторная аутентификация** (2FA)
- **API ключи** и Webhooks
- **Управление биллингом**

---

## 🛠 Технологический стек

### Backend

| Компонент                 | Версия | Назначение         |
| ------------------------- | ------ | ------------------ |
| **Django**                | 5.2+   | Web фреймворк      |
| **Django REST Framework** | 3.14+  | REST API           |
| **TimescaleDB**           | 2.17+  | Временные ряды     |
| **PostgreSQL**            | 16+    | Основная БД        |
| **Redis**                 | 7.0+   | Кэш и очереди      |
| **Celery**                | 5.4+   | Асинхронные задачи |
| **Django Channels**       | 4.1+   | WebSocket          |
| **Ollama**                | Latest | Локальные LLM      |
| **LangChain**             | 0.3+   | RAG pipeline       |
| **FAISS**                 | 1.7+   | Векторный поиск    |

### Frontend

| Компонент        | Версия | Назначение            |
| ---------------- | ------ | --------------------- |
| **Nuxt**         | 4.x    | Full-stack фреймворк  |
| **Vue**          | 3.x    | Реактивные компоненты |
| **Tailwind CSS** | 3.4+   | Стилинг               |
| **Chart.js**     | 4.x    | Графики и диаграммы   |
| **TypeScript**   | 5.0+   | Типизация             |
| **Vite**         | 5.x+   | Сборка и HMR          |

### Инфраструктура

- **Docker** & **Docker Compose** — контейнеризация
- **Nginx** — reverse proxy, балансировщик, gzip
- **GitHub Actions** — CI/CD, автоматические тесты
- **Pre-commit hooks** — проверка качества кода

---

## 🚀 Быстрый старт

### 1. Клонирование

```bash
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas
cp backend/.env.example backend/.env
```

### 2. Запуск с Docker Compose

```bash
# Режим разработки
make dev

# Или вручную
docker-compose -f docker-compose.dev.yml up -d
```

### 3. Инициализация

```bash
# Миграции БД
make migrate

# Создание суперпользователя
make superuser

# Тестовые данные + RAG система
make init-data
```

### 4. Доступ к приложению

| Сервис          | URL                         | Учётные данные |
| --------------- | --------------------------- | -------------- |
| **Frontend**    | http://localhost:3000       | Авто           |
| **Backend API** | http://localhost:8000/api   | Swagger docs   |
| **Admin Panel** | http://localhost:8000/admin | Superuser      |
| **Ollama UI**   | http://localhost:11434      | Локально       |

---

## 🏗️ Архитектура проекта

### Структура каталогов

```
hydraulic-diagnostic-saas/
├── backend/                          # Django 5.2 приложение
│   ├── apps/
│   │   ├── diagnostics/             # Мониторинг и диагностика
│   │   │   ├── models.py            # System, Equipment, SensorData
│   │   │   ├── views.py             # API endpoints
│   │   │   ├── services.py          # Бизнес-логика
│   │   │   ├── ai_engine.py         # ML анализ аномалий
│   │   │   ├── websocket_consumers.py # Real-time обновления
│   │   │   └── timescale_tasks.py   # Celery задачи
│   │   ├── rag_assistant/           # RAG система
│   │   │   ├── models.py            # Document, RagSystem
│   │   │   ├── rag_system.py        # Логика RAG
│   │   │   └── views.py             # API /rag/query/
│   │   └── users/                   # Управление пользователями
│   ├── core/                        # Django settings & конфиг
│   │   ├── settings.py              # Настройки проекта
│   │   ├── urls.py                  # Корневые URL
│   │   └── wsgi.py                  # Production WSGI
│   ├── manage.py                    # Django CLI
│   ├── requirements.txt             # Python зависимости
│   ├── pytest.ini                   # Pytest конфиг
│   ├── Makefile                     # Быстрые команды
│   └── tests/                       # Unit и integration тесты
│
├── nuxt_frontend/                    # Nuxt 4 фронтенд
│   ├── pages/                        # File-based routing
│   │   ├── index.vue                # Лендинг (/)
│   │   ├── auth/
│   │   │   ├── login.vue            # /auth/login
│   │   │   └── register.vue         # /auth/register
│   │   ├── dashboard.vue            # /dashboard (главная app)
│   │   ├── chat.vue                 # /chat (ИИ чат)
│   │   ├── diagnostics/
│   │   │   └── index.vue            # /diagnostics
│   │   ├── reports/                 # Отчёты
│   │   │   ├── index.vue            # /reports
│   │   │   └── [reportId]/
│   │   │       ├── index.vue        # /reports/123
│   │   │       └── details.vue      # /reports/123/details
│   │   ├── settings/                # Настройки пользователя
│   │   │   ├── index.vue            # /settings
│   │   │   ├── profile.vue          # /settings/profile
│   │   │   ├── notifications.vue    # /settings/notifications
│   │   │   ├── security.vue         # /settings/security
│   │   │   └── billing.vue          # /settings/billing
│   │   └── systems/                 # ⭐ ГЛАВНЫЙ МОДУЛЬ
│   │       ├── index.vue            # /systems
│   │       └── [systemId]/
│   │           ├── index.vue        # /systems/123 (pill-tabs)
│   │           ├── equipments/
│   │           │   ├── index.vue    # /systems/123/equipments
│   │           │   └── [equipmentId].vue # /systems/123/equipments/456
│   │           └── sensors/         # Датчики системы!
│   │               ├── index.vue    # /systems/123/sensors
│   │               └── [sensorId]/
│   │                   ├── index.vue       # /systems/123/sensors/789
│   │                   ├── data.vue        # /systems/123/sensors/789/data
│   │                   ├── calibration.vue # .../calibration
│   │                   └── alerts.vue      # .../alerts
│   ├── layouts/
│   │   ├── default.vue              # Dashboard layout
│   │   ├── landing.vue              # Публичный лендинг
│   │   └── auth.vue                 # Страницы авторизации
│   ├── components/                  # Переиспользуемые компоненты
│   │   ├── ui/                      # UI библиотека
│   │   │   ├── AppNavbar.vue        # Навбар приложения
│   │   │   └── ...
│   │   └── dashboard/               # Dashboard компоненты
│   ├── composables/                 # Vue 3 Composition API
│   ├── stores/                      # Pinia state management
│   ├── plugins/                     # Vue плагины
│   ├── nuxt.config.ts               # Конфиг Nuxt
│   ├── tailwind.config.js           # Tailwind CSS конфиг
│   └── package.json                 # npm зависимости
│
├── docker/                          # Docker конфигурация
│   ├── nginx/                       # Nginx reverse proxy
│   │   └── nginx.conf               # Production конфиг
│   ├── entrypoint.sh                # Инициализация контейнеров
│   └── init-timescale.sql           # TimescaleDB setup
│
├── scripts/                         # Вспомогательные скрипты
├── tools/                           # Утилиты разработки
├── data/indexes/                    # FAISS индексы (локально)
│
├── docker-compose.dev.yml           # Dev конфиг
├── docker-compose.prod.yml          # Production конфиг
├── .github/workflows/
│   ├── ci.yml                       # CI/CD пайплайн
│   └── rag_smoke.yml                # RAG smoke тесты
├── .pre-commit-config.yaml          # Pre-commit хуки
├── Makefile                         # Команды разработки
├── README.md                        # Этот файл
└── LICENSE                          # MIT лицензия
```

### Иерархия навигации (UX flow)

```
🏠 Главная (/)
 ├─ 🔐 Авторизация (/auth/*)
 └─ 📊 Приложение (Dashboard)
     ├─ 🔧 Системы (/systems)
     │   └─ Система #123 (/systems/123) [pill-tabs]
     │       ├─ ⚙️ Оборудование
     │       │   └─ Оборудование #456
     │       └─ 📡 Датчики
     │           └─ Датчик #789
     │               ├─ Данные
     │               ├─ Калибровка
     │               └─ События
     ├─ 🔍 Диагностика (/diagnostics)
     ├─ 📈 Отчёты (/reports)
     ├─ ⚙️ Настройки (/settings/*)
     └─ 💬 ИИ Чат (/chat)
```

**🎨 Ключевые компоненты UI:**

- **Pill-tabs**: Активные разделы (Оборудование | Датчики)
- **Breadcrumbs**: Навигация по иерархии
- **Dark/Light mode**: Полная поддержка Tailwind CSS
- **Responsive дизайн**: Mobile-first, все размеры

---

## 🔌 API

### Основные endpoints

```
GET    /api/systems/                    # Список систем
POST   /api/systems/                    # Создать систему
GET    /api/systems/{id}/               # Детали системы
GET    /api/systems/{id}/equipments/    # Оборудование системы
GET    /api/systems/{id}/sensors/       # Датчики системы
POST   /api/sensor-data/                # Отправить данные датчика
POST   /api/diagnostics/                # Запустить диагностику
GET    /api/diagnostics/{id}/results/   # Результаты
POST   /api/rag/query/                  # RAG запрос к ассистенту
```

### Swagger документация

```
http://localhost:8000/api/schema/swagger-ui/
```

---

## 🔧 Разработка

### Makefile команды

```bash
# 🚀 Запуск
make dev                # Dev сервер
make prod               # Production
make logs               # Логи контейнеров

# 🧪 Тестирование
make test               # Все тесты
make test-backend       # Только backend
make test-rag           # Тест RAG
make smoke-test         # Smoke тесты

# 🎨 Качество кода
make lint               # Проверка линтером
make format             # Автоформатирование
make check              # Pre-commit проверка

# 🛠 Управление
make migrate            # Миграции БД
make superuser          # Создать админа
make init-data          # Тестовые данные + RAG
make shell              # Django shell
```

### Backend разработка

```bash
cd backend

# Установка зависимостей
pip install -r requirements.txt

# Запуск миграций
python manage.py migrate

# Dev сервер
python manage.py runserver

# Создание админа
python manage.py createsuperuser
```

### Frontend разработка

```bash
cd nuxt_frontend

# Установка зависимостей
npm install

# Dev сервер с HMR
npm run dev

# Сборка production
npm run build

# Запуск production
npm run start
```

### Качество кода

```bash
# Backend
cd backend
flake8 .
black --check .
isort --check-only .
mypy .

# Frontend
cd nuxt_frontend
npm run lint
npm run typecheck
```

---

## 🧪 Тестирование

### Запуск тестов

```bash
# Все тесты с покрытием
make test

# Параллельные тесты
pytest -n auto

# Только быстрые
pytest -m "not slow"

# С профилированием
pytest --durations=10
```

Подробности в `TESTING.md`

---

## 🛡️ Безопасность

✅ **SQL Injection защита** — параметризованные запросы  
✅ **Pre-commit хуки** — автоматическая проверка  
✅ **GitHub Actions CI** — автоматические тесты  
✅ **Type Hints** — поддержка mypy  
✅ **Code Coverage** — покрытие > 80%

```bash
# Проверка безопасности
bandit -r backend/ -c .bandit
```

---

## 🤖 AI и RAG система

### Компоненты

- **Qwen3:8b** (Ollama) — локальная LLM для генерации
- **nomic-embed-text** — embeddings
- **FAISS** — векторный поиск в базе знаний
- **LangChain** — оркестрация RAG pipeline

### Пример RAG запроса

```python
import requests

response = requests.post('http://localhost:8000/api/rag/query/', {
    'question': 'Почему упало давление в системе?',
    'system_id': 1
})

print(response.json()['answer'])
```

Подробности в `backend/BACKEND_ARCHITECTURE_REVIEW.md`

---

## 📊 TimescaleDB оптимизация

Проект использует **TimescaleDB** для высокопроизводительного хранения временных рядов:

- **Гипертаблицы** с автоматическими чанками
- **Chunk interval** = 7 дней
- **Compression policy** = 30 дней
- **Retention policy** = 365 дней
- **BRIN индексы** для timestamp

---

## 📈 Дорожная карта

### ✅ Завершено

- ✓ Django 5.2 + TimescaleDB архитектура
- ✓ Nuxt 4 с file-based routing
- ✓ RAG система на Qwen3 + LangChain
- ✓ WebSocket для real-time обновлений
- ✓ Иерархический роутинг систем/оборудования/датчиков
- ✓ Pill-tabs и breadcrumbs навигация
- ✓ Dark/Light theme поддержка
- ✓ Docker контейнеризация
- ✓ CI/CD пайплайн

### 🟡 В процессе

- ML алгоритмы для прогнозирования
- Расширенная аналитика
- Мобильное приложение
- Интеграция с промышленными датчиками

### 🔴 Планируется

- Поддержка разных типов оборудования
- Экспорт отчётов в PDF/Excel
- AI видеоанализ гидросистем
- Microservices архитектура

---

## 🤝 Вклад в проект

```bash
# 1. Fork репозитория
git checkout -b feature/amazing-feature

# 2. Commit с понятным сообщением
git commit -m 'feat: добавить потрясающую фичу'

# 3. Push ветку
git push origin feature/amazing-feature

# 4. Создать Pull Request
```

**Требования к PR:**

- ✅ Code passes lint и typecheck
- ✅ Tests зелёные
- ✅ Conventional commits
- ✅ Документированы API изменения

---

## 📄 Лицензия

MIT License — смотрите `LICENSE` файл

---

## 👨‍💻 Контакты

- 📧 **Email**: a.s.plotnikov85@gmail.com
- 🐙 **GitHub**: [@Shukik85](https://github.com/Shukik85)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Shukik85/hydraulic-diagnostic-saas/discussions)

---

<div align="center">

### ⭐ Star на GitHub, если проект полезен!

**Спасибо за ваш интерес к Hydraulic Diagnostic SaaS!**

</div>
