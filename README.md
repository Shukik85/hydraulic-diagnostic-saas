# Hydraulic Diagnostic SaaS Platform

**Enterprise гидравлическая диагностика с ML-аналитикой**

## 🎯 Обзор Проекта

Платформа для real-time мониторинга и аномалий в гидравлических системах с использованием machine learning.

### Текущий Статус (ноябрь 2025):

**✅ РАБОТАЕТ:**
- **Frontend**: Nuxt 4 + Tailwind v4, полная RU/EN локализация
- **ML Service**: CatBoost модель, FastAPI, Redis кеширование
- **Infrastructure**: TimescaleDB, Docker Compose, Celery
- **Testing**: UCI hydraulic тесты (100% success rate)

**⚠️ В РАЗРАБОТКЕ:**
- **Sensor Ingestion API**: Modbus, OPC UA протоколы
- **TimescaleDB Integration**: Hypertables, compression
- **Real-time Dashboard**: WebSocket, графики, alerts
- **DRF API**: связка Django с ML сервисом

**❌ НЕ РЕАЛИЗОВАНО:**
- **99.99% AUC** - маркетинговая метрика
- **4 ML модели** - только CatBoost реально работает
- **<100ms latency** - фактически ~1100ms
- **Production Monitoring** - базовые health checks

## 🏗️ Архитектура

### Frontend (Nuxt 4)
- **✅ UI Framework**: Nuxt 4 + Tailwind v4
- **✅ Локализация**: Полная RU/EN поддержка
- **✅ Dashboard**: Responsive, mobile-friendly
- **⚠️ Real-time**: WebSocket в разработке

### Backend (Django DRF)
- **✅ Framework**: Django + DRF
- **✅ Database**: PostgreSQL/TimescaleDB готов
- **✅ Caching**: Redis
- **✅ Tasks**: Celery
- **⚠️ API**: базовые endpoints

### ML Service (FastAPI)
- **✅ Model**: CatBoost аномалий detection
- **✅ API**: FastAPI async
- **✅ Caching**: Redis TTL 5мин
- **❌ Ensemble**: только 1 модель

### Infrastructure
- **✅ Containerization**: Docker Compose
- **✅ Database**: TimescaleDB 2.15
- **✅ Monitoring**: базовые health checks
- **⚠️ Production**: требует доработки

## 🚀 Быстрый старт

### Предварительные требования
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+
- 8GB RAM минимум

### Запуск Development
```bash
# 1. Клонирование
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas

# 2. Конфигурация
cp .env.example .env
# Отредактировать по необходимости

# 3. Запуск сервисов
docker-compose -f docker-compose.dev.yml up -d

# 4. Проверка
curl http://localhost:8000/health  # Django
curl http://localhost:8001/health  # ML Service
curl http://localhost:3000         # Nuxt Frontend
```

### Полезные команды
```bash
# Логи сервисов
docker-compose logs -f backend
docker-compose logs -f ml_service

# Миграции Django
docker-compose exec backend python manage.py migrate

# Тесты ML
cd ml_service && python scripts/push_to_api.py
```

## 📊 Производительность

**Фактические показатели:**
- **ML Latency**: ~1100ms p50 (цель: <100ms)
- **Success Rate**: 100% на UCI тестах
- **Models**: 1 CatBoost (вместо заявленных 4)
- **Cache Hit**: 90%+ после прогрева

## 🎯 Roadmap до Production (15 ноября)

### Критические Задачи (9 дней):

**Дни 1-2 (6-7 ноября):**
- ✅ TimescaleDB hypertables + compression
- ✅ Django модели для sensor data
- ✅ Retention policy (5 лет)

**Дни 3-4 (8-9 ноября):**
- ⚠️ Ingestion API (Modbus, OPC UA MVP)
- ⚠️ Validation + quarantine pipeline
- ⚠️ DRF endpoints для sensor data

**Дни 5-8 (10-13 ноября):**
- ❌ E2E pipeline: данные → ML → API → UI
- ❌ WebSocket real-time alerts
- ❌ Оптимизация latency (<50ms p95)

**День 9 (14 ноября):**
- ❌ Production health/readiness checks
- ❌ Prometheus + Grafana monitoring
- ❌ Security hardening

### Опционально:
- XGBoost/RandomForest реализация
- A/B тестирование моделей
- Advanced reporting system

## 🚨 Известные Проблемы

1. **Производительность ML**: Latency 1100ms вместо <100ms
2. **Единственная Модель**: Только CatBoost реально работает
3. **Sensor Integration**: Нет реальных протоколов Modbus/OPC UA
4. **Real-time UI**: WebSocket не реализован
5. **Monitoring**: Базовые health checks, нет SLA метрик

## 📚 Документация

- [Development Quickstart](DEVELOPMENT_QUICKSTART.md)
- [Windows Setup Guide](WINDOWS_SETUP.md)
- [DoD Checklists](DoD_CHECKLISTS.md)
- [ML Service README](ml_service/README.md)
- [Incremental Roadmap](ROADMAP_INCREMENTAL.md)

## 🔧 Технологии

**Frontend:**
- Nuxt 4, Vue 3, Tailwind CSS v4
- TypeScript, i18n (RU/EN)
- WebSocket (в разработке)

**Backend:**
- Django 5.0, DRF, Celery
- PostgreSQL + TimescaleDB 2.15
- Redis, Docker

**ML Service:**
- FastAPI, CatBoost
- Pydantic, structlog
- Redis caching

**Infrastructure:**
- Docker Compose
- Prometheus (планы)
- GitOps (планы)

## 👥 Контакты

**Разработчик:** Plotnikov Aleksandr  
**Email:** shukik85@ya.ru  
**GitHub:** [@Shukik85](https://github.com/Shukik85)  

---

**⚠️ Отказ от ответственности:**

Данный README отражает **ФАКТИЧЕСКОЕ состояние** проекта на 5 ноября 2025. 

Маркетинговые заявления о 99.99% AUC и 4 ML моделях **не соответствуют реализации**.
