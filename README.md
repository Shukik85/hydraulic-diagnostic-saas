# Hydraulic Diagnostic SaaS

Enterprise SaaS платформа для диагностики гидравлических систем с машинным обучением.

## 🎯 Текущий MVP: Anomaly Detection (14 дней)

**Цель:** End-to-end поток обнаружения аномалий в гидравлических системах:
- **Data Ingestion** → TimescaleDB hypertables + retention + compression  
- **ML Inference** → 4 модели (RandomForest, XGBoost, HELM, Adaptive) + ансамбль
- **Backend API** → DRF endpoints для аномалий, трендов, анализа
- **Frontend UI** → компоненты визуализации аномалий + i18n RU/EN

**Целевые метрики:**
- Accuracy ≥ 99.5% (на UC Irvine dataset)
- Inference latency p90 < 100ms  
- False Positive Rate < 10% (на прод-валидации)

## 🏗️ Архитектура

**Frontend:** Nuxt 4 + Tailwind v4 + Premium UI tokens
**Backend:** Django + DRF + TimescaleDB + Celery + Redis
**ML Stack:** FastAPI + scikit-learn + XGBoost + joblib
**Observability:** Prometheus + Grafana + структурированные логи

## 🚀 Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt -r requirements-dev.txt
python manage.py migrate
python manage.py runserver

# Frontend
cd nuxt_frontend
npm install
npm run dev

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## 📊 CI/CD

**GitHub Actions:**
- `ci-frontend.yml` → ESLint + Prettier + TypeScript
- `ci-backend.yml` → Ruff + Black + Bandit + pytest  
- `notifications.yml` → Telegram уведомления

**Линтеры:**
- **Python:** Ruff (вместо flake8), Black, Bandit, pip-audit
- **Frontend:** ESLint + Prettier
- **Общее:** pre-commit hooks, Hadolint (Docker)

## 📱 Telegram Notifications

Уведомления в Telegram о статусе разработки:

**Триггеры:**
- Коммиты с префиксом `READY:` 
- PR помечен как `ready_for_review`
- CI падает (`failure`)
- Issues закрыты

**Setup:**
1. Создать бота через @BotFather
2. Добавить в GitHub Secrets:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. Коммиты с `READY:` автоматически отправят уведомление

**Отключить уведомления:** добавить label `[no-notify]` в PR

**Пример коммита:**
```bash
git commit -m "READY: TimescaleDB ingestion completed, tests green"
```

## 📋 Development Workflow

**Ветки:**
- `main` → production-ready код
- `chore/lint-fixes-ci-green` → активная разработка  
- Feature branches → по задачам

**Процесс:**
1. Атомарные коммиты с информативными сообщениями
2. Pre-commit hooks обязательны (ruff, black, bandit)
3. PR review для всех изменений
4. "READY:" коммиты для уведомлений о готовности

## 🔒 Security

- Все секреты через GitHub Secrets / .env
- Параметризованные SQL запросы (защита от инъекций)
- Rate limiting на критичных эндпоинтах
- Audit trail для всех операций
- HTTPS + secure cookies в production

## 📚 Documentation

- `ROADMAP_INCREMENTAL.md` → план развития платформы
- `DoD_CHECKLISTS.md` → критерии приемки этапов
- `backend/BACKEND_IMPLEMENTATION_PLAN.md` → детальный план backend
- `nuxt_frontend/IMPLEMENTATION_PLAN.md` → план frontend

## 🤝 Contributing

1. Установить pre-commit: `pre-commit install`
2. Следовать архитектурным принципам (инкрементальность, совместимость)
3. Покрывать изменения тестами
4. Обновлять документацию при изменении контрактов

---

**Enterprise Features Roadmap:**
- 100+ датчиков, 20+ промышленных протоколов
- Predictive maintenance + RUL (Remaining Useful Life)
- Advanced reporting + compliance
- Multi-tenant SaaS + enterprise integrations
- 99.9% SLA + horizontal scaling