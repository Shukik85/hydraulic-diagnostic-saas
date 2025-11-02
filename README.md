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

## 🤖 Hybrid Bot Operations System

**Новая система безопасных автоматических операций с контролем критичных изменений:**

### **Smart Auto-Approval (80% операций):**
- ✅ **Documentation updates** (*.md, README, docs/)
- ✅ **Test additions** (test_*.py, *_test.py)
- ✅ **Lint fixes** (ruff/black/prettier changes)
- ✅ **Comments and docstrings**
- ✅ **Dependencies updates** (requirements.txt, package.json)

### **Manual Approval Required (20% операций):**
- ⚠️ **Workflow changes** (.github/workflows/)
- ⚠️ **Database migrations** (Django migrations)
- ⚠️ **File deletions** (любые удаления)
- ⚠️ **Production configs** (docker-compose, .env)
- ⚠️ **Security-sensitive** (токены, ключи, пароли)

### **Как использовать:**

```bash
# 1. Начать сессию разработки (в PR комментариях)
/start-session {"goal": "timescale-ingestion-mvp", "duration": "4h"}

# 2. Одобрить операцию (если требуется)
/approve {"files": [{"path": "workflow.yml", "action": "create"}]}

# 3. Откатить операции
/rollback {"last": 3}

# 4. Статус сессии
/bot-status
```

### **Transparent Audit Trail:**
- 📋 Все операции логируются в `.bot-operations/`
- 🔍 Превью diff'ов перед выполнением критичных изменений
- ↩️ Rollback capability для любых операций
- 📱 Telegram уведомления обо всех действиях

## 📊 CI/CD

**GitHub Actions:**
- `ci-frontend.yml` → ESLint + Prettier + TypeScript
- `ci-backend.yml` → Ruff + Black + Bandit + pytest  
- `notifications.yml` → Telegram уведомления
- `bot-hybrid.yml` → Bot operations с approval

**Validation & Security:**
- **actionlint** — статический анализ GitHub Actions
- **workflow validator** — проверка опасных команд и permissions
- **bot risk classifier** — автоматическая классификация операций
- **pre-commit hooks** — ruff, black, bandit, prettier, actionlint

## 📱 Telegram Notifications

Уведомления в Telegram о статусе разработки:

**Триггеры:**
- Коммиты с префиксом `READY:` 
- PR помечен как `ready_for_review`
- CI падает (`failure`)
- Issues закрыты
- Bot operations (approval required, completed, failed)

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
2. Pre-commit hooks обязательны (ruff, black, bandit, actionlint)
3. PR review для всех изменений
4. "READY:" коммиты для уведомлений о готовности
5. Bot operations для автоматизации рутинных задач

**Bot Operations Workflow:**
1. Безопасные операции выполняются автоматически
2. Критичные операции требуют `/approve` команды
3. Все действия логируются и могут быть откачены
4. Сессии разработки с auto-approval настройками

## 🔒 Security

- Все секреты через GitHub Secrets / .env
- Параметризованные SQL запросы (защита от инъекций)
- Rate limiting на критичных эндпоинтах
- Audit trail для всех операций
- HTTPS + secure cookies в production
- **Bot operations security:**
  - Risk classification для всех автоматических операций
  - Validation опасных команд в workflows
  - Approval gates для критичных изменений
  - Rollback capability с restore points

## 📚 Documentation

- `ROADMAP_INCREMENTAL.md` → план развития платформы
- `DoD_CHECKLISTS.md` → критерии приемки этапов
- `backend/BACKEND_IMPLEMENTATION_PLAN.md` → детальный план backend
- `nuxt_frontend/IMPLEMENTATION_PLAN.md` → план frontend
- **Bot Operations:**
  - `scripts/bot_risk_classifier.py` → алгоритм классификации операций
  - `scripts/bot_session_manager.py` → управление сессиями разработки
  - `scripts/validate_workflows.py` → валидация GitHub Actions

---

## 🧪 Quick Test (Bot & CI)

1. В PR комментариях:
```
/start-session {"goal":"hybrid-demo","duration":"1h"}
```
2. Статус:
```
/bot-status
```
Если бот молчит:
- Убедитесь, что PR не draft (нажмите Ready for review)
- Settings → Actions → Workflow permissions: Read and write + Allow approvals

Diagnostics:
- Откройте вкладку Actions
- Откройте последний run нужного workflow
