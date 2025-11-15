# CI/CD Pipeline Documentation

## 🚀 Overview

Комплексный CI/CD pipeline для автоматизации проверок качества кода, тестирования и деплоя.

## 📊 Pipeline Stages

### Stage 1: Code Quality (⚡ ~3 min)

**Параллельные проверки**:
- **Ruff Linter**: Быстрая проверка стиля кода
- **Ruff Formatter**: Проверка форматирования
- **MyPy**: Статическая проверка типов

**Quality Gates**:
- ✅ No linting errors
- ✅ Code properly formatted
- ✅ Type hints correct

### Stage 2: Security (🔒 ~2 min)

**Проверки безопасности**:
- **Bandit**: Сканирование кода на уязвимости
- **Safety**: Проверка зависимостей на CVE
- **pip-audit**: Аудит пакетов

**Outputs**:
- JSON reports в artifacts
- Комментарии в PR с найденными проблемами

### Stage 3: Backend Tests (🧪 ~5-7 min)

**Окружение**:
- PostgreSQL 16
- Redis 7
- Python 3.14

**Тесты**:
- Unit tests
- Integration tests
- Coverage >= 85%

**Outputs**:
- Coverage report в Codecov
- XML coverage для анализа

### Stage 4: Frontend Tests (⚙️ ~3-4 min)

**Проверки**:
- ESLint
- TypeScript type checking
- Unit tests (Vitest)
- Build verification

### Stage 5: Docker Build (🐳 ~4-6 min)

**Условия**: Only on push to main branches

**Процесс**:
1. Build multi-arch image (amd64, arm64)
2. Push to GitHub Container Registry
3. Tag: branch name + SHA
4. Cache layers for speed

### Stage 6: Performance (📊 ~5 min)

**Условия**: Only on Pull Requests

**Benchmarks**:
- API response time
- Database query performance
- Memory usage

**Outputs**:
- Performance comparison vs base branch
- Comment in PR with results

### Stage 7: Deploy Staging (🚀 ~3 min)

**Условия**: Only on push to `staging` branch

**Процесс**:
1. SSH to staging server
2. Pull latest images
3. Run migrations
4. Restart services
5. Health check

**Environments**:
- `staging`: https://staging.hydraulic-diagnostics.com

### Stage 8: Notifications (📢 ~1 min)

**Каналы**:
- GitHub PR comments
- Slack/Discord (опционально)
- Email (on failures)

---

## 🛠️ Local Testing

### Запуск всех проверок локально

```bash
# В директории services/backend

# 1. Code quality
ruff check .
ruff format --check .
mypy apps/ config/

# 2. Security
bandit -r apps/ config/
safety check

# 3. Tests
pytest --cov=apps --cov-report=term-missing

# 4. Всё сразу
pre-commit run --all-files
```

### Используя Docker Compose

```bash
# Запустить тесты в изоляции
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# С coverage
docker-compose -f docker-compose.test.yml run backend pytest --cov
```

---

## 🐛 Troubleshooting

### Pipeline fails on "Code Quality"

**Причина**: Код не соответствует стандартам

**Решение**:
```bash
# Auto-fix большинства проблем
ruff check --fix .
ruff format .

# Проверить типы
mypy apps/ --show-error-codes
```

### Pipeline fails on "Security"

**Причина**: Найдены уязвимости

**Решение**:
```bash
# Обновить зависимости
pip install --upgrade -r requirements.txt

# Проверить конкретные пакеты
pip-audit --desc

# Игнорировать false-positives (только если уверены!)
bandit -r apps/ -ll  # Только high + medium
```

### Pipeline fails on "Backend Tests"

**Причина**: Тесты падают или coverage < 85%

**Решение**:
```bash
# Запустить тесты локально
pytest -v --tb=short

# Показать непокрытый код
pytest --cov=apps --cov-report=html
open htmlcov/index.html

# Запустить только упавшие тесты
pytest --lf
```

### Docker build fails

**Причина**: Проблемы с dependencies или Dockerfile

**Решение**:
```bash
# Локальная сборка с выводом
cd services/backend
docker build -t test-backend . --progress=plain

# Проверить слои
docker history test-backend

# Проверить requirements
pip-compile requirements.txt --upgrade
```

### Deployment fails

**Причина**: SSH ключ, migrations, или health check

**Проверить**:
1. Секреты в GitHub Settings → Secrets
2. SSH доступ: `ssh deploy@staging-host`
3. Migrations: `python manage.py showmigrations`
4. Health: `curl https://staging.../api/health/`

---

## 📊 Metrics & Monitoring

### GitHub Actions Dashboard

**Просмотр**: Repository → Actions

**Метрики**:
- Success rate
- Average runtime
- Failure trends

### Coverage Trends

**Codecov Dashboard**: https://codecov.io/gh/Shukik85/hydraulic-diagnostic-saas

**Цели**:
- Overall: >= 85%
- New code: >= 90%
- Critical paths: 100%

### Performance Benchmarks

**Отслеживаемые метрики**:
- API response time (p50, p95, p99)
- Database query count
- Memory usage
- Docker image size

---

## ⚙️ Configuration

### Required Secrets

**В GitHub Settings → Secrets → Actions**:

```bash
# Staging deployment
STAGING_DEPLOY_KEY=<SSH private key>
STAGING_HOST=staging.hydraulic-diagnostics.com

# Docker registry (already configured via GITHUB_TOKEN)
# GHCR_TOKEN=${{ secrets.GITHUB_TOKEN }}

# Notifications (optional)
SLACK_WEBHOOK=<Slack incoming webhook URL>
DISCORD_WEBHOOK=<Discord webhook URL>

# External services
CODECOV_TOKEN=<Codecov upload token>
```

### Branch Protection Rules

**Для `master` и `staging`**:

```yaml
Required status checks:
  - code-quality (ruff-lint)
  - code-quality (ruff-format)
  - code-quality (mypy)
  - security
  - backend-tests
  - frontend-tests

Require pull request reviews: 1
Dismiss stale reviews: true
Require review from Code Owners: true
Restrict pushes: admins only
```

---

## 📝 Best Practices

### Before Creating PR

1. **Run checks locally**:
   ```bash
   pre-commit run --all-files
   pytest
   ```

2. **Update tests** if adding new features

3. **Check coverage**:
   ```bash
   pytest --cov --cov-report=term-missing
   ```

4. **Write clear commit messages**:
   ```
   🐛 fix: Resolve race condition in API key generation
   
   - Add database-level unique constraint
   - Implement retry logic in save method
   - Add integration test for concurrent requests
   
   Fixes: #123
   ```

### During Code Review

- Wait for all checks to pass (✅)
- Address reviewer comments
- Keep PR focused (< 500 lines)
- Update docs if needed

### After Merge

- Delete feature branch
- Monitor staging deployment
- Check error tracking (Sentry)
- Verify metrics (Codecov, performance)

---

## 🚀 Deployment Process

### Staging Deployment

**Trigger**: Push to `staging` branch

```bash
# Merge PR to staging
git checkout staging
git merge master
git push origin staging

# Watch deployment
# GitHub Actions → Deploy to Staging

# Verify
curl https://staging.hydraulic-diagnostics.com/api/health/
```

### Production Deployment

**Trigger**: Manual (via GitHub Actions)

```bash
# Create release
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin v1.2.0

# Trigger workflow
# GitHub Actions → Deploy to Production (manual)

# Monitor
# - Application logs
# - Error rates (Sentry)
# - Performance metrics
```

---

## 📚 Related Documentation

- [Pre-commit Hooks](../services/backend/.pre-commit-config.yaml)
- [Docker Build](../services/backend/Dockerfile)
- [Testing Guide](./TESTING.md) (if exists)
- [Deployment Guide](./DEPLOYMENT.md) (if exists)

---

**✨ CI/CD Pipeline maintained by @Shukik85**
