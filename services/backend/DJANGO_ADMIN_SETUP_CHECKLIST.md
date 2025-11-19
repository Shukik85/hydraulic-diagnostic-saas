# Django Admin - Финальный Чеклист Настройки

## ✅ Уже выполнено

- [x] Создана система документации (`apps/docs`)
- [x] Настроены admin интерфейсы для всех приложений
- [x] Добавлен custom дизайн (metallic/teal theme)
- [x] Реализован биллинг через Stripe
- [x] Исправлены critical файлы с type hints
- [x] Создан скрипт автоисправления (`fix_ruff_errors.py`)
- [x] Написана документация по исправлению ошибок (`RUFF_FIXES.md`)

## 📝 Требуется выполнить

### 1. Добавить docs в INSTALLED_APPS

**Файл:** `services/backend/config/settings.py`

```python
INSTALLED_APPS = [
    # Django core
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party
    "rest_framework",
    "rest_framework_simplejwt",
    "corsheaders",
    "drf_spectacular",
    "django_celery_beat",
    "django_celery_results",
    "django_prometheus",
    # Local apps
    "apps.core",
    "apps.users",
    "apps.subscriptions",
    "apps.equipment",
    "apps.notifications",
    "apps.monitoring",
    "apps.support",
    "apps.docs",  # ← ДОБАВИТЬ
]
```

### 2. Подключить URLs документации

**Файл:** `services/backend/config/urls.py`

```python
urlpatterns = [
    path('admin/', admin.site.urls),
    path('admin/docs/', include('apps.docs.urls')),  # ← ДОБАВИТЬ
    path('api/support/', include('apps.support.urls')),
    path('health/', include('apps.monitoring.urls')),
]
```

### 3. Исправить ошибки ruff

```bash
cd services/backend

# 1. Автоисправления
ruff check . --fix

# 2. Исправить models.py
python fix_ruff_errors.py

# 3. Форматирование
ruff format .
```

**Оставшиеся ручные исправления:**
- `apps/support/admin.py` - добавить ClassVar
- `apps/users/admin.py` - добавить ClassVar  
- `apps/equipment/admin.py` - добавить ClassVar
- `apps/support/models.py` - убрать null=True с CharField
- `apps/users/models.py` - убрать null=True с CharField
- `apps/support/tasks.py` - переместить imports наверх

См. подробности в `RUFF_FIXES.md`

### 4. Создать миграции для docs

```bash
python manage.py makemigrations docs
python manage.py migrate docs
```

### 5. Загрузить начальные данные

```bash
python manage.py loaddata apps/docs/fixtures/initial_docs.json
```

### 6. Собрать статические файлы

```bash
python manage.py collectstatic --noinput --clear
```

### 7. Обновить .env файл

**Файл:** `services/backend/.env`

```bash
# Сгенерировать SECRET_KEY
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"

# Основные настройки
DJANGO_SECRET_KEY=<сгенерированный_ключ>
DEBUG=False
ALLOWED_HOSTS=yourdomain.com

# Database
DATABASE_PASSWORD=<сильный_пароль>

# Redis
REDIS_PASSWORD=<сильный_пароль>

# Stripe (live keys)
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Email
EMAIL_HOST_PASSWORD=<api_key>

# Sentry
SENTRY_DSN=https://...@sentry.io/...
```

### 8. Добавить rate limiting middleware

**Файл:** `services/backend/apps/core/middleware.py` (создать)

```python
from django.core.cache import cache
from django.http import JsonResponse

class RateLimitMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            key = f"rate_limit_{request.user.id}"
            limit = 1000
        else:
            key = f"rate_limit_{self.get_client_ip(request)}"
            limit = 100
        
        count = cache.get(key, 0)
        if count >= limit:
            return JsonResponse({"error": "Rate limit exceeded"}, status=429)
        
        cache.set(key, count + 1, 3600)
        return self.get_response(request)

    @staticmethod
    def get_client_ip(request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
```

**Добавить в settings.py:**

```python
MIDDLEWARE = [
    "django_prometheus.middleware.PrometheusBeforeMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "apps.core.middleware.RateLimitMiddleware",  # ← ДОБАВИТЬ
    # ... остальные
]
```

### 9. Обновить settings.py

**Файл:** `services/backend/config/settings.py`

Добавить после REST_FRAMEWORK:

```python
# Rate Limiting
REST_FRAMEWORK = {
    # ... существующие настройки ...
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "100/hour",
        "user": "1000/hour",
    },
}

# Celery Logging
CELERY_WORKER_LOG_FORMAT = "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
CELERY_WORKER_TASK_LOG_FORMAT = "[%(asctime)s: %(levelname)s/%(processName)s] [%(task_name)s(%(task_id)s)] %(message)s"

# Frontend URL
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
```

### 10. Создать superuser

```bash
python manage.py createsuperuser
```

### 11. Запустить тесты

```bash
# Проверка кода
ruff check .
ruff format .
mypy apps/ config/

# Проверка миграций
python manage.py makemigrations --check --dry-run

# Проверка безопасности
python manage.py check --deploy

# Тесты (если есть)
pytest --cov=apps
```

### 12. Тестовый запуск

```bash
# Запустить сервер
python manage.py runserver

# Или через Docker
docker-compose up backend

# Проверить админку
curl http://localhost:8000/admin/

# Проверить healthcheck
curl http://localhost:8000/health/

# Проверить документацию
curl http://localhost:8000/admin/docs/
```

## 🚀 Production Deployment

### Pre-deployment checklist

- [ ] Все ошибки ruff исправлены
- [ ] Миграции применены
- [ ] Статика собрана
- [ ] `.env` настроен с prod значениями
- [ ] Superuser создан
- [ ] Rate limiting включен
- [ ] Sentry настроен
- [ ] Тесты прошли
- [ ] Security check прошёл
- [ ] Backup БД настроен

### Docker deployment

```bash
# Build
docker-compose build backend

# Run migrations
docker-compose run --rm backend python manage.py migrate

# Collect static
docker-compose run --rm backend python manage.py collectstatic --noinput

# Create superuser
docker-compose run --rm backend python manage.py createsuperuser

# Start services
docker-compose up -d backend celery celery-beat redis postgres

# Check logs
docker-compose logs -f backend
```

## 📊 Мониторинг

### Endpoints для проверки

- Admin: http://localhost:8000/admin/
- Docs: http://localhost:8000/admin/docs/
- Health: http://localhost:8000/health/
- Metrics: http://localhost:8000/metrics
- API: http://localhost:8000/api/

### Grafana Dashboards

- http://localhost:3001 (Grafana)
- http://localhost:9090 (Prometheus)

Дефолтный логин: `admin` / пароль из `GRAFANA_PASSWORD` env

## 🆘 Troubleshooting

### Проблема: Миграции не применяются

```bash
python manage.py showmigrations
python manage.py migrate --fake-initial
```

### Проблема: Статика не загружается

```bash
python manage.py collectstatic --clear --noinput
ls -la staticfiles/
```

### Проблема: Ошибки импорта

```bash
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name '*.pyc' -delete
pip install --force-reinstall -r requirements.txt
```

### Проблема: База данных не подключается

```bash
# Проверить PostgreSQL
psql -h localhost -U postgres -d hydraulic_db

# Проверить настройки в .env
cat .env | grep DATABASE
```

## 📚 Документация

- [Backend README](README.md)
- [Ruff Fixes Guide](RUFF_FIXES.md)
- [API Documentation](http://localhost:8000/api/docs/)
- [Django Admin Docs](http://localhost:8000/admin/docs/)

## ✅ Финальная проверка

После выполнения всех шагов:

```bash
# 1. Код чистый
ruff check .  # Должно быть: All checks passed!

# 2. Админка работает
curl -I http://localhost:8000/admin/  # Должно быть: 200 OK

# 3. Документация доступна
curl -I http://localhost:8000/admin/docs/  # Должно быть: 200 OK

# 4. Health check работает
curl http://localhost:8000/health/  # Должно быть: {"status":"ok"}
```

## 🎉 Готово!

После завершения всех пунктов:

1. Закоммитить изменения:
   ```bash
   git add .
   git commit -m "feat: Complete Django Admin setup with docs and fixes"
   git push origin feature/django-admin-docs-app
   ```

2. Создать Pull Request в `master`

3. После ревью и мерджа - задеплоить на production

---

**Версия:** 1.0.0  
**Дата:** 2025-11-16  
**Автор:** Backend Team
