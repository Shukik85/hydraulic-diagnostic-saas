# Self-Service API - Authentication & Support

Полная реализация self-service функций для пользователей.

## 📦 Что включено

### FastAPI Backend Routers (services/backend_fastapi/app/routers/)
1. **auth.py** - Authentication & Password Reset
   - POST /auth/password-reset-request
   - POST /auth/password-reset-confirm
   - POST /auth/api-key-reset
   - POST /auth/verify-token

2. **support.py** - Customer Support
   - POST /support/tickets
   - GET /support/tickets
   - GET /support/tickets/{id}
   - PATCH /support/tickets/{id}

3. **account.py** - Account Management
   - GET /account/me
   - POST /account/export-data (GDPR)
   - DELETE /account/me

### Celery Tasks (services/backend_fastapi/app/tasks/)
- **email.py** - Email sending tasks
  - send_password_reset_email()
  - send_new_api_key_email()
  - send_support_ticket_notification()

- **data_export.py** - Data export task (GDPR)
  - export_user_data_task()

### Models (services/backend_fastapi/app/models/)
- **support.py** - SupportTicket model
- **data_export.py** - DataExportRequest model

### Email Templates (services/backend_fastapi/app/templates/emails/)
- **password_reset.html** - Password reset email
- **new_api_key.html** - New API key email

### Django Backend (services/backend/apps/support/)
- **models.py** - SupportTicket & DataExportRequest (synced with FastAPI)
- **admin.py** - Django Admin interface with badges

---

## 🚀 Установка

### 1. Распаковать архив
```bash
cd /h/hydraulic-diagnostic-saas
unzip self_service_api.zip
```

### 2. Установить зависимости
```bash
cd services/backend_fastapi
pip install celery redis boto3  # Если ещё не установлены
```

### 3. Настроить .env
```bash
# Email Configuration
EMAIL_HOST=smtp.sendgrid.net
EMAIL_PORT=587
EMAIL_HOST_USER=apikey
EMAIL_HOST_PASSWORD=your-sendgrid-api-key
DEFAULT_FROM_EMAIL=noreply@hydraulic-diagnostics.com
SUPPORT_EMAIL=support@hydraulic-diagnostics.com

# Celery (Redis)
REDIS_URL=redis://redis:6379/0
```

### 4. Выполнить миграции
```bash
# FastAPI (Alembic)
cd services/backend_fastapi
alembic revision --autogenerate -m "Add support tables"
alembic upgrade head

# Django
cd services/backend
python manage.py makemigrations
python manage.py migrate
```

### 5. Запустить Celery
```bash
# В отдельном терминале
celery -A app.celery_app worker -l info

# Celery Beat (если нужны периодические задачи)
celery -A app.celery_app beat -l info
```

---

## 📝 Использование

### Password Reset Flow

**Frontend:**
```typescript
// 1. Запрос на сброс пароля
await $fetch('/api/auth/password-reset-request', {
  method: 'POST',
  body: { email: 'user@example.com' }
})

// 2. Пользователь получает email с токеном
// 3. Подтверждение с новым паролем
await $fetch('/api/auth/password-reset-confirm', {
  method: 'POST',
  body: {
    token: 'reset-token-from-email',
    new_password: 'new-secure-password'
  }
})
```

### API Key Reset

```typescript
await $fetch('/api/auth/api-key-reset', {
  method: 'POST',
  headers: {
    Authorization: `Bearer ${accessToken}`
  }
})
// Новый ключ будет отправлен на email
```

### Support Ticket

```typescript
const ticket = await $fetch('/api/support/tickets', {
  method: 'POST',
  headers: { Authorization: `Bearer ${accessToken}` },
  body: {
    subject: 'Need help with API',
    message: 'I cannot connect to the API...',
    priority: 'high'
  }
})
```

### Data Export (GDPR)

```typescript
await $fetch('/api/account/export-data', {
  method: 'POST',
  headers: { Authorization: `Bearer ${accessToken}` }
})
// Download link придёт на email через ~30-60 минут
```

---

## 🎨 Django Admin

После установки Support Tickets доступны в Django Admin:

```
http://localhost:8000/admin/support/supportticket/
```

**Возможности:**
- Просмотр всех тикетов
- Фильтрация по status, priority
- Ответ на тикеты (response field)
- Назначение оператору (assigned_to)

---

## 🔒 Security

### Rate Limiting
Добавь rate limiting для auth endpoints:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/password-reset-request")
@limiter.limit("5/hour")  # Максимум 5 запросов в час
async def request_password_reset(...):
    ...
```

### Email Validation
Используй email-validator:
```python
from pydantic import EmailStr  # Уже используется
```

### Token Security
- Reset tokens живут 1 час (хранятся в Redis)
- После использования токен удаляется
- Токены не раскрываются в логах

---

## 🧪 Тестирование

```bash
# Тест password reset flow
curl -X POST http://localhost:8100/api/auth/password-reset-request \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com"}'

# Тест support ticket
curl -X POST http://localhost:8100/api/support/tickets \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"subject":"Test","message":"Test ticket","priority":"medium"}'
```

---

## 📧 Email Templates

Кастомизируй templates в:
```
services/backend_fastapi/app/templates/emails/
```

Добавь свои стили, логотипы, footer.

---

## 🔄 Next Steps

### 1. Добавить SMS notifications (Twilio)
### 2. Two-Factor Authentication (2FA)
### 3. Email verification при регистрации
### 4. Webhooks для third-party integrations

---

## 💬 Support

Questions: support@hydraulic-diagnostics.com
