# Django Backend - Customer Support & Operations Hub

## 🎯 Purpose

Django Admin Panel для операторов и support team:
- 👤 User management (restore access, reset passwords)
- 💳 Subscription & billing management (Stripe integration)
- 📧 Email campaigns & notifications
- 📊 Monitoring & analytics
- 🔧 Equipment data viewing (read-only)
- 🆘 Support tools & quick actions

## 🚀 Quick Start

### Local Development

```bash
cd services/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run server
python manage.py runserver 0.0.0.0:8000
```

### Docker

```bash
cd /path/to/hydraulic-diagnostic-saas

# Build and run
docker-compose up --build backend_django

# Create superuser
docker-compose exec backend_django python manage.py createsuperuser

# Access admin
http://localhost:8000/admin
```

## 📦 Structure

```
services/backend/
├── config/                  # Django configuration
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── celery.py
│
├── apps/
│   ├── users/              # User management
│   ├── subscriptions/      # Billing & subscriptions
│   ├── equipment/          # Equipment viewing (read-only)
│   ├── notifications/      # Email campaigns
│   ├── monitoring/         # Logs & metrics
│   └── support/            # Support tools
│
├── requirements.txt        # Single file, all deps
├── Dockerfile
└── manage.py
```

## 🎨 Admin Features

### 1. User Management
- Search by email, name, API key
- Filter by subscription tier, status
- Bulk actions (export, email)
- Quick password reset

### 2. Subscription Dashboard
- View all subscriptions
- Upgrade/downgrade plans
- Extend trials
- View payment history

### 3. Email Campaigns
- Create targeted campaigns
- Schedule sends
- Track opens & clicks
- Template management

### 4. Monitoring
- API request logs
- Error tracking
- Performance metrics
- User activity

### 5. Support Tools
- Quick actions (reset password, extend trial)
- Support action history
- Data export (GDPR)

## 🔧 Management Commands

```bash
# Create superuser
python manage.py createsuperuser

# Migrations
python manage.py makemigrations
python manage.py migrate

# Collect static files
python manage.py collectstatic

# Run Celery worker (separate terminal)
celery -A config worker -l info

# Run Celery beat (scheduler)
celery -A config beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
```

## 🔒 Security

### Production Checklist
- [ ] Change DJANGO_SECRET_KEY
- [ ] Set DEBUG=False
- [ ] Configure ALLOWED_HOSTS
- [ ] Enable HTTPS (SECURE_SSL_REDIRECT=True)
- [ ] Secure cookies (SESSION_COOKIE_SECURE=True)
- [ ] Configure CORS properly
- [ ] Set up rate limiting
- [ ] Enable 2FA for superusers

## 🧪 Testing

```bash
# Run tests
python manage.py test

# With coverage
coverage run --source='.' manage.py test
coverage report
```

## 📊 Database Schema

### Managed by Django:
- `users` - User model with subscription
- `subscriptions` - Subscription details
- `payments` - Payment history
- `email_campaigns` - Email campaigns
- `notifications` - System notifications
- `api_logs` - API request logs
- `error_logs` - Error tracking
- `support_actions` - Support action history

### Shared with FastAPI (read-only):
- `equipment` - Equipment metadata (managed by FastAPI)

## 🔌 Integrations

### Stripe
```python
# Webhook endpoint: /api/webhooks/stripe
# Events handled:
# - payment_intent.succeeded
# - subscription.updated
# - subscription.deleted
```

### SendGrid/Mailgun
```python
# Email templates in templates/emails/
# Sent via Celery tasks
```

### Redis
```python
# Celery broker: REDIS_URL
# Cache backend: REDIS_URL
```

## 📝 Environment Variables

See `.env.example` for all required variables.

## 🆘 Troubleshooting

### "Could not open requirements file"
**Fixed!** Now using single `requirements.txt` without nested imports.

### Static files not loading
```bash
python manage.py collectstatic
```

### Database connection error
Check `DATABASE_*` environment variables.

## 📞 Support

Questions: shukik85@ya.ru
