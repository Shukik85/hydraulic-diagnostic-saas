# Django Backend - Customer Support & Operations Hub

## 🎯 Purpose

Django Admin Panel для операторов и support team:
- 👤 User management (restore access, reset passwords)
- 💳 Subscription & billing management (Stripe integration)
- 📧 Email campaigns & notifications
- 📊 Monitoring & analytics
- 🔧 Equipment data viewing (read-only)
- 🆘 Support tools & quick actions

## 📋 Requirements

- **Python 3.14+** (required for modern type hints and match/case)
- **PostgreSQL 14+**
- **Redis 7+**
- **pip >= 23.0**

## 🚀 Quick Start

### Local Development

```bash
cd services/backend

# Create virtual environment
python3.14 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .[dev,test]

# Install pre-commit hooks
pre-commit install

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run development server
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
│   ├── core/               # Shared utilities (enums, types)
│   ├── users/              # User management
│   ├── subscriptions/      # Billing & subscriptions
│   ├── equipment/          # Equipment viewing (read-only)
│   ├── notifications/      # Email campaigns
│   ├── monitoring/         # Logs & metrics
│   └── support/            # Support tools
│
├── pyproject.toml          # Project config & dependencies
├── .pre-commit-config.yaml # Pre-commit hooks
├── requirements.txt        # Legacy requirements (use pyproject.toml)
├── Dockerfile
└── manage.py
```

## 🔧 Development Tools

### Type Checking

```bash
# Run mypy type checker
mypy apps/

# With strict mode (recommended)
mypy --strict apps/
```

### Linting & Formatting

```bash
# Run ruff linter
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Security Scanning

```bash
# Run bandit security scanner
bandit -r apps/

# Check dependencies for vulnerabilities
safety check
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
# Run all hooks
pre-commit run --all-files

# Run specific hook
pre-commit run mypy --all-files
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

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=apps --cov-report=html

# Run specific test file
pytest tests/test_users.py

# Run with verbose output
pytest -v
```

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

# Check code quality
python manage.py check
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
- [ ] Run security audit (`bandit -r apps/`)
- [ ] Check dependencies (`safety check`)

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

## 🤝 Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.

### Code Style

- **Type hints**: All functions must have type annotations
- **Docstrings**: Google-style docstrings for public APIs
- **Formatting**: Handled by ruff (max line length: 100)
- **Linting**: Must pass `ruff check` and `mypy --strict`
- **Testing**: Minimum 80% coverage required

### Pull Request Process

1. Create feature branch from `master`
2. Make changes with proper type hints and tests
3. Run pre-commit hooks: `pre-commit run --all-files`
4. Ensure tests pass: `pytest`
5. Submit PR with clear description

## 🆘 Troubleshooting

### "Could not open requirements file"
**Fixed!** Now using `pyproject.toml`. Install with:
```bash
pip install -e .[dev,test]
```

### Static files not loading
```bash
python manage.py collectstatic
```

### Database connection error
Check `DATABASE_*` environment variables.

### Type checking errors
```bash
# Install type stubs
pip install django-stubs types-redis types-pillow

# Run with less strict mode if needed
mypy --no-strict-optional apps/
```

## 📞 Support

Questions: shukik85@ya.ru

## 📚 Additional Resources

- [Django Documentation](https://docs.djangoproject.com/)
- [Python 3.14 What's New](https://docs.python.org/3.14/whatsnew/3.14.html)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [ruff Documentation](https://docs.astral.sh/ruff/)
