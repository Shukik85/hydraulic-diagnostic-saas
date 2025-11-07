# 🏗️ Backend Structure Reorganization

## 🎯 Цель

Упростить структуру backend, сделав её похожей на ml_service и современные best practices Django.

## 🔄 Изменения

### До (сложная структура):

```
backend/
├── apps/
│   ├── users/
│   │   ├── apps.py (class UsersConfig: name="apps.users")
│   │   ├── models.py
│   │   └── ...
│   ├── diagnostics/
│   │   ├── apps.py (class DiagnosticsConfig: name="apps.diagnostics")
│   │   └── ...
│   ├── sensors/
│   └── rag_assistant/
├── core/
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
└── manage.py
```

**Проблемы:**
- Сложная адресация: `"apps.users.apps.UsersConfig"`
- Длинные импорты: `from apps.users.models import User`
- Неочевидное название `core` (может быть `config` или `project`)
- Лишний уровень вложенности `apps/`

### После (простая структура):

```
backend/
├── users/               # Прямо в корне backend
│   ├── __init__.py
│   ├── apps.py (class UsersConfig: name="users")
│   ├── models.py
│   ├── views.py
│   ├── serializers.py
│   ├── urls.py
│   └── migrations/
├── diagnostics/
├── sensors/
├── rag_assistant/
├── config/              # Было core, теперь config
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
└── manage.py
```

**Преимущества:**
- ✅ Простые импорты: `from users.models import User`
- ✅ Короткие INSTALLED_APPS: `["users", "diagnostics", ...]`
- ✅ Ясное название `config` для конфигурации
- ✅ Соответствует современным Django best practices
- ✅ Похоже на структуру ml_service

## 🛠️ Использование

### 1️⃣ Dry Run (проверка без изменений)

```powershell
# Windows PowerShell
.\scripts\reorganize_backend.ps1 -DryRun
```

```bash
# Linux/macOS
python scripts/reorganize_backend.py --dry-run
```

Это покажет все изменения, но **ничего не изменит**.

### 2️⃣ Применение изменений

```powershell
# Windows PowerShell
.\scripts\reorganize_backend.ps1
```

```bash
# Linux/macOS
python scripts/reorganize_backend.py
```

### 3️⃣ Проверка и тестирование

```powershell
# Проверь изменения
git status
git diff

# Пересобери Docker образы
docker-compose build

# Запусти сервисы
docker-compose up -d

# Проверь логи
docker-compose logs backend --tail=50

# Проверь что всё работает
curl http://localhost:8000/health/
```

### 4️⃣ Фиксация изменений

```bash
# Если всё работает
git add .
git commit -m "refactor: simplify backend structure (remove apps/ nesting, rename core to config)"
git push
```

## 🔍 Что делает скрипт

### 1. Перемещение приложений

```
backend/apps/users/     →  backend/users/
backend/apps/diagnostics/  →  backend/diagnostics/
backend/apps/sensors/    →  backend/sensors/
backend/apps/rag_assistant/ → backend/rag_assistant/
```

### 2. Переименование core в config

```
backend/core/  →  backend/config/
```

### 3. Обновление импортов во всех .py файлах

**До:**
```python
from apps.users.models import User
from apps.diagnostics.views import DiagnosticView
from core.settings import DEBUG
import apps.sensors.tasks
```

**После:**
```python
from users.models import User
from diagnostics.views import DiagnosticView
from config.settings import DEBUG
import sensors.tasks
```

### 4. Упрощение INSTALLED_APPS

**До:**
```python
LOCAL_APPS = [
    "apps.users.apps.UsersConfig",
    "apps.diagnostics.apps.DiagnosticsConfig",
    "apps.sensors.apps.SensorsConfig",
    "apps.rag_assistant.apps.RagAssistantConfig",
]
```

**После:**
```python
LOCAL_APPS = [
    "users",
    "diagnostics",
    "sensors",
    "rag_assistant",
]
```

### 5. Обновление manage.py, wsgi.py, asgi.py

**До:**
```python
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
```

**После:**
```python
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
```

### 6. Очистка __pycache__

Удаляет все `__pycache__/` директории, чтобы избежать проблем с кешированием.

## 🚫 Что НЕ изменяется

- ✅ Модели, представления, сериалайзеры
- ✅ Миграции базы данных
- ✅ Бизнес-логика
- ✅ API endpoints
- ✅ Тесты (только импорты обновятся)

## ⚠️ Важно

### Перед запуском:

1. **Сделай backup** или commit текущих изменений
2. **Останови Docker контейнеры**: `docker-compose down`
3. **Запусти dry-run**: `.\scripts\reorganize_backend.ps1 -DryRun`

### После запуска:

1. **Проверь git diff** чтобы убедиться что всё корректно
2. **Пересобери Docker**: `docker-compose build`
3. **Запусти тесты**: `docker-compose run --rm backend python manage.py test`
4. **Запусти приложение**: `docker-compose up -d`

## 🐛 Troubleshooting

### Ошибка: "No module named 'apps'"

**Решение:** Пересобери Docker образы:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Ошибка: Импорты не обновились

**Решение:** Удали __pycache__ и перезапусти:
```powershell
Get-ChildItem -Path backend -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
docker-compose restart
```

### Ошибка: INSTALLED_APPS не обновился

**Решение:** Ручно обнови `config/settings.py`:
```python
LOCAL_APPS = [
    "users",
    "diagnostics",
    "sensors",
    "rag_assistant",
]
```

## 🔙 Rollback (откат)

Если что-то пошло не так:

```bash
# Откат через git
git checkout .
git clean -fd

# Или конкретный commit
git reset --hard HEAD~1
```

## 📚 Дополнительные ресурсы

- [Django Best Practices](https://docs.djangoproject.com/en/stable/)
- [Two Scoops of Django](https://www.feldroy.com/books/two-scoops-of-django-3-x)
- [Cookiecutter Django](https://github.com/cookiecutter/cookiecutter-django)

---

🚀 **После реорганизации код станет проще, чище и понятнее!**