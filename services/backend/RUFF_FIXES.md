# Ruff Error Fixes - Implementation Guide

Этот документ содержит инструкции по исправлению всех оставшихся ошибок ruff в проекте.

## Статус исправлений

### ✅ Уже исправлено (закоммичено)

- `apps/monitoring/admin.py` - добавлены ClassVar аннотации
- `apps/subscriptions/admin.py` - добавлены ClassVar аннотации  
- `apps/notifications/admin.py` - добавлены ClassVar аннотации
- `apps/support/views.py` - убраны кириллические комментарии, добавлены type hints
- `apps/monitoring/views.py` - добавлены type hints, помечен unused параметр

### 🔧 Требуют исправления

#### 1. Автоматические исправления (3 ошибки)

```bash
ruff check . --fix
```

Это исправит:
- SIM108 (ternary operator)
- SIM102 (combined if statements)  
- SIM113 (enumerate usage)

#### 2. Models.py файлы (RUF012 - ClassVar для ordering/indexes)

**Автоматическое исправление:**

```bash
python fix_ruff_errors.py
```

Или **вручную** добавить в каждый models.py:

```python
from typing import ClassVar

class YourModel(models.Model):
    # Было:
    # ordering = ["-created_at"]
    
    # Стало:
    ordering: ClassVar[list[str]] = ["-created_at"]
    indexes: ClassVar[list] = [...]
```

**Файлы требующие изменений:**
- `apps/monitoring/models.py`
- `apps/notifications/models.py`
- `apps/subscriptions/models.py`
- `apps/support/models.py`
- `apps/users/models.py`

#### 3. Admin.py файлы (RUF012 - ClassVar для list_display, etc.)

**Вручную исправить:**

- `apps/support/admin.py` (большой файл)
- `apps/users/admin.py`
- `apps/equipment/admin.py`

**Шаблон исправления:**

```python
from typing import ClassVar

class YourAdmin(admin.ModelAdmin):
    # Было:
    # list_display = ["field1", "field2"]
    
    # Стало:
    list_display: ClassVar[list[str]] = ["field1", "field2"]
    list_filter: ClassVar[list[str]] = [...]
    search_fields: ClassVar[list[str]] = [...]
    readonly_fields: ClassVar[list[str]] = [...]
    inlines: ClassVar[list] = [...]
    actions: ClassVar[list[str]] = [...]
```

#### 4. Специфичные исправления

**apps/support/models.py** (3 места):

1. DJ001 - `null=True` на CharField:
```python
# Было:
verification_method = models.CharField(
    max_length=20,
    null=True,  # Удалить это
    blank=True,
)

# Стало:
verification_method = models.CharField(
    max_length=20,
    blank=True,
    default="",  # Или сделать необязательным через validators
)
```

2. SIM108 - использовать тернарный оператор:
```python
# Было:
if last_ticket:
    seq = int(last_ticket.ticket_number.split("-")[-1]) + 1
else:
    seq = 1

# Стало:
seq = int(last_ticket.ticket_number.split("-")[-1]) + 1 if last_ticket else 1
```

3. SIM102 - объединить if statements:
```python
# Было:
if self.status not in [self.Status.RESOLVED, self.Status.CLOSED]:
    if timezone.now() > self.sla_due_date:
        self.sla_breached = True

# Стало:
if (
    self.status not in [self.Status.RESOLVED, self.Status.CLOSED]
    and timezone.now() > self.sla_due_date
):
    self.sla_breached = True
```

**apps/support/tasks.py**:

1. E402 - переместить import наверх:
```python
# Переместить в начало файла
from django.db import models
from django.contrib.auth import get_user_model
```

2. N806 - lowercase variable name:
```python
# Было:
User = get_user_model()

# Стало:
user_model = get_user_model()
```

3. SIM113 - использовать enumerate:
```python
# Было:
agent_index = 0
for ticket in unassigned_tickets:
    # ...
    agent_index += 1

# Стало:
for agent_index, ticket in enumerate(unassigned_tickets):
    # ...
```

**apps/users/models.py**:

1. DJ001 - убрать `null=True` с CharField:
```python
# Было:
stripe_customer_id = models.CharField(max_length=255, blank=True, null=True)

# Стало:
stripe_customer_id = models.CharField(max_length=255, blank=True, default="")
```

2. RUF012 - REQUIRED_FIELDS:
```python
# Было:
REQUIRED_FIELDS = []

# Стало:
REQUIRED_FIELDS: ClassVar[list[str]] = []
```

## Порядок выполнения

1. **Автоисправления:**
   ```bash
   cd services/backend
   ruff check . --fix
   ```

2. **Исправить models.py:**
   ```bash
   python fix_ruff_errors.py
   ```

3. **Вручную исправить admin.py файлы** (support, users, equipment)

4. **Применить специфичные исправления** из раздела 4

5. **Проверить результат:**
   ```bash
   ruff check .
   ruff format .
   ```

6. **Закоммитить изменения:**
   ```bash
   git add .
   git commit -m "fix: Resolve all ruff linting errors"
   git push origin feature/django-admin-docs-app
   ```

## Ожидаемый результат

```bash
$ ruff check .
All checks passed!
```

## Дополнительно

### Настройка pre-commit hook

Чтобы автоматически проверять код перед коммитом:

```bash
pre-commit install
pre-commit run --all-files
```

### Игнорирование специфичных ошибок

Если какие-то ошибки нельзя исправить, добавить в `pyproject.toml`:

```toml
[tool.ruff]
ignore = [
    "RUF012",  # Если нужно игнорировать ClassVar warnings
]
```

Или inline для конкретной строки:

```python
list_display = [...]  # noqa: RUF012
```

## Вопросы?

Обратитесь к документации:
- [Ruff Rules](https://docs.astral.sh/ruff/rules/)
- [Django Type Hints](https://docs.djangoproject.com/en/5.1/ref/contrib/admin/#django.contrib.admin.ModelAdmin.list_display)
