# Django Deployment Fixes

## Обзор исправленных ошибок

В процессе запуска `python manage.py check --deploy` были обнаружены критические ошибки и предупреждения. Все проблемы успешно решены.

## Критическая ошибка: drf_spectacular.E001

**Проблема:**
```
drf_spectacular.E001) Schema generation threw exception "'Meta.fields' must not contain non-model field names: criticality"
```

**Причина:** Поле `criticality` было указано в `Meta.fields` сериализатора, но не существовало в модели.

**Решение:**
- Удалено несуществующее поле `criticality` из всех сериализаторов
- Проверена согласованность между моделями и сериализаторами

## Предупреждения drf_spectacular.W001

**Проблема:**
```
drf_spectacular.W001) unable to resolve type hint for function "get_latest_activity". Consider using a type hint or @extend_schema_field.
```

**Решение:**
- Добавлены декораторы `@extend_schema_field` для всех `SerializerMethodField`
- Настроена правильная типизация для OpenAPI схемы
- Добавлены импорты `Optional` и `extend_schema_field`

**Пример исправления:**
```python
from drf_spectacular.utils import extend_schema_field
from typing import Optional

class HydraulicSystemListSerializer(ChoiceDisplayMixin, serializers.ModelSerializer):
    @extend_schema_field(serializers.CharField())
    def get_system_type_display(self, obj) -> Optional[str]:
        return self.get_choice_display(obj, "system_type")
```

## Security ошибки и предупреждения

### Критические настройки безопасности:

#### security.W004 - SECURE_HSTS_SECONDS
**Решение:** Настроено `SECURE_HSTS_SECONDS = 31536000` (1 год)

#### security.W008 - SECURE_SSL_REDIRECT
**Решение:** Настроено `SECURE_SSL_REDIRECT = True`

#### security.W009 - SECRET_KEY
**Проблема:** Слабый SECRET_KEY (<50 символов, <5 уникальных символов)
**Решение:** Создан сильный SECRET_KEY в `deploy_settings.py`

#### security.W012 - SESSION_COOKIE_SECURE
**Решение:** Настроено `SESSION_COOKIE_SECURE = True`

#### security.W016 - CSRF_COOKIE_SECURE
**Решение:** Настроено `CSRF_COOKIE_SECURE = True`

#### security.W018 - DEBUG
**Решение:** Настроено `DEBUG = False` в `deploy_settings.py`

## Новые файлы

### `backend/core/deploy_settings.py`
Продакшн конфигурация Django с усиленной безопасностью:
- Сильный SECRET_KEY (50+ символов)
- HTTPS/HSTS настройки
- Secure cookies
- Отключенные отладочные приложения

## Обновленные файлы

### `backend/apps/diagnostics/serializers.py`
- Удалено несуществующее поле `criticality`
- Добавлены декораторы `@extend_schema_field`
- Правильная типизация

### `.github/workflows/ci-backend.yml`
- Настроен `DJANGO_SETTINGS_MODULE=core.deploy_settings`
- Добавлен сильный SECRET_KEY для CI
- Настроены отсутствующие переменные окружения

## Использование

### Локальная разработка
```bash
# Обычная разработка
cd backend
python manage.py runserver

# Проверка production настроек
DJANGO_SETTINGS_MODULE=core.deploy_settings python manage.py check --deploy
```

### CI/CD
```bash
# CI теперь автоматически использует deploy_settings
# и проходит все проверки безопасности
```

### Production развертывание
```bash
# Настройка переменных окружения
export DJANGO_SETTINGS_MODULE=core.deploy_settings
export SECRET_KEY="your-super-strong-secret-key-50-plus-characters"
export DATABASE_NAME=hydraulic_prod
export DATABASE_USER=hydraulic_user
export DATABASE_PASSWORD=strong_password
export DATABASE_HOST=db.company.com
export REDIS_URL=redis://redis.company.com:6379/0

# Запуск проверок
python manage.py check --deploy

# Запуск продакшн сервера
gunicorn core.wsgi:application --bind 0.0.0.0:8000
```

## Результат

✅ **Все критические ошибки устранены**
✅ **Все security предупреждения устранены**
✅ **CI/CD настроен для production конфигурации**
✅ **OpenAPI/Swagger схема работает корректно**

Проект теперь соответствует enterprise-стандартам безопасности и готов к production развертыванию.
