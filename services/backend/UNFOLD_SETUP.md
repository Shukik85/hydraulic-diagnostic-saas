# Django Unfold Integration - Setup Guide

## 🎉 Что установлено

**Django Unfold** - современная тема для Django Admin на Tailwind CSS.

### Возможности:
- ✅ Современный Tailwind CSS дизайн
- 🌓 Dark/Light режимы с переключателем
- 📱 Полностью responsive (мобильные устройства)
- 🎯 Кастомная навигация в сайдбаре с Material Icons
- 📊 Dashboard с виджетами и метриками
- 🔍 Улучшенные фильтры и формы
- ⚡ Production-ready - используется в enterprise проектах

## 🚀 Установка

### 1. Установите зависимости

```bash
cd /h/hydraulic-diagnostic-saas/services/backend
source ../../.venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Соберите статику

```bash
python manage.py collectstatic --noinput
```

### 3. Запустите сервер

```bash
python manage.py runserver
```

### 4. Откройте админку

Перейдите по адресу: http://127.0.0.1:8000/admin/

Нажмите **Ctrl+F5** для очистки кэша браузера.

## 🏪 Что изменилось

### Файлы:
- `requirements.txt` - добавлен `django-unfold>=0.38.0`
- `config/settings.py` - настроен `UNFOLD` конфигурация
- `apps/users/admin.py` - мигрирован на `unfold.admin.ModelAdmin`
- `apps/support/admin.py` - мигрирован на `unfold.admin.ModelAdmin`
- `apps/core/admin.py` - добавлен `dashboard_callback` с метриками
- `apps/core/utils.py` - добавлен `environment_callback`

### Удалено:
- `templates/admin/base_site.html` - больше не нужен (используется Unfold шаблон)

## 🔧 Дополнительная настройка

### Миграция остальных admin.py

Все остальные `ModelAdmin` классы нужно мигрировать:

**Было:**
```python
from django.contrib import admin

class MyAdmin(admin.ModelAdmin):
    pass
```

**Стало:**
```python
from unfold.admin import ModelAdmin

class MyAdmin(ModelAdmin):
    pass
```

### Файлы для миграции:
- `apps/equipment/admin.py`
- `apps/subscriptions/admin.py`
- `apps/notifications/admin.py`
- `apps/monitoring/admin.py`
- `apps/gnn_config/admin.py`
- `apps/docs/admin.py`

### Использование @display decorator

Для badges и кастомных полей:

```python
from unfold.decorators import display

@display(description="Status", label=True)
def status_badge(self, obj):
    return format_html('<span class="badge bg-success">Активен</span>')
```

## 📚 Документация

Полная документация: https://unfoldadmin.com/

### Полезные ссылки:
- Quickstart: https://unfoldadmin.com/docs/quickstart/
- Settings: https://unfoldadmin.com/docs/settings/
- Dashboard: https://unfoldadmin.com/docs/dashboard/
- Navigation: https://unfoldadmin.com/docs/navigation/
- Actions: https://unfoldadmin.com/docs/actions/

## ✅ Результат

После установки вы получите:

- 🎨 **Современный дизайн** - Tailwind CSS
- 🌓 **Тёмная/светлая тема** - переключатель в хедере
- 🧭 **Кастомную навигацию** - с Material Icons
- 📊 **Dashboard с виджетами** - живые метрики
- 📍 **Environment badge** - DEVELOPMENT/STAGING/PRODUCTION
- ✨ **Улучшенные фильтры** - быстрый поиск и фильтрация

## 🛠️ Troubleshooting

### Стили не применяются?

```bash
python manage.py collectstatic --noinput --clear
```

### Ошибка импорта?

Убедитесь, что `unfold` указан в `INSTALLED_APPS` **ПЕРЕД** `django.contrib.admin`.

### Навигация не отображается?

Проверьте, что все URL в `UNFOLD["SIDEBAR"]["navigation"]` существуют и зарегистрированы в `urls.py`.

## 📝 Примечания

- Все кастомные шаблоны admin удалены
- Unfold предоставляет свои шаблоны
- Для кастомизации используйте `UNFOLD` настройки в settings.py
