# 🎨 Metallic Industrial Theme - Django Admin

**Version:** 1.0  
**Date:** November 16, 2025  
**Status:** ✅ Production Ready

---

## 📝 Overview

Металлический промышленный дизайн для Django Admin, адаптированный с frontend METALLIC_THEME_GUIDE.

### Ключевые особенности:
- ✅ **Brushed metal gradients** - шлифованные металлические градиенты
- ✅ **Inset shadows** - вдавленные тени для глубины
- ✅ **Muted indigo primary** - приглушенный индиго вместо яркого синего
- ✅ **Industrial status colors** - промышленные цвета статусов
- ✅ **Responsive design** - адаптивный дизайн

---

## 🚀 Quick Start

### 1. Синхронизируй репозиторий

```bash
cd services/backend
git pull origin feature/django-admin-docs-app
```

### 2. Собери статику

```bash
python manage.py collectstatic --noinput
```

### 3. Перезапусти сервер

```bash
python manage.py runserver
```

### 4. Открой админку

http://127.0.0.1:8000/admin/

🎉 **Готово!** Металлическая тема применена!

---

## 🎨 Цветовая палитра

### Metal Shades
```css
--metal-dark: #1a1d24
--metal-medium: #2d3139
--metal-light: #3f4451
--metal-lighter: #52596b
```

### Steel Accents
```css
--steel-dark: #464d5e
--steel-medium: #6b7280
--steel-light: #9ca3af
--steel-shine: #d1d5db
```

### Primary (Muted Indigo)
```css
--primary-500: #6366f1
--primary-600: #4f46e5
--primary-700: #4338ca
```

### Status Colors
```css
--status-success: #10b981  /* Green */
--status-warning: #f59e0b  /* Orange */
--status-error: #ef4444    /* Red */
--status-info: #3b82f6     /* Blue */
```

---

## 🛠️ Файловая структура

```
services/backend/
├── static/admin/css/
│   └── metallic_admin.css      # Основная тема
│
├── templates/admin/
│   └── base_site.html          # Custom темплейт
│
└── METALLIC_ADMIN_THEME.md  # Этот файл
```

---

## ⚙️ Детали реализации

### Основные компоненты

#### 1. **Header**
- Gradient: `primary-700` → `primary-900`
- Shine effect на h2
- Text shadow для глубины

#### 2. **Breadcrumbs**
- Metal background с inset shadows
- Primary цвет для ссылок
- Rounded corners

#### 3. **Modules (Cards)**
- Metal gradient background
- Inset shadows
- Primary gradient headers
- Box shadow для глубины

#### 4. **Buttons**
- Metal gradient для обычных
- Primary gradient для default
- Error gradient для delete
- Hover: lift effect + glow

#### 5. **Forms**
- Dark metal inputs
- Inset shadows
- Primary border on focus
- Glow effect

#### 6. **Tables**
- Primary gradient headers
- Striped rows
- Hover effect
- Border separation

#### 7. **Messages**
- Status color gradients
- Rounded corners
- Box shadows
- Icons

---

## 🔧 Кастомизация

### Изменить первичный цвет

Редактируй `static/admin/css/metallic_admin.css`:

```css
:root {
    --primary-500: #your-color;
    --primary-600: #darker-variant;
    --primary-700: #darkest-variant;
}
```

### Изменить metal shades

```css
:root {
    --metal-dark: #your-dark-shade;
    --metal-medium: #your-medium-shade;
    --metal-light: #your-light-shade;
}
```

### Добавить кастомные стили

Создай `static/admin/css/custom.css`:

```css
/* Твои дополнительные стили */
```

Подключи в `templates/admin/base_site.html`:

```django
{% block extrastyle %}
{{ block.super }}
<link rel="stylesheet" href="{% static 'admin/css/custom.css' %}">
{% endblock %}
```

---

## 💡 Best Practices

### 1. Используй CSS переменные

✅ **Good:**
```css
background: var(--metal-dark);
color: var(--text-primary);
```

❌ **Bad:**
```css
background: #1a1d24;
color: #f9fafb;
```

### 2. Используй gradients

✅ **Good:**
```css
background: linear-gradient(135deg, var(--primary-600) 0%, var(--primary-700) 100%);
```

### 3. Добавляй shadows

✅ **Good:**
```css
box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
```

### 4. Используй transitions

✅ **Good:**
```css
transition: all 0.2s ease;
```

---

## 🐞 Troubleshooting

### Стили не применяются?

1. **Очисти кэш браузера:**
   - Ctrl+Shift+R (Windows/Linux)
   - Cmd+Shift+R (Mac)

2. **Проверь статику:**
   ```bash
   python manage.py collectstatic --noinput --clear
   ```

3. **Проверь загрузку CSS:**
   - F12 → Network → ищи `metallic_admin.css`
   - Должен быть статус 200

### Цвета неправильные?

1. Проверь CSS переменные в DevTools
2. Убедись что `metallic_admin.css` загружается последним

### Навигация не работает?

Проверь URL patterns в `config/urls.py`:

```python
urlpatterns = [
    path('admin/', admin.site.urls),
    path('admin/docs/', include('apps.docs.urls')),
]
```

---

## 📚 Документация

- [Frontend Theme Guide](../frontend/METALLIC_THEME_GUIDE.md)
- [Django Admin Documentation](https://docs.djangoproject.com/en/5.1/ref/contrib/admin/)
- [CSS Variables](https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_custom_properties)

---

## ✅ Checklist

Перед production:

- [ ] Статика собрана: `collectstatic`
- [ ] Тема применена в всех браузерах
- [ ] Адаптив проверен на mobile
- [ ] Контраст достаточен (accessibility)
- [ ] Все ссылки в навигации работают

---

**🎉 Готово! Metallic Industrial Theme для Django Admin применен!**

---

**Version:** 1.0  
**Author:** Backend Team  
**Date:** 2025-11-16
