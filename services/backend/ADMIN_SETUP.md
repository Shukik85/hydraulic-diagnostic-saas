# 🚀 Django Admin - Quick Setup

> Modern, friendly guide to get all your models into Django Admin in 30 seconds

---

## ⚡ Super Quick Start

```bash
# 1. Pull latest
git pull origin feature/django-admin-docs-app

# 2. Run magic script 🪄
python register_all_models.py

# 3. Restart server
python manage.py runserver

# 4. Open admin panel
# http://127.0.0.1:8000/admin/
```

**That's it!** 🎉 All models are now in admin!

---

## 🧪 What Just Happened?

The script automatically:

- ✨ Found all your models across all apps
- 📝 Created `admin.py` files where missing
- 🎯 Registered every model with smart defaults
- ✅ Added proper type hints (ruff-compliant)
- 📦 Ready to use out-of-the-box

---

## 👀 What You'll See

After running the script, admin panel shows:

```
👥 USERS
  • Users
  • User Profiles

💳 SUBSCRIPTIONS
  • Subscriptions
  • Payments
  • Invoices

⚙️ EQUIPMENT
  • Equipment Systems (read-only)

🔔 NOTIFICATIONS
  • Notifications
  • Email Campaigns

📊 MONITORING
  • Access Logs (read-only)
  • Error Logs (read-only)

🎟️ SUPPORT
  • Support Tickets
  • Ticket Messages
  • Access Recovery

📚 DOCS
  • Documentation Categories
  • Documents
  • User Progress

🧠 GNN CONFIG
  • GNN Models
  • Training Jobs
```

---

## ✏️ Customize (Optional)

Want to make it yours? Easy!

### Add More Fields to Display

```python
# apps/yourapp/admin.py
list_display: ClassVar[list[str]] = [
    'id',           # ⬅️ Add this
    'name',         # ⬅️ And this
    'created_at',   # ⬅️ And this
]
```

### Add Search

```python
search_fields: ClassVar[list[str]] = ['name', 'email']
```

### Add Filters

```python
list_filter: ClassVar[list[str]] = ['status', 'created_at']
```

### Make Read-Only

```python
readonly_fields: ClassVar[list[str]] = ['id', 'created_at']
```

---

## 🚫 Hide Unnecessary Fields

### Option 1: Remove from List

```python
list_display: ClassVar[list[str]] = ['id']  # Only show ID
```

### Option 2: Disable Actions

```python
def has_add_permission(self, request):  # noqa: ARG002
    return False  # Can't create

def has_delete_permission(self, request, obj=None):  # noqa: ARG002
    return False  # Can't delete
```

### Option 3: Exclude Completely

Just remove the `@admin.register` decorator!

---

## 🎯 Pro Tips

### 💎 **Tip 1:** Smart Ordering

```python
ordering: ClassVar[list[str]] = ['-created_at']  # Newest first!
```

### 📊 **Tip 2:** Custom Actions

```python
@admin.action(description='Activate items')
def make_active(self, request, queryset):
    queryset.update(is_active=True)
    self.message_user(request, f"{queryset.count()} items activated!")

actions: ClassVar = [make_active]
```

### 📦 **Tip 3:** Group Fields

```python
fieldsets = (
    ('🔑 Basic Info', {
        'fields': ('name', 'email'),
    }),
    ('⚙️ Settings', {
        'fields': ('is_active', 'permissions'),
        'classes': ('collapse',),  # Collapsed by default
    }),
)
```

---

## 🐛 Troubleshooting

### Models not showing?

```bash
# 1. Check if admin.py exists
ls apps/yourapp/admin.py

# 2. Restart server
python manage.py runserver

# 3. Check in shell
python manage.py shell
>>> from django.contrib import admin
>>> admin.site._registry  # Should show your models
```

### Ruff errors?

```bash
# Auto-fix most issues
ruff check apps/*/admin.py --fix

# Format code
ruff format apps/*/admin.py
```

### Missing ClassVar?

Add this import at the top:

```python
from typing import ClassVar
```

---

## 📚 Learn More

- [Django Admin Docs](https://docs.djangoproject.com/en/5.1/ref/contrib/admin/) - Official guide
- [ModelAdmin Options](https://docs.djangoproject.com/en/5.1/ref/contrib/admin/#modeladmin-options) - All options
- [Metallic Theme](./METALLIC_ADMIN_THEME.md) - Style customization

---

## ✅ Done!

**Your admin panel is ready!** 🎉

Now open http://127.0.0.1:8000/admin/ and start managing your data!

Need help? Check out the examples above or [ask for help](https://docs.djangoproject.com/en/5.1/ref/contrib/admin/).

---

**Made with ❤️ for Hydraulic Diagnostics SaaS**
