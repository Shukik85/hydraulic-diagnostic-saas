# 📚 Django Admin Documentation System - Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Django Admin Documentation System to your hydraulic-diagnostic-saas project.

## Prerequisites

- Django 5.1+
- PostgreSQL database
- Python 3.14+
- Existing Django Admin setup

## Deployment Steps

### Step 1: Merge Pull Request

```bash
# Review and merge PR #88
gh pr view 88
gh pr merge 88
```

Or via GitHub web interface:
1. Go to https://github.com/Shukik85/hydraulic-diagnostic-saas/pull/88
2. Review changes
3. Click "Merge pull request"

### Step 2: Update Local Repository

```bash
cd hydraulic-diagnostic-saas
git checkout master
git pull origin master
```

### Step 3: Update Django Settings

Edit `services/backend/config/settings.py`:

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
    "apps.docs",  # ← ADD THIS LINE
]
```

### Step 4: Configure URLs

Edit `services/backend/config/urls.py`:

```python
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("admin/docs/", include("apps.docs.urls")),  # ← ADD THIS LINE
    # ... other URL patterns
]
```

### Step 5: Run Migrations

```bash
# If using Docker
docker-compose exec backend python manage.py migrate docs

# Or locally
python manage.py migrate docs
```

Expected output:
```
Operations to perform:
  Apply all migrations: docs
Running migrations:
  Applying docs.0001_initial... OK
```

### Step 6: Load Sample Documents

```bash
# If using Docker
docker-compose exec backend python manage.py loaddata initial_docs

# Or locally
python manage.py loaddata initial_docs
```

Expected output:
```
Installed 11 object(s) from 1 fixture(s)
```

This creates:
- 6 categories
- 5 sample documents

### Step 7: Collect Static Files

```bash
# If using Docker
docker-compose exec backend python manage.py collectstatic --noinput

# Or locally
python manage.py collectstatic --noinput
```

### Step 8: Update Admin Navigation (Optional)

To add a prominent "Documentation" link in the admin navigation, edit your admin base template.

If you're using custom admin templates, add to `services/backend/templates/admin/base_site.html`:

```django
{% block nav-global %}
<nav class="admin-nav">
  <a href="{% url 'admin:index' %}">Dashboard</a>
  <a href="{% url 'docs:index' %}">📚 Documentation</a>
  <!-- other nav items -->
</nav>
{% endblock %}
```

### Step 9: Restart Services

```bash
# If using Docker
docker-compose restart backend

# Or if using systemd
sudo systemctl restart hydraulic-backend
```

### Step 10: Verify Installation

1. Access admin panel: `http://localhost:8000/admin/`
2. Navigate to: `http://localhost:8000/admin/docs/`
3. You should see:
   - Documentation homepage
   - 6 categories
   - Featured documents
   - Search functionality

## Testing Checklist

After deployment, verify:

- [ ] Can access `/admin/docs/`
- [ ] Homepage displays categories and featured docs
- [ ] Can click on a category to see documents
- [ ] Can open individual documents
- [ ] Markdown renders correctly with syntax highlighting
- [ ] Search works (try searching for "api" or "equipment")
- [ ] Code copy buttons work
- [ ] Can mark documents as complete
- [ ] Mobile responsive (test on phone/tablet)
- [ ] Admin interface shows "Documents" and "Document Categories"
- [ ] Can create new documents via admin

## Creating Your First Document

1. Log into admin: `/admin/`
2. Go to: **Documentation > Documents > Add Document**
3. Fill in:
   ```
   Title: Your First Guide
   Category: Quick Start
   Summary: A brief description
   Content: 
   # Welcome
   
   This is your first document!
   
   ## Getting Started
   
   Write content here in Markdown format.
   ```
4. Check **Is published**
5. Save

6. View at: `/admin/docs/doc/your-first-guide/`

## Common Issues

### Issue: "No module named 'apps.docs'"

**Solution**: Make sure you added `"apps.docs"` to `INSTALLED_APPS` and restarted the server.

### Issue: Static files not loading

**Solution**: Run `python manage.py collectstatic` and check that `STATIC_ROOT` is configured.

### Issue: Documents not showing

**Solution**: 
1. Check `is_published=True` on documents
2. Check `is_active=True` on categories
3. Verify migrations ran: `python manage.py showmigrations docs`

### Issue: Search returns no results

**Solution**: 
1. Make sure query is at least 3 characters
2. Check documents are published
3. Try rebuilding search index (future feature)

## Configuration Options

### Customize Colors

Edit `services/backend/static/docs/css/docs.css`:

```css
:root {
  --color-primary: #21808D;  /* Change to your brand color */
}
```

### Add Custom Categories

Via Admin UI:
1. Go to **Documentation > Document Categories**
2. Click **Add Document Category**
3. Set:
   - Name: "My Category"
   - Icon: "🔥" (emoji or SVG class)
   - Order: 10
4. Save

### Disable Features

To disable features, edit `services/backend/static/docs/js/docs.js`:

```javascript
function init() {
  // Comment out features you don't want
  // addCopyButtons();
  // generateTableOfContents();
  highlightSearchResults();
  // initReadingProgress();
  initKeyboardShortcuts();
}
```

## Performance Optimization

### Enable Caching

In `settings.py`:

```python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://redis:6379/2',
    }
}
```

Then in views:

```python
from django.views.decorators.cache import cache_page

@cache_page(60 * 15)  # Cache for 15 minutes
def docs_index(request):
    # ...
```

### Database Indexing

Indexes are already created in migrations:
- `slug` fields
- `is_published`
- `is_featured`
- `category` + `order`

### Full-Text Search

For better search performance with PostgreSQL:

```python
# Future enhancement
from django.contrib.postgres.search import SearchVector

Document.objects.annotate(
    search=SearchVector('title', 'content', 'tags')
).filter(search=query)
```

## Security Considerations

### Authentication

All views are protected with `@staff_member_required`:
- Only admin users can access documentation
- Regular users cannot view

### CSRF Protection

The mark-complete endpoint uses CSRF tokens:
```javascript
'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
```

### XSS Prevention

- Markdown is rendered client-side with `marked.js`
- All user input is escaped
- External links open in new tabs with `rel="noopener noreferrer"`

## Backup and Restore

### Backup Documents

```bash
python manage.py dumpdata docs > docs_backup.json
```

### Restore Documents

```bash
python manage.py loaddata docs_backup.json
```

### Export to Markdown

Create a management command:

```python
# services/backend/apps/docs/management/commands/export_docs.py
from django.core.management.base import BaseCommand
from apps.docs.models import Document

class Command(BaseCommand):
    def handle(self, *args, **options):
        for doc in Document.objects.all():
            filename = f"docs_{doc.slug}.md"
            with open(filename, 'w') as f:
                f.write(f"# {doc.title}\n\n")
                f.write(doc.content)
```

Run: `python manage.py export_docs`

## Monitoring

### View Statistics

Check in admin:
1. Go to **Documentation > Documents**
2. See **View Count** column
3. Sort by most viewed

### User Progress

Check in admin:
1. Go to **Documentation > User Progress Records**
2. Filter by user or document
3. See completion rates

### Analytics Integration

Add to templates:

```django
{% block extrajs %}
{{ block.super }}
<script>
  // Track page views
  gtag('event', 'page_view', {
    page_title: '{{ document.title }}',
    page_path: '{{ request.path }}'
  });
</script>
{% endblock %}
```

## Future Enhancements

Potential additions:

- [ ] **Level C Features**:
  - [ ] Interactive tours (Shepherd.js)
  - [ ] Video tutorials
  - [ ] Interactive code playground
  - [ ] AI-powered search
  - [ ] Versioning system
  - [ ] Translation support (i18n)
  - [ ] Feedback system (thumbs up/down)
  - [ ] Related documents recommendations
  - [ ] Export to PDF
  - [ ] Real-time collaboration

- [ ] **Integrations**:
  - [ ] Slack notifications for new docs
  - [ ] GitHub sync (docs as code)
  - [ ] Algolia search integration
  - [ ] OpenAPI spec auto-generation

## Support

For issues or questions:

1. Check `services/backend/apps/docs/README.md`
2. Review PR #88: https://github.com/Shukik85/hydraulic-diagnostic-saas/pull/88
3. Contact development team
4. Submit GitHub issue

## Version History

- **v1.0.0** (2025-11-16): Initial release
  - Core functionality (Level B)
  - 6 categories, 5 sample docs
  - Full-text search
  - Markdown rendering
  - User progress tracking

## License

Internal use only - Hydraulic Diagnostic SaaS

---

**Deployed by**: [Your Name]  
**Date**: [Deployment Date]  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
