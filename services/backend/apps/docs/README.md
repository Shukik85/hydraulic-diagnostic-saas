# Django Admin Documentation System

## Overview

This Django app provides a comprehensive documentation and guide system integrated into the Django Admin panel. It features a modern, industrial metallic design matching the existing admin interface.

## Features

### Core Functionality
- ✅ Document categories with icons and descriptions
- ✅ Markdown-based document content
- ✅ Full-text search across all documents
- ✅ User progress tracking (mark as completed)
- ✅ View count statistics
- ✅ Featured documents on homepage
- ✅ Tag-based organization

### UI/UX Features
- ✅ Responsive sidebar navigation
- ✅ Breadcrumb navigation
- ✅ Code syntax highlighting (highlight.js)
- ✅ Copy-to-clipboard for code blocks
- ✅ Auto-generated table of contents
- ✅ Reading progress indicator
- ✅ Keyboard shortcuts (Cmd/Ctrl+K for search)
- ✅ Mobile-friendly design
- ✅ Print-optimized styles

### Admin Features
- ✅ Rich admin interface for managing documents
- ✅ Auto-slug generation from titles
- ✅ Author tracking
- ✅ Publish/unpublish documents
- ✅ Ordering and categorization
- ✅ Preview links in admin

## Installation

### 1. Add to INSTALLED_APPS

Edit `services/backend/config/settings.py`:

```python
INSTALLED_APPS = [
    # ...
    "apps.docs",  # Add this line
]
```

### 2. Configure URLs

Edit `services/backend/config/urls.py`:

```python
from django.urls import include, path

urlpatterns = [
    # ...
    path("admin/docs/", include("apps.docs.urls")),
]
```

### 3. Run Migrations

```bash
python manage.py migrate docs
```

### 4. Load Initial Data (Optional)

Load sample documents:

```bash
python manage.py loaddata initial_docs
```

### 5. Create Superuser (if needed)

```bash
python manage.py createsuperuser
```

### 6. Collect Static Files

```bash
python manage.py collectstatic --noinput
```

## Usage

### Accessing Documentation

1. Log into Django Admin: `http://localhost:8000/admin/`
2. Click **📚 Documentation** in the navigation
3. Browse categories or use search

### Creating Documents

1. Go to **Admin > Documentation > Documents**
2. Click **Add Document**
3. Fill in:
   - Title (auto-generates slug)
   - Category
   - Summary (optional, for search results)
   - Content (Markdown format)
   - Tags (comma-separated)
4. Check **Is published** to make visible
5. Check **Is featured** to show on homepage
6. Save

### Managing Categories

1. Go to **Admin > Documentation > Document Categories**
2. Click **Add Document Category**
3. Fill in:
   - Name
   - Slug (auto-generated)
   - Icon (emoji or SVG class)
   - Description
   - Order (for sorting)
4. Save

## Markdown Features

Supported Markdown syntax:

- Headers: `# H1`, `## H2`, `### H3`
- Bold: `**bold text**`
- Italic: `*italic text*`
- Links: `[text](url)`
- Lists: `- item` or `1. item`
- Code: `` `inline code` ``
- Code blocks:
  ````markdown
  ```python
  def hello():
      print("Hello!")
  ```
  ````
- Blockquotes: `> quote`
- Tables:
  ```markdown
  | Header 1 | Header 2 |
  |----------|----------|
  | Cell 1   | Cell 2   |
  ```

## Keyboard Shortcuts

- `Cmd/Ctrl + K`: Focus search input
- `Esc`: Close search (when focused)

## API Endpoints

- `GET /admin/docs/` - Documentation homepage
- `GET /admin/docs/search/?q=query` - Search documents
- `GET /admin/docs/category/<slug>/` - Category page
- `GET /admin/docs/doc/<slug>/` - Document detail
- `POST /admin/docs/api/mark-complete/<slug>/` - Mark as completed

## Customization

### Styling

Edit `services/backend/static/docs/css/docs.css` to customize:

- Colors (change `--color-primary` to your brand color)
- Spacing
- Typography
- Layout

### JavaScript

Edit `services/backend/static/docs/js/docs.js` to customize:

- Table of contents generation
- Copy button behavior
- Keyboard shortcuts
- Mobile menu

### Templates

Override templates in `services/backend/templates/docs/`:

- `base.html` - Base layout
- `index.html` - Homepage
- `category.html` - Category listing
- `detail.html` - Document detail
- `search.html` - Search results

## Models

### DocumentCategory

```python
class DocumentCategory(models.Model):
    name = CharField(max_length=100)
    slug = SlugField(unique=True)
    icon = CharField(max_length=50, blank=True)
    description = TextField(blank=True)
    order = IntegerField(default=0)
    is_active = BooleanField(default=True)
```

### Document

```python
class Document(models.Model):
    title = CharField(max_length=200)
    slug = SlugField(unique=True)
    category = ForeignKey(DocumentCategory)
    summary = CharField(max_length=300, blank=True)
    content = TextField()  # Markdown
    tags = CharField(max_length=200, blank=True)
    order = IntegerField(default=0)
    is_published = BooleanField(default=True)
    is_featured = BooleanField(default=False)
    author = ForeignKey(User, null=True)
    view_count = PositiveIntegerField(default=0)
```

### UserProgress

```python
class UserProgress(models.Model):
    user = ForeignKey(User)
    document = ForeignKey(Document)
    completed = BooleanField(default=False)
    last_viewed_at = DateTimeField(auto_now=True)
```

## Troubleshooting

### Documents not showing

1. Check `is_published=True`
2. Check category `is_active=True`
3. Run migrations: `python manage.py migrate docs`

### Static files not loading

1. Run: `python manage.py collectstatic`
2. Check `STATIC_URL` and `STATIC_ROOT` in settings
3. Verify `whitenoise` is configured

### Search not working

1. Check query length (minimum 3 characters)
2. Verify documents are published
3. Check database indexes

## Contributing

To add new features:

1. Create a new branch
2. Make changes
3. Test thoroughly
4. Submit pull request

## License

Internal use only - Hydraulic Diagnostic SaaS project

## Support

For questions or issues:
- Check existing documentation
- Contact development team
- Submit GitHub issue

---

**Version**: 1.0.0  
**Last Updated**: November 16, 2025  
**Maintained by**: Backend Team
