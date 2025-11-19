#!/usr/bin/env python
"""Auto-register all models in Django Admin.

This script will create admin.py files for all apps that don't have them yet,
and register all models with default ModelAdmin settings.
"""

import os
import django
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.apps import apps
from django.contrib import admin


def create_admin_for_app(app_label: str):
    """Create admin.py for an app and register all its models."""
    app_config = apps.get_app_config(app_label)
    models = list(app_config.get_models())  # Convert generator to list
    
    if not models:
        print(f"⏭️  {app_label}: No models found")
        return
    
    # Get app path
    app_path = Path(app_config.path)
    admin_file = app_path / 'admin.py'
    
    # Check if admin.py exists
    if admin_file.exists():
        print(f"✓ {app_label}: admin.py already exists ({len(models)} models)")
        return
    
    # Generate admin.py content
    imports = ['from django.contrib import admin', 'from typing import ClassVar']
    model_imports = [f"from .models import {model.__name__}" for model in models]
    
    admin_classes = []
    for model in models:
        class_name = f"{model.__name__}Admin"
        admin_classes.append(f"""
@admin.register({model.__name__})
class {class_name}(admin.ModelAdmin):
    \"\"\"Admin interface for {model.__name__} model.\"\"\"
    
    list_display: ClassVar[list[str]] = ['id']  # Add your fields
    list_filter: ClassVar[list[str]] = []
    search_fields: ClassVar[list[str]] = []
    ordering: ClassVar[list[str]] = ['-id']
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs
""")
    
    # Write file
    content = "\n".join(imports + [''] + model_imports + [''] + admin_classes)
    admin_file.write_text(content, encoding='utf-8')
    
    print(f"✅ {app_label}: Created admin.py with {len(models)} models")
    for model in models:
        print(f"   - {model.__name__}")


def main():
    """Register all models in admin."""
    print("🔧 Auto-registering models in Django Admin...\n")
    
    # Get all local apps (skip django.contrib)
    local_apps = [
        app.label for app in apps.get_app_configs() 
        if app.name.startswith('apps.')
    ]
    
    for app_label in sorted(local_apps):
        create_admin_for_app(app_label)
    
    print("\n✅ Done! Run 'python manage.py runserver' to see changes.")
    print("📝 Note: Edit the generated admin.py files to customize display fields.")


if __name__ == '__main__':
    main()
