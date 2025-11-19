#!/usr/bin/env python
import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.apps import apps
from django.contrib import admin

print(" Checking Django Admin registration...\n")

total_models = 0
registered = 0
missing = []

for app_config in apps.get_app_configs():
    if app_config.name.startswith("apps."):
        models = list(app_config.get_models())
        if models:
            print(f" {app_config.label}:")
            for model in models:
                total_models += 1
                is_registered = model in admin.site._registry
                if is_registered:
                    registered += 1
                    print(f"   {model.__name__}")
                else:
                    missing.append((app_config.label, model.__name__))
                    print(f"  {model.__name__} - NOT REGISTERED")

print("\n Summary:")
print(f"  Total models: {total_models}")
print(f"  Registered: {registered}")
print(f"  Missing: {len(missing)}")

if missing:
    print("\n Models not registered:")
    for app_label, model_name in missing:
        print(f"  - {app_label}.{model_name}")
