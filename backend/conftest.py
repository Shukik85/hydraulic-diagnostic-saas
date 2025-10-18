import pytest
from django.conf import settings
from rest_framework.test import APIClient


@pytest.fixture(autouse=True)
def disable_cache(monkeypatch):
    """Отключить кэширование в тестах."""
    dummy = {"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}}
    monkeypatch.setattr(settings, "CACHES", dummy)
    yield


@pytest.fixture
def api_client():
    return APIClient()
