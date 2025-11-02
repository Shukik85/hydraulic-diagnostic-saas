"""Pytest configuration and fixtures for Django tests."""

import os
from pathlib import Path
import sys

import django
from django.conf import settings
from django.core.management import execute_from_command_line
from django.test import Client
import pytest

# Add backend directory to Python path
BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Configure Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

# Setup Django
if not settings.configured:
    django.setup()


@pytest.fixture(scope="session")
def django_db_setup(django_db_setup, django_db_blocker):
    """Setup test database with migrations."""
    with django_db_blocker.unblock():
        # Run migrations
        execute_from_command_line(["manage.py", "migrate", "--verbosity=1"])


@pytest.fixture
def client() -> Client:
    """Django test client fixture."""
    return Client()


@pytest.fixture
def api_client() -> Client:
    """API test client with JSON content type."""
    client = Client()
    client.defaults = {
        "HTTP_CONTENT_TYPE": "application/json",
        "HTTP_ACCEPT": "application/json",
    }
    return client


@pytest.fixture(autouse=True)
def enable_db_access_for_all_tests(db):
    """Enable database access for all tests."""
    pass


@pytest.fixture(scope="session")
def django_db_modify_db_settings():
    """Modify database settings for tests."""

    settings.DATABASES["default"]["NAME"] = ":memory:"


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables and configurations."""
    # Set test-specific environment variables
    os.environ.update(
        {
            "TESTING": "True",
            "DEBUG": "False",
            "CELERY_TASK_ALWAYS_EAGER": "True",
            "CELERY_TASK_EAGER_PROPAGATES": "True",
        }
    )

    yield

    # Cleanup after tests
    test_vars = ["TESTING", "CELERY_TASK_ALWAYS_EAGER", "CELERY_TASK_EAGER_PROPAGATES"]
    for var in test_vars:
        os.environ.pop(var, None)


# Collection hooks for pytest
def pytest_configure(config):
    """Configure pytest settings."""
    # Ensure Django is setup before running tests
    if not settings.configured:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
        django.setup()


def pytest_collection_modifyitems(config, items):
    """Modify test collection items."""
    # Mark slow tests
    for item in items:
        # Mark TimescaleDB tests as slow
        if "timescale" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid.lower() or "test_integration" in item.name:
            item.add_marker(pytest.mark.integration)

        # Mark RAG assistant tests
        if "rag" in item.nodeid.lower():
            item.add_marker(pytest.mark.rag)


# Custom markers for better test organization
pytest_plugins = [
    "pytest_django",
    "pytest_cov",
]
