import os
import sys
from typing import Generator

from django.conf import settings
from django.test import Client

import pytest

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


@pytest.fixture()
def client() -> Client:
    return Client()


@pytest.fixture(autouse=True)
def _setup_django_settings() -> Generator[None, None, None]:
    # Example autouse fixture to ensure settings are loaded
    assert hasattr(settings, "INSTALLED_APPS")
    yield
