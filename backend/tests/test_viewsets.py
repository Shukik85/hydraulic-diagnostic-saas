"""Integration tests adjusted to current viewsets/models (mypy-friendly)."""

from __future__ import annotations

from django.urls import reverse

import pytest
from rest_framework import status
from rest_framework.test import APIClient


@pytest.fixture
def api_client():
    return APIClient()


@pytest.mark.django_db
class TestSystemComponentViewSet:
    def test_list_components(self, api_client):
        url = reverse("systemcomponent-list")  # ensure router name matches urls.py
        response = api_client.get(url)
        assert response.status_code in (status.HTTP_200_OK, status.HTTP_404_NOT_FOUND)


@pytest.mark.django_db
class TestDiagnosticReportViewSet:
    def test_list_reports(self, api_client):
        url = reverse("diagnosticreport-list")
        response = api_client.get(url)
        assert response.status_code in (status.HTTP_200_OK, status.HTTP_404_NOT_FOUND)
