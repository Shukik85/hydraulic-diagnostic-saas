"""Integration tests for viewsets."""

import pytest
from apps.diagnostics.models import Diagnosis, Equipment
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient


@pytest.fixture
def api_client():
    """Provide API client for testing."""
    return APIClient()


@pytest.mark.django_db
class TestEquipmentViewSet:
    """Integration tests for EquipmentViewSet."""

    def test_list_equipment(self, api_client, equipment_factory):
        """Test listing all equipment via API."""
        equipment_factory.create_batch(3)
        url = reverse("equipment-list")  # Adjust endpoint name as per urls.py
        response = api_client.get(url)

        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 3

    def test_create_equipment(self, api_client):
        """Test creating equipment via POST."""
        url = reverse("equipment-list")
        data = {
            "name": "New Hydraulic Motor",
            "equipment_type": "motor",
            "manufacturer": "MotorCo",
            "model": "MC-200",
        }
        response = api_client.post(url, data, format="json")

        assert response.status_code == status.HTTP_201_CREATED
        assert Equipment.objects.count() == 1
        assert Equipment.objects.first().name == "New Hydraulic Motor"

    def test_retrieve_equipment(self, api_client, equipment_factory):
        """Test retrieving single equipment detail."""
        equipment = equipment_factory(name="Test Equipment")
        url = reverse("equipment-detail", args=[equipment.id])
        response = api_client.get(url)

        assert response.status_code == status.HTTP_200_OK
        assert response.data["name"] == "Test Equipment"

    def test_update_equipment(self, api_client, equipment_factory):
        """Test updating equipment via PATCH."""
        equipment = equipment_factory()
        url = reverse("equipment-detail", args=[equipment.id])
        data = {"name": "Updated Name"}
        response = api_client.patch(url, data, format="json")

        assert response.status_code == status.HTTP_200_OK
        equipment.refresh_from_db()
        assert equipment.name == "Updated Name"

    def test_delete_equipment(self, api_client, equipment_factory):
        """Test deleting equipment."""
        equipment = equipment_factory()
        url = reverse("equipment-detail", args=[equipment.id])
        response = api_client.delete(url)

        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert Equipment.objects.count() == 0


@pytest.mark.django_db
class TestDiagnosisViewSet:
    """Integration tests for DiagnosisViewSet."""

    def test_list_diagnoses(self, api_client, diagnosis_factory):
        """Test listing all diagnoses."""
        diagnosis_factory.create_batch(5)
        url = reverse("diagnosis-list")
        response = api_client.get(url)

        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) >= 5

    def test_create_diagnosis(self, api_client, equipment_factory):
        """Test creating diagnosis for equipment."""
        equipment = equipment_factory()
        url = reverse("diagnosis-list")
        data = {
            "equipment": equipment.id,
            "symptom": "Unusual noise",
            "status": "pending",
            "priority": "high",
        }
        response = api_client.post(url, data, format="json")

        assert response.status_code == status.HTTP_201_CREATED
        assert Diagnosis.objects.count() == 1
        assert Diagnosis.objects.first().symptom == "Unusual noise"

    def test_filter_diagnoses_by_status(self, api_client, diagnosis_factory):
        """Test filtering diagnoses by status."""
        diagnosis_factory(status="pending")
        diagnosis_factory(status="completed")
        diagnosis_factory(status="pending")

        url = reverse("diagnosis-list")
        response = api_client.get(url, {"status": "pending"})

        assert response.status_code == status.HTTP_200_OK
        # All returned diagnoses should have pending status
        for diagnosis in response.data:
            if isinstance(diagnosis, dict):
                assert diagnosis.get("status") == "pending"

    def test_update_diagnosis_status(self, api_client, diagnosis_factory):
        """Test workflow: updating diagnosis status."""
        diagnosis = diagnosis_factory(status="pending")
        url = reverse("diagnosis-detail", args=[diagnosis.id])
        data = {"status": "in_progress"}
        response = api_client.patch(url, data, format="json")

        assert response.status_code == status.HTTP_200_OK
        diagnosis.refresh_from_db()
        assert diagnosis.status == "in_progress"
