"""Unit tests for models."""

from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.utils import timezone

import pytest

from apps.diagnostics.models import Diagnosis, Equipment


@pytest.mark.django_db
class TestEquipmentModel:
    """Test cases for Equipment model."""

    def test_equipment_creation(self):
        """Test creating a new equipment instance."""
        equipment = Equipment.objects.create(
            name="Hydraulic Pump A",
            equipment_type="pump",
            manufacturer="HydroTech",
            model="HT-3000",
            serial_number="SN123456",
        )
        assert equipment.name == "Hydraulic Pump A"
        assert equipment.equipment_type == "pump"
        assert equipment.serial_number == "SN123456"
        assert str(equipment) == "Hydraulic Pump A (HT-3000)"

    def test_equipment_required_fields(self):
        """Test that required fields are enforced."""
        # Используем конкретное исключение вместо общего Exception
        with pytest.raises(ValidationError):
            equipment = Equipment(name="")
            equipment.full_clean()  # Это вызовет ValidationError

        # Альтернативно: проверка IntegrityError при сохранении
        with pytest.raises(IntegrityError):
            Equipment.objects.create(name="")


@pytest.mark.django_db
class TestDiagnosisModel:
    """Test cases for Diagnosis model."""

    def test_diagnosis_creation(self, equipment_factory):
        """Test creating a diagnosis with factory."""
        equipment = equipment_factory()
        diagnosis = Diagnosis.objects.create(
            equipment=equipment,
            symptom="High temperature",
            status="pending",
            priority="high",
        )
        assert diagnosis.symptom == "High temperature"
        assert diagnosis.status == "pending"
        assert diagnosis.equipment == equipment
        assert diagnosis.created_at is not None

    def test_diagnosis_status_workflow(self, diagnosis_factory):
        """Test diagnosis status transitions."""
        diagnosis = diagnosis_factory(status="pending")
        diagnosis.status = "in_progress"
        diagnosis.save()
        assert diagnosis.status == "in_progress"

        diagnosis.status = "completed"
        diagnosis.completed_at = timezone.now()
        diagnosis.save()
        assert diagnosis.status == "completed"
        assert diagnosis.completed_at is not None

    def test_diagnosis_invalid_status(self):
        """Test that invalid status values are rejected."""
        equipment = Equipment.objects.create(
            name="Test Equipment", equipment_type="pump"
        )

        # Если в модели есть валидация статусов, используем ValidationError
        with pytest.raises(ValidationError):
            diagnosis = Diagnosis(
                equipment=equipment,
                symptom="Test symptom",
                status="invalid_status",  # Неверный статус
                priority="high",
            )
            diagnosis.full_clean()
