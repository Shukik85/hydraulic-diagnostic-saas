"""Unit tests for serializers."""

import pytest
from apps.diagnostics.serializers import DiagnosisSerializer, EquipmentSerializer


@pytest.mark.django_db
class TestEquipmentSerializer:
    """Test cases for EquipmentSerializer."""

    def test_serialize_equipment(self, equipment_factory):
        """Test serializing equipment model to JSON."""
        equipment = equipment_factory(
            name="Test Pump",
            equipment_type="pump",
            manufacturer="TestCo",
            model="TC-500",
        )
        serializer = EquipmentSerializer(equipment)
        data = serializer.data

        assert data["name"] == "Test Pump"
        assert data["equipment_type"] == "pump"
        assert data["manufacturer"] == "TestCo"
        assert data["model"] == "TC-500"
        assert "id" in data

    def test_deserialize_valid_equipment(self):
        """Test creating equipment from valid data."""
        data = {
            "name": "New Pump",
            "equipment_type": "pump",
            "manufacturer": "NewCo",
            "model": "NC-1000",
            "serial_number": "SN789",
        }
        serializer = EquipmentSerializer(data=data)
        assert serializer.is_valid()
        equipment = serializer.save()
        assert equipment.name == "New Pump"
        assert equipment.serial_number == "SN789"

    def test_deserialize_invalid_equipment(self):
        """Test validation for invalid equipment data."""
        data = {"name": ""}  # Missing required fields
        serializer = EquipmentSerializer(data=data)
        assert not serializer.is_valid()
        assert "name" in serializer.errors or "equipment_type" in serializer.errors


@pytest.mark.django_db
class TestDiagnosisSerializer:
    """Test cases for DiagnosisSerializer."""

    def test_serialize_diagnosis(self, diagnosis_factory):
        """Test serializing diagnosis model to JSON."""
        diagnosis = diagnosis_factory(symptom="Overheating")
        serializer = DiagnosisSerializer(diagnosis)
        data = serializer.data

        assert data["symptom"] == "Overheating"
        assert "status" in data
        assert "priority" in data
        assert "equipment" in data

    def test_create_diagnosis_with_nested_equipment(self, equipment_factory):
        """Test creating diagnosis with equipment relationship."""
        equipment = equipment_factory()
        data = {
            "equipment": equipment.id,
            "symptom": "Pressure drop",
            "status": "pending",
            "priority": "medium",
        }
        serializer = DiagnosisSerializer(data=data)
        assert serializer.is_valid(), serializer.errors
        diagnosis = serializer.save()
        assert diagnosis.symptom == "Pressure drop"
        assert diagnosis.equipment == equipment

    def test_update_diagnosis_status(self, diagnosis_factory):
        """Test updating diagnosis status via serializer."""
        diagnosis = diagnosis_factory(status="pending")
        data = {"status": "in_progress"}
        serializer = DiagnosisSerializer(diagnosis, data=data, partial=True)
        assert serializer.is_valid()
        updated = serializer.save()
        assert updated.status == "in_progress"
