"""Pytest configuration and fixtures for backend tests."""
import pytest
from django.conf import settings
from rest_framework.test import APIClient


# Configure Django settings for pytest
pytest_plugins = ['pytest_django']


@pytest.fixture
def api_client():
    """Provide DRF API client."""
    return APIClient()


@pytest.fixture
def equipment_factory(db):
    """Factory for creating Equipment instances."""
    from apps.diagnostics.models import Equipment
    
    def create_equipment(**kwargs):
        defaults = {
            'name': 'Test Equipment',
            'equipment_type': 'pump',
            'manufacturer': 'TestManufacturer',
            'model': 'TEST-001'
        }
        defaults.update(kwargs)
        return Equipment.objects.create(**defaults)
    
    # Support batch creation
    def create_batch(size, **kwargs):
        return [create_equipment(**kwargs) for _ in range(size)]
    
    create_equipment.create_batch = create_batch
    return create_equipment


@pytest.fixture
def diagnosis_factory(db, equipment_factory):
    """Factory for creating Diagnosis instances."""
    from apps.diagnostics.models import Diagnosis
    
    def create_diagnosis(**kwargs):
        if 'equipment' not in kwargs:
            kwargs['equipment'] = equipment_factory()
        
        defaults = {
            'symptom': 'Test symptom',
            'status': 'pending',
            'priority': 'medium'
        }
        defaults.update(kwargs)
        return Diagnosis.objects.create(**defaults)
    
    # Support batch creation
    def create_batch(size, **kwargs):
        return [create_diagnosis(**kwargs) for _ in range(size)]
    
    create_diagnosis.create_batch = create_batch
    return create_diagnosis
