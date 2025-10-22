"""Unit tests for serializers (updated to current models)."""
from __future__ import annotations

import pytest

from apps.diagnostics.serializers import DiagnosticReportSerializer


@pytest.mark.django_db
class TestDiagnosticReportSerializer:
    def test_create_report_minimal(self, django_user_model):
        from apps.diagnostics.models import DiagnosticReport, HydraulicSystem

        owner = django_user_model.objects.create(email="u@example.com", username="u")
        hs = HydraulicSystem.objects.create(name="Sys", system_type="industrial", owner=owner)
        data = {"system": str(hs.id), "title": "R1", "severity": "info"}
        ser = DiagnosticReportSerializer(data=data)
        assert ser.is_valid(), ser.errors
        inst = ser.save()
        assert isinstance(inst, DiagnosticReport)
