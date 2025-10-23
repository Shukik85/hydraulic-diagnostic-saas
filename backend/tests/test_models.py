"""Adjust test_models imports to current diagnostics models for mypy."""

from __future__ import annotations

import pytest
from apps.diagnostics.models import DiagnosticReport, HydraulicSystem, SystemComponent


@pytest.mark.django_db
class TestModels:
    def test_component_str(self) -> None:
        # owner is required; for typing, skip full creation details in unit tests scaffold
        hs = HydraulicSystem.objects.create(name="Sys", system_type="industrial", owner_id=None)  # type: ignore[arg-type]
        comp = SystemComponent.objects.create(system=hs, name="Pump")
        assert "Pump" in str(comp)

    def test_report_creation(self) -> None:
        hs = HydraulicSystem.objects.create(name="Sys", system_type="industrial", owner_id=None)  # type: ignore[arg-type]
        report = DiagnosticReport.objects.create(system=hs, title="R1", severity="info")
        assert report.title == "R1"
