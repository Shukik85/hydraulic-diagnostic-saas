"""Diagnostics models package."""

from .diagnostic_results import (
    IntegratedDiagnosticResult,
    MathematicalModelResult,
    PhasePortraitResult,
    TribodiagnosticResult,
)

# Re-export from main models file for backward compatibility
from ..models import (
    DiagnosticReport,
    HydraulicSystem,
    SensorData,
    SystemComponent,
)

__all__ = [
    # Core models
    "DiagnosticReport",
    "HydraulicSystem",
    "SensorData",
    "SystemComponent",
    # Diagnostic results models
    "IntegratedDiagnosticResult",
    "MathematicalModelResult",
    "PhasePortraitResult",
    "TribodiagnosticResult",
]
