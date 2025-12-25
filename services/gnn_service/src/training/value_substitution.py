"""Value substitution engine for filling missing sensor data.

Physics-based estimation using:
- Darcy-Weisbach equation for pressure drop
- Conservation of mass for flow rate
- Thermal models for temperature
- Nominal values from topology as fallback

Python 3.14 features:
    - Native union types
    - Improved numpy typing
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.schemas.requests import FlexibleInferenceRequest, HybridInferenceRequest
    from src.schemas.sensor_coverage import SensorCoverageConfig
    from src.schemas.topology import TopologyConfig

from src.config import settings

__all__ = ["ValueSubstitutionEngine"]


class ValueSubstitutionEngine:
    """Fill missing sensor values using physics-based estimation.
    
    Priority order:
    1. Real measurement (from FlexibleInferenceRequest)
    2. Physics-based estimation (Darcy-Weisbach, conservation laws)
    3. Nominal value (from TopologyConfig)
    4. Reasonable default (component-specific)
    """
    
    def __init__(
        self,
        topology: TopologyConfig,
        sensor_coverage: SensorCoverageConfig
    ):
        """Initialize substitution engine.
        
        Args:
            topology: Equipment topology with nominal values
            sensor_coverage: Installed sensor configuration
        """
        self.topology = topology
        self.sensor_coverage = sensor_coverage
        
        # Physics constants from config
        self.fluid_density = settings.fluid_density_kg_m3
        self.fluid_viscosity = settings.fluid_viscosity_pas
        self.material_roughness = {
            "steel": settings.material_roughness_steel,
            "rubber": settings.material_roughness_rubber,
            "composite": settings.material_roughness_composite
        }
    
    def substitute_missing_values(
        self,
        measured: FlexibleInferenceRequest
    ) -> HybridInferenceRequest:
        """Fill missing values in inference request.
        
        Args:
            measured: Request with ONLY measured values
        
        Returns:
            Complete request with all fields filled
        
        TODO: Implement full substitution logic (Day 2-3)
        """
        raise NotImplementedError(
            "ValueSubstitutionEngine.substitute_missing_values() "
            "will be implemented on Day 2-3"
        )
    
    # ========================================================================
    # PRESSURE ESTIMATION
    # ========================================================================
    
    def _estimate_pressure_inlet(
        self,
        edge_id: str,
        measured: FlexibleInferenceRequest
    ) -> float | None:
        """Estimate inlet pressure from upstream measurements.
        
        TODO: Implement (Day 2)
        """
        raise NotImplementedError("Day 2")
    
    def _estimate_pressure_outlet(
        self,
        edge_id: str,
        measured: FlexibleInferenceRequest
    ) -> float | None:
        """Estimate outlet pressure using Darcy-Weisbach.
        
        TODO: Implement (Day 2)
        """
        raise NotImplementedError("Day 2")
    
    def _calculate_pressure_drop(
        self,
        flow_lpm: float,
        diameter_mm: float,
        length_m: float,
        material: str
    ) -> float:
        """Calculate pressure drop using Darcy-Weisbach equation.
        
        Darcy-Weisbach: ΔP = f * (L/D) * (ρ * v²/2)
        
        Where:
            f = friction factor (from Reynolds number)
            L = pipe length (m)
            D = pipe diameter (m)
            ρ = fluid density (kg/m³)
            v = fluid velocity (m/s)
        
        Args:
            flow_lpm: Flow rate in liters per minute
            diameter_mm: Pipe internal diameter in mm
            length_m: Pipe length in meters
            material: Pipe material (steel/rubber/composite)
        
        Returns:
            Pressure drop in bar
        
        TODO: Implement full Darcy-Weisbach (Day 2)
        """
        raise NotImplementedError("Day 2 - Darcy-Weisbach implementation")
    
    # ========================================================================
    # FLOW ESTIMATION
    # ========================================================================
    
    def _estimate_flow_rate(
        self,
        edge_id: str,
        measured: FlexibleInferenceRequest
    ) -> float | None:
        """Estimate flow rate using conservation of mass.
        
        TODO: Implement (Day 2)
        """
        raise NotImplementedError("Day 2")
    
    # ========================================================================
    # TEMPERATURE ESTIMATION
    # ========================================================================
    
    def _estimate_temperature(
        self,
        edge_id: str,
        measured: FlexibleInferenceRequest
    ) -> float | None:
        """Estimate temperature from tank or upstream measurements.
        
        TODO: Implement (Day 2)
        """
        raise NotImplementedError("Day 2")
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _has_internal_sensors(self, comp_id: str) -> bool:
        """Check if component has any installed internal sensors."""
        return self.sensor_coverage.has_component_sensor(comp_id)
