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

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.schemas.requests import FlexibleInferenceRequest, HybridInferenceRequest
    from src.schemas.sensor_coverage import SensorCoverageConfig
    from src.schemas.topology import EdgeConfiguration, TopologyConfig

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
        
        # Build edge lookup for faster access
        self._edge_map: dict[str, EdgeConfiguration] = {
            f"{edge.source_id}__{edge.target_id}": edge
            for edge in topology.edges
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
        
        Flow regimes:
            - Laminar: Re < 2300 → f = 64/Re
            - Transitional: 2300 < Re < 4000 → use turbulent formula
            - Turbulent: Re > 4000 → Colebrook-White equation
        
        Args:
            flow_lpm: Flow rate in liters per minute
            diameter_mm: Pipe internal diameter in mm
            length_m: Pipe length in meters
            material: Pipe material ("steel", "rubber", "composite")
        
        Returns:
            Pressure drop in bar (positive value)
        
        Examples:
            >>> engine = ValueSubstitutionEngine(topology, coverage)
            >>> dp = engine._calculate_pressure_drop(
            ...     flow_lpm=120.0,
            ...     diameter_mm=25.0,
            ...     length_m=3.0,
            ...     material="steel"
            ... )
            >>> assert 0.5 <= dp <= 5.0  # Typical range
        """
        # Handle edge cases
        if flow_lpm <= 0:
            return 0.0  # No flow = no pressure drop
        
        if diameter_mm <= 0 or length_m <= 0:
            warnings.warn(
                f"Invalid pipe dimensions: diameter={diameter_mm}mm, length={length_m}m. "
                "Returning zero pressure drop.",
                UserWarning,
                stacklevel=2
            )
            return 0.0
        
        # Get material roughness
        roughness_mm = self.material_roughness.get(
            material.lower(),
            self.material_roughness["steel"]  # Default to steel
        )
        
        # Unit conversions
        diameter_m = diameter_mm / 1000.0  # mm → m
        flow_m3s = flow_lpm / 60000.0      # L/min → m³/s
        roughness_m = roughness_mm / 1000.0  # mm → m
        
        # Calculate cross-sectional area
        area_m2 = math.pi * (diameter_m ** 2) / 4.0
        
        # Calculate velocity
        velocity_ms = flow_m3s / area_m2
        
        # Calculate Reynolds number: Re = (ρ * v * D) / μ
        reynolds = (self.fluid_density * velocity_ms * diameter_m) / self.fluid_viscosity
        
        # Calculate friction factor based on flow regime
        if reynolds < 2300:
            # Laminar flow: f = 64/Re
            friction_factor = 64.0 / reynolds
            
        else:
            # Turbulent flow: Colebrook-White equation
            # 1/√f = -2 * log10((ε/D)/3.71 + 2.51/(Re*√f))
            # Solved iteratively
            
            relative_roughness = roughness_m / diameter_m
            
            # Initial guess using Swamee-Jain approximation
            # f ≈ 0.25 / [log10((ε/D)/3.7 + 5.74/Re^0.9)]²
            f_guess = 0.25 / (
                math.log10(relative_roughness / 3.7 + 5.74 / (reynolds ** 0.9))
            ) ** 2
            
            # Iterative refinement (Newton-Raphson)
            friction_factor = f_guess
            for _ in range(10):  # 10 iterations sufficient for convergence
                sqrt_f = math.sqrt(friction_factor)
                
                # Colebrook-White equation
                f_new = 1.0 / (
                    -2.0 * math.log10(relative_roughness / 3.71 + 2.51 / (reynolds * sqrt_f))
                ) ** 2
                
                # Check convergence
                if abs(f_new - friction_factor) < 1e-6:
                    break
                
                friction_factor = f_new
        
        # Apply Darcy-Weisbach equation: ΔP = f * (L/D) * (ρ * v²/2)
        pressure_drop_pa = friction_factor * (length_m / diameter_m) * (
            self.fluid_density * velocity_ms ** 2 / 2.0
        )
        
        # Convert Pa → bar (1 bar = 100000 Pa)
        pressure_drop_bar = pressure_drop_pa / 100000.0
        
        # Validate result
        if pressure_drop_bar > 20.0:
            warnings.warn(
                f"Very high pressure drop calculated: {pressure_drop_bar:.1f} bar "
                f"(flow={flow_lpm:.1f} L/min, diameter={diameter_mm:.1f}mm, "
                f"length={length_m:.1f}m, material={material}). "
                "Check for blockage, restriction, or sensor malfunction.",
                UserWarning,
                stacklevel=2
            )
        
        return pressure_drop_bar
    
    # ========================================================================
    # FLOW ESTIMATION
    # ========================================================================
    
    def _estimate_flow_rate(
        self,
        edge_id: str,
        measured: dict[str, dict[str, float]]
    ) -> float | None:
        """Estimate flow rate using conservation of mass.
        
        Conservation of mass:
            - Single path: flow_in = flow_out (continuity equation)
            - Branching point: sum(inflows) = sum(outflows)
            - Cylinder: account for rod volume displacement
        
        Estimation priority:
            1. Check upstream edge for measured flow
            2. Check downstream edge for measured flow  
            3. If branching point: balance from other branches
            4. Fallback to nominal value from topology
        
        Args:
            edge_id: Edge identifier ("source__target")
            measured: Measured edge readings {edge_id: {"flow_rate_lpm": value, ...}}
        
        Returns:
            Estimated flow rate in L/min, or None if can't estimate
        
        Examples:
            >>> # Simple propagation
            >>> measured = {
            ...     "pump__valve": {"flow_rate_lpm": 120.0}
            ... }
            >>> flow = engine._estimate_flow_rate("valve__cylinder", measured)
            >>> assert flow == 120.0  # Propagated from upstream
            >>>
            >>> # Branching point
            >>> measured = {
            ...     "pump__junction": {"flow_rate_lpm": 150.0},
            ...     "junction__cylinder_left": {"flow_rate_lpm": 90.0}
            ... }
            >>> flow = engine._estimate_flow_rate("junction__cylinder_right", measured)
            >>> assert flow == 60.0  # 150 - 90 = 60 (mass balance)
        """
        # Parse edge_id
        parts = edge_id.split("__")
        if len(parts) != 2:
            warnings.warn(
                f"Invalid edge_id format: '{edge_id}'. Expected 'source__target'.",
                UserWarning,
                stacklevel=2
            )
            return None
        
        source_id, target_id = parts
        
        # Check if edge exists in topology
        if edge_id not in self._edge_map:
            warnings.warn(
                f"Edge '{edge_id}' not found in topology. Cannot estimate flow.",
                UserWarning,
                stacklevel=2
            )
            return None
        
        # Strategy 1: Check upstream edges (edges ending at source_id)
        upstream_edges = self._get_upstream_edges(source_id)
        for up_edge_id in upstream_edges:
            if up_edge_id in measured and "flow_rate_lpm" in measured[up_edge_id]:
                flow = measured[up_edge_id]["flow_rate_lpm"]
                if flow > 0:
                    return flow  # Propagate from upstream
        
        # Strategy 2: Check downstream edges (edges starting from target_id)
        downstream_edges = self._get_downstream_edges(target_id)
        for down_edge_id in downstream_edges:
            if down_edge_id in measured and "flow_rate_lpm" in measured[down_edge_id]:
                flow = measured[down_edge_id]["flow_rate_lpm"]
                if flow > 0:
                    return flow  # Propagate from downstream
        
        # Strategy 3: Branching point - mass balance
        # Get all edges at source (junction)
        source_inlets = self._get_upstream_edges(source_id)
        source_outlets = self._get_downstream_edges(source_id)
        
        if len(source_outlets) > 1:  # Source is a branching point
            # Try to balance: inlet_flow = sum(outlet_flows)
            inlet_flow = 0.0
            for inlet_edge in source_inlets:
                if inlet_edge in measured and "flow_rate_lpm" in measured[inlet_edge]:
                    inlet_flow += measured[inlet_edge]["flow_rate_lpm"]
            
            if inlet_flow > 0:
                # Sum known outlet flows
                known_outlet_flow = 0.0
                for outlet_edge in source_outlets:
                    if outlet_edge != edge_id and outlet_edge in measured:
                        if "flow_rate_lpm" in measured[outlet_edge]:
                            known_outlet_flow += measured[outlet_edge]["flow_rate_lpm"]
                
                # Remaining flow goes to this edge
                estimated_flow = inlet_flow - known_outlet_flow
                if estimated_flow >= 0:
                    return estimated_flow
        
        # Strategy 4: Check target (junction) mass balance
        target_inlets = self._get_upstream_edges(target_id)
        target_outlets = self._get_downstream_edges(target_id)
        
        if len(target_inlets) > 1:  # Target is a merging point
            # Try to balance: sum(inlet_flows) = outlet_flow
            outlet_flow = 0.0
            for outlet_edge in target_outlets:
                if outlet_edge in measured and "flow_rate_lpm" in measured[outlet_edge]:
                    outlet_flow += measured[outlet_edge]["flow_rate_lpm"]
            
            if outlet_flow > 0:
                # Sum known inlet flows
                known_inlet_flow = 0.0
                for inlet_edge in target_inlets:
                    if inlet_edge != edge_id and inlet_edge in measured:
                        if "flow_rate_lpm" in measured[inlet_edge]:
                            known_inlet_flow += measured[inlet_edge]["flow_rate_lpm"]
                
                # Remaining flow comes from this edge
                estimated_flow = outlet_flow - known_inlet_flow
                if estimated_flow >= 0:
                    return estimated_flow
        
        # Strategy 5: Fallback to nominal (if available)
        # This would require nominal flow values in TopologyConfig
        # For now, return None
        
        return None
    
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
    
    def _get_upstream_edges(self, component_id: str) -> list[str]:
        """Get all edges ending at this component (inlets).
        
        Args:
            component_id: Component identifier
        
        Returns:
            List of edge_ids ("source__target") ending at component_id
        
        Example:
            >>> edges = engine._get_upstream_edges("valve_1")
            >>> # ["pump_1__valve_1", "tank__valve_1"]
        """
        upstream = []
        for edge in self.topology.edges:
            if edge.target_id == component_id:
                edge_id = f"{edge.source_id}__{edge.target_id}"
                upstream.append(edge_id)
        return upstream
    
    def _get_downstream_edges(self, component_id: str) -> list[str]:
        """Get all edges starting from this component (outlets).
        
        Args:
            component_id: Component identifier
        
        Returns:
            List of edge_ids ("source__target") starting from component_id
        
        Example:
            >>> edges = engine._get_downstream_edges("pump_1")
            >>> # ["pump_1__valve_1", "pump_1__filter_1"]
        """
        downstream = []
        for edge in self.topology.edges:
            if edge.source_id == component_id:
                edge_id = f"{edge.source_id}__{edge.target_id}"
                downstream.append(edge_id)
        return downstream
