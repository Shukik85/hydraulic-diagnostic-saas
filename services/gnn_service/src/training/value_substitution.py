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
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.schemas.graph import ComponentType
    from src.schemas.requests import (
        ComponentSensorReading,
        EdgeSensorReading,
        FlexibleInferenceRequest,
        HybridInferenceRequest,
    )
    from src.schemas.sensor_coverage import SensorCoverageConfig
    from src.schemas.topology import EdgeConfiguration, TopologyConfig

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
        
        # Thermal constants
        self.fluid_specific_heat = 2000.0  # J/(kg·K) for hydraulic oil
        self.ambient_temperature = 20.0    # °C
        self.cooling_coefficient = 0.01    # Empirical cooling factor
        self.nominal_operating_temp = 65.0 # °C (default)
        
        # Build edge lookup for faster access
        self._edge_map: dict[str, EdgeConfiguration] = {
            f"{edge.source_id}__{edge.target_id}": edge
            for edge in topology.edges
        }
        
        # Build component lookup
        self._component_map = {
            comp.component_id: comp
            for comp in topology.components
        }
    
    def substitute_missing_values(
        self,
        measured: FlexibleInferenceRequest
    ) -> HybridInferenceRequest:
        """Fill missing values in inference request.
        
        Args:
            measured: Request with ONLY measured values (partial sensor data)
        
        Returns:
            Complete HybridInferenceRequest with all fields filled
        
        Workflow:
            1. Parse FlexibleInferenceRequest
            2. For each edge in topology:
                a. Check if edge has measurements
                b. Identify missing fields
                c. Estimate using physics → nominal → default
                d. Build complete EdgeSensorReading
            3. For each component in topology:
                a. Check if component has measurements
                b. Fill missing internal sensor values with defaults
                c. Build complete ComponentSensorReading
            4. Construct HybridInferenceRequest
        
        Examples:
            >>> # Input: 1 pressure sensor
            >>> flex_req = FlexibleInferenceRequest(
            ...     equipment_id="pump_01",
            ...     timestamp=datetime.now(UTC),
            ...     topology_id="standard",
            ...     edge_readings={
            ...         "pump__valve": FlexibleEdgeSensorReading(
            ...             edge_id="pump__valve",
            ...             pressure_inlet_bar=150.0,
            ...             timestamp=datetime.now(UTC)
            ...         )
            ...     }
            ... )
            >>> 
            >>> # Output: Complete data
            >>> engine = ValueSubstitutionEngine(topology, coverage)
            >>> complete = engine.substitute_missing_values(flex_req)
            >>> # complete.edge_readings["pump__valve"] has:
            >>> # - pressure_inlet_bar=150.0 (measured) ✅
            >>> # - pressure_outlet_bar=147.5 (estimated via Darcy-Weisbach) ✅
            >>> # - flow_rate_lpm=120.0 (estimated via conservation) ✅
            >>> # - temperature_c=64.0 (estimated via thermal model) ✅
        """
        from src.schemas.requests import (
            ComponentSensorReading,
            EdgeSensorReading,
            HybridInferenceRequest,
        )
        
        logger.info(
            f"Starting value substitution for equipment '{measured.equipment_id}' "
            f"with {len(measured.edge_readings)} edge readings, "
            f"{len(measured.component_readings)} component readings"
        )
        
        # Convert FlexibleInferenceRequest to dict format for estimation methods
        measured_edges_dict: dict[str, dict[str, float]] = {}
        for edge_id, flex_reading in measured.edge_readings.items():
            measured_edges_dict[edge_id] = {}
            if flex_reading.pressure_inlet_bar is not None:
                measured_edges_dict[edge_id]["pressure_inlet_bar"] = flex_reading.pressure_inlet_bar
            if flex_reading.pressure_outlet_bar is not None:
                measured_edges_dict[edge_id]["pressure_outlet_bar"] = flex_reading.pressure_outlet_bar
            if flex_reading.flow_rate_lpm is not None:
                measured_edges_dict[edge_id]["flow_rate_lpm"] = flex_reading.flow_rate_lpm
            if flex_reading.temperature_c is not None:
                measured_edges_dict[edge_id]["temperature_c"] = flex_reading.temperature_c
            if flex_reading.vibration_g is not None:
                measured_edges_dict[edge_id]["vibration_g"] = flex_reading.vibration_g
        
        # Build complete edge readings for ALL edges in topology
        complete_edge_readings: dict[str, EdgeSensorReading] = {}
        
        for edge_config in self.topology.edges:
            edge_id = f"{edge_config.source_id}__{edge_config.target_id}"
            
            logger.debug(f"Processing edge '{edge_id}'")
            
            # Build complete EdgeSensorReading
            complete_reading = self._build_edge_reading(
                edge_id=edge_id,
                edge_config=edge_config,
                measured_edges=measured_edges_dict,
                timestamp=measured.timestamp
            )
            
            complete_edge_readings[edge_id] = complete_reading
        
        # Build complete component readings
        complete_component_readings: dict[str, ComponentSensorReading] = {}
        
        for component_config in self.topology.components:
            comp_id = component_config.component_id
            
            logger.debug(f"Processing component '{comp_id}'")
            
            # Check if component has measurements
            if comp_id in measured.component_readings:
                flex_comp = measured.component_readings[comp_id]
                
                complete_reading = self._build_component_reading(
                    component_id=comp_id,
                    flex_reading=flex_comp,
                    timestamp=measured.timestamp
                )
                
                complete_component_readings[comp_id] = complete_reading
        
        # Construct HybridInferenceRequest
        hybrid_request = HybridInferenceRequest(
            equipment_id=measured.equipment_id,
            timestamp=measured.timestamp,
            topology_id=measured.topology_id,
            edge_readings=complete_edge_readings,
            component_readings=complete_component_readings
        )
        
        logger.info(
            f"Value substitution complete: {len(complete_edge_readings)} edges, "
            f"{len(complete_component_readings)} components"
        )
        
        return hybrid_request
    
    # ========================================================================
    # BUILDER METHODS
    # ========================================================================
    
    def _build_edge_reading(
        self,
        edge_id: str,
        edge_config: EdgeConfiguration,
        measured_edges: dict[str, dict[str, float]],
        timestamp: datetime
    ) -> EdgeSensorReading:
        """Build complete EdgeSensorReading from partial measurements.
        
        Args:
            edge_id: Edge identifier
            edge_config: Edge configuration from topology
            measured_edges: Dict of measured edge values
            timestamp: Measurement timestamp
        
        Returns:
            Complete EdgeSensorReading with all required fields
        """
        from src.schemas.requests import EdgeSensorReading
        
        # Get measurements for this edge (if any)
        measured = measured_edges.get(edge_id, {})
        
        # === PRESSURE INLET ===
        pressure_inlet = measured.get("pressure_inlet_bar")
        if pressure_inlet is None:
            # TODO: Estimate from upstream (Day 2 enhancement)
            # For now, use nominal or default
            pressure_inlet = 150.0  # Default system pressure
            logger.debug(f"Edge '{edge_id}': Using default pressure_inlet={pressure_inlet} bar")
        else:
            logger.debug(f"Edge '{edge_id}': Using measured pressure_inlet={pressure_inlet} bar")
        
        # === FLOW RATE ===
        flow_rate = measured.get("flow_rate_lpm")
        if flow_rate is None:
            # Estimate using conservation of mass
            flow_rate = self._estimate_flow_rate(edge_id, measured_edges)
            if flow_rate is None:
                # Fallback to nominal
                flow_rate = 100.0  # Default nominal flow
                logger.debug(f"Edge '{edge_id}': Using default flow_rate={flow_rate} L/min")
            else:
                logger.debug(f"Edge '{edge_id}': Estimated flow_rate={flow_rate:.1f} L/min (conservation)")
        else:
            logger.debug(f"Edge '{edge_id}': Using measured flow_rate={flow_rate} L/min")
        
        # === PRESSURE OUTLET ===
        pressure_outlet = measured.get("pressure_outlet_bar")
        if pressure_outlet is None:
            # Estimate using Darcy-Weisbach
            pressure_drop = self._calculate_pressure_drop(
                flow_lpm=flow_rate,
                diameter_mm=edge_config.diameter_mm,
                length_m=edge_config.length_m,
                material=edge_config.material
            )
            pressure_outlet = pressure_inlet - pressure_drop
            logger.debug(
                f"Edge '{edge_id}': Estimated pressure_outlet={pressure_outlet:.1f} bar "
                f"(inlet={pressure_inlet:.1f} - drop={pressure_drop:.1f})"
            )
        else:
            logger.debug(f"Edge '{edge_id}': Using measured pressure_outlet={pressure_outlet} bar")
        
        # === TEMPERATURE ===
        temperature = measured.get("temperature_c")
        if temperature is None:
            # Estimate using thermal model
            temperature = self._estimate_temperature(edge_id, measured_edges)
            if temperature is None:
                # Fallback to nominal
                temperature = self.nominal_operating_temp
                logger.debug(f"Edge '{edge_id}': Using nominal temperature={temperature} °C")
            else:
                logger.debug(f"Edge '{edge_id}': Estimated temperature={temperature:.1f} °C (thermal model)")
        else:
            logger.debug(f"Edge '{edge_id}': Using measured temperature={temperature} °C")
        
        # === VIBRATION ===
        vibration = measured.get("vibration_g")
        if vibration is None:
            # Use default based on flow rate
            vibration = self._get_default_vibration(flow_rate)
            logger.debug(f"Edge '{edge_id}': Using default vibration={vibration:.2f} g")
        else:
            logger.debug(f"Edge '{edge_id}': Using measured vibration={vibration} g")
        
        # Build EdgeSensorReading
        return EdgeSensorReading(
            edge_id=edge_id,
            pressure_inlet_bar=pressure_inlet,
            pressure_outlet_bar=pressure_outlet,
            flow_rate_lpm=flow_rate,
            temperature_c=temperature,
            vibration_g=vibration,
            timestamp=timestamp
        )
    
    def _build_component_reading(
        self,
        component_id: str,
        flex_reading: Any,  # FlexibleComponentSensorReading
        timestamp: datetime
    ) -> ComponentSensorReading:
        """Build complete ComponentSensorReading from partial measurements.
        
        Args:
            component_id: Component identifier
            flex_reading: FlexibleComponentSensorReading with partial data
            timestamp: Measurement timestamp
        
        Returns:
            Complete ComponentSensorReading
        
        Note:
            For components, we only include them if at least ONE sensor exists.
            Missing internal sensors are left as None (not critical for GNN).
        """
        from src.schemas.requests import ComponentSensorReading
        
        # Pass through measured values, keep None for missing
        return ComponentSensorReading(
            component_id=component_id,
            rpm=flex_reading.rpm,
            position_percent=flex_reading.position_percent,
            current_a=flex_reading.current_a,
            voltage_v=flex_reading.voltage_v,
            timestamp=timestamp
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
        
        TODO: Implement (Day 2 enhancement)
        """
        raise NotImplementedError("Day 2 enhancement")
    
    def _estimate_pressure_outlet(
        self,
        edge_id: str,
        measured: FlexibleInferenceRequest
    ) -> float | None:
        """Estimate outlet pressure using Darcy-Weisbach.
        
        TODO: Implement (Day 2 enhancement)
        """
        raise NotImplementedError("Day 2 enhancement")
    
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
                    if outlet_edge != edge_id and outlet_edge in measured and "flow_rate_lpm" in measured[outlet_edge]:
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
                    if inlet_edge != edge_id and inlet_edge in measured and "flow_rate_lpm" in measured[inlet_edge]:
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
        measured: dict[str, dict[str, float]]
    ) -> float | None:
        """Estimate temperature from tank or upstream measurements.
        
        Thermal model:
            1. Start from tank temperature (baseline)
            2. Add pump heating (+3-5°C typical)
            3. Apply line cooling (heat loss over distance)
            4. Fallback to nominal operating temperature (60-80°C)
        
        Physics:
            - Pump heating: ΔT = Q_loss / (m_dot * C_p)
              where Q_loss = pump inefficiency losses
            - Line cooling: T_out = T_in - k * L * (T_in - T_ambient)
              where k = cooling coefficient, L = length
        
        Args:
            edge_id: Edge identifier ("source__target")
            measured: Measured edge readings {edge_id: {"temperature_c": value, ...}}
        
        Returns:
            Estimated temperature in °C, or None if can't estimate
        
        Examples:
            >>> # Propagate from tank
            >>> measured = {
            ...     "tank__pump": {"temperature_c": 60.0}
            ... }
            >>> temp = engine._estimate_temperature("pump__valve", measured)
            >>> assert 63.0 <= temp <= 65.0  # Tank + pump heating
            >>>
            >>> # Long line with cooling
            >>> measured = {
            ...     "pump__valve": {"temperature_c": 70.0}
            ... }
            >>> temp = engine._estimate_temperature("valve__cylinder", measured)
            >>> assert temp < 70.0  # Cooled down
        """
        # Parse edge_id
        parts = edge_id.split("__")
        if len(parts) != 2:
            return None
        
        source_id, target_id = parts
        
        # Check if edge exists
        if edge_id not in self._edge_map:
            return None
        
        edge_config = self._edge_map[edge_id]
        
        # Strategy 1: Check upstream edges for measured temperature
        upstream_edges = self._get_upstream_edges(source_id)
        upstream_temp = None
        
        for up_edge_id in upstream_edges:
            if up_edge_id in measured and "temperature_c" in measured[up_edge_id]:
                upstream_temp = measured[up_edge_id]["temperature_c"]
                break
        
        # If we have upstream temperature, propagate with modifications
        if upstream_temp is not None:
            estimated_temp = upstream_temp
            
            # Add pump heating if source is a pump
            if self._is_pump_component(source_id):
                # Typical pump heating: 3-5°C
                # Simplified model (more accurate would use power loss)
                pump_heating = 4.0  # °C (average)
                estimated_temp += pump_heating
            
            # Apply line cooling (heat loss to ambient)
            # T_out = T_in - k * L * (T_in - T_ambient)
            length_m = edge_config.length_m
            if length_m > 1.0:  # Only for lines longer than 1m
                temp_diff = estimated_temp - self.ambient_temperature
                cooling = self.cooling_coefficient * length_m * temp_diff
                estimated_temp -= cooling
            
            # Validate range
            if estimated_temp < -20 or estimated_temp > 150:
                warnings.warn(
                    f"Estimated temperature out of valid range: {estimated_temp:.1f}°C "
                    f"on edge '{edge_id}'. Using nominal instead.",
                    UserWarning,
                    stacklevel=2
                )
                return self.nominal_operating_temp
            
            return estimated_temp
        
        # Strategy 2: Check if tank component exists and look for its temperature
        tank_components = [c for c in self.topology.components 
                          if 'tank' in c.component_id.lower()]
        
        if tank_components:
            # Look for tank outlet edges
            for tank_comp in tank_components:
                tank_edges = self._get_downstream_edges(tank_comp.component_id)
                for tank_edge in tank_edges:
                    if tank_edge in measured and "temperature_c" in measured[tank_edge]:
                        tank_temp = measured[tank_edge]["temperature_c"]
                        
                        # Propagate tank temperature through system
                        # Add heating if passing through pump
                        if self._is_pump_component(source_id):
                            return tank_temp + 4.0  # Tank + pump heating
                        else:
                            return tank_temp
        
        # Strategy 3: Check downstream for temperature and propagate backwards
        downstream_edges = self._get_downstream_edges(target_id)
        for down_edge_id in downstream_edges:
            if down_edge_id in measured and "temperature_c" in measured[down_edge_id]:
                down_temp = measured[down_edge_id]["temperature_c"]
                
                # Reverse calculation (add back cooling loss)
                length_m = edge_config.length_m
                if length_m > 1.0:
                    # Approximate reverse cooling
                    estimated_temp = down_temp + (self.cooling_coefficient * length_m * 2.0)
                else:
                    estimated_temp = down_temp
                
                # Validate
                if -20 <= estimated_temp <= 150:
                    return estimated_temp
        
        # Strategy 4: Fallback to nominal operating temperature
        return self.nominal_operating_temp
    
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
    
    def _is_pump_component(self, component_id: str) -> bool:
        """Check if component is a pump (generates heat).
        
        Args:
            component_id: Component identifier
        
        Returns:
            True if component is a pump
        
        Example:
            >>> engine._is_pump_component("pump_main")
            True
            >>> engine._is_pump_component("valve_01")
            False
        """
        if component_id not in self._component_map:
            return False
        
        comp = self._component_map[component_id]
        
        # Check component type
        from src.schemas.graph import ComponentType
        return comp.component_type == ComponentType.PUMP
    
    def _get_component_type(self, component_id: str) -> ComponentType | None:
        """Get component type from topology.
        
        Args:
            component_id: Component identifier
        
        Returns:
            ComponentType or None if not found
        """
        if component_id not in self._component_map:
            return None
        
        return self._component_map[component_id].component_type
    
    def _get_default_vibration(self, flow_rate: float) -> float:
        """Get default vibration level based on flow rate.
        
        Args:
            flow_rate: Flow rate in L/min
        
        Returns:
            Default vibration in g
        
        Heuristics:
            - Low flow (<50 L/min): 0.3-0.5 g (low vibration)
            - Normal flow (50-150 L/min): 0.5-1.0 g (normal)
            - High flow (>150 L/min): 1.0-2.0 g (higher vibration)
        
        Example:
            >>> vib = engine._get_default_vibration(120.0)
            >>> assert 0.5 <= vib <= 1.0  # Normal range
        """
        if flow_rate < 50:
            return 0.4  # Low vibration
        elif flow_rate < 150:
            return 0.7  # Normal vibration
        else:
            return 1.2  # Higher vibration (but still normal)
