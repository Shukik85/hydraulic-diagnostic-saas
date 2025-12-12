"""Edge feature computation module.

Computes dynamic edge features from sensor data and static topology.
Uses physics-based models (Darcy-Weisbach) for flow rate estimation.

Features computed:
1. Flow rate (L/min) - from pressure drop + geometry
2. Pressure drop (bar) - direct from sensors
3. Temperature delta (°C) - direct from sensors
4. Vibration level (g) - averaged from adjacent components
5. Age (hours) - from installation date
6. Maintenance score [0, 1] - decay-based

Author: GNN Service Team
Python: 3.14+
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.schemas.graph import EdgeMaterial, EdgeSpec

if TYPE_CHECKING:
    from datetime import datetime

    from src.schemas.requests import ComponentSensorReading

# ============================================================================
# Fluid Properties (Hydraulic Oil ISO VG 46)
# ============================================================================


def get_fluid_density(temperature_c: float) -> float:
    """Get hydraulic oil density at given temperature.

    Uses linear approximation for ISO VG 46 hydraulic oil.
    Density decreases with temperature.

    Args:
        temperature_c: Temperature in °C

    Returns:
        Density in kg/m³

    Examples:
        >>> get_fluid_density(20)  # Room temperature
        875.0
        >>> get_fluid_density(80)  # Hot oil
        845.0
    """
    # ISO VG 46 at 15°C: ~880 kg/m³
    # Temperature coefficient: ~0.7 kg/m³ per °C
    rho_15 = 880.0
    temp_coeff = -0.7

    rho = rho_15 + temp_coeff * (temperature_c - 15.0)

    # Clamp to reasonable range
    return np.clip(rho, 800.0, 900.0)


def get_fluid_viscosity(temperature_c: float) -> float:
    """Get hydraulic oil kinematic viscosity at given temperature.

    Uses Walther equation for ISO VG 46 hydraulic oil.
    Viscosity decreases exponentially with temperature.

    Args:
        temperature_c: Temperature in °C

    Returns:
        Kinematic viscosity in mm²/s (cSt)

    Examples:
        >>> get_fluid_viscosity(40)  # Standard test temperature
        46.0
        >>> get_fluid_viscosity(100)  # High temperature
        6.8
    """
    # Walther equation parameters for ISO VG 46
    # log(log(ν + 0.8)) = A - B * log(T_K)

    # Known points: ν(40°C) = 46 cSt, ν(100°C) = 6.8 cSt
    if temperature_c <= 40:
        # Linear extrapolation below 40°C
        nu_40 = 46.0
        slope = -1.5  # cSt per °C (approximation)
        nu = nu_40 + slope * (40 - temperature_c)
    elif temperature_c >= 100:
        # Linear extrapolation above 100°C
        nu_100 = 6.8
        slope = -0.08  # cSt per °C (approximation)
        nu = nu_100 + slope * (temperature_c - 100)
    else:
        # Logarithmic interpolation
        t = (temperature_c - 40) / (100 - 40)
        log_nu_40 = math.log(46.0)
        log_nu_100 = math.log(6.8)
        log_nu = log_nu_40 + t * (log_nu_100 - log_nu_40)
        nu = math.exp(log_nu)

    # Clamp to reasonable range
    return np.clip(nu, 1.0, 1000.0)


def get_material_roughness(material: EdgeMaterial | str) -> float:
    """Get surface roughness for pipe material.

    Absolute roughness (ε) in mm for different materials.
    Used in friction factor calculation.

    Args:
        material: Pipe/hose material

    Returns:
        Absolute roughness in mm

    Examples:
        >>> get_material_roughness(EdgeMaterial.STEEL)
        0.045
        >>> get_material_roughness("rubber")
        0.15
    """
    roughness_map = {
        EdgeMaterial.STEEL: 0.045,  # Commercial steel pipe
        EdgeMaterial.RUBBER: 0.15,  # Rubber hose (smooth inner)
        EdgeMaterial.COMPOSITE: 0.10,  # Composite materials
        EdgeMaterial.THERMOPLASTIC: 0.08,  # Thermoplastic hoses
        "steel": 0.045,
        "rubber": 0.15,
        "composite": 0.10,
        "thermoplastic": 0.08,
    }

    return roughness_map.get(material, 0.10)  # Default: medium roughness


# ============================================================================
# Flow Rate Estimation (Darcy-Weisbach)
# ============================================================================


def estimate_friction_factor(relative_roughness: float, reynolds_number: float) -> float:
    """Estimate Darcy friction factor using Haaland approximation.

    Valid for turbulent flow (Re > 4000).
    More accurate than simple Moody chart lookup.

    Args:
        relative_roughness: ε/D (dimensionless)
        reynolds_number: Re = ρvD/μ

    Returns:
        Darcy friction factor f

    Examples:
        >>> estimate_friction_factor(0.001, 50000)
        0.0245
    """
    if reynolds_number < 2300:
        # Laminar flow: f = 64/Re
        return 64.0 / reynolds_number
    if reynolds_number < 4000:
        # Transition region: linear interpolation
        f_laminar = 64.0 / 2300
        f_turbulent = estimate_friction_factor(relative_roughness, 4000)
        t = (reynolds_number - 2300) / (4000 - 2300)
        return f_laminar + t * (f_turbulent - f_laminar)
    # Turbulent flow: Haaland approximation
    # 1/√f = -1.8 * log₁₀[(ε/D/3.7)^1.11 + 6.9/Re]
    term1 = (relative_roughness / 3.7) ** 1.11
    term2 = 6.9 / reynolds_number

    inv_sqrt_f = -1.8 * math.log10(term1 + term2)
    return 1.0 / (inv_sqrt_f**2)


def estimate_flow_rate_darcy_weisbach(
    pressure_drop_pa: float,
    diameter_m: float,
    length_m: float,
    temperature_c: float,
    material: EdgeMaterial | str,
) -> float:
    """Estimate flow rate using Darcy-Weisbach equation.

    Solves: ΔP = f * (L/D) * (ρ * v²/2)
    For flow rate: Q = A * v

    Iterative solution since f depends on Re which depends on v.

    Args:
        pressure_drop_pa: Pressure drop in Pascals (ΔP > 0)
        diameter_m: Pipe diameter in meters
        length_m: Pipe length in meters
        temperature_c: Average fluid temperature in °C
        material: Pipe material (for roughness)

    Returns:
        Flow rate in L/min (0 if ΔP ≤ 0)

    Examples:
        >>> estimate_flow_rate_darcy_weisbach(
        ...     pressure_drop_pa=200000,  # 2 bar
        ...     diameter_m=0.025,         # 25 mm
        ...     length_m=5.0,
        ...     temperature_c=60.0,
        ...     material=EdgeMaterial.STEEL
        ... )
        115.3
    """
    if pressure_drop_pa <= 0:
        return 0.0

    # Fluid properties
    rho = get_fluid_density(temperature_c)  # kg/m³
    nu = get_fluid_viscosity(temperature_c)  # mm²/s = 1e-6 m²/s
    mu = rho * nu * 1e-6  # Dynamic viscosity (Pa·s)

    # Geometry
    area = math.pi * (diameter_m / 2) ** 2  # m²
    roughness = get_material_roughness(material)  # mm
    relative_roughness = (roughness / 1000) / diameter_m  # dimensionless

    # Initial guess for velocity (assuming f ≈ 0.02)
    f_guess = 0.02
    v_guess = math.sqrt((2 * pressure_drop_pa) / (rho * f_guess * (length_m / diameter_m)))

    # Iterative refinement (max 10 iterations)
    velocity = v_guess
    for _ in range(10):
        # Reynolds number
        Re = (rho * velocity * diameter_m) / mu

        # Friction factor
        f = estimate_friction_factor(relative_roughness, Re)

        # Updated velocity
        velocity_new = math.sqrt((2 * pressure_drop_pa) / (rho * f * (length_m / diameter_m)))

        # Check convergence
        if abs(velocity_new - velocity) / velocity < 0.01:  # 1% tolerance
            velocity = velocity_new
            break

        velocity = velocity_new

    # Flow rate
    Q_m3s = area * velocity  # m³/s
    return Q_m3s * 60000  # L/min


# ============================================================================
# EdgeFeatureComputer
# ============================================================================


class EdgeFeatureComputer:
    """Compute dynamic edge features from sensor data.

    Takes sensor readings and static edge configuration, computes:
    - Flow rate (physics-based estimation)
    - Pressure drop (direct calculation)
    - Temperature delta (direct calculation)
    - Vibration level (average from adjacent components)
    - Age (from install date)
    - Maintenance score (decay-based)

    Examples:
        >>> computer = EdgeFeatureComputer()
        >>> features = computer.compute_edge_features(
        ...     edge=EdgeSpec(...),
        ...     sensor_readings={
        ...         "pump_1": ComponentSensorReading(pressure_bar=150, ...),
        ...         "valve_1": ComponentSensorReading(pressure_bar=148, ...)
        ...     },
        ...     current_time=datetime.now()
        ... )
        >>> features.keys()
        dict_keys(['flow_rate_lpm', 'pressure_drop_bar', ...])
    """

    def __init__(self):
        """Initialize EdgeFeatureComputer."""

    def compute_edge_features(
        self,
        edge: EdgeSpec,
        sensor_readings: dict[str, ComponentSensorReading],
        current_time: datetime,
    ) -> dict[str, float]:
        """Compute all dynamic edge features.

        Args:
            edge: Edge specification with static properties
            sensor_readings: Sensor readings per component
            current_time: Current timestamp

        Returns:
            Dictionary with 6 dynamic features

        Raises:
            KeyError: If source or target component not in sensor_readings
        """
        # Get sensor data for source and target
        src_sensors = sensor_readings.get(edge.source_id)
        tgt_sensors = sensor_readings.get(edge.target_id)

        if not src_sensors or not tgt_sensors:
            msg = f"Missing sensor data for edge {edge.source_id}->{edge.target_id}"
            raise KeyError(msg)

        # 1. Pressure drop (direct calculation)
        pressure_drop = self._compute_pressure_drop(src_sensors, tgt_sensors)

        # 2. Temperature delta (direct calculation)
        temp_delta = self._compute_temperature_delta(src_sensors, tgt_sensors)

        # 3. Average temperature (for flow estimation)
        avg_temp = (src_sensors.temperature_c + tgt_sensors.temperature_c) / 2

        # 4. Flow rate (Darcy-Weisbach estimation)
        flow_rate = self._estimate_flow_rate(
            edge=edge, pressure_drop_bar=pressure_drop, temperature_c=avg_temp
        )

        # 5. Vibration level (average)
        vibration = self._compute_vibration_level(src_sensors, tgt_sensors)

        # 6. Age (from install date or edge.age_hours)
        age_hours = edge.get_age_hours(current_time)

        # 7. Maintenance score (from last maintenance or edge method)
        maintenance_score = edge.get_maintenance_score(current_time)

        return {
            "flow_rate_lpm": flow_rate,
            "pressure_drop_bar": pressure_drop,
            "temperature_delta_c": temp_delta,
            "vibration_level_g": vibration,
            "age_hours": age_hours,
            "maintenance_score": maintenance_score,
        }

    def _compute_pressure_drop(
        self, src_sensors: ComponentSensorReading, tgt_sensors: ComponentSensorReading
    ) -> float:
        """Compute pressure drop across edge.

        ΔP = P_source - P_target

        Args:
            src_sensors: Source component sensors
            tgt_sensors: Target component sensors

        Returns:
            Pressure drop in bar (can be negative for backflow)
        """
        return src_sensors.pressure_bar - tgt_sensors.pressure_bar

    def _compute_temperature_delta(
        self, src_sensors: ComponentSensorReading, tgt_sensors: ComponentSensorReading
    ) -> float:
        """Compute temperature difference across edge.

        ΔT = T_source - T_target

        Args:
            src_sensors: Source component sensors
            tgt_sensors: Target component sensors

        Returns:
            Temperature delta in °C (can be negative)
        """
        return src_sensors.temperature_c - tgt_sensors.temperature_c

    def _estimate_flow_rate(
        self, edge: EdgeSpec, pressure_drop_bar: float, temperature_c: float
    ) -> float:
        """Estimate flow rate using Darcy-Weisbach.

        Args:
            edge: Edge with geometry
            pressure_drop_bar: Pressure drop in bar
            temperature_c: Average temperature in °C

        Returns:
            Flow rate in L/min (0 if ΔP ≤ 0)
        """
        if pressure_drop_bar <= 0:
            return 0.0

        # Convert units
        pressure_drop_pa = pressure_drop_bar * 1e5  # bar → Pa
        diameter_m = edge.diameter_mm / 1000  # mm → m

        # Call physics-based estimator
        return estimate_flow_rate_darcy_weisbach(
            pressure_drop_pa=pressure_drop_pa,
            diameter_m=diameter_m,
            length_m=edge.length_m,
            temperature_c=temperature_c,
            material=edge.material,
        )

    def _compute_vibration_level(
        self, src_sensors: ComponentSensorReading, tgt_sensors: ComponentSensorReading
    ) -> float:
        """Compute average vibration level at connection.

        Takes average of source and target vibration sensors.
        Returns 0 if no vibration sensors available.

        Args:
            src_sensors: Source component sensors
            tgt_sensors: Target component sensors

        Returns:
            Average vibration in g
        """
        vibrations = []

        if src_sensors.vibration_g is not None:
            vibrations.append(src_sensors.vibration_g)

        if tgt_sensors.vibration_g is not None:
            vibrations.append(tgt_sensors.vibration_g)

        if not vibrations:
            return 0.0

        return sum(vibrations) / len(vibrations)

    def compute_all_edges(
        self,
        edges: list[EdgeSpec],
        sensor_readings: dict[str, ComponentSensorReading],
        current_time: datetime,
    ) -> dict[str, dict[str, float]]:
        """Compute features for all edges in topology.

        Args:
            edges: List of edge specifications
            sensor_readings: Sensor readings per component
            current_time: Current timestamp

        Returns:
            Dictionary {edge_id: features}

        Examples:
            >>> computer = EdgeFeatureComputer()
            >>> all_features = computer.compute_all_edges(
            ...     edges=[edge1, edge2],
            ...     sensor_readings={...},
            ...     current_time=datetime.now()
            ... )
            >>> all_features["pump_1->valve_1"]["flow_rate_lpm"]
            115.3
        """
        all_features = {}

        for edge in edges:
            edge_id = f"{edge.source_id}->{edge.target_id}"

            try:
                features = self.compute_edge_features(
                    edge=edge, sensor_readings=sensor_readings, current_time=current_time
                )
                all_features[edge_id] = features
            except KeyError:
                # Log warning but continue
                all_features[edge_id] = self._get_default_features()

        return all_features

    def _get_default_features(self) -> dict[str, float]:
        """Get default features when computation fails.

        Returns all features as 0 (or neutral values).

        Returns:
            Dictionary with default values
        """
        return {
            "flow_rate_lpm": 0.0,
            "pressure_drop_bar": 0.0,
            "temperature_delta_c": 0.0,
            "vibration_level_g": 0.0,
            "age_hours": 0.0,
            "maintenance_score": 0.5,  # Neutral
        }


# ============================================================================
# Convenience Functions
# ============================================================================


def create_edge_feature_computer() -> EdgeFeatureComputer:
    """Create EdgeFeatureComputer instance.

    Factory function for consistency with other modules.

    Returns:
        Configured EdgeFeatureComputer

    Examples:
        >>> computer = create_edge_feature_computer()
        >>> features = computer.compute_edge_features(...)
    """
    return EdgeFeatureComputer()
