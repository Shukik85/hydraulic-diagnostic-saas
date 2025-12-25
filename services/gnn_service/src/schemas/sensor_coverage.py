"""Sensor coverage configuration schemas.

Defines WHICH sensors are physically installed on equipment.
Used by ValueSubstitutionEngine to determine measured vs estimated values.

Python 3.14 features:
    - PEP 649: Deferred annotation evaluation
    - PEP 692: TypedDict with totality
    - Native union types (T | None)

Architecture:
    - Minimal implementation with extension points
    - Validates at least some sensors installed
    - Provides helper methods for coverage queries

Examples:
    >>> # Level 3 equipment (minimal production)
    >>> config = SensorCoverageConfig(
    ...     equipment_id="excavator_001",
    ...     topology_id="boom_circuit",
    ...     installed_sensors={
    ...         "edges": {
    ...             "pump_main__valve_boom": EdgeSensorCoverage(
    ...                 pressure_inlet=True,
    ...                 pressure_outlet=True,
    ...                 flow_meter=True
    ...             )
    ...         },
    ...         "components": {
    ...             "pump_main": ComponentSensorCoverage(
    ...                 rpm=True,
    ...                 current=True
    ...             )
    ...         }
    ...     }
    ... )
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "EdgeSensorCoverage",
    "ComponentSensorCoverage",
    "SensorCoverageConfig",
]


class EdgeSensorCoverageDict(TypedDict, total=False):
    """TypedDict for flexible edge sensor configuration."""

    pressure_inlet: bool
    pressure_outlet: bool
    flow_meter: bool
    temperature: bool
    vibration: bool


class EdgeSensorCoverage(BaseModel):
    """Installed sensors on hydraulic line (edge).

    Minimal implementation with extension points for future sensors.

    Physical Reality:
        Most hydraulic sensors are installed IN hydraulic lines:
        - Pressure transducers at connection points
        - Flow meters through pipes/hoses
        - Temperature sensors in lines
        - Vibration sensors on pipe/hose

    Attributes:
        pressure_inlet: Pressure transducer at edge inlet (source outlet)
        pressure_outlet: Pressure transducer at edge outlet (target inlet)
        flow_meter: Flow meter on edge (expensive, optional)
        temperature: Temperature sensor on edge (optional)
        vibration: Vibration sensor on edge (optional)

    Examples:
        >>> # Level 3: Minimal critical coverage
        >>> coverage = EdgeSensorCoverage(
        ...     pressure_inlet=True,
        ...     pressure_outlet=True,
        ...     flow_meter=True
        ... )
        >>>
        >>> # Level 4: Full coverage
        >>> coverage = EdgeSensorCoverage(
        ...     pressure_inlet=True,
        ...     pressure_outlet=True,
        ...     flow_meter=True,
        ...     temperature=True,
        ...     vibration=True
        ... )
    """

    # Core sensors (Level 3 minimum)
    pressure_inlet: bool = Field(
        default=False, description="Pressure sensor at edge inlet (source component outlet)"
    )
    pressure_outlet: bool = Field(
        default=False, description="Pressure sensor at edge outlet (target component inlet)"
    )
    flow_meter: bool = Field(default=False, description="Flow meter measuring flow through edge")

    # Advanced sensors (Level 4+)
    temperature: bool = Field(
        default=False, description="Temperature sensor measuring fluid temp in edge"
    )
    vibration: bool = Field(
        default=False, description="Vibration sensor monitoring pipe/hose vibration"
    )

    # EXTENSION POINT: Add new sensor types here in future
    # acoustic_emission: bool = False  # Phase 2
    # particle_count: bool = False      # Phase 2

    def has_any_sensor(self) -> bool:
        """Check if at least one sensor installed on this edge.

        Returns:
            True if any sensor is installed, False otherwise

        Examples:
            >>> coverage = EdgeSensorCoverage()
            >>> coverage.has_any_sensor()
            False
            >>> coverage = EdgeSensorCoverage(pressure_inlet=True)
            >>> coverage.has_any_sensor()
            True
        """
        return any(
            [
                self.pressure_inlet,
                self.pressure_outlet,
                self.flow_meter,
                self.temperature,
                self.vibration,
            ]
        )

    def has_pressure_monitoring(self) -> bool:
        """Check if edge has pressure drop monitoring capability.

        Pressure drop monitoring requires BOTH inlet and outlet pressure sensors.
        This enables detection of:
        - Leaks (excessive pressure drop)
        - Blockages (excessive pressure drop + reduced flow)
        - Wear/contamination (increasing pressure drop over time)

        Returns:
            True if both pressure sensors installed

        Examples:
            >>> coverage = EdgeSensorCoverage(
            ...     pressure_inlet=True,
            ...     pressure_outlet=True
            ... )
            >>> coverage.has_pressure_monitoring()
            True
        """
        return self.pressure_inlet and self.pressure_outlet

    def sensor_count(self) -> int:
        """Count number of installed sensors.

        Returns:
            Total number of installed sensors on this edge

        Examples:
            >>> coverage = EdgeSensorCoverage(
            ...     pressure_inlet=True,
            ...     pressure_outlet=True,
            ...     flow_meter=True
            ... )
            >>> coverage.sensor_count()
            3
        """
        return sum(
            [
                self.pressure_inlet,
                self.pressure_outlet,
                self.flow_meter,
                self.temperature,
                self.vibration,
            ]
        )


class ComponentSensorCoverage(BaseModel):
    """Installed internal sensors on hydraulic component.

    Minimal implementation with extension points.

    Physical Reality:
        Component internal sensors monitor:
        - Pumps: RPM, current, vibration
        - Valves: Position feedback, current
        - Cylinders: Stroke position, chamber pressure
        - Motors: RPM, current, voltage

    Attributes:
        rpm: RPM sensor on rotating components (pumps, motors)
        position: Position feedback on actuators/valves
        current: Current sensor on electric motors
        voltage: Voltage sensor on electric motors
        vibration: Vibration sensor on rotating components

    Examples:
        >>> # Pump monitoring
        >>> coverage = ComponentSensorCoverage(
        ...     rpm=True,
        ...     current=True,
        ...     vibration=True
        ... )
        >>>
        >>> # Proportional valve monitoring
        >>> coverage = ComponentSensorCoverage(
        ...     position=True,
        ...     current=True
        ... )
    """

    # Core sensors
    rpm: bool = Field(default=False, description="RPM sensor on rotating components (pumps, motors)")
    position: bool = Field(
        default=False, description="Position sensor on actuators/valves (stroke, spool position)"
    )
    current: bool = Field(
        default=False, description="Current sensor on electric motors/solenoids"
    )
    voltage: bool = Field(default=False, description="Voltage sensor on electric components")
    vibration: bool = Field(
        default=False, description="Vibration sensor on rotating/moving components"
    )

    # EXTENSION POINT: Add new sensor types here
    # temperature_internal: bool = False  # Phase 2
    # torque: bool = False                # Phase 2

    def has_any_sensor(self) -> bool:
        """Check if component has any internal sensors.

        Returns:
            True if any sensor is installed

        Examples:
            >>> coverage = ComponentSensorCoverage()
            >>> coverage.has_any_sensor()
            False
            >>> coverage = ComponentSensorCoverage(rpm=True)
            >>> coverage.has_any_sensor()
            True
        """
        return any([self.rpm, self.position, self.current, self.voltage, self.vibration])

    def sensor_count(self) -> int:
        """Count installed sensors.

        Returns:
            Total number of installed sensors

        Examples:
            >>> coverage = ComponentSensorCoverage(
            ...     rpm=True,
            ...     current=True
            ... )
            >>> coverage.sensor_count()
            2
        """
        return sum([self.rpm, self.position, self.current, self.voltage, self.vibration])


class SensorCoverageConfig(BaseModel):
    """Complete sensor installation coverage for equipment.

    Defines WHICH sensors are physically installed on equipment.
    Used by ValueSubstitutionEngine to determine which values are
    measured vs estimated.

    Minimal implementation:
        - Validates at least some sensors installed
        - Provides helper methods for querying coverage
        - Extension points for future analytics

    Architecture:
        Three-layer model:
        1. Physical Layer: Actual sensors on equipment
        2. Mapping Layer: SensorCoverageConfig (THIS)
        3. Logical Layer: ValueSubstitutionEngine uses this config

    Attributes:
        equipment_id: Equipment identifier (matches TopologyConfig)
        topology_id: Topology identifier (matches TopologyConfig)
        version: Configuration version for tracking changes
        installed_sensors: Dict mapping element IDs to sensor coverage

    Examples:
        >>> # Level 3 equipment (minimal production)
        >>> config = SensorCoverageConfig(
        ...     equipment_id="excavator_001",
        ...     topology_id="boom_circuit",
        ...     installed_sensors={
        ...         "edges": {
        ...             "pump_main__valve_boom": EdgeSensorCoverage(
        ...                 pressure_inlet=True,
        ...                 pressure_outlet=True,
        ...                 flow_meter=True
        ...             ),
        ...             "valve_boom__cylinder": EdgeSensorCoverage(
        ...                 pressure_inlet=True  # Only inlet
        ...             ),
        ...         },
        ...         "components": {
        ...             "pump_main": ComponentSensorCoverage(
        ...                 rpm=True, current=True
        ...             )
        ...         },
        ...     },
        ... )
        >>> config.total_sensor_count()
        5
    """

    equipment_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Equipment identifier (must match TopologyConfig)",
    )

    topology_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Topology identifier (must match TopologyConfig)",
    )

    version: str = Field(
        default="v1.0", pattern=r"^v\d+\.\d+$", description="Configuration version (vX.Y format)"
    )

    installed_sensors: dict[str, dict[str, EdgeSensorCoverage | ComponentSensorCoverage]] = Field(
        ...,
        description="Nested dict: {edges: {edge_id: coverage}, components: {comp_id: coverage}}",
    )

    @model_validator(mode="after")
    def validate_minimal_coverage(self) -> SensorCoverageConfig:
        """Validate at least some sensors installed.

        Minimal requirement: At least 1 sensor on ANY element.
        Recommended: 2+ pressure sensors + 1 flow meter (Level 3).

        Raises:
            ValueError: If no sensors installed or invalid coverage types

        Returns:
            Self if validation passes
        """
        total_sensors = 0

        # Count edge sensors
        for edge_id, edge_coverage in self.installed_sensors.get("edges", {}).items():
            if not isinstance(edge_coverage, EdgeSensorCoverage):
                raise ValueError(
                    f"Edge '{edge_id}' coverage must be EdgeSensorCoverage, "
                    f"got {type(edge_coverage).__name__}"
                )
            total_sensors += edge_coverage.sensor_count()

        # Count component sensors
        for comp_id, comp_coverage in self.installed_sensors.get("components", {}).items():
            if not isinstance(comp_coverage, ComponentSensorCoverage):
                raise ValueError(
                    f"Component '{comp_id}' coverage must be ComponentSensorCoverage, "
                    f"got {type(comp_coverage).__name__}"
                )
            total_sensors += comp_coverage.sensor_count()

        if total_sensors == 0:
            raise ValueError(
                "At least one sensor must be installed. "
                "SensorCoverageConfig requires minimal sensor coverage."
            )

        return self

    # HELPER METHODS (extension point for future analytics)

    def total_sensor_count(self) -> int:
        """Count total installed sensors across all elements.

        Returns:
            Total number of sensors

        Examples:
            >>> config.total_sensor_count()
            8
        """
        total = 0
        for coverage in self.installed_sensors.get("edges", {}).values():
            total += coverage.sensor_count()
        for coverage in self.installed_sensors.get("components", {}).values():
            total += coverage.sensor_count()
        return total

    def get_edge_coverage(self, edge_id: str) -> EdgeSensorCoverage | None:
        """Get sensor coverage for specific edge.

        Args:
            edge_id: Edge identifier (e.g., "pump__valve")

        Returns:
            EdgeSensorCoverage if edge exists, None otherwise

        Examples:
            >>> coverage = config.get_edge_coverage("pump__valve")
            >>> if coverage and coverage.has_pressure_monitoring():
            ...     print("Can detect leaks!")
        """
        return self.installed_sensors.get("edges", {}).get(edge_id)

    def get_component_coverage(self, component_id: str) -> ComponentSensorCoverage | None:
        """Get sensor coverage for specific component.

        Args:
            component_id: Component identifier (e.g., "pump_main")

        Returns:
            ComponentSensorCoverage if component exists, None otherwise
        """
        return self.installed_sensors.get("components", {}).get(component_id)

    def has_edge_sensor(self, edge_id: str) -> bool:
        """Check if edge has any installed sensors.

        Args:
            edge_id: Edge identifier

        Returns:
            True if edge has at least one sensor

        Examples:
            >>> if config.has_edge_sensor("pump__valve"):
            ...     # Use measured values
            ... else:
            ...     # Use estimated values
        """
        coverage = self.get_edge_coverage(edge_id)
        return coverage is not None and coverage.has_any_sensor()

    def has_component_sensor(self, component_id: str) -> bool:
        """Check if component has any internal sensors.

        Args:
            component_id: Component identifier

        Returns:
            True if component has at least one sensor
        """
        coverage = self.get_component_coverage(component_id)
        return coverage is not None and coverage.has_any_sensor()
