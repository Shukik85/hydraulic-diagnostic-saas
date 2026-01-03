"""Feature definitions for hydraulic system graphs.

Defines the 34-dimensional node features and 14-dimensional edge features
used in hydraulic system modeling.

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NodeFeatures:
    """34-dimensional node feature vector for hydraulic components.
    
    Features are organized into physical, operational, and state categories.
    """
    
    # Physical measurements (10 features)
    pressure: float  # bar (0-400)
    flow_rate: float  # l/min (0-500)
    temperature: float  # °C (20-120)
    position: float  # % (0-100) - for actuators
    velocity: float  # mm/s (0-1000) - for actuators
    torque: float  # Nm (0-5000) - for motors
    force: float  # kN (0-500) - for cylinders
    vibration: float  # m/s² (0-50)
    noise_level: float  # dB (0-100)
    power: float  # kW (0-200)
    
    # Valve states (5 features)
    valve_opening: float  # % (0-100)
    valve_pressure_drop: float  # bar (0-50)
    valve_flow_coefficient: float  # dimensionless (0-1)
    pilot_pressure: float  # bar (0-30)
    spool_position: float  # % (0-100)
    
    # Component health indicators (6 features)
    wear_level: float  # % (0-100)
    contamination: float  # ISO code (0-25)
    seal_condition: float  # % (0-100)
    internal_leakage: float  # l/min (0-20)
    external_leakage: float  # l/min (0-5)
    efficiency: float  # % (0-100)
    
    # Temporal features (4 features)
    pressure_rate_change: float  # bar/s (-100 to 100)
    flow_rate_change: float  # l/min/s (-200 to 200)
    temperature_rate_change: float  # °C/s (-5 to 5)
    position_rate_change: float  # %/s (-50 to 50)
    
    # Component type encoding (5 features - one-hot)
    is_pump: float  # binary
    is_valve: float  # binary
    is_actuator: float  # binary
    is_sensor: float  # binary
    is_filter: float  # binary
    
    # Operational context (4 features)
    load_factor: float  # % (0-150)
    duty_cycle: float  # % (0-100)
    operating_hours: float  # normalized (0-1)
    ambient_temperature: float  # °C (-20 to 50)
    
    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Return list of all feature names."""
        return [
            # Physical
            'pressure', 'flow_rate', 'temperature', 'position', 'velocity',
            'torque', 'force', 'vibration', 'noise_level', 'power',
            # Valve states
            'valve_opening', 'valve_pressure_drop', 'valve_flow_coefficient',
            'pilot_pressure', 'spool_position',
            # Health
            'wear_level', 'contamination', 'seal_condition',
            'internal_leakage', 'external_leakage', 'efficiency',
            # Temporal
            'pressure_rate_change', 'flow_rate_change',
            'temperature_rate_change', 'position_rate_change',
            # Type encoding
            'is_pump', 'is_valve', 'is_actuator', 'is_sensor', 'is_filter',
            # Operational
            'load_factor', 'duty_cycle', 'operating_hours', 'ambient_temperature'
        ]
    
    @classmethod
    def get_physical_ranges(cls) -> dict[str, tuple[float, float]]:
        """Return valid physical ranges for each feature."""
        return {
            'pressure': (0.0, 400.0),
            'flow_rate': (0.0, 500.0),
            'temperature': (20.0, 120.0),
            'position': (0.0, 100.0),
            'velocity': (0.0, 1000.0),
            'torque': (0.0, 5000.0),
            'force': (0.0, 500.0),
            'vibration': (0.0, 50.0),
            'noise_level': (0.0, 100.0),
            'power': (0.0, 200.0),
            'valve_opening': (0.0, 100.0),
            'valve_pressure_drop': (0.0, 50.0),
            'valve_flow_coefficient': (0.0, 1.0),
            'pilot_pressure': (0.0, 30.0),
            'spool_position': (0.0, 100.0),
            'wear_level': (0.0, 100.0),
            'contamination': (0.0, 25.0),
            'seal_condition': (0.0, 100.0),
            'internal_leakage': (0.0, 20.0),
            'external_leakage': (0.0, 5.0),
            'efficiency': (0.0, 100.0),
            'pressure_rate_change': (-100.0, 100.0),
            'flow_rate_change': (-200.0, 200.0),
            'temperature_rate_change': (-5.0, 5.0),
            'position_rate_change': (-50.0, 50.0),
            'is_pump': (0.0, 1.0),
            'is_valve': (0.0, 1.0),
            'is_actuator': (0.0, 1.0),
            'is_sensor': (0.0, 1.0),
            'is_filter': (0.0, 1.0),
            'load_factor': (0.0, 150.0),
            'duty_cycle': (0.0, 100.0),
            'operating_hours': (0.0, 1.0),
            'ambient_temperature': (-20.0, 50.0),
        }
    
    @classmethod
    def dimension(cls) -> int:
        """Return total feature dimension."""
        return 34


@dataclass
class EdgeFeatures:
    """14-dimensional edge feature vector for hydraulic connections.
    
    Represents flow paths between components.
    """
    
    # Flow characteristics (6 features)
    flow_rate: float  # l/min (0-500)
    pressure_drop: float  # bar (0-100)
    reynolds_number: float  # dimensionless (normalized 0-1)
    flow_velocity: float  # m/s (0-10)
    pipe_diameter: float  # mm (6-100)
    pipe_length: float  # m (0.1-50)
    
    # Connection state (3 features)
    is_active: float  # binary (0 or 1)
    valve_restriction: float  # % (0-100)
    flow_direction: float  # -1 (reverse), 0 (blocked), 1 (forward)
    
    # Fluid properties (3 features)
    fluid_viscosity: float  # cSt (10-100)
    fluid_density: float  # kg/m³ (850-950)
    fluid_temperature: float  # °C (20-120)
    
    # Connection health (2 features)
    pipe_roughness: float  # mm (0.001-0.5)
    contamination_level: float  # ISO code (0-25)
    
    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Return list of all feature names."""
        return [
            # Flow
            'flow_rate', 'pressure_drop', 'reynolds_number', 'flow_velocity',
            'pipe_diameter', 'pipe_length',
            # State
            'is_active', 'valve_restriction', 'flow_direction',
            # Fluid
            'fluid_viscosity', 'fluid_density', 'fluid_temperature',
            # Health
            'pipe_roughness', 'contamination_level'
        ]
    
    @classmethod
    def get_physical_ranges(cls) -> dict[str, tuple[float, float]]:
        """Return valid physical ranges for each feature."""
        return {
            'flow_rate': (0.0, 500.0),
            'pressure_drop': (0.0, 100.0),
            'reynolds_number': (0.0, 1.0),  # normalized
            'flow_velocity': (0.0, 10.0),
            'pipe_diameter': (6.0, 100.0),
            'pipe_length': (0.1, 50.0),
            'is_active': (0.0, 1.0),
            'valve_restriction': (0.0, 100.0),
            'flow_direction': (-1.0, 1.0),
            'fluid_viscosity': (10.0, 100.0),
            'fluid_density': (850.0, 950.0),
            'fluid_temperature': (20.0, 120.0),
            'pipe_roughness': (0.001, 0.5),
            'contamination_level': (0.0, 25.0),
        }
    
    @classmethod
    def dimension(cls) -> int:
        """Return total feature dimension."""
        return 14


@dataclass
class GraphLabels:
    """Multi-label annotations for training."""
    
    # Node-level labels (5 classes)
    node_health_status: list[int] = field(default_factory=list)
    # 0: healthy, 1: degraded, 2: worn, 3: leaking, 4: failed
    
    # Graph-level labels (4 classes)
    system_anomaly_type: int = 0
    # 0: normal, 1: parallel_overload, 2: sequential_cascade, 3: cavitation
    
    @classmethod
    def get_node_class_names(cls) -> list[str]:
        """Return node health class names."""
        return ['healthy', 'degraded', 'worn', 'leaking', 'failed']
    
    @classmethod
    def get_graph_class_names(cls) -> list[str]:
        """Return system anomaly class names."""
        return ['normal', 'parallel_overload', 'sequential_cascade', 'cavitation']
    
    @classmethod
    def num_node_classes(cls) -> int:
        """Return number of node-level classes."""
        return 5
    
    @classmethod
    def num_graph_classes(cls) -> int:
        """Return number of graph-level classes."""
        return 4
