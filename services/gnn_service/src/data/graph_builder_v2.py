"""GraphBuilderV2: Edge-Centric Graph Construction.

Phase 3.2: Transition from node-centric to edge-centric sensor placement.

Key Changes from GraphBuilder:
    - Edge features: Primary (rich 14-116D with edge sensors)
    - Node features: Secondary (minimal 29D internal sensors only)
    - Accepts HybridInferenceRequest (edge_readings + component_readings)
    - Physical reality: Sensors IN pipes (edges), not ON components (nodes)

Architecture:
    Nodes (29D):
        - 4D: Internal component sensors (rpm, position, current, voltage)
        - 25D: Component type one-hot encoding (25 real hydraulic types)
    
    Edges (14-116D based on config):
        - 8D: Static physical features (diameter, length, material, ...)
        - 6D: Dynamic instant features (pressure_drop, flow, temp, vibration)
        - 34D per sensor: Time-series statistical features (optional)

Physical Correctness:
    ❌ Node-centric (WRONG):
        pump: [pressure=150bar, flow=115lpm]  ← Where? Inlet/Outlet unclear!
    
    ✅ Edge-centric (CORRECT):
        pump__valve: EdgeSensorReading(
            pressure_inlet_bar=150,   ← At pump outlet (clear!)
            pressure_outlet_bar=148,  ← At valve inlet (clear!)
            pressure_drop_bar=2.0,    ← Direct measurement!
            flow_rate_lpm=115         ← Flow THROUGH pipe (physical!)
        )

Expected Improvements:
    - Anomaly Detection: +40-60%
    - Leak Localization: +70% (edge-level precision)
    - Pressure Drop Detection: +80% (direct measurement)
    - RUL Prediction: +50% (flow-based per-edge)

References:
    - EDGE_CENTRIC_MIGRATION.md: Architecture details
    - PRODUCTION_ROADMAP.md: Week 1 implementation plan
    - schemas/requests.py: HybridInferenceRequest, EdgeSensorReading

Examples:
    >>> from src.schemas.requests import HybridInferenceRequest, EdgeSensorReading, ComponentSensorReading
    >>> from src.data.graph_builder_v2 import GraphBuilderV2
    >>> from datetime import datetime, UTC
    >>> 
    >>> # Create builder
    >>> builder = GraphBuilderV2(
    ...     feature_engineer=feature_engineer,
    ...     feature_config=FeatureConfig(edge_in_dim=14)  # 8 static + 6 dynamic
    ... )
    >>> 
    >>> # Hybrid request (edge-centric!)
    >>> request = HybridInferenceRequest(
    ...     equipment_id="excavator_001",
    ...     timestamp=datetime.now(UTC),
    ...     topology_id="boom_circuit",
    ...     edge_readings={
    ...         "pump_main__valve_boom": EdgeSensorReading(
    ...             edge_id="pump_main__valve_boom",
    ...             pressure_inlet_bar=250.2,
    ...             pressure_outlet_bar=248.5,
    ...             flow_rate_lpm=180.5,
    ...             temperature_c=68.3,
    ...             timestamp=datetime.now(UTC)
    ...         )
    ...     },
    ...     component_readings={
    ...         "pump_main": ComponentSensorReading(
    ...             component_id="pump_main",
    ...             rpm=1800,
    ...             current_a=35.2,
    ...             timestamp=datetime.now(UTC)
    ...         )
    ...     }
    ... )
    >>> 
    >>> # Build graph
    >>> graph = builder.build_graph_hybrid(request, topology)
    >>> print(graph.x.shape)  # [N, 29] - Minimal node features
    >>> print(graph.edge_attr.shape)  # [E, 14] - Rich edge features
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.edge_features import EdgeFeatureComputer, create_edge_feature_computer
from src.data.feature_config import FeatureConfig
from src.data.feature_engineer import FeatureEngineer
from src.data.normalization import EdgeFeatureNormalizer, create_edge_feature_normalizer

if TYPE_CHECKING:
    import pandas as pd

    from src.schemas import EdgeSpec, GraphTopology
    from src.schemas.requests import (
        ComponentSensorReading,
        EdgeSensorReading,
        HybridInferenceRequest,
    )

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Node feature dimensions
NODE_INTERNAL_SENSORS_DIM = 4  # rpm, position, current, voltage
NODE_COMPONENT_TYPE_DIM = 25  # One-hot encoding for component types (EXPANDED!)
NODE_FEATURES_TOTAL_DIM = NODE_INTERNAL_SENSORS_DIM + NODE_COMPONENT_TYPE_DIM  # 29D

# Component type encoding (25 types - REAL HYDRAULIC COMPONENTS)
# Based on ISO 5598:2020 Fluid power systems and components
# Covers mobile + industrial hydraulics
COMPONENT_TYPE_MAPPING = {
    # === ENERGY CONVERSION (Pumps & Motors) ===
    "pump": 0,  # General hydraulic pump
    "gear_pump": 1,  # Fixed displacement gear pump
    "piston_pump": 2,  # Variable displacement piston pump (A10VSO, etc.)
    "vane_pump": 3,  # Variable displacement vane pump
    "hydraulic_motor": 4,  # Rotational hydraulic motor
    "orbital_motor": 5,  # Low-speed high-torque orbital motor
    
    # === CONTROL VALVES ===
    "directional_valve": 6,  # 2/2, 3/2, 4/3, etc. directional control valve
    "proportional_valve": 7,  # Proportional directional valve (variable flow)
    "servo_valve": 8,  # High-precision servo valve
    "relief_valve": 9,  # Pressure relief/safety valve
    "check_valve": 10,  # One-way check valve
    "flow_control_valve": 11,  # Flow control/throttle valve
    "pressure_reducing_valve": 12,  # Pressure reducing valve
    "sequence_valve": 13,  # Sequence valve
    
    # === ACTUATORS ===
    "cylinder": 14,  # Linear hydraulic cylinder
    "telescopic_cylinder": 15,  # Multi-stage telescopic cylinder
    
    # === CONDITIONING & STORAGE ===
    "accumulator": 16,  # Hydraulic accumulator (gas/bladder/piston)
    "filter": 17,  # Hydraulic filter (suction/pressure/return)
    "cooler": 18,  # Oil cooler/heat exchanger
    "heater": 19,  # Hydraulic oil heater
    "reservoir": 20,  # Hydraulic tank/reservoir
    
    # === DISTRIBUTION & MONITORING ===
    "manifold": 21,  # Hydraulic manifold block
    "pressure_sensor": 22,  # Pressure transducer/sensor node
    "flow_sensor": 23,  # Flow meter sensor node
    "temperature_sensor": 24,  # Temperature sensor node
}

# Reverse mapping for debugging
COMPONENT_TYPE_NAMES = {v: k for k, v in COMPONENT_TYPE_MAPPING.items()}

# Edge feature dimensions
EDGE_STATIC_DIM = 8  # Physical properties (diameter, length, material, ...)
EDGE_DYNAMIC_INSTANT_DIM = 6  # Instant measurements (pressure_drop, flow, temp, ...)
EDGE_TIMESERIES_DIM_PER_SENSOR = 34  # Statistical features per sensor type


# ============================================================================
# GRAPHBUILDERV2: EDGE-CENTRIC ARCHITECTURE
# ============================================================================


class GraphBuilderV2:
    """Phase 3.2: Edge-centric graph construction.

    Sensor placement:
        - Edges: Pressure, flow, temperature, vibration (PRIMARY!)
        - Nodes: RPM, position, current (SECONDARY, internal only)

    Features:
        - Edge features: 8 static + 6 dynamic + 34*N time-series = 14-116D
        - Node features: 4 internal sensors + 25 component type = 29D

    Args:
        feature_engineer: FeatureEngineer for time-series feature extraction
        feature_config: FeatureConfig with edge_in_dim setting
        edge_feature_computer: EdgeFeatureComputer for dynamic calculations
        edge_normalizer: EdgeFeatureNormalizer for normalization
        use_edge_timeseries: Enable time-series features for edges (default: False)

    Examples:
        >>> # Standard: 14D edges (static + dynamic)
        >>> builder = GraphBuilderV2(
        ...     feature_config=FeatureConfig(edge_in_dim=14),
        ...     use_edge_timeseries=False
        ... )
        >>>
        >>> # Advanced: 48D edges (static + dynamic + 1 sensor time-series)
        >>> builder = GraphBuilderV2(
        ...     feature_config=FeatureConfig(edge_in_dim=48),
        ...     use_edge_timeseries=True
        ... )
    """

    def __init__(
        self,
        feature_engineer: FeatureEngineer | None = None,
        feature_config: FeatureConfig | None = None,
        edge_feature_computer: EdgeFeatureComputer | None = None,
        edge_normalizer: EdgeFeatureNormalizer | None = None,
        use_edge_timeseries: bool = False,
    ):
        """Initialize GraphBuilderV2.

        Args:
            feature_engineer: FeatureEngineer instance (default: create new)
            feature_config: FeatureConfig instance (default: FeatureConfig())
            edge_feature_computer: EdgeFeatureComputer (default: create new)
            edge_normalizer: EdgeFeatureNormalizer (default: create new)
            use_edge_timeseries: Enable time-series features (default: False)
        """
        self.feature_config = feature_config or FeatureConfig()
        self.feature_engineer = feature_engineer or FeatureEngineer(self.feature_config)
        self.edge_feature_computer = edge_feature_computer or create_edge_feature_computer()
        self.edge_normalizer = edge_normalizer or create_edge_feature_normalizer()
        self.use_edge_timeseries = use_edge_timeseries

        logger.info(
            f"GraphBuilderV2 initialized: "
            f"edge_in_dim={self.feature_config.edge_in_dim}, "
            f"use_timeseries={use_edge_timeseries}, "
            f"component_types={NODE_COMPONENT_TYPE_DIM}"
        )

    # ========================================================================
    # NODE FEATURES (MINIMAL 29D)
    # ========================================================================

    def build_node_features_v2(
        self,
        component_id: str,
        component_reading: ComponentSensorReading | None,
        component_type: str | None = None,
    ) -> torch.Tensor:
        """Build MINIMAL node features from internal component sensors.

        Node features (29D total):
            [0-3]: Internal sensor features (4D)
                - rpm (normalized to 0-1, max 3000 RPM)
                - position_percent (normalized to 0-1)
                - current_a (normalized to 0-1, max 100A)
                - voltage_v (normalized to 0-1, max 500V)

            [4-28]: Component type one-hot (25D)
                - 25 real hydraulic component types (ISO 5598:2020)
                - Covers: pumps, motors, valves, actuators, conditioning, sensors

        Args:
            component_id: Component identifier (e.g., "pump_main", "proportional_valve_boom")
            component_reading: ComponentSensorReading with internal sensors (optional)
            component_type: Component type string (optional, inferred from component_id)

        Returns:
            features: Tensor [29] with minimal node features

        Examples:
            >>> # Piston pump with RPM and current
            >>> from datetime import datetime, UTC
            >>> reading = ComponentSensorReading(
            ...     component_id="piston_pump_main",
            ...     rpm=1450,
            ...     current_a=25.5,
            ...     timestamp=datetime.now(UTC)
            ... )
            >>> features = builder.build_node_features_v2("piston_pump_main", reading, "piston_pump")
            >>> print(features.shape)  # torch.Size([29])
            >>> print(features[:4])  # [0.483, 0.0, 0.255, 0.0] (rpm, pos, current, voltage)
            >>> print(features[4:])  # [0, 0, 1, 0, ...] (piston_pump one-hot at index 2)
            >>>
            >>> # Proportional valve with position
            >>> reading = ComponentSensorReading(
            ...     component_id="proportional_valve_01",
            ...     position_percent=65.5,
            ...     timestamp=datetime.now(UTC)
            ... )
            >>> features = builder.build_node_features_v2("proportional_valve_01", reading, "proportional_valve")
            >>> print(features[:4])  # [0.0, 0.655, 0.0, 0.0]
            >>> print(features[4:])  # [0, 0, 0, 0, 0, 0, 0, 1, ...] (proportional_valve at index 7)
            >>>
            >>> # Pressure sensor (passive, no internal sensors)
            >>> features = builder.build_node_features_v2("pressure_sensor_01", None, "pressure_sensor")
            >>> print(features[:4])  # [0.0, 0.0, 0.0, 0.0] (all zeros)
            >>> print(features[4:])  # [..., 0, 0, 1, 0, 0] (pressure_sensor at index 22)
        """
        features = []

        # ====================================================================
        # PART 1: Internal Sensor Features (4D)
        # ====================================================================

        if component_reading:
            # 1. RPM (normalized to 3000 RPM max)
            rpm_norm = (
                component_reading.rpm / 3000.0 if component_reading.rpm is not None else 0.0
            )
            rpm_norm = float(np.clip(rpm_norm, 0.0, 1.0))  # Clamp to [0, 1]

            # 2. Position percent (already 0-100, normalize to 0-1)
            position_norm = (
                component_reading.position_percent / 100.0
                if component_reading.position_percent is not None
                else 0.0
            )
            position_norm = float(np.clip(position_norm, 0.0, 1.0))

            # 3. Current (normalized to 100A max)
            current_norm = (
                component_reading.current_a / 100.0
                if component_reading.current_a is not None
                else 0.0
            )
            current_norm = float(np.clip(current_norm, 0.0, 1.0))

            # 4. Voltage (normalized to 500V max)
            voltage_norm = (
                component_reading.voltage_v / 500.0
                if component_reading.voltage_v is not None
                else 0.0
            )
            voltage_norm = float(np.clip(voltage_norm, 0.0, 1.0))

            features.extend([rpm_norm, position_norm, current_norm, voltage_norm])
        else:
            # No component reading → all zeros
            features.extend([0.0, 0.0, 0.0, 0.0])

        # ====================================================================
        # PART 2: Component Type One-Hot (25D)
        # ====================================================================

        # Infer component type from component_id if not provided
        if component_type is None:
            component_type = self._infer_component_type(component_id)

        # Create one-hot encoding
        component_type_onehot = [0.0] * NODE_COMPONENT_TYPE_DIM
        type_idx = COMPONENT_TYPE_MAPPING.get(component_type.lower(), 0)  # Default to pump
        component_type_onehot[type_idx] = 1.0

        features.extend(component_type_onehot)

        # ====================================================================
        # VALIDATION & RETURN
        # ====================================================================

        assert len(features) == NODE_FEATURES_TOTAL_DIM, (
            f"Node features dimension mismatch: {len(features)} != {NODE_FEATURES_TOTAL_DIM}"
        )

        return torch.tensor(features, dtype=torch.float32)

    def _infer_component_type(self, component_id: str) -> str:
        """Infer component type from component_id.

        Heuristic: Look for type keywords in component_id (prioritize specific types).

        Args:
            component_id: Component identifier (e.g., "piston_pump_main", "proportional_valve_boom")

        Returns:
            component_type: Inferred type (default: "pump")

        Examples:
            >>> builder._infer_component_type("piston_pump_main")
            'piston_pump'
            >>> builder._infer_component_type("gear_pump_aux")
            'gear_pump'
            >>> builder._infer_component_type("proportional_valve_boom_01")
            'proportional_valve'
            >>> builder._infer_component_type("servo_valve_steering")
            'servo_valve'
            >>> builder._infer_component_type("telescopic_cylinder_left")
            'telescopic_cylinder'
            >>> builder._infer_component_type("unknown_component")
            'pump'  # Default
        """
        component_id_lower = component_id.lower()

        # Priority order: Specific types BEFORE general types
        # (e.g., "piston_pump" before "pump")
        priority_types = [
            # Specific pumps/motors
            "piston_pump", "gear_pump", "vane_pump", "orbital_motor", "hydraulic_motor",
            # Specific valves
            "proportional_valve", "servo_valve", "relief_valve", "check_valve",
            "flow_control_valve", "pressure_reducing_valve", "sequence_valve", "directional_valve",
            # Specific cylinders
            "telescopic_cylinder",
            # Specific sensors
            "pressure_sensor", "flow_sensor", "temperature_sensor",
            # General types (fallback)
            "pump", "motor", "valve", "cylinder", "accumulator", "filter",
            "cooler", "heater", "reservoir", "manifold", "sensor",
        ]

        for type_name in priority_types:
            # Check for exact word match (e.g., "piston_pump" not "pump_piston")
            # SIM102: Combine nested if into single condition
            if type_name.replace("_", "") in component_id_lower.replace("_", "") and type_name in COMPONENT_TYPE_MAPPING:
                return type_name

        # Default to pump if no match
        logger.warning(
            f"Could not infer component type for '{component_id}', defaulting to 'pump'"
        )
        return "pump"

    # ========================================================================
    # EDGE FEATURES (RICH 14-116D)
    # ========================================================================

    def build_edge_features_v2(
        self,
        edge_spec: EdgeSpec,
        edge_reading: EdgeSensorReading | None,
        edge_history: pd.DataFrame | None = None,
    ) -> torch.Tensor:
        """Build RICH edge features from edge sensors.

        Edge features (variable dimension based on config):
            [0-7]: Static physical features (8D)
                - diameter_mm (normalized)
                - length_m (normalized)
                - cross_section_area (computed)
                - pressure_loss_coeff (computed)
                - pressure_rating_bar (normalized)
                - material_onehot (3D: steel, rubber, composite)

            [8-13]: Dynamic instant features (6D)
                - pressure_drop_bar (inlet - outlet)
                - flow_rate_lpm
                - temperature_c
                - vibration_g
                - age_hours
                - maintenance_score

            [14+]: Time-series features (34D per sensor type, optional)
                - Statistical: mean, std, min, max, percentiles
                - Frequency: FFT components, dominant frequency
                - Temporal: trend, seasonality, autocorrelation

        Total dimensions:
            - 14D: Static + Dynamic (default)
            - 48D: Static + Dynamic + 1 sensor time-series
            - 116D: Static + Dynamic + 3 sensor time-series (pressure, flow, temp)

        Args:
            edge_spec: EdgeSpec with physical properties
            edge_reading: EdgeSensorReading with instant measurements (optional)
            edge_history: DataFrame with time-series data (optional)
                         Columns: [timestamp, pressure, flow, temperature, ...]

        Returns:
            features: Tensor [edge_in_dim] with rich edge features

        Examples:
            >>> # Basic: 14D (static + dynamic)
            >>> from datetime import datetime, UTC
            >>> edge_reading = EdgeSensorReading(
            ...     edge_id="pump__valve",
            ...     pressure_inlet_bar=150.0,
            ...     pressure_outlet_bar=148.0,
            ...     flow_rate_lpm=115.5,
            ...     temperature_c=65.0,
            ...     timestamp=datetime.now(UTC)
            ... )
            >>> features = builder.build_edge_features_v2(edge_spec, edge_reading)
            >>> print(features.shape)  # torch.Size([14])
            >>>
            >>> # Advanced: 48D (static + dynamic + time-series)
            >>> import pandas as pd
            >>> edge_history = pd.DataFrame({
            ...     "pressure": [...],  # Time-series data
            ...     "flow": [...],
            ...     "temperature": [...]
            ... })
            >>> builder_ts = GraphBuilderV2(use_edge_timeseries=True)
            >>> features = builder_ts.build_edge_features_v2(
            ...     edge_spec, edge_reading, edge_history
            ... )
            >>> print(features.shape)  # torch.Size([48]) or more
        """
        all_features = []

        # ====================================================================
        # PART 1: Static Physical Features (8D)
        # ====================================================================

        static_features = self._build_static_edge_features(edge_spec)
        all_features.append(static_features)

        # ====================================================================
        # PART 2: Dynamic Instant Features (6D)
        # ====================================================================

        if edge_reading is not None:
            dynamic_features = self._build_dynamic_edge_features(edge_spec, edge_reading)
        else:
            # No edge reading → all zeros
            dynamic_features = np.zeros(EDGE_DYNAMIC_INSTANT_DIM, dtype=np.float32)
            logger.debug(f"No edge reading for {edge_spec.edge_id}, using zeros for dynamic features")

        all_features.append(dynamic_features)

        # ====================================================================
        # PART 3: Time-Series Features (34D per sensor, optional)
        # ====================================================================

        if self.use_edge_timeseries and edge_history is not None:
            timeseries_features = self._build_timeseries_edge_features(edge_history)
            all_features.append(timeseries_features)

        # ====================================================================
        # CONCATENATE & PAD/TRUNCATE
        # ====================================================================

        # Concatenate all parts
        all_features_array = np.concatenate(all_features)

        # Pad or truncate to match config.edge_in_dim
        if len(all_features_array) < self.feature_config.edge_in_dim:
            # Pad with zeros
            padding = np.zeros(
                self.feature_config.edge_in_dim - len(all_features_array),
                dtype=np.float32,
            )
            all_features_array = np.concatenate([all_features_array, padding])
        elif len(all_features_array) > self.feature_config.edge_in_dim:
            # Truncate
            logger.warning(
                f"Edge features ({len(all_features_array)}D) exceed config.edge_in_dim "
                f"({self.feature_config.edge_in_dim}D), truncating"
            )
            all_features_array = all_features_array[: self.feature_config.edge_in_dim]

        return torch.from_numpy(all_features_array)

    def _build_static_edge_features(self, edge_spec: EdgeSpec) -> np.ndarray:
        """Build static edge features (8D).

        Same as GraphBuilder.build_edge_features_static for compatibility.

        Returns:
            features: Array [8]
        """
        features = []

        # 1. Diameter (normalized to 6-50mm range)
        diameter_norm = edge_spec.diameter_mm / 50.0
        features.append(diameter_norm)

        # 2. Length (normalized to 0.1-10m range)
        length_norm = edge_spec.length_m / 10.0
        features.append(length_norm)

        # 3. Cross-section area (computed)
        radius_m = (edge_spec.diameter_mm / 1000.0) / 2.0
        area = np.pi * radius_m**2
        area_norm = area / 0.002  # Normalize to typical max area
        features.append(area_norm)

        # 4. Pressure loss coefficient (Darcy-Weisbach approx)
        pressure_loss_coeff = edge_spec.length_m / (edge_spec.diameter_mm**4 + 1e-6)
        pressure_loss_coeff_norm = float(np.clip(pressure_loss_coeff * 1000, 0, 1))
        features.append(pressure_loss_coeff_norm)

        # 5. Pressure rating (normalized to 100-400 bar)
        if edge_spec.pressure_rating_bar is not None:
            rating_norm = edge_spec.pressure_rating_bar / 400.0
            features.append(rating_norm)
        else:
            features.append(0.5)  # Default mid-range

        # 6. Material one-hot (3D: steel, rubber, composite)
        material_map = {
            "steel": [1.0, 0.0, 0.0],
            "rubber": [0.0, 1.0, 0.0],
            "composite": [0.0, 0.0, 1.0],
            "thermoplastic": [0.0, 0.0, 1.0],  # Treat as composite
        }
        material = edge_spec.material or "steel"
        # Handle enum
        if hasattr(material, "value"):
            material = material.value
        material_encoding = material_map.get(material.lower(), [1.0, 0.0, 0.0])  # Default steel
        features.extend(material_encoding)

        return np.array(features, dtype=np.float32)

    def _build_dynamic_edge_features(self, edge_spec: EdgeSpec, edge_reading: EdgeSensorReading) -> np.ndarray:
        """Build dynamic edge features (6D) from EdgeSensorReading.

        Features:
            - pressure_drop_bar (direct measurement!)
            - flow_rate_lpm
            - temperature_c
            - vibration_g
            - age_hours (from edge_spec)
            - maintenance_score (from edge_spec)

        Args:
            edge_spec: EdgeSpec with metadata
            edge_reading: EdgeSensorReading with instant measurements

        Returns:
            features: Array [6] (normalized)
        """
        # Raw features
        raw_features = {
            "pressure_drop_bar": (
                edge_reading.pressure_drop_bar
                if edge_reading.pressure_drop_bar is not None
                else (edge_reading.pressure_inlet_bar - edge_reading.pressure_outlet_bar)
            ),
            "flow_rate_lpm": edge_reading.flow_rate_lpm or 0.0,
            "temperature_c": edge_reading.temperature_c or 0.0,
            "vibration_level_g": edge_reading.vibration_g or 0.0,
            "age_hours": edge_spec.age_hours or 0.0,
            "maintenance_score": (
                edge_spec.get_maintenance_score(edge_reading.timestamp)
                if hasattr(edge_spec, "get_maintenance_score")
                else 1.0  # Default perfect maintenance
            ),
        }

        # Normalize
        normalized = self.edge_normalizer.normalize_all(raw_features)

        # Return as ordered array
        features = [
            normalized["pressure_drop_bar"],
            normalized["flow_rate_lpm"],
            normalized["temperature_c"],
            normalized["vibration_level_g"],
            normalized["age_hours"],
            normalized["maintenance_score"],
        ]

        return np.array(features, dtype=np.float32)

    def _build_timeseries_edge_features(self, edge_history: pd.DataFrame) -> np.ndarray:
        """Build time-series statistical features (34D per sensor).

        Extracts statistical/frequency/temporal features from edge sensor history.

        Args:
            edge_history: DataFrame with columns [timestamp, pressure, flow, temperature, ...]

        Returns:
            features: Array [34*N] where N = number of sensor columns
        """
        timeseries_features = []

        for sensor_col in edge_history.columns:
            if sensor_col == "timestamp":
                continue

            # Extract features for this sensor
            sensor_data = edge_history[[sensor_col]]
            features_34d = self.feature_engineer.extract_all_features(sensor_data)

            timeseries_features.append(features_34d)

        if not timeseries_features:
            logger.warning("No time-series columns found in edge_history")
            return np.array([], dtype=np.float32)

        return np.concatenate(timeseries_features)

    # ========================================================================
    # HYBRID GRAPH CONSTRUCTION
    # ========================================================================

    def build_graph_hybrid(
        self,
        request: HybridInferenceRequest,
        topology: GraphTopology,
        edge_history: dict[str, pd.DataFrame] | None = None,
    ) -> Data:
        """Build graph from HybridInferenceRequest (edge-centric!).

        Process:
            1. Build minimal node features (29D) from component_readings
            2. Build rich edge features (14-116D) from edge_readings + edge_history
            3. Create PyG Data object
            4. Validate graph structure

        Args:
            request: HybridInferenceRequest with edge_readings + component_readings
            topology: GraphTopology with components and edges
            edge_history: Optional dict mapping edge_id → time-series DataFrame
                         {"pump__valve": DataFrame[timestamp, pressure, flow, ...]}

        Returns:
            graph: PyG Data object with:
                - x: [N, 29] minimal node features
                - edge_index: [2, E]
                - edge_attr: [E, edge_in_dim] rich edge features

        Raises:
            ValueError: If topology is invalid or graph construction fails

        Examples:
            >>> request = HybridInferenceRequest(...)
            >>> topology = GraphTopology(...)
            >>> graph = builder.build_graph_hybrid(request, topology)
            >>> print(graph)
            Data(x=[10, 29], edge_index=[2, 20], edge_attr=[20, 14])
        """
        # TODO: Implement in Day 3 (Wednesday)
        # Placeholder for now
        raise NotImplementedError("build_graph_hybrid will be implemented in Day 3")


# ============================================================================
# BACKWARD COMPATIBILITY HELPERS
# ============================================================================


def convert_node_to_hybrid(
    node_centric_request,  # MinimalInferenceRequest
    topology: GraphTopology,
) -> HybridInferenceRequest:
    """Convert node-centric request to edge-centric HybridInferenceRequest.

    Heuristic conversion:
        - Edge pressure_inlet = source component pressure
        - Edge pressure_outlet = target component pressure
        - Edge flow = source component flow (if available)
        - Edge temperature = average of source and target

    Args:
        node_centric_request: MinimalInferenceRequest (old API)
        topology: GraphTopology

    Returns:
        HybridInferenceRequest (new API)

    Examples:
        >>> # Old API
        >>> old_request = MinimalInferenceRequest(
        ...     equipment_id="pump_sys_01",
        ...     sensor_readings={
        ...         "pump_1": ComponentSensorReading(pressure_bar=150, ...),
        ...         "valve_1": ComponentSensorReading(pressure_bar=148, ...)
        ...     }
        ... )
        >>>
        >>> # Convert to new API
        >>> new_request = convert_node_to_hybrid(old_request, topology)
        >>> print(new_request.edge_readings)
        >>> # {"pump_1__valve_1": EdgeSensorReading(pressure_inlet_bar=150, ...)}
    """
    # TODO: Implement in Day 5 (Friday) for backward compatibility
    raise NotImplementedError("convert_node_to_hybrid will be implemented in Day 5")
