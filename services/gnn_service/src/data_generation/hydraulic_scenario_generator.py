"""Hydraulic scenario generator for GNN training data.

Generates realistic hydraulic system scenarios with:
- Parallel operations (700 graphs)
- Sequential cascades (400 graphs)
- Temporal sequences (100 sequences × 10 timesteps)

Phase 2 Multi-Task Architecture:
- 6-task predictions (4 graph + 2 component)
- Multi-label anomaly detection (9 classes)
- RUL estimation (hours until failure)
- Component-level health monitoring

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .topologies import (
    ComponentNode,
    ComponentType,
    Connection,
    TopologyDefinitions,
    TopologyType,
)
from .validators import PhysicalValidator

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for scenario generation."""
    
    # Dataset sizes
    num_parallel_graphs: int = 700
    num_sequential_graphs: int = 400
    num_temporal_sequences: int = 100
    temporal_sequence_length: int = 10
    
    # Noise and variation
    noise_level: float = 0.05  # 5% noise
    degradation_probability: float = 0.3  # 30% chance of degraded components
    anomaly_probability: float = 0.4  # 40% chance of system anomaly
    
    # Physical parameters
    nominal_pressure_range: tuple[float, float] = (200.0, 350.0)  # bar
    nominal_flow_range: tuple[float, float] = (50.0, 300.0)  # l/min
    temperature_range: tuple[float, float] = (40.0, 90.0)  # °C
    
    # RUL simulation (Phase 2)
    rul_healthy_range: tuple[float, float] = (500.0, 1000.0)  # hours
    rul_degraded_range: tuple[float, float] = (10.0, 500.0)  # hours
    
    # Anomaly classes (9 types - Phase 2)
    anomaly_classes: list[str] = None
    
    # Validation
    validate_physics: bool = True
    strict_validation: bool = False  # Warnings only
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        """Initialize anomaly classes if not provided."""
        if self.anomaly_classes is None:
            self.anomaly_classes = [
                'overload',           # 0: Load exceeds capacity
                'pressure_spike',     # 1: Sudden pressure increase
                'cavitation',         # 2: Bubble formation
                'contamination',      # 3: Fluid contamination
                'leakage',            # 4: Internal/external leak
                'valve_stuck',        # 5: Valve malfunction
                'pump_degradation',   # 6: Pump wear
                'thermal_runaway',    # 7: Overheating
                'vibration_anomaly',  # 8: Abnormal vibration
            ]


class HydraulicScenarioGenerator:
    """Generate realistic hydraulic scenarios for GNN training.
    
    Phase 2 Architecture:
    - Multi-level predictions (graph + component)
    - Multi-label anomaly detection (9 classes)
    - RUL estimation
    - Temporal degradation progression
    
    Creates datasets with:
    - Parallel operations: simultaneous actuator movements with load sensing
    - Sequential cascades: safety valve cascades during overload
    - Temporal patterns: degradation progression over time
    
    Examples:
        >>> config = GeneratorConfig(num_parallel_graphs=700)
        >>> generator = HydraulicScenarioGenerator(config)
        >>> 
        >>> # Generate parallel operation scenarios
        >>> parallel_data = generator.generate_parallel_scenarios()
        >>> graph = parallel_data['graphs'][0]
        >>> graph.y_graph_health  # [1] scalar
        >>> graph.y_component_anomaly  # [N, 9] multi-label
    """
    
    def __init__(self, config: GeneratorConfig | None = None) -> None:
        """Initialize generator.
        
        Args:
            config: Generator configuration. Uses defaults if None.
        """
        self.config = config or GeneratorConfig()
        self.validator = PhysicalValidator(strict=self.config.strict_validation)
        
        # Set random seed
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        
        logger.info("HydraulicScenarioGenerator (Phase 2) initialized")
        logger.info("Multi-task targets: 4 graph + 2 component = 6 tasks")
        logger.info("Anomaly classes: %d", len(self.config.anomaly_classes))
    
    def _generate_base_node_features(
        self,
        component_type: ComponentType,
        nominal_pressure: float,
        nominal_flow: float,
        health_score: float = 1.0  # 1.0=healthy, 0.0=failed
    ) -> np.ndarray:
        """Generate base node features for a component.
        
        Args:
            component_type: Type of hydraulic component
            nominal_pressure: Nominal operating pressure (bar)
            nominal_flow: Nominal flow rate (l/min)
            health_score: Health score ∈ [0,1]
            
        Returns:
            Feature vector [34]
        """
        features = np.zeros(34, dtype=np.float32)
        
        # Degradation factor (inverse of health)
        degradation_factor = 1.0 - health_score
        
        # Physical measurements (indices 0-9)
        features[0] = nominal_pressure * (1.0 + np.random.uniform(-0.1, 0.1))  # pressure
        features[1] = nominal_flow * (1.0 + np.random.uniform(-0.1, 0.1))  # flow_rate
        features[2] = np.random.uniform(*self.config.temperature_range)  # temperature
        features[3] = np.random.uniform(0, 100)  # position (for actuators)
        features[4] = np.random.uniform(0, 500)  # velocity
        features[5] = nominal_flow * 10 * (1.0 + np.random.uniform(-0.2, 0.2))  # torque
        features[6] = nominal_pressure * 0.5 * (1.0 + np.random.uniform(-0.2, 0.2))  # force
        features[7] = np.random.uniform(0, 20) * (1 + degradation_factor)  # vibration (↑ when degraded)
        features[8] = np.random.uniform(40, 80)  # noise_level
        features[9] = (nominal_pressure * nominal_flow) / 600.0  # power (kW)
        
        # Valve states (indices 10-14)
        if component_type in [ComponentType.VALVE_LOAD_SENSING, ComponentType.VALVE_DIRECTIONAL, 
                               ComponentType.VALVE_RELIEF]:
            features[10] = np.random.uniform(20, 80)  # valve_opening
            features[11] = np.random.uniform(5, 30)  # valve_pressure_drop
            features[12] = np.random.uniform(0.3, 0.9) * health_score  # coefficient (↓ when degraded)
            features[13] = np.random.uniform(10, 25)  # pilot_pressure
            features[14] = features[10]  # spool_position matches opening
        
        # Health indicators (indices 15-20) - degraded based on health_score
        features[15] = degradation_factor * 100  # wear_level
        features[16] = 12 + degradation_factor * 10  # contamination (ISO code)
        features[17] = health_score * 100  # seal_condition
        features[18] = degradation_factor * 15  # internal_leakage
        features[19] = degradation_factor * 3  # external_leakage
        features[20] = health_score * 100  # efficiency
        
        # Temporal features (indices 21-24) - rates of change
        features[21] = np.random.uniform(-10, 10)  # pressure_rate_change
        features[22] = np.random.uniform(-20, 20)  # flow_rate_change
        features[23] = np.random.uniform(-1, 1)  # temperature_rate_change
        features[24] = np.random.uniform(-10, 10)  # position_rate_change
        
        # Component type one-hot (indices 25-29)
        type_encoding = [0.0, 0.0, 0.0, 0.0, 0.0]
        if component_type == ComponentType.PUMP:
            type_encoding[0] = 1.0
        elif component_type in [ComponentType.VALVE_RELIEF, ComponentType.VALVE_LOAD_SENSING,
                                ComponentType.VALVE_DIRECTIONAL, ComponentType.SHOCK_VALVE,
                                ComponentType.MAKEUP_VALVE]:
            type_encoding[1] = 1.0
        elif component_type in [ComponentType.ACTUATOR_CYLINDER, ComponentType.ACTUATOR_MOTOR]:
            type_encoding[2] = 1.0
        elif component_type == ComponentType.FILTER:
            type_encoding[4] = 1.0
        features[25:30] = type_encoding
        
        # Operational context (indices 30-33)
        features[30] = np.random.uniform(50, 120)  # load_factor
        features[31] = np.random.uniform(40, 90)  # duty_cycle
        features[32] = np.random.uniform(0.1, 0.9)  # operating_hours (normalized)
        features[33] = np.random.uniform(10, 40)  # ambient_temperature
        
        # Add noise
        noise = np.random.normal(0, self.config.noise_level, 34)
        features = features * (1.0 + noise)
        
        return features
    
    def _generate_edge_features(
        self,
        connection: Connection,
        source_pressure: float,
        target_pressure: float,
        flow_rate: float
    ) -> np.ndarray:
        """Generate edge features for a connection.
        
        Args:
            connection: Connection specification
            source_pressure: Pressure at source node (bar)
            target_pressure: Pressure at target node (bar)
            flow_rate: Flow through connection (l/min)
            
        Returns:
            Feature vector [8] (static only - model will project to 14D)
        """
        features = np.zeros(8, dtype=np.float32)
        
        # Static edge features (8D)
        features[0] = connection.pipe_diameter  # diameter (mm)
        features[1] = connection.pipe_length  # length (m)
        features[2] = np.pi * (connection.pipe_diameter / 2000) ** 2  # cross-sectional area (m²)
        features[3] = np.random.uniform(0.01, 0.05)  # loss_coefficient
        features[4] = np.random.uniform(150, 350)  # pressure_rating (bar)
        
        # Material encoding (3D one-hot)
        material = np.random.choice([0, 1, 2])  # steel, rubber, composite
        material_enc = [0.0, 0.0, 0.0]
        material_enc[material] = 1.0
        features[5:8] = material_enc
        
        return features
    
    def _compute_rul(self, component_health_scores: list[float]) -> float:
        """Compute graph-level remaining useful life.
        
        Args:
            component_health_scores: Health scores for all components [0,1]
            
        Returns:
            RUL in hours (0 to 1000+)
        """
        # Average health
        avg_health = np.mean(component_health_scores)
        
        # RUL inversely proportional to degradation
        if avg_health > 0.8:
            # Healthy system: 500-1000 hours
            rul = np.random.uniform(*self.config.rul_healthy_range)
        elif avg_health > 0.5:
            # Degrading: 100-500 hours
            rul = np.random.uniform(100, 500)
        else:
            # Critical: 10-100 hours
            rul = np.random.uniform(10, 100)
        
        # Add correlation with health
        rul = rul * avg_health
        
        return max(0.0, rul)
    
    def _sample_anomaly_flags(self, num_classes: int, base_probability: float) -> np.ndarray:
        """Sample multi-label anomaly flags.
        
        Args:
            num_classes: Number of anomaly classes (9)
            base_probability: Base probability for each class
            
        Returns:
            Binary flags [num_classes]
        """
        flags = np.zeros(num_classes, dtype=np.float32)
        
        # Sample each class independently
        for i in range(num_classes):
            if np.random.random() < base_probability:
                flags[i] = 1.0
        
        return flags
    
    def _assign_multi_task_labels(
        self, 
        num_nodes: int,
        scenario_type: str = 'normal'
    ) -> dict[str, torch.Tensor]:
        """Assign multi-task labels for Phase 2.
        
        Args:
            num_nodes: Number of nodes in graph
            scenario_type: 'normal', 'parallel_overload', 'sequential_cascade'
            
        Returns:
            Dictionary with all 6 targets:
            - y_graph_health: [1]
            - y_graph_degradation: [1]
            - y_graph_anomaly: [9]
            - y_graph_rul: [1]
            - y_component_health: [N]
            - y_component_anomaly: [N, 9]
        """
        # === Component-level health ===
        component_health = []
        for _ in range(num_nodes):
            if np.random.random() < self.config.degradation_probability:
                # Degraded: 0.2-0.8
                health = np.random.uniform(0.2, 0.8)
            else:
                # Healthy: 0.8-1.0
                health = np.random.uniform(0.8, 1.0)
            component_health.append(health)
        
        # === Component-level anomaly (multi-label) ===
        component_anomaly = []
        for health in component_health:
            # Probability increases with degradation
            anomaly_prob = (1.0 - health) * 0.5  # 0-50% chance
            flags = self._sample_anomaly_flags(9, anomaly_prob)
            component_anomaly.append(flags)
        
        # === Graph-level health (average) ===
        graph_health = np.mean(component_health)
        
        # === Graph-level degradation rate ===
        # Inversely proportional to health
        graph_degradation = 1.0 - graph_health + np.random.uniform(-0.1, 0.1)
        graph_degradation = np.clip(graph_degradation, 0.0, 1.0)
        
        # === Graph-level anomaly (multi-label) ===
        graph_anomaly_prob = 0.5 if scenario_type != 'normal' else 0.2
        graph_anomaly = self._sample_anomaly_flags(9, graph_anomaly_prob)
        
        # Set specific anomalies based on scenario
        if scenario_type == 'parallel_overload':
            graph_anomaly[0] = 1.0  # overload
        elif scenario_type == 'sequential_cascade':
            graph_anomaly[1] = 1.0  # pressure_spike
            graph_anomaly[5] = 1.0  # valve_stuck
        
        # === Graph-level RUL ===
        graph_rul = self._compute_rul(component_health)
        
        return {
            'y_graph_health': torch.tensor([graph_health], dtype=torch.float32),
            'y_graph_degradation': torch.tensor([graph_degradation], dtype=torch.float32),
            'y_graph_anomaly': torch.from_numpy(graph_anomaly),
            'y_graph_rul': torch.tensor([graph_rul], dtype=torch.float32),
            'y_component_health': torch.tensor(component_health, dtype=torch.float32),
            'y_component_anomaly': torch.from_numpy(np.stack(component_anomaly)),
        }
    
    def _create_pyg_graph(
        self,
        nodes: list[ComponentNode],
        edges: list[Connection],
        scenario_type: str = 'normal'
    ) -> Data:
        """Create PyTorch Geometric Data object with Phase 2 multi-task labels.
        
        Args:
            nodes: List of component nodes
            edges: List of connections
            scenario_type: Type of scenario for label generation
            
        Returns:
            PyG Data object with 6 targets + batch tensor
        """
        num_nodes = len(nodes)
        
        # Generate multi-task labels
        labels = self._assign_multi_task_labels(num_nodes, scenario_type)
        
        # Generate node features (use component health from labels)
        node_features_list = []
        for i, node in enumerate(nodes):
            health_score = labels['y_component_health'][i].item()
            features = self._generate_base_node_features(
                node.component_type,
                node.nominal_pressure,
                node.nominal_flow,
                health_score=health_score
            )
            node_features_list.append(features)
        
        x = torch.from_numpy(np.stack(node_features_list))
        
        # Build edge_index and edge_attr
        edge_index_list = []
        edge_attr_list = []
        
        for edge in edges:
            edge_index_list.append([edge.source, edge.target])
            
            # Get pressures for edge feature computation
            source_pressure = node_features_list[edge.source][0]
            target_pressure = node_features_list[edge.target][0]
            flow_rate = node_features_list[edge.source][1]
            
            edge_features = self._generate_edge_features(
                edge, source_pressure, target_pressure, flow_rate
            )
            edge_attr_list.append(edge_features)
        
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.from_numpy(np.stack(edge_attr_list))
        
        # Create Data object with Phase 2 multi-task labels
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # Graph-level targets (4)
            y_graph_health=labels['y_graph_health'],
            y_graph_degradation=labels['y_graph_degradation'],
            y_graph_anomaly=labels['y_graph_anomaly'],
            y_graph_rul=labels['y_graph_rul'],
            # Component-level targets (2)
            y_component_health=labels['y_component_health'],
            y_component_anomaly=labels['y_component_anomaly'],
            # Batch tensor (for DataLoader)
            batch=torch.zeros(num_nodes, dtype=torch.long)
        )
        
        # Validate if enabled
        if self.config.validate_physics:
            self.validator.validate_graph(data)
        
        return data
    
    def generate_parallel_scenarios(self) -> dict[str, list]:
        """Generate parallel operation scenarios.
        
        Creates 700 graphs with 7-node topology:
        - 2 pumps → load-sensing valve → 4 actuators
        - Simulates simultaneous boom + swing operations
        - Models dynamic flow distribution
        
        Returns:
            Dictionary with 'graphs' and metadata
        """
        logger.info("Generating %d parallel operation scenarios...", self.config.num_parallel_graphs)
        
        graphs = []
        nodes, edges = TopologyDefinitions.get_medium_7_node_parallel_topology()
        
        for _ in tqdm(range(self.config.num_parallel_graphs), desc="Parallel scenarios"):
            # Determine scenario type
            if np.random.random() < self.config.anomaly_probability:
                scenario_type = 'parallel_overload'
            else:
                scenario_type = 'normal'
            
            # Create graph
            graph = self._create_pyg_graph(nodes, edges, scenario_type)
            graphs.append(graph)
        
        logger.info("Generated %d parallel scenarios", len(graphs))
        
        return {
            'graphs': graphs,
            'topology_type': 'parallel_7_nodes',
            'num_graphs': len(graphs)
        }
    
    def generate_sequential_scenarios(self) -> dict[str, list]:
        """Generate sequential cascade scenarios.
        
        Creates 400 graphs with 10-node topology:
        - pump → relief → distributor → section relief → motor → shock/makeup
        - Simulates safety valve cascade during overload
        - Models temporal cascade patterns
        
        Returns:
            Dictionary with 'graphs' and metadata
        """
        logger.info("Generating %d sequential cascade scenarios...", self.config.num_sequential_graphs)
        
        graphs = []
        nodes, edges = TopologyDefinitions.get_large_10_node_sequential_topology()
        
        for _ in tqdm(range(self.config.num_sequential_graphs), desc="Sequential scenarios"):
            # Determine scenario type
            if np.random.random() < self.config.anomaly_probability:
                scenario_type = 'sequential_cascade'
            else:
                scenario_type = 'normal'
            
            # Create graph
            graph = self._create_pyg_graph(nodes, edges, scenario_type)
            graphs.append(graph)
        
        logger.info("Generated %d sequential scenarios", len(graphs))
        
        return {
            'graphs': graphs,
            'topology_type': 'sequential_10_nodes',
            'num_graphs': len(graphs)
        }
    
    def generate_temporal_sequences(self) -> dict[str, list]:
        """Generate temporal sequences for LSTM training.
        
        Creates 100 sequences with 10 timesteps each:
        - Uses mixed topologies (3, 7, 10 nodes)
        - Simulates gradual degradation over time
        - Each sequence shows progression from healthy to failed
        
        Returns:
            Dictionary with 'sequences' and metadata
        """
        logger.info(
            "Generating %d temporal sequences (length=%d)...",
            self.config.num_temporal_sequences,
            self.config.temporal_sequence_length
        )
        
        sequences = []
        
        topologies = [
            (TopologyType.SMALL_3_NODES, 3),
            (TopologyType.MEDIUM_7_NODES, 7),
            (TopologyType.LARGE_10_NODES, 10)
        ]
        
        for _ in tqdm(range(self.config.num_temporal_sequences), desc="Temporal sequences"):
            # Randomly select topology
            topology_type, num_nodes = topologies[np.random.randint(0, 3)]
            nodes, edges = TopologyDefinitions.get_topology(topology_type)
            
            sequence_graphs = []
            
            # Generate degradation progression
            for t in range(self.config.temporal_sequence_length):
                # Progress from normal → degraded → anomalous
                degradation_progress = t / self.config.temporal_sequence_length
                
                if degradation_progress < 0.3:
                    scenario_type = 'normal'
                elif degradation_progress < 0.7:
                    scenario_type = 'parallel_overload' if np.random.random() > 0.5 else 'normal'
                else:
                    scenario_type = 'sequential_cascade'
                
                graph = self._create_pyg_graph(nodes, edges, scenario_type)
                sequence_graphs.append(graph)
            
            sequences.append(sequence_graphs)
        
        logger.info("Generated %d temporal sequences", len(sequences))
        
        return {
            'sequences': sequences,  # List[List[Data]] - [num_seq, seq_len]
            'sequence_length': self.config.temporal_sequence_length,
            'num_sequences': len(sequences)
        }
    
    def generate_all(self, save_dir: Path | str | None = None) -> dict[str, dict]:
        """Generate all datasets and optionally save to disk.
        
        Args:
            save_dir: Directory to save datasets. If None, only returns data.
            
        Returns:
            Dictionary with all generated datasets
        """
        logger.info("Starting full dataset generation (Phase 2)...")
        
        # Generate all scenarios
        parallel_data = self.generate_parallel_scenarios()
        sequential_data = self.generate_sequential_scenarios()
        temporal_data = self.generate_temporal_sequences()
        
        all_data = {
            'parallel': parallel_data,
            'sequential': sequential_data,
            'temporal': temporal_data,
            'config': self.config
        }
        
        # Save if directory provided
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("Saving datasets to %s", save_path)
            
            torch.save(parallel_data, save_path / "parallel_operations_700graphs.pt")
            torch.save(sequential_data, save_path / "sequential_cascade_400graphs.pt")
            torch.save(temporal_data, save_path / "temporal_sequences_100x10.pt")
            
            logger.info("Datasets saved successfully")
        
        total_graphs = (
            len(parallel_data['graphs']) +
            len(sequential_data['graphs']) +
            len(temporal_data['sequences']) * self.config.temporal_sequence_length
        )
        
        logger.info(
            "Total graphs generated: %d (%d parallel + %d sequential + %d temporal)",
            total_graphs,
            len(parallel_data['graphs']),
            len(sequential_data['graphs']),
            len(temporal_data['sequences']) * self.config.temporal_sequence_length
        )
        
        return all_data
