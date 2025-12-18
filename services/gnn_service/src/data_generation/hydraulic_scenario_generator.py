"""Hydraulic scenario generator for GNN training data.

Generates realistic hydraulic system scenarios with:
- Parallel operations (700 graphs)
- Sequential cascades (400 graphs)
- Temporal sequences (100 sequences × 10 timesteps)

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
    
    # Validation
    validate_physics: bool = True
    strict_validation: bool = False  # Warnings only
    
    # Random seed for reproducibility
    random_seed: int = 42


class HydraulicScenarioGenerator:
    """Generate realistic hydraulic scenarios for GNN training.
    
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
        >>> len(parallel_data['graphs'])  # 700
        >>> 
        >>> # Generate sequential cascades
        >>> sequential_data = generator.generate_sequential_scenarios()
        >>> 
        >>> # Generate temporal sequences
        >>> temporal_data = generator.generate_temporal_sequences()
        >>> temporal_data['graphs'][0].shape  # [10, num_nodes, 34]
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
        
        logger.info("HydraulicScenarioGenerator initialized with config: %s", self.config)
    
    def _generate_base_node_features(
        self,
        component_type: ComponentType,
        nominal_pressure: float,
        nominal_flow: float,
        health_status: int = 0  # 0=healthy by default
    ) -> np.ndarray:
        """Generate base node features for a component.
        
        Args:
            component_type: Type of hydraulic component
            nominal_pressure: Nominal operating pressure (bar)
            nominal_flow: Nominal flow rate (l/min)
            health_status: Health status (0-4)
            
        Returns:
            Feature vector [34]
        """
        features = np.zeros(34, dtype=np.float32)
        
        # Physical measurements (indices 0-9)
        features[0] = nominal_pressure * (1.0 + np.random.uniform(-0.1, 0.1))  # pressure
        features[1] = nominal_flow * (1.0 + np.random.uniform(-0.1, 0.1))  # flow_rate
        features[2] = np.random.uniform(*self.config.temperature_range)  # temperature
        features[3] = np.random.uniform(0, 100)  # position (for actuators)
        features[4] = np.random.uniform(0, 500)  # velocity
        features[5] = nominal_flow * 10 * (1.0 + np.random.uniform(-0.2, 0.2))  # torque
        features[6] = nominal_pressure * 0.5 * (1.0 + np.random.uniform(-0.2, 0.2))  # force
        features[7] = np.random.uniform(0, 20)  # vibration
        features[8] = np.random.uniform(40, 80)  # noise_level
        features[9] = (nominal_pressure * nominal_flow) / 600.0  # power (kW)
        
        # Valve states (indices 10-14)
        if component_type in [ComponentType.VALVE_LOAD_SENSING, ComponentType.VALVE_DIRECTIONAL, 
                               ComponentType.VALVE_RELIEF]:
            features[10] = np.random.uniform(20, 80)  # valve_opening
            features[11] = np.random.uniform(5, 30)  # valve_pressure_drop
            features[12] = np.random.uniform(0.3, 0.9)  # valve_flow_coefficient
            features[13] = np.random.uniform(10, 25)  # pilot_pressure
            features[14] = features[10]  # spool_position matches opening
        
        # Health indicators (indices 15-20) - degraded based on health_status
        degradation_factor = health_status / 4.0  # 0.0 to 1.0
        features[15] = degradation_factor * 100  # wear_level
        features[16] = 12 + degradation_factor * 10  # contamination (ISO code)
        features[17] = 100 - degradation_factor * 50  # seal_condition
        features[18] = degradation_factor * 15  # internal_leakage
        features[19] = degradation_factor * 3  # external_leakage
        features[20] = 100 - degradation_factor * 30  # efficiency
        
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
            Feature vector [14]
        """
        features = np.zeros(14, dtype=np.float32)
        
        # Flow characteristics (indices 0-5)
        features[0] = flow_rate * (1.0 + np.random.uniform(-0.05, 0.05))  # flow_rate
        features[1] = abs(source_pressure - target_pressure)  # pressure_drop
        
        # Reynolds number (simplified)
        velocity = flow_rate / (np.pi * (connection.pipe_diameter / 2000) ** 2) / 60  # m/s
        features[2] = min(velocity / 10.0, 1.0)  # normalized reynolds (proxy)
        features[3] = velocity  # flow_velocity
        features[4] = connection.pipe_diameter  # pipe_diameter
        features[5] = connection.pipe_length  # pipe_length
        
        # Connection state (indices 6-8)
        features[6] = 1.0  # is_active
        features[7] = np.random.uniform(0, 20)  # valve_restriction
        features[8] = 1.0  # flow_direction (forward)
        
        # Fluid properties (indices 9-11)
        features[9] = np.random.uniform(30, 60)  # fluid_viscosity (cSt)
        features[10] = np.random.uniform(870, 900)  # fluid_density (kg/m³)
        features[11] = np.random.uniform(40, 80)  # fluid_temperature
        
        # Connection health (indices 12-13)
        features[12] = np.random.uniform(0.01, 0.1)  # pipe_roughness
        features[13] = np.random.uniform(10, 18)  # contamination_level
        
        return features
    
    def _assign_health_labels(self, num_nodes: int) -> list[int]:
        """Assign health status labels to nodes.
        
        Args:
            num_nodes: Number of nodes in graph
            
        Returns:
            List of health labels (0-4)
        """
        labels = [0] * num_nodes  # Start with all healthy
        
        # Randomly degrade some components
        for i in range(num_nodes):
            if np.random.random() < self.config.degradation_probability:
                # Weighted towards less severe degradation
                labels[i] = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
        
        return labels
    
    def _create_pyg_graph(
        self,
        nodes: list[ComponentNode],
        edges: list[Connection],
        node_health_labels: list[int],
        graph_anomaly_label: int
    ) -> Data:
        """Create PyTorch Geometric Data object.
        
        Args:
            nodes: List of component nodes
            edges: List of connections
            node_health_labels: Node-level health labels
            graph_anomaly_label: Graph-level anomaly label
            
        Returns:
            PyG Data object
        """
        # Generate node features
        node_features_list = []
        for i, node in enumerate(nodes):
            features = self._generate_base_node_features(
                node.component_type,
                node.nominal_pressure,
                node.nominal_flow,
                health_status=node_health_labels[i]
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
        
        # Create Data object with labels
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_node=torch.tensor(node_health_labels, dtype=torch.long),
            y_graph=torch.tensor(graph_anomaly_label, dtype=torch.long)
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
            Dictionary with 'graphs', 'node_labels', 'graph_labels'
        """
        logger.info("Generating %d parallel operation scenarios...", self.config.num_parallel_graphs)
        
        graphs = []
        node_labels_list = []
        graph_labels_list = []
        
        nodes, edges = TopologyDefinitions.get_medium_7_node_parallel_topology()
        
        for _ in tqdm(range(self.config.num_parallel_graphs), desc="Parallel scenarios"):
            # Assign health labels
            node_health = self._assign_health_labels(len(nodes))
            
            # Determine graph-level anomaly
            if np.random.random() < self.config.anomaly_probability:
                graph_anomaly = 1  # parallel_overload
            else:
                graph_anomaly = 0  # normal
            
            # Create graph
            graph = self._create_pyg_graph(nodes, edges, node_health, graph_anomaly)
            
            graphs.append(graph)
            node_labels_list.append(node_health)
            graph_labels_list.append(graph_anomaly)
        
        logger.info("Generated %d parallel scenarios", len(graphs))
        
        return {
            'graphs': graphs,
            'node_labels': node_labels_list,
            'graph_labels': graph_labels_list,
            'topology_type': 'parallel_7_nodes'
        }
    
    def generate_sequential_scenarios(self) -> dict[str, list]:
        """Generate sequential cascade scenarios.
        
        Creates 400 graphs with 10-node topology:
        - pump → relief → distributor → section relief → motor → shock/makeup
        - Simulates safety valve cascade during overload
        - Models temporal cascade patterns
        
        Returns:
            Dictionary with 'graphs', 'node_labels', 'graph_labels'
        """
        logger.info("Generating %d sequential cascade scenarios...", self.config.num_sequential_graphs)
        
        graphs = []
        node_labels_list = []
        graph_labels_list = []
        
        nodes, edges = TopologyDefinitions.get_large_10_node_sequential_topology()
        
        for _ in tqdm(range(self.config.num_sequential_graphs), desc="Sequential scenarios"):
            # Assign health labels
            node_health = self._assign_health_labels(len(nodes))
            
            # Determine graph-level anomaly
            if np.random.random() < self.config.anomaly_probability:
                graph_anomaly = 2  # sequential_cascade
            else:
                graph_anomaly = 0  # normal
            
            # Create graph
            graph = self._create_pyg_graph(nodes, edges, node_health, graph_anomaly)
            
            graphs.append(graph)
            node_labels_list.append(node_health)
            graph_labels_list.append(graph_anomaly)
        
        logger.info("Generated %d sequential scenarios", len(graphs))
        
        return {
            'graphs': graphs,
            'node_labels': node_labels_list,
            'graph_labels': graph_labels_list,
            'topology_type': 'sequential_10_nodes'
        }
    
    def generate_temporal_sequences(self) -> dict[str, list]:
        """Generate temporal sequences for LSTM training.
        
        Creates 100 sequences with 10 timesteps each:
        - Uses mixed topologies (3, 7, 10 nodes)
        - Simulates gradual degradation over time
        - Each sequence shows progression from healthy to failed
        
        Returns:
            Dictionary with 'sequences', 'node_labels', 'graph_labels'
        """
        logger.info(
            "Generating %d temporal sequences (length=%d)...",
            self.config.num_temporal_sequences,
            self.config.temporal_sequence_length
        )
        
        sequences = []
        node_labels_sequences = []
        graph_labels_sequences = []
        
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
            sequence_node_labels = []
            sequence_graph_labels = []
            
            # Generate degradation progression
            for t in range(self.config.temporal_sequence_length):
                # Increase degradation over time
                degradation_progress = t / self.config.temporal_sequence_length
                
                node_health = []
                for _ in range(num_nodes):
                    if np.random.random() < degradation_progress * 0.8:
                        # Health degrades: 0 -> 1 -> 2 -> 3 -> 4
                        max_health = min(4, int(degradation_progress * 5))
                        node_health.append(np.random.randint(0, max_health + 1))
                    else:
                        node_health.append(0)  # healthy
                
                # Graph anomaly appears later in sequence
                if degradation_progress > 0.6 and np.random.random() < 0.5:
                    graph_anomaly = np.random.choice([1, 2, 3])  # some anomaly
                else:
                    graph_anomaly = 0  # normal
                
                graph = self._create_pyg_graph(nodes, edges, node_health, graph_anomaly)
                
                sequence_graphs.append(graph)
                sequence_node_labels.append(node_health)
                sequence_graph_labels.append(graph_anomaly)
            
            sequences.append(sequence_graphs)
            node_labels_sequences.append(sequence_node_labels)
            graph_labels_sequences.append(sequence_graph_labels)
        
        logger.info("Generated %d temporal sequences", len(sequences))
        
        return {
            'sequences': sequences,  # List[List[Data]] - [num_seq, seq_len]
            'node_labels': node_labels_sequences,
            'graph_labels': graph_labels_sequences,
            'sequence_length': self.config.temporal_sequence_length
        }
    
    def generate_all(self, save_dir: Path | str | None = None) -> dict[str, dict]:
        """Generate all datasets and optionally save to disk.
        
        Args:
            save_dir: Directory to save datasets. If None, only returns data.
            
        Returns:
            Dictionary with all generated datasets
        """
        logger.info("Starting full dataset generation...")
        
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
        
        logger.info(
            "Total graphs generated: %d parallel + %d sequential + %d temporal sequences",
            len(parallel_data['graphs']),
            len(sequential_data['graphs']),
            len(temporal_data['sequences']) * self.config.temporal_sequence_length
        )
        
        return all_data
