"""Hydraulic system topology definitions.

Defines standard topologies (3, 7, 10 nodes) for parallel and sequential operations.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict
import networkx as nx


class TopologyType(Enum):
    """Supported topology types."""
    SMALL_3_NODES = "small_3_nodes"
    MEDIUM_7_NODES = "medium_7_nodes"
    LARGE_10_NODES = "large_10_nodes"
    CUSTOM = "custom"


class ComponentType(Enum):
    """Hydraulic component types."""
    PUMP = "pump"
    VALVE_RELIEF = "valve_relief"
    VALVE_LOAD_SENSING = "valve_load_sensing"
    VALVE_DIRECTIONAL = "valve_directional"
    ACTUATOR_CYLINDER = "actuator_cylinder"
    ACTUATOR_MOTOR = "actuator_motor"
    DISTRIBUTOR = "distributor"
    FILTER = "filter"
    RESERVOIR = "reservoir"
    SHOCK_VALVE = "shock_valve"
    MAKEUP_VALVE = "makeup_valve"


@dataclass
class ComponentNode:
    """Hydraulic component node definition."""
    node_id: int
    component_type: ComponentType
    name: str
    nominal_pressure: float  # bar
    nominal_flow: float  # l/min


@dataclass
class Connection:
    """Edge connection between components."""
    source: int
    target: int
    pipe_diameter: float  # mm
    pipe_length: float  # m


class TopologyDefinitions:
    """Standard hydraulic topology definitions."""
    
    @staticmethod
    def get_small_3_node_topology() -> Tuple[List[ComponentNode], List[Connection]]:
        """Simple 3-node topology: pump -> valve -> actuator.
        
        Used for basic sequential operations.
        """
        nodes = [
            ComponentNode(0, ComponentType.PUMP, "main_pump", 250.0, 100.0),
            ComponentNode(1, ComponentType.VALVE_DIRECTIONAL, "control_valve", 250.0, 100.0),
            ComponentNode(2, ComponentType.ACTUATOR_CYLINDER, "cylinder_1", 250.0, 100.0),
        ]
        
        edges = [
            Connection(0, 1, 25.0, 2.0),  # pump to valve
            Connection(1, 2, 20.0, 1.5),  # valve to actuator
        ]
        
        return nodes, edges
    
    @staticmethod
    def get_medium_7_node_parallel_topology() -> Tuple[List[ComponentNode], List[Connection]]:
        """7-node parallel operations topology.
        
        Topology: 2 pumps -> load-sensing valve -> 4 actuators
        Scenario: simultaneous boom + swing operations
        """
        nodes = [
            # Pumps
            ComponentNode(0, ComponentType.PUMP, "pump_1", 280.0, 150.0),
            ComponentNode(1, ComponentType.PUMP, "pump_2", 280.0, 150.0),
            # Load sensing valve
            ComponentNode(2, ComponentType.VALVE_LOAD_SENSING, "ls_valve", 280.0, 300.0),
            # Actuators
            ComponentNode(3, ComponentType.ACTUATOR_CYLINDER, "boom_cylinder", 250.0, 120.0),
            ComponentNode(4, ComponentType.ACTUATOR_MOTOR, "swing_motor", 250.0, 150.0),
            ComponentNode(5, ComponentType.ACTUATOR_CYLINDER, "stick_cylinder", 250.0, 100.0),
            ComponentNode(6, ComponentType.ACTUATOR_CYLINDER, "bucket_cylinder", 220.0, 80.0),
        ]
        
        edges = [
            # Pumps to LS valve
            Connection(0, 2, 32.0, 1.0),
            Connection(1, 2, 32.0, 1.0),
            # LS valve to actuators (parallel distribution)
            Connection(2, 3, 25.0, 2.5),  # to boom
            Connection(2, 4, 25.0, 2.0),  # to swing
            Connection(2, 5, 20.0, 2.2),  # to stick
            Connection(2, 6, 16.0, 2.8),  # to bucket
        ]
        
        return nodes, edges
    
    @staticmethod
    def get_large_10_node_sequential_topology() -> Tuple[List[ComponentNode], List[Connection]]:
        """10-node sequential safety cascade topology.
        
        Topology: pump -> main relief -> distributor -> section relief -> motor -> shock + makeup
        Scenario: temporal cascade during overload
        """
        nodes = [
            # Primary circuit
            ComponentNode(0, ComponentType.PUMP, "main_pump", 300.0, 200.0),
            ComponentNode(1, ComponentType.VALVE_RELIEF, "main_relief", 300.0, 200.0),
            ComponentNode(2, ComponentType.DISTRIBUTOR, "flow_distributor", 280.0, 200.0),
            
            # Section 1: motor circuit with safety chain
            ComponentNode(3, ComponentType.VALVE_RELIEF, "section_relief_1", 280.0, 120.0),
            ComponentNode(4, ComponentType.ACTUATOR_MOTOR, "hydraulic_motor_1", 250.0, 120.0),
            ComponentNode(5, ComponentType.SHOCK_VALVE, "shock_valve_1", 250.0, 120.0),
            ComponentNode(6, ComponentType.MAKEUP_VALVE, "makeup_valve_1", 250.0, 120.0),
            
            # Section 2: cylinder circuit
            ComponentNode(7, ComponentType.VALVE_RELIEF, "section_relief_2", 280.0, 80.0),
            ComponentNode(8, ComponentType.ACTUATOR_CYLINDER, "cylinder_2", 250.0, 80.0),
            ComponentNode(9, ComponentType.FILTER, "return_filter", 10.0, 280.0),
        ]
        
        edges = [
            # Main line
            Connection(0, 1, 40.0, 0.5),  # pump to main relief
            Connection(1, 2, 40.0, 1.0),  # main relief to distributor
            
            # Section 1 cascade
            Connection(2, 3, 32.0, 1.5),  # distributor to section relief 1
            Connection(3, 4, 32.0, 1.0),  # section relief to motor
            Connection(4, 5, 32.0, 0.8),  # motor to shock valve
            Connection(5, 6, 32.0, 0.5),  # shock to makeup
            Connection(6, 9, 32.0, 3.0),  # makeup to filter (return)
            
            # Section 2
            Connection(2, 7, 25.0, 2.0),  # distributor to section relief 2
            Connection(7, 8, 25.0, 1.2),  # section relief to cylinder
            Connection(8, 9, 25.0, 2.5),  # cylinder to filter (return)
        ]
        
        return nodes, edges
    
    @staticmethod
    def create_networkx_graph(
        nodes: List[ComponentNode],
        edges: List[Connection]
    ) -> nx.DiGraph:
        """Convert topology to NetworkX directed graph.
        
        Args:
            nodes: List of component nodes
            edges: List of connections
            
        Returns:
            NetworkX directed graph with node and edge attributes
        """
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in nodes:
            G.add_node(
                node.node_id,
                component_type=node.component_type.value,
                name=node.name,
                nominal_pressure=node.nominal_pressure,
                nominal_flow=node.nominal_flow
            )
        
        # Add edges with attributes
        for edge in edges:
            G.add_edge(
                edge.source,
                edge.target,
                pipe_diameter=edge.pipe_diameter,
                pipe_length=edge.pipe_length
            )
        
        return G
    
    @staticmethod
    def get_topology(
        topology_type: TopologyType
    ) -> Tuple[List[ComponentNode], List[Connection]]:
        """Get topology by type.
        
        Args:
            topology_type: Type of topology to retrieve
            
        Returns:
            Tuple of (nodes, edges)
        """
        if topology_type == TopologyType.SMALL_3_NODES:
            return TopologyDefinitions.get_small_3_node_topology()
        elif topology_type == TopologyType.MEDIUM_7_NODES:
            return TopologyDefinitions.get_medium_7_node_parallel_topology()
        elif topology_type == TopologyType.LARGE_10_NODES:
            return TopologyDefinitions.get_large_10_node_sequential_topology()
        else:
            raise ValueError(f"Unsupported topology type: {topology_type}")
    
    @staticmethod
    def visualize_topology(
        nodes: List[ComponentNode],
        edges: List[Connection],
        save_path: str = None
    ):
        """Visualize topology using NetworkX and matplotlib.
        
        Args:
            nodes: List of component nodes
            edges: List of connections
            save_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for visualization")
        
        G = TopologyDefinitions.create_networkx_graph(nodes, edges)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw
        plt.figure(figsize=(12, 8))
        nx.draw(
            G, pos,
            with_labels=True,
            labels={n: G.nodes[n]['name'] for n in G.nodes()},
            node_color='lightblue',
            node_size=2000,
            font_size=8,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            width=2
        )
        
        plt.title("Hydraulic System Topology", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
