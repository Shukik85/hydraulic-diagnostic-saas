"""Node-centric hydraulic system topology.

Physical Model:
  SENSORS ARE NODES (17 nodes, one per physical sensor location)
  CONNECTIONS ARE EDGES (flow paths and diagnostic correlations)

Physical Layout (UCI Hydraulic System):

  Pump P1 (PS1) ----> [PS4: Pump-Motor Feed]
                        |
                        v
  Motor M1 (PS2) <--- [FS1/FS2: flow through motor]
                        |
                        v
  Accumulator A1 (PS5) - [stores energy]

  Cooler C1 (TS1,TS3) --- [returns to tank]

  Solenoid SE controls pump on/off

Key Insight:
  - PS1 (pump outlet) is HIGH pressure (0-350 bar)
  - PS2 (motor inlet) is MEDIUM pressure (0-200 bar)
  - PS3 (pump inlet) is LOW pressure (-0.5-10 bar)
  - Temperature sensors (TS) are distributed along return line
  - Flow sensors (FS) measure pump and motor flow

Edges (diagnostic connections):
  - Pump P1 → Motor M1: pressure/flow correlation
  - Motor M1 → Cooler C1: heat generation
  - Cooler C1 → Tank (implicit): energy dissipation
  - Accumulator A1 ↔ Motor M1: energy storage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


class Edge(NamedTuple):
    """Directed edge in hydraulic system."""

    source: str  # sensor name (PS1, TS2, etc.)
    target: str  # sensor name
    edge_type: str  # diagnostic, physical, thermal, etc.
    description: str  # human-readable


@dataclass
class NodeCentricTopology:
    """Node-centric graph topology for hydraulic diagnostics.

    17 nodes (sensors) connected by meaningful diagnostic edges.
    """

    # 17 Nodes: sensors (in order)
    nodes: list[str] = field(
        default_factory=lambda: [
            "PS1",  # Pump outlet pressure (HIGH)
            "PS2",  # Motor inlet pressure
            "PS3",  # Pump inlet pressure (LOW)
            "PS4",  # Pump-motor feed pressure
            "PS5",  # Accumulator pressure
            "PS6",  # Return line backpressure
            "TS1",  # Pump oil temperature
            "TS2",  # Motor oil temperature
            "TS3",  # System inlet temperature
            "TS4",  # Motor case temperature
            "FS1",  # Pump flow rate
            "FS2",  # Motor flow rate
            "VS1",  # Vibration (bearing health)
            "SE",   # Solenoid (pump control)
            "CE",   # Cumulative energy
            "CP",   # Cumulative power
            "EPS1", # Electrical power signature
        ]
    )

    # Edges: diagnostic connections
    edges: list[Edge] = field(
        default_factory=lambda: [
            # PUMP-MOTOR CHAIN (main flow path)
            Edge("PS1", "PS2", "pressure_drop", "Pump outlet → Motor inlet"),
            Edge("PS2", "FS1", "flow_correlation", "Motor inlet pressure → Pump flow"),
            Edge("PS1", "FS1", "pump_performance", "Pump pressure → Pump flow (volumetric efficiency)"),
            Edge("PS2", "FS2", "motor_performance", "Motor pressure → Motor flow (mechanical efficiency)"),

            # RETURN-TANK PATH (cooler circuit)
            Edge("PS6", "TS1", "backpressure_thermal", "Return backpressure → Pump temperature (cooler effectiveness)"),
            Edge("TS1", "TS3", "thermal_gradient", "Pump T → Inlet T (heat dissipation)"),
            Edge("TS1", "TS2", "thermal_crosstalk", "Pump T → Motor T (shared heat sources)"),

            # ACCUMULATOR CIRCUIT
            Edge("PS5", "PS2", "accumulator_assist", "Accumulator pressure ↔ Motor pressure (energy supply)"),
            Edge("PS5", "CE", "energy_storage", "Accumulator pressure → Cumulative energy"),

            # PRESSURE-TEMPERATURE DIAGNOSTICS
            Edge("PS1", "TS1", "pump_health", "Pump pressure → Pump temperature (wear correlation)"),
            Edge("PS2", "TS2", "motor_health", "Motor pressure → Motor temperature (bearing stress)"),
            Edge("PS3", "TS3", "filter_clogging", "Pump inlet pressure → Inlet temperature (restriction)"),

            # SOLENOID CONTROL
            Edge("SE", "PS1", "pump_control", "Solenoid state → Pump outlet pressure"),
            Edge("SE", "FS1", "flow_control", "Solenoid state → Pump flow"),
            Edge("SE", "FS2", "motor_control", "Solenoid state → Motor flow"),

            # ENERGY-POWER DIAGNOSTICS
            Edge("CP", "PS1", "power_pressure", "Instantaneous power → Pump pressure"),
            Edge("CP", "FS1", "power_flow", "Instantaneous power → Pump flow"),
            Edge("CE", "CP", "cumulative_power", "Cumulative energy → Instantaneous power"),

            # VIBRATION (BEARING HEALTH)
            Edge("PS2", "VS1", "motor_vibration", "Motor pressure → Vibration (bearing wear)"),
            Edge("FS2", "VS1", "flow_vibration", "Motor flow → Vibration (cavitation risk)"),
            Edge("TS2", "VS1", "thermal_vibration", "Motor temperature → Vibration (thermal stress)"),

            # ELECTRICAL SIGNATURE (MOTOR CURRENT)
            Edge("EPS1", "FS2", "motor_current", "Electrical signature → Motor flow (motor load)"),
            Edge("EPS1", "PS2", "current_pressure", "Electrical signature → Motor pressure (electrical load)"),
        ]
    )

    def get_node_index(self, node_name: str) -> int:
        """Get node index by name.

        Args:
            node_name: Sensor name (PS1, TS2, etc.)

        Returns:
            index: Position in nodes list

        Raises:
            ValueError: If node not found
        """
        try:
            return self.nodes.index(node_name)
        except ValueError as e:
            msg = f"Unknown node: {node_name}. Known nodes: {self.nodes}"
            raise ValueError(msg) from e

    def get_edge_index(self) -> tuple[list[int], list[int]]:
        """Get edge_index in PyG format.

        Returns:
            (source_indices, target_indices): Lists of node indices

        Example:
            >>> source, target = topology.get_edge_index()
            >>> edge_index = torch.tensor([source, target], dtype=torch.long)
        """
        source_indices = []
        target_indices = []

        for edge in self.edges:
            src_idx = self.get_node_index(edge.source)
            tgt_idx = self.get_node_index(edge.target)
            source_indices.append(src_idx)
            target_indices.append(tgt_idx)

        return source_indices, target_indices

    def get_adjacent_edges(self, node_name: str) -> list[Edge]:
        """Get all edges connected to a node.

        Args:
            node_name: Sensor name

        Returns:
            edges: List of Edge objects
        """
        node_edges = [e for e in self.edges if e.source == node_name or e.target == node_name]
        return node_edges

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NodeCentricTopology(\n"
            f"  nodes={len(self.nodes)},\n"
            f"  edges={len(self.edges)},\n"
            f"  density={2*len(self.edges)/(len(self.nodes)*(len(self.nodes)-1)):.2%}\n"
            f")"
        )
