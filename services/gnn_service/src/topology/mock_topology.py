"""Mock hydraulic system topology for testing.

Defines a 10-component hydraulic system with physical connections.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GraphTopology:
    """Mock graph topology for hydraulic system.
    
    Represents a typical hydraulic circuit:
    Pump → Valve → Actuator chain with support components
    
    Examples:
        >>> topology = GraphTopology.create_mock_excavator()
        >>> print(len(topology.components))
        10
        >>> print(len(topology.connections))
        12
    """

    def __init__(self, components: dict, connections: list[dict]):
        """Initialize topology.
        
        Args:
            components: Dict of {component_id: metadata}
            connections: List of {from: id, to: id, type: str}
        """
        self.components = components
        self.connections = connections
        logger.info(
            "🔧 Created GraphTopology: %d components, %d connections",
            len(components),
            len(connections),
        )

    @classmethod
    def create_mock_excavator(cls, equipment_id: str = "pump_001") -> GraphTopology:
        """Create mock excavator hydraulic topology.
        
        Args:
            equipment_id: Equipment identifier prefix
            
        Returns:
            GraphTopology instance
        """
        # Define components
        components = {
            f"{equipment_id}_pump": {
                "type": "pump",
                "criticality": 1.0,
                "location": "powerpack",
            },
            f"{equipment_id}_valve_main": {
                "type": "valve",
                "criticality": 0.9,
                "location": "control_block",
            },
            f"{equipment_id}_valve_relief": {
                "type": "valve",
                "criticality": 0.8,
                "location": "control_block",
            },
            f"{equipment_id}_pipe_supply": {
                "type": "pipe",
                "criticality": 0.6,
                "location": "hydraulic_line",
            },
            f"{equipment_id}_pipe_return": {
                "type": "pipe",
                "criticality": 0.5,
                "location": "hydraulic_line",
            },
            f"{equipment_id}_tank": {
                "type": "tank",
                "criticality": 0.7,
                "location": "powerpack",
            },
            f"{equipment_id}_filter": {
                "type": "filter",
                "criticality": 0.6,
                "location": "powerpack",
            },
            f"{equipment_id}_actuator": {
                "type": "actuator",
                "criticality": 0.9,
                "location": "boom",
            },
            f"{equipment_id}_cooler": {
                "type": "cooler",
                "criticality": 0.4,
                "location": "powerpack",
            },
            f"{equipment_id}_accumulator": {
                "type": "accumulator",
                "criticality": 0.3,
                "location": "control_block",
            },
        }

        # Define connections (physical flow)
        connections = [
            # Main circuit
            {"from": f"{equipment_id}_pump", "to": f"{equipment_id}_valve_main", "type": "hydraulic"},
            {"from": f"{equipment_id}_valve_main", "to": f"{equipment_id}_pipe_supply", "type": "hydraulic"},
            {"from": f"{equipment_id}_pipe_supply", "to": f"{equipment_id}_actuator", "type": "hydraulic"},
            {"from": f"{equipment_id}_actuator", "to": f"{equipment_id}_pipe_return", "type": "hydraulic"},
            {"from": f"{equipment_id}_pipe_return", "to": f"{equipment_id}_tank", "type": "hydraulic"},
            {"from": f"{equipment_id}_tank", "to": f"{equipment_id}_filter", "type": "hydraulic"},
            {"from": f"{equipment_id}_filter", "to": f"{equipment_id}_pump", "type": "hydraulic"},
            
            # Relief valve
            {"from": f"{equipment_id}_valve_main", "to": f"{equipment_id}_valve_relief", "type": "hydraulic"},
            {"from": f"{equipment_id}_valve_relief", "to": f"{equipment_id}_tank", "type": "hydraulic"},
            
            # Cooler
            {"from": f"{equipment_id}_pipe_return", "to": f"{equipment_id}_cooler", "type": "thermal"},
            {"from": f"{equipment_id}_cooler", "to": f"{equipment_id}_tank", "type": "thermal"},
            
            # Accumulator
            {"from": f"{equipment_id}_valve_main", "to": f"{equipment_id}_accumulator", "type": "hydraulic"},
        ]

        return cls(components=components, connections=connections)
