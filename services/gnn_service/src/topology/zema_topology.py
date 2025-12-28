"""ZeMA hydraulic system topology for the Condition Monitoring of Hydraulic Systems dataset.

This topology is a *logical* graph tailored for mapping ZeMA sensor channels (PS/FS/TS/VS/EPS/CE/CP/SE)
onto edges/components.

Design constraints (agreed mapping hypotheses):
- PS1: pump -> valve_main (main pressure supply)
- PS2: valve_main -> actuator (load line)
- PS3: drain line (relief / leakage / drain)
- PS4 + TS2: separate cooling contour behavior
- PS5-PS6: filter monitoring (before/after filter)
- TS4: tank temperature (slow trend)

Note:
The public dataset documentation describes sensor channels and sampling rates, but does not provide an
authoritative "sensor -> physical location" schematic. This topology is therefore *engineering-driven* and
intended to be iterated.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GraphTopology:
    """Graph topology representation (same structure as mock_topology.GraphTopology)."""

    def __init__(self, components: dict, connections: list[dict]):
        self.components = components
        self.connections = connections
        logger.info(
            "🔧 Created ZeMA GraphTopology: %d components, %d connections",
            len(components),
            len(connections),
        )

    @classmethod
    def create_zema_benchmark(cls, equipment_id: str = "zema_001") -> "GraphTopology":
        """Create a ZeMA-specific logical topology."""

        components = {
            f"{equipment_id}_pump": {"type": "pump", "criticality": 1.0, "location": "powerpack"},
            f"{equipment_id}_valve_main": {"type": "valve", "criticality": 0.9, "location": "control_block"},
            f"{equipment_id}_actuator": {"type": "actuator", "criticality": 0.9, "location": "load"},
            f"{equipment_id}_tank": {"type": "tank", "criticality": 0.7, "location": "powerpack"},
            # Drain / relief path
            f"{equipment_id}_drain_line": {"type": "pipe", "criticality": 0.6, "location": "drain"},
            # Filtration branch
            f"{equipment_id}_filter": {"type": "filter", "criticality": 0.7, "location": "powerpack"},
            # Cooling branch (separate contour)
            f"{equipment_id}_cooler": {"type": "cooler", "criticality": 0.5, "location": "powerpack"},
            f"{equipment_id}_cooling_line": {"type": "pipe", "criticality": 0.4, "location": "cooling"},
        }

        connections = [
            # Main hydraulic path
            {"from": f"{equipment_id}_pump", "to": f"{equipment_id}_valve_main", "type": "hydraulic"},
            {"from": f"{equipment_id}_valve_main", "to": f"{equipment_id}_actuator", "type": "hydraulic"},
            {"from": f"{equipment_id}_actuator", "to": f"{equipment_id}_tank", "type": "hydraulic"},

            # Drain (PS3)
            {"from": f"{equipment_id}_valve_main", "to": f"{equipment_id}_drain_line", "type": "hydraulic"},
            {"from": f"{equipment_id}_drain_line", "to": f"{equipment_id}_tank", "type": "hydraulic"},

            # Filtration (PS5/PS6)
            {"from": f"{equipment_id}_tank", "to": f"{equipment_id}_filter", "type": "hydraulic"},
            {"from": f"{equipment_id}_filter", "to": f"{equipment_id}_pump", "type": "hydraulic"},

            # Cooling contour (PS4 + TS2)
            {"from": f"{equipment_id}_tank", "to": f"{equipment_id}_cooling_line", "type": "thermal"},
            {"from": f"{equipment_id}_cooling_line", "to": f"{equipment_id}_cooler", "type": "thermal"},
            {"from": f"{equipment_id}_cooler", "to": f"{equipment_id}_tank", "type": "thermal"},
        ]

        return cls(components=components, connections=connections)
