"""Graph Builder для конструирования equipment topology."""

import structlog
from django.db.models import Avg
from datetime import timedelta
from django.utils import timezone

from equipment.models import Equipment, HydraulicComponent
from sensors.models import SensorData

logger = structlog.get_logger(__name__)


class GraphBuilder:
    """Builder для equipment graph construction."""
    
    async def build_graph(
        self,
        equipment_id: int,
        window_seconds: int = 20,
    ) -> dict:
        """Build graph from equipment topology + sensor data.
        
        Args:
            equipment_id: Equipment ID
            window_seconds: Time window для sensor aggregation
        
        Returns:
            Graph dict with node_features, edge_index, edge_attr, component_names
        """
        equipment = await Equipment.objects.aget(id=equipment_id)
        
        # Get hydraulic components
        components = await HydraulicComponent.objects.filter(
            equipment=equipment
        ).order_by("id").values(
            "id",
            "component_type",
            "name",
        ).all()
        
        component_list = [c async for c in components]
        
        # Build node features (aggregate sensor data)
        node_features = []
        component_names = []
        
        end_time = timezone.now()
        start_time = end_time - timedelta(seconds=window_seconds)
        
        for comp in component_list:
            # Aggregate sensors for this component
            sensors = await SensorData.objects.filter(
                equipment=equipment,
                component_id=comp["id"],
                timestamp__gte=start_time,
                timestamp__lte=end_time,
            ).values(
                "sensor_type",
            ).annotate(
                avg_value=Avg("value"),
            ).all()
            
            sensor_dict = {s["sensor_type"]: s["avg_value"] async for s in sensors}
            
            # Feature vector: [pressure, temperature, flow, vibration, ...]
            features = [
                sensor_dict.get("pressure", 0.0),
                sensor_dict.get("temperature", 0.0),
                sensor_dict.get("flow_rate", 0.0),
                sensor_dict.get("vibration", 0.0),
                sensor_dict.get("position", 0.0),
                sensor_dict.get("speed", 0.0),
                sensor_dict.get("current", 0.0),
                sensor_dict.get("voltage", 0.0),
                sensor_dict.get("power", 0.0),
                sensor_dict.get("efficiency", 0.0),
            ]
            
            node_features.append(features)
            component_names.append(comp["name"] or comp["component_type"])
        
        # Build edge_index (hydraulic connections)
        edge_index = self._build_topology(component_list)
        
        # Build edge_attr (pipe characteristics)
        edge_attr = self._build_edge_attributes(component_list, edge_index)
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "component_names": component_names,
        }
    
    def _build_topology(self, components: list[dict]) -> list[list[int]]:
        """Build edge_index from hydraulic topology.
        
        Simplified: linear chain (pump → actuators).
        TODO: Load from equipment.topology_config JSON.
        """
        num_nodes = len(components)
        
        if num_nodes < 2:
            return [[], []]
        
        # Simple chain: 0 → 1 → 2 → ...
        sources = list(range(num_nodes - 1))
        targets = list(range(1, num_nodes))
        
        # Bidirectional edges
        edge_index = [
            sources + targets,
            targets + sources,
        ]
        
        return edge_index
    
    def _build_edge_attributes(
        self,
        components: list[dict],
        edge_index: list[list[int]],
    ) -> list[list[float]]:
        """Build edge attributes (pipe diameter, flow, pressure drop).
        
        Simplified: dummy attributes.
        TODO: Load from equipment.pipe_configs.
        """
        num_edges = len(edge_index[0]) if edge_index else 0
        
        # Dummy attributes: [diameter_mm, nominal_flow, pressure_drop]
        edge_attr = [[50.0, 180.0, 5.0] for _ in range(num_edges)]
        
        return edge_attr
