# services/gnn_service/graph_builder.py
"""
Dynamic graph topology builder utility for GNN service.
"""
from schemas import EquipmentMetadata, ComponentRole, GraphTopology

def build_dynamic_graph(metadata: EquipmentMetadata):
    comp_to_idx = {c.id: i for i, c in enumerate(metadata.components)}
    edges = []
    if metadata.connections:
        for conn in metadata.connections:
            src = comp_to_idx[conn.from_component]
            dst = comp_to_idx[conn.to_component]
            edges.append([src, dst])
            if conn.bidirectional:
                edges.append([dst, src])
    else:
        # auto topology: energy_source->actuators, weak actuator-actuator
        sources = [c for c in metadata.components if c.role == ComponentRole.ENERGY_SOURCE]
        actuators = [c for c in metadata.components if c.role == ComponentRole.ACTUATOR]
        for source in sources:
            for a in actuators:
                edges.extend([[comp_to_idx[source.id], comp_to_idx[a.id]],[comp_to_idx[a.id], comp_to_idx[source.id]]])
        for i, a1 in enumerate(actuators):
            for a2 in actuators[i+1:]:
                edges.extend([[comp_to_idx[a1.id], comp_to_idx[a2.id]],[comp_to_idx[a2.id], comp_to_idx[a1.id]]])
    return edges
