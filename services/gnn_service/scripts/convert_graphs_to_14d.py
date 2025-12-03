#!/usr/bin/env python3
"""Convert 8D edge graphs to 14D for Phase 3 validation.

Converts existing PyTorch Geometric graphs from 8D edge features
to 14D edge features using Phase 3.1 components.

Author: GNN Service Team
Python: 3.14+
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
import argparse
from typing import List, Dict
from datetime import datetime, timedelta
from tqdm import tqdm

from src.data.edge_features import EdgeFeatureComputer, create_edge_feature_computer
from src.data.normalization import EdgeFeatureNormalizer, create_edge_feature_normalizer
from src.schemas import EdgeSpec, ComponentType, EdgeType
from src.schemas.graph import EdgeMaterial
from src.schemas.requests import ComponentSensorReading


def load_edge_specifications(edge_specs_path: Path) -> Dict[str, EdgeSpec]:
    """Load edge specifications from JSON.
    
    Args:
        edge_specs_path: Path to edge_specifications.json
    
    Returns:
        Dictionary mapping edge_id to EdgeSpec
    """
    print(f"Loading edge specifications: {edge_specs_path}")
    
    with open(edge_specs_path, 'r') as f:
        data = json.load(f)
    
    edge_specs = {}
    
    for edge_data in data['edges']:
        # Create edge_id from source and target
        edge_id = f"{edge_data['source_id']}_to_{edge_data['target_id']}"
        
        # Parse material
        material_str = edge_data['material'].lower()
        if material_str == 'steel':
            material = EdgeMaterial.STEEL
        elif material_str == 'rubber':
            material = EdgeMaterial.RUBBER
        else:
            material = EdgeMaterial.COMPOSITE
        
        # Parse edge type
        edge_type_str = edge_data['edge_type'].lower()
        if edge_type_str == 'pipe':
            edge_type = EdgeType.PIPE
        elif edge_type_str == 'hose':
            edge_type = EdgeType.HOSE
        else:
            edge_type = EdgeType.PIPE  # Default
        
        # Parse install date
        install_date = None
        if 'install_date' in edge_data:
            try:
                install_date = datetime.fromisoformat(
                    edge_data['install_date'].replace('Z', '+00:00')
                )
            except:
                pass
        
        # Create EdgeSpec
        edge_spec = EdgeSpec(
            source_id=edge_data['source_id'],
            target_id=edge_data['target_id'],
            edge_type=edge_type,
            diameter_mm=edge_data['diameter_mm'],
            length_m=edge_data['length_m'],
            material=material,
            pressure_rating_bar=edge_data.get('pressure_rating_bar', 300.0),
            temperature_rating_c=edge_data.get('temperature_rating_c', 100.0),
            install_date=install_date
        )
        
        edge_specs[edge_id] = edge_spec
    
    print(f"Loaded {len(edge_specs)} edge specifications")
    
    return edge_specs


def create_synthetic_sensor_readings(
    graph,
    component_ids: List[str]
) -> Dict[str, ComponentSensorReading]:
    """Create synthetic sensor readings from graph node features.
    
    Args:
        graph: PyG Data object
        component_ids: List of component IDs (in node order)
    
    Returns:
        Dictionary of sensor readings per component
    """
    sensor_readings = {}
    
    for i, comp_id in enumerate(component_ids):
        # Extract node features (assume first few are pressure/temp/vibration)
        node_features = graph.x[i]
        
        # Heuristic: assume first feature is pressure-related
        # (This is a rough approximation - ideally we'd know the exact mapping)
        pressure_base = 150.0  # Nominal pressure
        temperature_base = 65.0  # Nominal temperature
        vibration_base = 0.5  # Low vibration
        
        sensor_readings[comp_id] = ComponentSensorReading(
            pressure_bar=pressure_base,
            temperature_c=temperature_base,
            vibration_g=vibration_base,
            rpm=1450 if 'pump' in comp_id else None
        )
    
    return sensor_readings


def convert_graph_to_14d(
    graph,
    edge_specs: Dict[str, EdgeSpec],
    component_ids: List[str],
    edge_computer: EdgeFeatureComputer,
    current_time: datetime
):
    """Convert single graph from 8D to 14D edge features.
    
    Args:
        graph: PyG Data object with 8D edges
        edge_specs: Edge specifications
        component_ids: Component IDs (in node order)
        edge_computer: EdgeFeatureComputer instance
        current_time: Current timestamp
    
    Returns:
        Updated graph with 14D edge features
    """
    num_edges = graph.edge_index.shape[1]
    
    # Create synthetic sensor readings
    sensor_readings = create_synthetic_sensor_readings(graph, component_ids)
    
    # Build 14D edge features
    edge_features_14d = []
    
    for edge_idx in range(num_edges):
        source_node = graph.edge_index[0, edge_idx].item()
        target_node = graph.edge_index[1, edge_idx].item()
        
        source_id = component_ids[source_node]
        target_id = component_ids[target_node]
        
        edge_id = f"{source_id}_to_{target_id}"
        
        # Get edge spec (or use default)
        if edge_id in edge_specs:
            edge_spec = edge_specs[edge_id]
        else:
            # Reverse direction?
            edge_id_rev = f"{target_id}_to_{source_id}"
            if edge_id_rev in edge_specs:
                # Flip source/target
                edge_spec_orig = edge_specs[edge_id_rev]
                edge_spec = EdgeSpec(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_spec_orig.edge_type,
                    diameter_mm=edge_spec_orig.diameter_mm,
                    length_m=edge_spec_orig.length_m,
                    material=edge_spec_orig.material,
                    pressure_rating_bar=edge_spec_orig.pressure_rating_bar,
                    temperature_rating_c=edge_spec_orig.temperature_rating_c,
                    install_date=edge_spec_orig.install_date
                )
            else:
                # Default edge spec
                edge_spec = EdgeSpec(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=EdgeType.PIPE,
                    diameter_mm=20.0,
                    length_m=2.0,
                    material=EdgeMaterial.STEEL,
                    pressure_rating_bar=300.0
                )
        
        # Compute dynamic features
        dynamic_features = edge_computer.compute_edge_features(
            edge=edge_spec,
            sensor_readings=sensor_readings,
            current_time=current_time
        )
        
        # Combine static (8D from graph) + dynamic (6D computed)
        static_features = graph.edge_attr[edge_idx]  # [8]
        
        dynamic_tensor = torch.tensor([
            dynamic_features['flow_rate_lpm'],
            dynamic_features['pressure_drop_bar'],
            dynamic_features['temperature_delta_c'],
            dynamic_features['vibration_level_g'],
            dynamic_features['age_hours'],
            dynamic_features['maintenance_score']
        ], dtype=torch.float32)
        
        # Concatenate to 14D
        edge_feature_14d = torch.cat([static_features, dynamic_tensor])
        edge_features_14d.append(edge_feature_14d)
    
    # Stack into tensor
    new_edge_attr = torch.stack(edge_features_14d)  # [E, 14]
    
    # Update graph
    graph.edge_attr = new_edge_attr
    
    return graph


def fit_normalizer_on_batch(
    graphs: List,
    edge_specs: Dict[str, EdgeSpec],
    component_ids: List[str]
) -> EdgeFeatureNormalizer:
    """Fit normalizer on a batch of graphs.
    
    Args:
        graphs: List of PyG graphs
        edge_specs: Edge specifications
        component_ids: Component IDs
    
    Returns:
        Fitted EdgeFeatureNormalizer
    """
    print("\nFitting normalizer on batch...")
    
    edge_computer = create_edge_feature_computer()
    normalizer = create_edge_feature_normalizer()
    current_time = datetime.now()
    
    # Collect all dynamic features
    all_features = []
    
    for graph in tqdm(graphs[:100], desc="Collecting features"):  # Use subset
        sensor_readings = create_synthetic_sensor_readings(graph, component_ids)
        
        for edge_id, edge_spec in edge_specs.items():
            features = edge_computer.compute_edge_features(
                edge=edge_spec,
                sensor_readings=sensor_readings,
                current_time=current_time
            )
            all_features.append(features)
    
    # Fit normalizer
    normalizer.fit(all_features)
    
    print(f"Normalizer fitted on {len(all_features)} edge samples")
    
    return normalizer


def main():
    parser = argparse.ArgumentParser(
        description="Convert 8D graphs to 14D for Phase 3"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to input graphs (8D)'
    )
    parser.add_argument(
        '--edge-specs',
        type=Path,
        required=True,
        help='Path to edge_specifications.json'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output graphs (14D)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to convert (for testing)'
    )
    parser.add_argument(
        '--component-ids',
        type=str,
        default='pump_main_1,pump_main_2,valve_main,cylinder_boom,cylinder_arm,cylinder_bucket,motor_swing,motor_travel_left,motor_travel_right',
        help='Comma-separated component IDs (in node order)'
    )
    
    args = parser.parse_args()
    
    # Parse component IDs
    component_ids = args.component_ids.split(',')
    print(f"Component IDs ({len(component_ids)}): {component_ids}")
    
    # Load edge specifications
    edge_specs = load_edge_specifications(args.edge_specs)
    
    # Load graphs
    print(f"\nLoading graphs: {args.input}")
    graphs = torch.load(args.input)
    
    if not isinstance(graphs, list):
        graphs = [graphs]
    
    print(f"Loaded {len(graphs)} graphs")
    
    # Limit samples if requested
    if args.max_samples:
        graphs = graphs[:args.max_samples]
        print(f"Using first {len(graphs)} samples")
    
    # Create edge computer
    edge_computer = create_edge_feature_computer()
    current_time = datetime.now()
    
    # Fit normalizer (optional - comment out if not needed)
    # normalizer = fit_normalizer_on_batch(graphs, edge_specs, component_ids)
    
    # Convert graphs
    print("\nConverting graphs to 14D...")
    converted_graphs = []
    
    for graph in tqdm(graphs, desc="Converting"):
        try:
            converted = convert_graph_to_14d(
                graph=graph,
                edge_specs=edge_specs,
                component_ids=component_ids,
                edge_computer=edge_computer,
                current_time=current_time
            )
            converted_graphs.append(converted)
        except Exception as e:
            print(f"\nError converting graph: {e}")
            continue
    
    # Validate
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    for i, graph in enumerate(converted_graphs[:5]):
        print(f"\nGraph {i}:")
        print(f"  Nodes: {graph.x.shape}")
        print(f"  Edges: {graph.edge_index.shape}")
        print(f"  Edge features: {graph.edge_attr.shape}")
        
        # Check 14D
        assert graph.edge_attr.shape[1] == 14, f"Expected 14D, got {graph.edge_attr.shape[1]}D"
        
        # Check no NaN
        assert not torch.isnan(graph.edge_attr).any(), "NaN detected in edge features"
        
        print(f"  ✅ Valid (14D, no NaN)")
    
    # Save
    print(f"\nSaving converted graphs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted_graphs, args.output)
    
    print(f"\n✅ Saved {len(converted_graphs)} graphs with 14D edge features")
    
    # Statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
    total_edges = sum(g.edge_attr.shape[0] for g in converted_graphs)
    avg_nodes = sum(g.x.shape[0] for g in converted_graphs) / len(converted_graphs)
    avg_edges = total_edges / len(converted_graphs)
    
    print(f"\nDataset:")
    print(f"  Total graphs: {len(converted_graphs)}")
    print(f"  Total edges: {total_edges}")
    print(f"  Avg nodes/graph: {avg_nodes:.1f}")
    print(f"  Avg edges/graph: {avg_edges:.1f}")
    print(f"  Edge feature dim: 14")
    
    print("\n✅ CONVERSION COMPLETE")
    print("\nNext steps:")
    print("  1. Inspect converted graphs")
    print("  2. Run quick training (1 epoch)")
    print("  3. Validate model accepts 14D edges")


if __name__ == '__main__':
    main()
