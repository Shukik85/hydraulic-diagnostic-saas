#!/usr/bin/env python3
"""Inspect prepared GNN dataset structure.

Check graph dimensions, edge features, and prepare for Universal GNN.

Usage:
    python scripts/inspect_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def inspect_dataset(data_path: str | Path) -> None:
    """Inspect dataset structure and dimensions.

    Args:
        data_path: Path to .pt dataset file
    """
    data_path = Path(data_path)

    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        return

    print(f"✨ Loading dataset: {data_path}")
    print(f"   Size: {data_path.stat().st_size / 1024 / 1024:.1f} MB\n")

    # Load dataset
    try:
        data = torch.load(data_path, map_location="cpu")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return

    # Check structure
    print(f"📊 Dataset Structure:")
    print(f"   Type: {type(data)}")

    if isinstance(data, dict):
        print(f"   Keys: {list(data.keys())}\n")

        # Inspect each key
        for key, value in data.items():
            print(f"\n🔑 Key: '{key}'")
            print(f"   Type: {type(value)}")

            if isinstance(value, torch.Tensor):
                print(f"   Shape: {value.shape}")
                print(f"   Dtype: {value.dtype}")
                print(f"   Device: {value.device}")

                # Sample values
                if value.numel() > 0:
                    print(f"   Min: {value.min().item():.4f}")
                    print(f"   Max: {value.max().item():.4f}")
                    print(f"   Mean: {value.float().mean().item():.4f}")

            elif isinstance(value, list):
                print(f"   Length: {len(value)}")
                if len(value) > 0:
                    print(f"   First item type: {type(value[0])}")

                    # If list of Data objects
                    if hasattr(value[0], "x"):
                        print(f"\n   📊 Graph Statistics (first 5):")
                        for i, graph in enumerate(value[:5]):
                            print(f"\n   Graph {i}:")
                            print(f"      Nodes: {graph.num_nodes}")
                            print(f"      Edges: {graph.num_edges}")

                            if hasattr(graph, "x") and graph.x is not None:
                                print(f"      Node features: {graph.x.shape[1]}")

                            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                                print(f"      Edge features: {graph.edge_attr.shape[1]} ⭐")
                            else:
                                print(f"      Edge features: None")

                            # Check for labels
                            if hasattr(graph, "y"):
                                print(f"      Labels: {graph.y.shape if hasattr(graph.y, 'shape') else type(graph.y)}")

                        # Statistics across all graphs
                        print(f"\n   📊 Dataset Statistics (all {len(value)} graphs):")

                        num_nodes_list = [g.num_nodes for g in value]
                        num_edges_list = [g.num_edges for g in value]

                        print(f"      Nodes: min={min(num_nodes_list)}, max={max(num_nodes_list)}, avg={sum(num_nodes_list)/len(num_nodes_list):.1f}")
                        print(f"      Edges: min={min(num_edges_list)}, max={max(num_edges_list)}, avg={sum(num_edges_list)/len(num_edges_list):.1f}")

                        # Check edge_attr consistency
                        edge_attr_dims = [
                            g.edge_attr.shape[1]
                            for g in value
                            if hasattr(g, "edge_attr") and g.edge_attr is not None
                        ]

                        if edge_attr_dims:
                            unique_dims = set(edge_attr_dims)
                            print(f"\n      ⭐ Edge Feature Dimensions:")
                            print(f"         Unique dimensions: {unique_dims}")
                            if len(unique_dims) == 1:
                                print(f"         ✅ All graphs have same edge_in_dim: {list(unique_dims)[0]}")
                            else:
                                print(f"         ⚠️  Mixed edge dimensions detected!")
                                for dim in unique_dims:
                                    count = edge_attr_dims.count(dim)
                                    print(f"            {dim}D: {count} graphs")

    elif isinstance(data, list):
        print(f"   Length: {len(data)}\n")
        if len(data) > 0 and hasattr(data[0], "x"):
            # List of Data objects
            print(f"📊 Graph List Statistics:")
            for i, graph in enumerate(data[:5]):
                print(f"\nGraph {i}:")
                print(f"   Nodes: {graph.num_nodes}")
                print(f"   Edges: {graph.num_edges}")

                if hasattr(graph, "x") and graph.x is not None:
                    print(f"   Node features: {graph.x.shape[1]}")

                if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                    print(f"   Edge features: {graph.edge_attr.shape[1]} ⭐")

    else:
        print(f"   Unknown structure: {type(data)}")

    print(f"\n✅ Inspection complete!")


if __name__ == "__main__":
    # Default dataset path
    dataset_path = Path(__file__).parent.parent / "data" / "gnn_graphs_multilabel.pt"

    # Allow custom path from command line
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])

    inspect_dataset(dataset_path)
