"""Attention weight extraction and visualization for GAT models.

Provides tools to:
- Extract attention weights from GAT layers
- Visualize attention flow in graphs
- Compute edge importance scores
- Generate interpretability reports

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AttentionWeightExtractor:
    """Extract and analyze attention weights from GAT layers.
    
    Examples:
        >>> model = UniversalTemporalGNN(...)
        >>> extractor = AttentionWeightExtractor(model)
        >>> 
        >>> # Forward pass
        >>> outputs, attention_weights = extractor.forward_with_attention(data)
        >>> 
        >>> # Analyze attention distribution
        >>> stats = extractor.compute_attention_statistics(attention_weights)
        >>> print(stats['mean_attention_per_edge'])
    """
    
    def __init__(self, model: nn.Module) -> None:
        """Initialize extractor.
        
        Args:
            model: GNN model with GAT layers
        """
        self.model = model
        self.attention_cache: dict[str, torch.Tensor] = {}
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward hooks to capture attention weights."""
        def attention_hook(module, input, output):
            # GAT returns (output, attention_weights)
            if isinstance(output, tuple) and len(output) == 2:
                _, attention = output
                layer_name = f"gat_layer_{len(self.attention_cache)}"
                self.attention_cache[layer_name] = attention.detach()
        
        # Register hooks on GAT layers
        for name, module in self.model.named_modules():
            if 'gat' in name.lower() or 'GATConv' in str(type(module)):
                module.register_forward_hook(attention_hook)
                logger.debug("Registered attention hook on %s", name)
    
    def forward_with_attention(
        self,
        data
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass with attention extraction.
        
        Args:
            data: PyG Data object
            
        Returns:
            Tuple of (model_output, attention_weights_dict)
        """
        # Clear cache
        self.attention_cache.clear()
        
        # Forward pass (hooks will populate cache)
        output = self.model(data)
        
        return output, self.attention_cache.copy()
    
    def compute_attention_statistics(
        self,
        attention_weights: dict[str, torch.Tensor]
    ) -> dict[str, float | torch.Tensor]:
        """Compute statistics from attention weights.
        
        Args:
            attention_weights: Dictionary of attention tensors per layer
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        for layer_name, attn in attention_weights.items():
            # Attention shape: [num_edges, num_heads]
            stats[f"{layer_name}_mean"] = attn.mean().item()
            stats[f"{layer_name}_std"] = attn.std().item()
            stats[f"{layer_name}_max"] = attn.max().item()
            stats[f"{layer_name}_min"] = attn.min().item()
            
            # Per-head statistics
            if attn.dim() == 2:
                stats[f"{layer_name}_head_means"] = attn.mean(dim=0)  # [num_heads]
        
        return stats
    
    def get_edge_importance(
        self,
        attention_weights: dict[str, torch.Tensor],
        aggregate: str = 'mean'
    ) -> torch.Tensor:
        """Compute edge importance scores from attention.
        
        Args:
            attention_weights: Attention weights per layer
            aggregate: How to aggregate across layers ('mean', 'max', 'sum')
            
        Returns:
            Edge importance scores [num_edges]
        """
        # Collect all attention tensors
        all_attn = []
        for attn in attention_weights.values():
            # Average across heads if multi-head
            if attn.dim() == 2:
                attn = attn.mean(dim=1)  # [num_edges]
            all_attn.append(attn)
        
        # Stack and aggregate
        stacked = torch.stack(all_attn, dim=0)  # [num_layers, num_edges]
        
        if aggregate == 'mean':
            importance = stacked.mean(dim=0)
        elif aggregate == 'max':
            importance = stacked.max(dim=0)[0]
        elif aggregate == 'sum':
            importance = stacked.sum(dim=0)
        else:
            msg = f"Unknown aggregation: {aggregate}"
            raise ValueError(msg)
        
        return importance
    
    def visualize_attention(
        self,
        data,
        attention_weights: dict[str, torch.Tensor],
        save_path: Path | str | None = None
    ) -> None:
        """Visualize attention weights on graph.
        
        Args:
            data: PyG Data object
            attention_weights: Attention weights to visualize
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            msg = "matplotlib and networkx required for visualization"
            raise ImportError(msg) from exc
        
        # Get edge importance
        importance = self.get_edge_importance(attention_weights, aggregate='mean')
        
        # Create NetworkX graph
        G = nx.DiGraph()
        edge_index = data.edge_index.cpu().numpy()
        
        # Add edges with importance as weight
        for i, (src, tgt) in enumerate(edge_index.T):
            G.add_edge(src.item(), tgt.item(), weight=importance[i].item())
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw
        plt.figure(figsize=(12, 8))
        
        # Nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=500
        )
        
        # Edges with varying width by attention
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1.0
        edge_widths = [5 * w / max_weight for w in edge_weights]
        
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color=edge_weights,
            edge_cmap=plt.cm.Reds,
            arrows=True,
            arrowsize=20
        )
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title("Attention Flow Visualization", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Saved attention visualization to %s", save_path)
        else:
            plt.show()
        
        plt.close()
    
    def save_attention_weights(
        self,
        attention_weights: dict[str, torch.Tensor],
        save_path: Path | str
    ) -> None:
        """Save attention weights to disk.
        
        Args:
            attention_weights: Attention weights to save
            save_path: Path to save file (.pt)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(attention_weights, save_path)
        logger.info("Saved attention weights to %s", save_path)
