"""Attention explainability module.

Генерирует human-readable объяснения для GNN predictions.
"""

import torch
import structlog

logger = structlog.get_logger(__name__)


class AttentionExplainer:
    """Explainer для attention-based causal reasoning."""
    
    def __init__(self, attention_threshold: float = 0.3):
        self.attention_threshold = attention_threshold
    
    def explain(
        self,
        attention_weights_list: list[tuple[torch.Tensor, torch.Tensor]],
        node_features: list[list[float]],
        edge_index: list[list[int]],
        component_names: list[str],
    ) -> dict:
        """Generate explainability.
        
        Args:
            attention_weights_list: List of (edge_index, attention) tuples per layer
            node_features: Original node features
            edge_index: Edge connectivity
            component_names: Human-readable component names
        
        Returns:
            Explanation dict
        """
        # Aggregate attention across layers (mean)
        all_attention_scores = []
        for edge_idx, attention in attention_weights_list:
            all_attention_scores.append(attention)
        
        # Average attention per edge
        avg_attention = torch.stack(all_attention_scores).mean(dim=0)
        
        # Node-level attention (aggregate incoming edges)
        num_nodes = len(component_names)
        node_attention = torch.zeros(num_nodes)
        
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        for i in range(num_nodes):
            # Find edges pointing to node i
            mask = edge_index_tensor[1] == i
            if mask.any():
                node_attention[i] = avg_attention[mask].mean().item()
        
        # Identify critical components
        critical_indices = (node_attention > self.attention_threshold).nonzero(as_tuple=True)[0]
        critical_components = [component_names[i] for i in critical_indices]
        
        # Build causal path (simplified heuristic)
        causal_path = self._build_causal_path(
            critical_indices=critical_indices,
            edge_index=edge_index,
            component_names=component_names,
            node_attention=node_attention,
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            critical_components=critical_components,
            causal_path=causal_path,
            node_features=node_features,
        )
        
        return {
            "critical_components": critical_components,
            "attention_scores": node_attention.tolist(),
            "causal_path": causal_path,
            "reasoning": reasoning,
        }
    
    def _build_causal_path(
        self,
        critical_indices: torch.Tensor,
        edge_index: list[list[int]],
        component_names: list[str],
        node_attention: torch.Tensor,
    ) -> list[str]:
        """Build causal chain.
        
        Heuristic: Follow edges from highest attention to lowest.
        """
        if len(critical_indices) == 0:
            return []
        
        # Sort by attention (descending)
        sorted_indices = critical_indices[torch.argsort(node_attention[critical_indices], descending=True)]
        
        path = []
        for idx in sorted_indices:
            comp_name = component_names[idx.item()]
            
            # Infer state from attention score
            score = node_attention[idx].item()
            if score > 0.7:
                state = "critical_failure"
            elif score > 0.5:
                state = "degradation"
            else:
                state = "anomaly"
            
            path.append(f"{comp_name}_{state}")
        
        return path
    
    def _generate_reasoning(
        self,
        critical_components: list[str],
        causal_path: list[str],
        node_features: list[list[float]],
    ) -> str:
        """Generate natural language reasoning."""
        if not critical_components:
            return "No critical components identified."
        
        # Simple template-based reasoning
        primary = critical_components[0]
        
        reasoning_parts = []
        reasoning_parts.append(f"Primary component affected: {primary}.")
        
        if len(critical_components) > 1:
            secondary = ", ".join(critical_components[1:])
            reasoning_parts.append(f"Secondary components: {secondary}.")
        
        if causal_path:
            chain = " → ".join(causal_path[:3])  # Limit to 3 steps
            reasoning_parts.append(f"Causal chain detected: {chain}.")
        
        reasoning_parts.append(
            "System-level anomaly detected with cascading effects. "
            "Immediate inspection recommended."
        )
        
        return " ".join(reasoning_parts)
