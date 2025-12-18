"""Gradient-balanced multi-task loss for GNN training.

Automatically balances task weights based on gradient magnitudes.
Prevents degradation of graph-level task in multi-task learning.

Reference:
    "Multi-Task Learning on Graphs with Node and Graph Level Labels"
    NeurIPS 2019 Workshop on Graph Representation Learning
    https://grlearning.github.io/papers/132.pdf

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradientBalancedMultiTaskLoss(nn.Module):
    """Multi-task loss with automatic gradient balancing.
    
    Problem:
    - In multi-task learning, tasks with larger gradients dominate
    - Graph-level task often has smaller gradients → gets ignored
    - Fixed weights (λ_node=1.0, λ_graph=1.0) don't adapt
    
    Solution:
    - Measure gradient magnitude per task
    - Adjust weights inversely proportional to gradient magnitude
    - Dynamic balancing during training
    
    Examples:
        >>> loss_fn = GradientBalancedMultiTaskLoss(
        ...     node_loss=nn.CrossEntropyLoss(),
        ...     graph_loss=nn.CrossEntropyLoss(),
        ...     alpha=0.16  # Balance strength
        ... )
        >>> 
        >>> # Training loop
        >>> outputs = model(data)
        >>> loss_dict = loss_fn(
        ...     outputs['node_logits'], data.y_node,
        ...     outputs['graph_logits'], data.y_graph,
        ...     shared_params=model.gat_layers.parameters()
        ... )
        >>> loss_dict['total'].backward()
    """
    
    def __init__(
        self,
        node_loss: nn.Module,
        graph_loss: nn.Module,
        alpha: float = 0.16,
        initial_node_weight: float = 1.0,
        initial_graph_weight: float = 1.0
    ) -> None:
        """Initialize gradient-balanced loss.
        
        Args:
            node_loss: Loss function for node-level task
            graph_loss: Loss function for graph-level task
            alpha: Balancing strength (higher = more aggressive)
            initial_node_weight: Initial weight for node task
            initial_graph_weight: Initial weight for graph task
        """
        super().__init__()
        
        self.node_loss_fn = node_loss
        self.graph_loss_fn = graph_loss
        self.alpha = alpha
        
        # Learnable task weights (initialized to provided values)
        self.node_weight = nn.Parameter(
            torch.tensor(initial_node_weight),
            requires_grad=False  # Updated manually, not via backprop
        )
        self.graph_weight = nn.Parameter(
            torch.tensor(initial_graph_weight),
            requires_grad=False
        )
        
        logger.info(
            "GradientBalancedMultiTaskLoss: alpha=%.3f",
            alpha
        )
    
    def forward(
        self,
        node_logits: torch.Tensor,
        node_labels: torch.Tensor,
        graph_logits: torch.Tensor,
        graph_labels: torch.Tensor,
        shared_params: list[torch.nn.Parameter] | None = None
    ) -> dict[str, torch.Tensor]:
        """Compute gradient-balanced multi-task loss.
        
        Args:
            node_logits: Node predictions [num_nodes, num_classes]
            node_labels: Node ground truth [num_nodes]
            graph_logits: Graph predictions [batch_size, num_classes]
            graph_labels: Graph ground truth [batch_size]
            shared_params: Shared parameters for gradient computation
            
        Returns:
            Dictionary with losses and weights
        """
        # Compute individual losses
        node_loss = self.node_loss_fn(node_logits, node_labels)
        graph_loss = self.graph_loss_fn(graph_logits, graph_labels)
        
        # Update weights based on gradients (if training)
        if self.training and shared_params is not None:
            self._update_weights(
                node_loss, graph_loss, shared_params
            )
        
        # Weighted sum
        total_loss = (
            self.node_weight * node_loss +
            self.graph_weight * graph_loss
        )
        
        return {
            'total': total_loss,
            'node_loss': node_loss.detach(),
            'graph_loss': graph_loss.detach(),
            'node_weight': self.node_weight.item(),
            'graph_weight': self.graph_weight.item()
        }
    
    def _update_weights(
        self,
        node_loss: torch.Tensor,
        graph_loss: torch.Tensor,
        shared_params: list[torch.nn.Parameter]
    ) -> None:
        """Update task weights based on gradient magnitudes.
        
        Args:
            node_loss: Node task loss
            graph_loss: Graph task loss
            shared_params: Shared parameters
        """
        # Compute gradients w.r.t. shared parameters
        node_grads = torch.autograd.grad(
            node_loss, shared_params,
            retain_graph=True, create_graph=False,
            allow_unused=True
        )
        
        graph_grads = torch.autograd.grad(
            graph_loss, shared_params,
            retain_graph=True, create_graph=False,
            allow_unused=True
        )
        
        # Compute L2 norm of gradients
        node_grad_norm = sum(
            g.norm(2) for g in node_grads if g is not None
        )
        graph_grad_norm = sum(
            g.norm(2) for g in graph_grads if g is not None
        )
        
        # Avoid division by zero
        node_grad_norm = node_grad_norm + 1e-8
        graph_grad_norm = graph_grad_norm + 1e-8
        
        # Compute relative gradient ratio
        # Inverse relationship: larger gradients → smaller weight
        avg_grad_norm = (node_grad_norm + graph_grad_norm) / 2
        
        new_node_weight = (avg_grad_norm / node_grad_norm) ** self.alpha
        new_graph_weight = (avg_grad_norm / graph_grad_norm) ** self.alpha
        
        # Update weights (exponential moving average)
        momentum = 0.9
        self.node_weight.data = (
            momentum * self.node_weight.data +
            (1 - momentum) * new_node_weight
        )
        self.graph_weight.data = (
            momentum * self.graph_weight.data +
            (1 - momentum) * new_graph_weight
        )
        
        logger.debug(
            "Gradient norms: node=%.4f, graph=%.4f | Weights: node=%.4f, graph=%.4f",
            node_grad_norm.item(), graph_grad_norm.item(),
            self.node_weight.item(), self.graph_weight.item()
        )
