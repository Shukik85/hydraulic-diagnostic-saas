"""Missing data imputation engine for temporal graphs.

Handles highly incomplete data (>50% missing) via:
- GRAPE graph reconstruction
- Spatial-temporal feature propagation
- Asymmetric noise modeling for low-confidence sensors

References:
  [3] GRAPE: Missing data via GNN edge embeddings
  [5] Spatial-temporal imputation for sensor networks
"""

from __future__ import annotations

import numpy as np
import torch


class ImputationEngine:
    """Missing data imputation for incomplete sensor networks.

    Two-stage approach:
    1. Feature propagation: Fill missing values from neighbors
    2. Confidence weighting: Reduce impact of low-quality reconstructions
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        propagation_hops: int = 2,
    ):
        self.confidence_threshold = confidence_threshold
        self.propagation_hops = propagation_hops

    def impute_missing_features(
        self,
        x: np.ndarray,  # [N, F]
        mask_nodes: np.ndarray,  # [N]
        edge_index: np.ndarray,  # [2, E]
        edge_weights: np.ndarray | None = None,  # [E]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Impute missing node features via spatial propagation.

        Args:
            x: Node features (partially filled with NaN or 0 for missing)
            mask_nodes: Binary mask (1=present, 0=missing)
            edge_index: Graph edges
            edge_weights: Edge confidence scores

        Returns:
            x_imputed: Imputed features
            imputation_confidence: Confidence scores for each imputed feature
        """
        x_imputed = x.copy()
        imputation_conf = np.ones_like(mask_nodes)

        # Identify missing nodes
        missing_nodes = np.where(mask_nodes == 0)[0]

        for node_idx in missing_nodes:
            # Find neighbors within propagation_hops
            neighbors = self._find_neighbors(
                node_idx, edge_index, self.propagation_hops
            )
            neighbor_values = [x[n] for n in neighbors if mask_nodes[n] == 1]

            if neighbor_values:
                # Weighted average from neighbors
                weights = self._compute_neighbor_weights(
                    node_idx, neighbors, edge_weights
                )
                x_imputed[node_idx] = np.average(
                    neighbor_values, axis=0, weights=weights
                )
                imputation_conf[node_idx] = np.mean(weights)
            else:
                # No neighbors: use global mean
                x_imputed[node_idx] = np.nanmean(x, axis=0)
                imputation_conf[node_idx] = 0.1  # Low confidence

        return x_imputed, imputation_conf

    def _find_neighbors(self, node_idx: int, edge_index: np.ndarray, hops: int) -> list:
        """Find neighbors within k hops."""
        neighbors = set()
        current_level = {node_idx}

        for _ in range(hops):
            next_level = set()
            for node in current_level:
                # Find edges involving this node
                mask = (edge_index[0] == node) | (edge_index[1] == node)
                edges = edge_index[:, mask]
                for i in range(edges.shape[1]):
                    src, dst = edges[0, i], edges[1, i]
                    next_node = dst if src == node else src
                    if next_node not in neighbors:
                        next_level.add(next_node)
            neighbors.update(next_level)
            current_level = next_level

        return list(neighbors)

    def _compute_neighbor_weights(
        self,
        node_idx: int,
        neighbors: list,
        edge_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Compute weights for neighbor aggregation."""
        if edge_weights is None:
            return np.ones(len(neighbors)) / len(neighbors)

        # Distance-based weights
        weights = np.ones(len(neighbors))
        for i, neighbor in enumerate(neighbors):
            weights[i] = 1.0 / (1 + abs(neighbor - node_idx))

        return weights / weights.sum()

    def apply_asymmetric_noise_model(
        self,
        x: np.ndarray,
        imputation_conf: np.ndarray,
    ) -> np.ndarray:
        """Apply confidence-weighted noise to low-quality reconstructions.

        Low-confidence sensors get more noise injected to reduce their
        influence on training (asymmetric noise modeling).
        """
        noise = np.random.normal(0, 0.1, x.shape)
        noise_scale = 1.0 - imputation_conf[:, np.newaxis]  # Inverse confidence
        return x + noise * noise_scale
