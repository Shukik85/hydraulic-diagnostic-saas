"""Imputation engine for missing sensor data.

Provides:
- K-NN based imputation
- Confidence scoring
- Asymmetric noise modeling
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class ImputationEngine:
    """Imputation engine for missing sensor data.

    Examples:
        >>> engine = ImputationEngine(confidence_threshold=0.5)
        >>> x_imputed, confidence = engine.impute_missing_features(
        ...     x=features,
        ...     mask_nodes=observed_mask,
        ...     edge_index=edges,
        ...     edge_weights=weights,
        ... )
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        propagation_hops: int = 2,
    ):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")

        self.confidence_threshold = confidence_threshold
        self.propagation_hops = propagation_hops

    def impute_missing_features(
        self,
        x: np.ndarray,
        mask_nodes: np.ndarray,
        edge_index: np.ndarray,
        edge_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Impute missing features via neighbor propagation.

        Args:
            x: Node features [N, F] (missing = 0)
            mask_nodes: Boolean mask [N] (True=observed, False=missing)
            edge_index: Edge indices [2, E]
            edge_weights: Edge weights [E]

        Returns:
            x_imputed: Imputed features [N, F]
            confidence: Confidence scores [N] (0-1)
        """
        n_nodes = x.shape[0]
        x_imputed = x.copy()
        confidence = np.ones(n_nodes)

        # Identify missing nodes
        missing_nodes = np.where(~mask_nodes)[0]

        if len(missing_nodes) == 0:
            return x_imputed, confidence

        # Build adjacency dict
        adjacency = self._build_adjacency_dict(edge_index, edge_weights, n_nodes)

        # Impute each missing node
        for node_idx in missing_nodes:
            # Find neighbors via BFS
            neighbors = self._find_neighbors_bfs(
                node_idx, adjacency, self.propagation_hops, mask_nodes
            )

            if not neighbors:
                # No neighbors - use global mean
                observed_features = x[mask_nodes]
                if len(observed_features) > 0:
                    x_imputed[node_idx] = observed_features.mean(axis=0)
                confidence[node_idx] = 0.1  # Low confidence
            else:
                # Weighted average of neighbors
                neighbor_features = []
                neighbor_weights = []

                for neighbor_idx, distance in neighbors:
                    neighbor_features.append(x[neighbor_idx])
                    # Weight decays with distance
                    weight = edge_weights[0] / (distance + 1)
                    neighbor_weights.append(weight)

                neighbor_features = np.array(neighbor_features)
                neighbor_weights = np.array(neighbor_weights)
                neighbor_weights /= neighbor_weights.sum()

                # Weighted average
                x_imputed[node_idx] = (neighbor_features.T @ neighbor_weights).T

                # Confidence based on number and distance of neighbors
                avg_distance = np.mean([d for _, d in neighbors])
                confidence[node_idx] = max(0.3, 1.0 - (avg_distance / self.propagation_hops))

        return x_imputed, confidence

    def _build_adjacency_dict(
        self, edge_index: np.ndarray, edge_weights: np.ndarray, n_nodes: int
    ) -> dict[int, list[tuple[int, float]]]:
        """Build adjacency dict with weights (vectorized)."""
        adjacency = {i: [] for i in range(n_nodes)}
        for (src, dst), weight in zip(edge_index.T, edge_weights, strict=False):
            adjacency[src].append((dst, weight))
        return adjacency

    def _find_neighbors_bfs(
        self,
        node_idx: int,
        adjacency: dict[int, list[tuple[int, float]]],
        max_hops: int,
        mask_nodes: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Find observed neighbors via BFS.

        Returns:
            List of (neighbor_idx, distance) tuples
        """
        visited = set()
        queue: deque[tuple[int, int]] = deque([(node_idx, 0)])
        neighbors = []

        while queue:
            current_node, distance = queue.popleft()

            if current_node in visited:
                continue

            visited.add(current_node)

            # Check if observed
            if current_node != node_idx and mask_nodes[current_node]:
                neighbors.append((current_node, distance))

            # Explore neighbors
            if distance < max_hops:
                for neighbor, _ in adjacency.get(current_node, []):
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))

        return neighbors

    def apply_asymmetric_noise_model(
        self, x: np.ndarray, confidence: np.ndarray, noise_scale: float = 0.1
    ) -> np.ndarray:
        """Apply asymmetric noise based on confidence.

        Low confidence nodes get more noise.

        Args:
            x: Features [N, F]
            confidence: Confidence scores [N]
            noise_scale: Base noise scale

        Returns:
            x_noisy: Features with asymmetric noise
        """
        assert 0 <= noise_scale <= 1, f"noise_scale must be in [0, 1], got {noise_scale}"

        # Noise inversely proportional to confidence
        noise_multiplier = 1.0 - confidence
        noise = np.random.randn(*x.shape) * noise_scale
        noise = noise * noise_multiplier[:, np.newaxis]

        x_noisy = x + noise
        return x_noisy
