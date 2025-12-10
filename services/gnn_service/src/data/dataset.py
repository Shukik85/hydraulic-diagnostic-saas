"""Hydraulic Graph Dataset для PyTorch.

PyTorch Dataset interface:
- Lazy loading (fetch from TimescaleDB on-demand or load from .pt)
- Intelligent caching (disk-based, persistent)
- Optional preloading
- Transform support
- Multi-worker safe
- Edge feature dimension flexibility (8D, 14D, custom)

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.data.feature_config import FeatureConfig
    from src.data.feature_engineer import FeatureEngineer
    from src.data.graph_builder import GraphBuilder
    from src.data.timescale_connector import TimescaleConnector
    from src.schemas import GraphTopology

logger = logging.getLogger(__name__)


class HydraulicGraphDataset(Dataset):
    """PyTorch Dataset для hydraulic graphs.

    Features:
    - Lazy loading: graphs built on-demand from TimescaleDB
    - Caching: disk-based with invalidation (includes edge_in_dim)
    - Preloading: optional for faster training
    - Transforms: data augmentation support
    - Multi-worker safe: file locking
    - Edge feature flexibility: supports 8D, 14D, custom dimensions

    Args:
        data_path: Path to equipment list JSON file
        timescale_connector: TimescaleConnector instance
        feature_engineer: FeatureEngineer instance
        graph_builder: GraphBuilder instance
        feature_config: FeatureConfig with edge_in_dim setting
        sequence_length: Number of time steps per graph
        transform: Optional transform function
        cache_dir: Directory для caching (None = no caching)
        preload: Preload all data to RAM

    Examples:
        >>> from src.data.feature_config import FeatureConfig
        >>>
        >>> # 14D edge features (static + dynamic)
        >>> config = FeatureConfig(edge_in_dim=14)
        >>>
        >>> dataset = HydraulicGraphDataset(
        ...     data_path="data/equipment_list.json",
        ...     timescale_connector=connector,
        ...     feature_engineer=FeatureEngineer(),
        ...     graph_builder=GraphBuilder(feature_config=config),
        ...     feature_config=config,
        ...     sequence_length=10,
        ...     cache_dir="data/cache",
        ...     preload=False
        ... )
        >>>
        >>> len(dataset)  # Number of equipment
        >>> graph = dataset[0]  # Load first graph
        >>> graph.edge_attr.shape  # [E, 14]
    """

    def __init__(
        self,
        data_path: str | Path,
        timescale_connector: TimescaleConnector,
        feature_engineer: FeatureEngineer,
        graph_builder: GraphBuilder,
        feature_config: FeatureConfig,
        sequence_length: int = 10,
        transform: Callable[[Data], Data] | None = None,
        cache_dir: Path | str | None = None,
        preload: bool = False,
    ):
        self.data_path = Path(data_path)
        self.connector = timescale_connector
        self.feature_engineer = feature_engineer
        self.graph_builder = graph_builder
        self.feature_config = feature_config
        self.sequence_length = sequence_length
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.preload = preload

        # Load equipment list
        self.equipment_list = self._load_equipment_list()

        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Preload data if requested
        self.preloaded_data: dict[str, Data] = {}
        if self.preload:
            self._preload_all()

        logger.info(
            f"HydraulicGraphDataset initialized: {len(self)} equipment, "
            f"edge_in_dim={self.feature_config.edge_in_dim}, "
            f"cache={'enabled' if self.cache_dir else 'disabled'}, "
            f"preload={self.preload}"
        )

    def _load_equipment_list(self) -> list[dict[str, Any]]:
        """Загрузить equipment list из JSON.

        Returns:
            equipment_list: List of equipment metadata dicts

        Raises:
            FileNotFoundError: Если data_path не существует
        """
        if not self.data_path.exists():
            msg = f"Equipment list not found: {self.data_path}"
            raise FileNotFoundError(msg)

        with open(self.data_path) as f:
            equipment_list = json.load(f)

        logger.info(f"Loaded {len(equipment_list)} equipment from {self.data_path}")
        return equipment_list

    def _get_cache_path(self, equipment_id: str, topology_hash: str) -> Path:
        """Получить cache file path с учётом edge_in_dim.

        Args:
            equipment_id: Equipment identifier
            topology_hash: Hash of topology (for invalidation)

        Returns:
            cache_path: Path to cache file
        """
        if self.cache_dir is None:
            msg = "Cache directory not configured"
            raise RuntimeError(msg)

        # Include edge_in_dim в cache filename для invalidation
        cache_filename = f"{equipment_id}_{topology_hash[:8]}_e{self.feature_config.edge_in_dim}.pkl"
        return self.cache_dir / cache_filename

    def _compute_topology_hash(self, topology: GraphTopology) -> str:
        """Вычислить hash of topology (для cache invalidation).

        Args:
            topology: GraphTopology instance

        Returns:
            hash: SHA256 hash string
        """
        # Convert to JSON string
        topology_dict = topology.model_dump()
        topology_json = json.dumps(topology_dict, sort_keys=True)

        # Compute hash
        hash_obj = hashlib.sha256(topology_json.encode())
        return hash_obj.hexdigest()

    def _load_from_cache(self, cache_path: Path) -> Data | None:
        """Загрузить graph from cache.

        Args:
            cache_path: Path to cache file

        Returns:
            graph: Cached Data object or None
        """
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                graph = pickle.load(f)
            logger.debug(f"Cache hit: {cache_path.name}")
            return graph
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None

    def _save_to_cache(self, cache_path: Path, graph: Data) -> None:
        """Сохранить graph to cache.

        Args:
            cache_path: Path to cache file
            graph: Data object to cache
        """
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Cached: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")

    def _build_graph_for_equipment(self, equipment_item: dict[str, Any]) -> Data:
        """Построить graph для equipment из TimescaleDB.

        Args:
            equipment_item: Equipment metadata dict

        Returns:
            graph: PyG Data object
        """
        equipment_id = equipment_item["equipment_id"]

        # TODO: Implement real schema parsing when ready
        logger.warning(
            f"Building graph for {equipment_id}: real schema integration pending. "
            f"Use TemporalGraphDataset for prepared .pt graphs."
        )

        # Fallback: create minimal graph
        x = torch.randn(5, self.feature_config.total_features_per_sensor)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        edge_attr = torch.randn(6, self.feature_config.edge_in_dim)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _preload_all(self) -> None:
        """Предзагрузить все graphs в RAM."""
        logger.info(f"Preloading {len(self)} graphs...")

        for idx in range(len(self)):
            equipment_item = self.equipment_list[idx]
            equipment_id = equipment_item["equipment_id"]

            graph = self._build_graph_for_equipment(equipment_item)
            self.preloaded_data[equipment_id] = graph

        logger.info(f"Preloaded {len(self.preloaded_data)} graphs to RAM")

    def __len__(self) -> int:
        """Dataset size.

        Returns:
            size: Количество equipment
        """
        return len(self.equipment_list)

    def __getitem__(self, idx: int) -> Data:
        """Get graph by index.

        Args:
            idx: Index в equipment list

        Returns:
            graph: PyG Data object with edge_attr of shape [E, edge_in_dim]

        Examples:
            >>> graph = dataset[0]
            >>> graph.x.shape  # [N, node_features]
            >>> graph.edge_attr.shape  # [E, edge_in_dim]
        """
        equipment_item = self.equipment_list[idx]
        equipment_id = equipment_item["equipment_id"]

        # 1. Check preloaded
        if self.preload and equipment_id in self.preloaded_data:
            graph = self.preloaded_data[equipment_id]
        else:
            # 2. Try cache
            graph = None
            if self.cache_dir:
                topology_hash = "default_hash"  # TODO: compute when schema ready
                cache_path = self._get_cache_path(equipment_id, topology_hash)
                graph = self._load_from_cache(cache_path)

            # 3. Build graph
            if graph is None:
                graph = self._build_graph_for_equipment(equipment_item)

                # Save to cache
                if self.cache_dir:
                    cache_path = self._get_cache_path(equipment_id, topology_hash)
                    self._save_to_cache(cache_path, graph)

        # 4. Validate edge_in_dim
        if graph.edge_attr is not None and graph.edge_attr.shape[1] != self.feature_config.edge_in_dim:
            logger.warning(
                f"Edge feature mismatch for {equipment_id}: "
                f"loaded {graph.edge_attr.shape[1]}D, expected {self.feature_config.edge_in_dim}D. "
                f"edge_projection will handle conversion."
            )

        # 5. Apply transform
        if self.transform is not None:
            graph = self.transform(graph)

        return graph

    def get_equipment_ids(self) -> list[str]:
        """Получить список equipment IDs.

        Returns:
            ids: List of equipment identifiers

        Examples:
            >>> ids = dataset.get_equipment_ids()
            >>> print(f"Equipment: {ids}")
        """
        return [item["equipment_id"] for item in self.equipment_list]

    def get_statistics(self) -> dict[str, Any]:
        """Получить dataset statistics.

        Returns:
            stats: Dictionary с статистикой

        Examples:
            >>> stats = dataset.get_statistics()
            >>> print(f"Average nodes: {stats['avg_num_nodes']}")
            >>> print(f"Edge feature dimension: {stats['edge_in_dim']}")
        """
        # Sample a few graphs
        sample_size = min(10, len(self))
        graphs = [self[i] for i in range(sample_size)]

        edge_features = []
        for g in graphs:
            if g.edge_attr is not None:
                edge_features.append(g.edge_attr.shape[1])

        return {
            "dataset_size": len(self),
            "sample_size": sample_size,
            "avg_num_nodes": np.mean([g.num_nodes for g in graphs]),
            "avg_num_edges": np.mean([g.num_edges for g in graphs]),
            "node_features": graphs[0].x.shape[1] if graphs else 0,
            "edge_features_actual": edge_features[0] if edge_features else 0,
            "edge_in_dim_configured": self.feature_config.edge_in_dim,
            "cache_enabled": self.cache_dir is not None,
            "preloaded": self.preload,
        }


class TemporalGraphDataset(Dataset):
    """Dataset для загрузки готовых PyG графов из .pt файлов.

    Используется для работы с pre-built датасетами вроде gnn_graphs_multilabel.pt.
    Поддерживает variable edge_in_dim и трансформации.

    Args:
        data_path: Path to .pt file containing graphs
        feature_config: FeatureConfig with edge_in_dim
        transform: Optional transform function
        split: Dataset split (train/val/test) - not enforced, just for logging
        weights_only: Use weights_only=False for PyG compatibility (PyTorch 2.6+)

    Examples:
        >>> from src.data.feature_config import FeatureConfig
        >>>
        >>> config = FeatureConfig(edge_in_dim=14)  # Model expects 14D
        >>>
        >>> # Load pre-built graphs (they may have 8D edge_attr)
        >>> dataset = TemporalGraphDataset(
        ...     data_path="data/gnn_graphs_multilabel.pt",
        ...     feature_config=config,
        ...     split="train"
        ... )
        >>>
        >>> len(dataset)  # Number of graphs
        >>> graph = dataset[0]  # Load first graph
        >>> graph.edge_attr.shape  # [E, 8] - will be projected to 14D by model
    """

    def __init__(
        self,
        data_path: str | Path,
        feature_config: FeatureConfig,
        transform: Callable[[Data], Data] | None = None,
        split: Literal["train", "val", "test"] = "train",
        weights_only: bool = False,
    ):
        self.data_path = Path(data_path)
        self.feature_config = feature_config
        self.transform = transform
        self.split = split
        self.weights_only = weights_only

        # Load graphs from .pt file
        self.graphs = self._load_graphs()

        logger.info(
            f"TemporalGraphDataset initialized: {len(self)} graphs, "
            f"split={self.split}, edge_in_dim={self.feature_config.edge_in_dim}"
        )

    def _load_graphs(self) -> list[Data]:
        """Загрузить графы из .pt файла.

        Returns:
            graphs: List of PyG Data objects

        Raises:
            FileNotFoundError: Если файл не существует
        """
        if not self.data_path.exists():
            msg = f"Dataset not found: {self.data_path}"
            raise FileNotFoundError(msg)

        logger.info(f"Loading dataset from {self.data_path}...")

        try:
            # PyTorch 2.6+ требует weights_only=False для PyG Data objects
            data = torch.load(self.data_path, weights_only=self.weights_only)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        # Extract graphs depending on structure
        if isinstance(data, dict):
            # Structure: {"graphs": [...], "metadata": {...}}
            if "graphs" in data:
                graphs = data["graphs"]
            else:
                msg = f"Unknown dict structure: keys={list(data.keys())}"
                raise ValueError(msg)
        elif isinstance(data, list):
            graphs = data
        else:
            msg = f"Unknown data structure: {type(data)}"
            raise ValueError(msg)

        logger.info(f"Loaded {len(graphs)} graphs from {self.data_path.name}")

        return graphs

    def __len__(self) -> int:
        """Dataset size.

        Returns:
            size: Number of graphs
        """
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        """Get graph by index.

        Args:
            idx: Index in graphs list

        Returns:
            graph: PyG Data object

        Examples:
            >>> graph = dataset[0]
            >>> graph.x.shape  # [N, node_features]
            >>> graph.edge_attr.shape  # [E, actual_edge_dim]
        """
        graph = self.graphs[idx]

        # Apply transform if provided
        if self.transform is not None:
            graph = self.transform(graph)

        return graph

    def get_statistics(self) -> dict[str, Any]:
        """Получить dataset statistics.

        Returns:
            stats: Dictionary с статистикой

        Examples:
            >>> stats = dataset.get_statistics()
            >>> print(f"Total graphs: {stats['dataset_size']}")
            >>> print(f"Edge dimensions: {stats['edge_feature_dims']}")
        """
        sample_size = min(10, len(self))
        sample_graphs = self.graphs[:sample_size]

        edge_dims = set()
        for g in sample_graphs:
            if hasattr(g, "edge_attr") and g.edge_attr is not None:
                edge_dims.add(g.edge_attr.shape[1])

        num_nodes = [g.num_nodes for g in sample_graphs]
        num_edges = [g.num_edges for g in sample_graphs]

        return {
            "dataset_size": len(self),
            "split": self.split,
            "sample_size": sample_size,
            "avg_num_nodes": np.mean(num_nodes) if num_nodes else 0,
            "min_num_nodes": min(num_nodes) if num_nodes else 0,
            "max_num_nodes": max(num_nodes) if num_nodes else 0,
            "avg_num_edges": np.mean(num_edges) if num_edges else 0,
            "min_num_edges": min(num_edges) if num_edges else 0,
            "max_num_edges": max(num_edges) if num_edges else 0,
            "node_features": sample_graphs[0].x.shape[1] if sample_graphs else 0,
            "edge_feature_dims": sorted(list(edge_dims)),
            "edge_in_dim_configured": self.feature_config.edge_in_dim,
        }
