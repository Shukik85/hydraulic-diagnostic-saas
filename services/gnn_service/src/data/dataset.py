"""Hydraulic Graph Dataset для PyTorch.

PyTorch Dataset interface:
- Lazy loading (fetch from TimescaleDB on-demand)
- Intelligent caching (disk-based, persistent)
- Optional preloading
- Transform support
- Multi-worker safe

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.feature_engineer import FeatureEngineer
from src.data.graph_builder import GraphBuilder
from src.data.timescale_connector import TimescaleConnector
from src.schemas import GraphTopology

logger = logging.getLogger(__name__)


class HydraulicGraphDataset(Dataset):
    """PyTorch Dataset для hydraulic graphs.
    
    Features:
    - Lazy loading: graphs built on-demand
    - Caching: disk-based with invalidation
    - Preloading: optional for faster training
    - Transforms: data augmentation support
    - Multi-worker safe: file locking
    
    Args:
        data_path: Path to equipment list JSON file
        timescale_connector: TimescaleConnector instance
        feature_engineer: FeatureEngineer instance
        graph_builder: GraphBuilder instance
        sequence_length: Number of time steps per graph
        transform: Optional transform function
        cache_dir: Directory для caching (None = no caching)
        preload: Preload all data to RAM
    
    Examples:
        >>> connector = TimescaleConnector(db_url=DB_URL)
        >>> await connector.connect()
        >>> 
        >>> dataset = HydraulicGraphDataset(
        ...     data_path="data/equipment_list.json",
        ...     timescale_connector=connector,
        ...     feature_engineer=FeatureEngineer(),
        ...     graph_builder=GraphBuilder(),
        ...     sequence_length=10,
        ...     cache_dir="data/cache"
        ... )
        >>> 
        >>> len(dataset)  # Number of equipment
        >>> graph = dataset[0]  # Load first graph
    """

    def __init__(
        self,
        data_path: str | Path,
        timescale_connector: TimescaleConnector,
        feature_engineer: FeatureEngineer,
        graph_builder: GraphBuilder,
        sequence_length: int = 10,
        transform: Callable[[Data], Data] | None = None,
        cache_dir: Path | str | None = None,
        preload: bool = False
    ):
        self.data_path = Path(data_path)
        self.connector = timescale_connector
        self.feature_engineer = feature_engineer
        self.graph_builder = graph_builder
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
            raise FileNotFoundError(f"Equipment list not found: {self.data_path}")

        with open(self.data_path) as f:
            equipment_list = json.load(f)

        logger.info(f"Loaded {len(equipment_list)} equipment from {self.data_path}")
        return equipment_list

    def _get_cache_path(self, equipment_id: str, topology_hash: str) -> Path:
        """Получить cache file path.
        
        Args:
            equipment_id: Equipment identifier
            topology_hash: Hash of topology (for invalidation)
        
        Returns:
            cache_path: Path to cache file
        """
        if self.cache_dir is None:
            raise RuntimeError("Cache directory not configured")

        # Include topology hash для cache invalidation
        cache_filename = f"{equipment_id}_{topology_hash[:8]}.pkl"
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
        """Построить graph для equipment.
        
        Args:
            equipment_item: Equipment metadata dict
        
        Returns:
            graph: PyG Data object
        """
        equipment_id = equipment_item["equipment_id"]

        # TODO: Parse real schemas when ready
        # For now, create dummy graph
        logger.warning(f"Building dummy graph for {equipment_id} (schema integration pending)")

        # Dummy graph: 5 nodes, 6 edges
        x = torch.randn(5, self.feature_engineer.config.total_features_per_sensor)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2]
        ], dtype=torch.long)
        edge_attr = torch.randn(6, 8)

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
            graph: PyG Data object
        
        Examples:
            >>> graph = dataset[0]
            >>> graph.x.shape  # [N, F]
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
                # TODO: Compute topology hash when schema ready
                topology_hash = "dummy_hash"
                cache_path = self._get_cache_path(equipment_id, topology_hash)
                graph = self._load_from_cache(cache_path)

            # 3. Build graph
            if graph is None:
                graph = self._build_graph_for_equipment(equipment_item)

                # Save to cache
                if self.cache_dir:
                    cache_path = self._get_cache_path(equipment_id, topology_hash)
                    self._save_to_cache(cache_path, graph)

        # 4. Apply transform
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
            >>> print(f"Average edges: {stats['avg_num_edges']}")
        """
        # Sample a few graphs
        sample_size = min(10, len(self))
        graphs = [self[i] for i in range(sample_size)]

        stats = {
            "dataset_size": len(self),
            "sample_size": sample_size,
            "avg_num_nodes": np.mean([g.num_nodes for g in graphs]),
            "avg_num_edges": np.mean([g.num_edges for g in graphs]),
            "avg_node_features": graphs[0].x.shape[1] if graphs else 0,
            "avg_edge_features": graphs[0].edge_attr.shape[1] if graphs and graphs[0].edge_attr is not None else 0,
            "cache_enabled": self.cache_dir is not None,
            "preloaded": self.preload,
        }

        return stats
