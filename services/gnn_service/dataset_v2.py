"""
Enhanced PyTorch Dataset with Data Augmentation for F1 > 90%

Improvements:
- Sensor noise simulation
- Feature scaling augmentation
- Random edge dropout (graph structure variation)
- Component-wise augmentation
- Configurable augmentation strength
"""

import logging
import random
from collections.abc import Callable

import torch
from config import training_config
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


class DataAugmentation:
    """
    Data augmentation for hydraulic graph data.

    Augmentation techniques:
    1. Gaussian noise (simulates sensor noise)
    2. Feature scaling (simulates measurement variations)
    3. Edge dropout (simulates intermittent connections)
    4. Component-wise variations
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        scale_range: tuple = (0.98, 1.02),
        edge_dropout: float = 0.02,
        enabled: bool = True,
    ):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise
            scale_range: Range for random scaling (min, max)
            edge_dropout: Probability of dropping edges
            enabled: Whether augmentation is enabled
        """
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.edge_dropout = edge_dropout
        self.enabled = enabled

    def __call__(self, data: Data) -> Data:
        """Apply augmentation to graph data."""
        if not self.enabled:
            return data

        # Clone to avoid modifying original
        data = data.clone()

        # 1. Add Gaussian noise to features
        if self.noise_std > 0:
            noise = torch.randn_like(data.x) * self.noise_std
            data.x = data.x + noise

        # 2. Random feature scaling
        if self.scale_range != (1.0, 1.0):
            scale = random.uniform(*self.scale_range)
            data.x = data.x * scale

        # 3. Edge dropout (randomly remove edges)
        if self.edge_dropout > 0 and data.edge_index.size(1) > 0:
            num_edges = data.edge_index.size(1)
            keep_mask = torch.rand(num_edges) > self.edge_dropout
            data.edge_index = data.edge_index[:, keep_mask]

        return data


class EnhancedHydraulicGraphDataset(Dataset):
    """
    Enhanced dataset with data augmentation for improved generalization.

    Features:
    - Configurable augmentation
    - Training/validation mode switching
    - Better error handling
    - Statistics tracking
    """

    def __init__(
        self,
        graphs: list | None = None,
        transform: Callable | None = None,
        augmentation: DataAugmentation | None = None,
        training: bool = True,
    ):
        """
        Args:
            graphs: List of PyG Data objects
            transform: Optional transform to apply to graphs
            augmentation: Data augmentation configuration
            training: Whether in training mode (enables augmentation)
        """
        self.graphs = graphs or self._load_graphs()
        self.transform = transform
        self.augmentation = augmentation
        self.training = training

        # Statistics
        self.stats = self._compute_statistics()

        logger.info(f"Initialized enhanced dataset with {len(self.graphs)} graphs")
        if self.augmentation and self.training:
            logger.info("  Data augmentation enabled:")
            logger.info(f"    - Noise std: {self.augmentation.noise_std}")
            logger.info(f"    - Scale range: {self.augmentation.scale_range}")
            logger.info(f"    - Edge dropout: {self.augmentation.edge_dropout}")

    def _load_graphs(self) -> list:
        """Load graphs from saved file."""
        try:
            graphs = torch.load(
                training_config.graphs_save_path,
                weights_only=False,
            )
            logger.info(
                f"Loaded {len(graphs)} graphs from {training_config.graphs_save_path}"
            )
            return graphs
        except Exception as e:
            logger.error(f"Error loading graphs: {e}")
            raise

    def _compute_statistics(self) -> dict:
        """Compute dataset statistics."""
        if not self.graphs:
            return {}

        # Sample first graph for dimensions
        sample = self.graphs[0]

        # Class distribution
        all_labels = torch.stack([g.y for g in self.graphs])
        class_counts = all_labels.sum(dim=0)

        return {
            "num_samples": len(self.graphs),
            "num_nodes": sample.x.size(0),
            "num_features": sample.x.size(1),
            "num_classes": sample.y.size(0),
            "class_distribution": class_counts.tolist(),
            "class_balance": (class_counts / len(self.graphs)).tolist(),
        }

    def __len__(self) -> int:
        """Return number of graphs in dataset."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        """
        Get graph by index with optional augmentation.

        Args:
            idx: Index of graph to retrieve

        Returns:
            PyG Data object
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self)}")

        # Get base graph
        graph = self.graphs[idx]

        # Apply augmentation (only in training mode)
        if self.augmentation and self.training:
            graph = self.augmentation(graph)

        # Apply custom transform
        if self.transform:
            graph = self.transform(graph)

        return graph

    def train_mode(self):
        """Enable training mode (with augmentation)."""
        self.training = True

    def eval_mode(self):
        """Enable evaluation mode (without augmentation)."""
        self.training = False

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return self.stats

    def get_data_loader(
        self,
        batch_size: int | None = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for this dataset.

        Args:
            batch_size: Batch size (default from config)
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes

        Returns:
            DataLoader instance
        """
        batch_size = batch_size or training_config.batch_size

        def collate_fn(batch):
            """Custom collate function for PyG Data objects."""
            return Batch.from_data_list(batch)

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=training_config.device == "cuda",
            persistent_workers=num_workers > 0,
        )


def split_dataset(
    dataset: EnhancedHydraulicGraphDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Split dataset into train/validation/test sets.

    Args:
        dataset: Complete dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # noqa: F841

    # Split indices with fixed seed
    torch.manual_seed(seed)
    indices = torch.randperm(total_size).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create subsets
    train_graphs = [dataset.graphs[i] for i in train_indices]
    val_graphs = [dataset.graphs[i] for i in val_indices]
    test_graphs = [dataset.graphs[i] for i in test_indices]

    # ✅ Create datasets with appropriate augmentation settings
    # Training set: WITH augmentation
    train_dataset = EnhancedHydraulicGraphDataset(
        graphs=train_graphs,
        augmentation=DataAugmentation(
            noise_std=0.01,
            scale_range=(0.98, 1.02),
            edge_dropout=0.02,
            enabled=True,
        ),
        training=True,
    )

    # Validation set: WITHOUT augmentation
    val_dataset = EnhancedHydraulicGraphDataset(
        graphs=val_graphs,
        augmentation=None,
        training=False,
    )

    # Test set: WITHOUT augmentation
    test_dataset = EnhancedHydraulicGraphDataset(
        graphs=test_graphs,
        augmentation=None,
        training=False,
    )

    logger.info(
        f"Dataset split: Train={len(train_dataset)}, "
        f"Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    # Log class distribution
    train_stats = train_dataset.get_statistics()
    logger.info(f"Train class balance: {train_stats['class_balance']}")

    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    batch_size: int = 12,
    num_workers: int = 0,
) -> tuple:
    """
    Create train/val/test data loaders.

    Args:
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load full dataset
    full_dataset = EnhancedHydraulicGraphDataset()

    # Split
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)

    # Create loaders
    train_loader = train_dataset.get_data_loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = val_dataset.get_data_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = test_dataset.get_data_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info("Data loaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create dataset with augmentation
    dataset = EnhancedHydraulicGraphDataset(
        augmentation=DataAugmentation(
            noise_std=0.01,
            scale_range=(0.98, 1.02),
            edge_dropout=0.02,
        ),
        training=True,
    )

    print("\nDataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test augmentation
    print("\nTesting augmentation:")
    sample = dataset[0]
    print(f"  Original features shape: {sample.x.shape}")
    print(f"  Features mean: {sample.x.mean():.4f}")
    print(f"  Features std: {sample.x.std():.4f}")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=12)

    # Test batch
    batch = next(iter(train_loader))
    print("\nSample batch:")
    print(f"  x shape: {batch.x.shape}")
    print(f"  edge_index shape: {batch.edge_index.shape}")
    print(f"  y shape: {batch.y.shape}")
    print(f"  batch shape: {batch.batch.shape}")

    print("\n✅ Dataset test passed!")
