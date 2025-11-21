"""DataLoader factory для hydraulic graphs.

PyG-compatible DataLoader:
- Custom collate для variable-size graphs
- Factory functions для train/val/test splits
- Optimal defaults для production

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data, Batch
import logging
from typing import Literal

from src.data.dataset import HydraulicGraphDataset
from src.data.feature_config import DataLoaderConfig

logger = logging.getLogger(__name__)


def hydraulic_collate_fn(batch: list[Data]) -> Batch:
    """Collate function для PyG graphs.
    
    Combines multiple Data objects into a single Batch:
    - Stacks node features
    - Concatenates edge_index with offset
    - Stacks edge attributes
    - Creates batch assignment vector
    
    Args:
        batch: List of Data objects
    
    Returns:
        batch: PyG Batch object
    
    Examples:
        >>> data1 = Data(x=torch.randn(3, 10), edge_index=...)
        >>> data2 = Data(x=torch.randn(5, 10), edge_index=...)
        >>> batch = hydraulic_collate_fn([data1, data2])
        >>> 
        >>> batch.x.shape  # [8, 10] - combined nodes
        >>> batch.num_graphs  # 2
    """
    return Batch.from_data_list(batch)


def create_dataloader(
    dataset: HydraulicGraphDataset,
    config: DataLoaderConfig | None = None,
    split: Literal["train", "val", "test"] = "train",
    **kwargs
) -> DataLoader:
    """Create DataLoader для hydraulic graphs.
    
    Args:
        dataset: HydraulicGraphDataset instance
        config: DataLoaderConfig (uses defaults if None)
        split: Dataset split (affects shuffle/drop_last)
        **kwargs: Override config parameters
    
    Returns:
        loader: PyTorch DataLoader
    
    Examples:
        >>> dataset = HydraulicGraphDataset(...)
        >>> 
        >>> # Training loader
        >>> train_loader = create_dataloader(
        ...     dataset,
        ...     split="train",
        ...     batch_size=32
        ... )
        >>> 
        >>> # Validation loader
        >>> val_loader = create_dataloader(
        ...     dataset,
        ...     split="val",
        ...     batch_size=64
        ... )
        >>> 
        >>> # Iterate
        >>> for batch in train_loader:
        ...     health, degradation, anomaly = model(
        ...         x=batch.x,
        ...         edge_index=batch.edge_index,
        ...         edge_attr=batch.edge_attr,
        ...         batch=batch.batch
        ...     )
    """
    if config is None:
        config = DataLoaderConfig()
    
    # Get base kwargs from config
    loader_kwargs = config.get_loader_kwargs(split)
    
    # Override with provided kwargs
    loader_kwargs.update(kwargs)
    
    # Add collate function
    loader_kwargs["collate_fn"] = hydraulic_collate_fn
    
    # Create DataLoader
    loader = DataLoader(dataset, **loader_kwargs)
    
    logger.info(
        f"Created {split} DataLoader: batch_size={loader_kwargs['batch_size']}, "
        f"num_workers={loader_kwargs['num_workers']}, shuffle={loader_kwargs['shuffle']}"
    )
    
    return loader


def create_train_val_loaders(
    dataset: HydraulicGraphDataset,
    config: DataLoaderConfig | None = None,
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.
    
    Args:
        dataset: HydraulicGraphDataset instance
        config: DataLoaderConfig
        train_ratio: Fraction для training (0-1)
        seed: Random seed для reproducibility
    
    Returns:
        loaders: (train_loader, val_loader)
    
    Examples:
        >>> dataset = HydraulicGraphDataset(...)
        >>> train_loader, val_loader = create_train_val_loaders(
        ...     dataset,
        ...     train_ratio=0.8
        ... )
        >>> 
        >>> len(train_loader.dataset)  # 80% of data
        >>> len(val_loader.dataset)   # 20% of data
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    
    # Split dataset
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )
    
    logger.info(f"Split dataset: train={train_size}, val={val_size}")
    
    # Create loaders
    train_loader = create_dataloader(train_dataset, config, split="train")
    val_loader = create_dataloader(val_dataset, config, split="val")
    
    return train_loader, val_loader


def create_train_val_test_loaders(
    dataset: HydraulicGraphDataset,
    config: DataLoaderConfig | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.
    
    Args:
        dataset: HydraulicGraphDataset instance
        config: DataLoaderConfig
        train_ratio: Fraction для training
        val_ratio: Fraction для validation (test = 1 - train - val)
        seed: Random seed
    
    Returns:
        loaders: (train_loader, val_loader, test_loader)
    
    Examples:
        >>> train_loader, val_loader, test_loader = create_train_val_test_loaders(
        ...     dataset,
        ...     train_ratio=0.7,
        ...     val_ratio=0.15  # test = 0.15
        ... )
    """
    if not 0 < train_ratio + val_ratio < 1:
        raise ValueError(f"train_ratio + val_ratio must be < 1, got {train_ratio + val_ratio}")
    
    # Split dataset
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    logger.info(f"Split dataset: train={train_size}, val={val_size}, test={test_size}")
    
    # Create loaders
    train_loader = create_dataloader(train_dataset, config, split="train")
    val_loader = create_dataloader(val_dataset, config, split="val")
    test_loader = create_dataloader(test_dataset, config, split="test")
    
    return train_loader, val_loader, test_loader
