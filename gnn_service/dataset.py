"""
PyTorch Dataset class for hydraulic graph data.
"""

import logging

import torch
from config import training_config
from torch.utils.data import Dataset
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)


class HydraulicGraphDataset(Dataset):
    """Dataset for hydraulic system graph data with multi-label classification."""

    def __init__(self, graphs: list | None = None, transform=None):
        """
        Initialize dataset.

        Args:
            graphs: List of PyG Data objects
            transform: Optional transform to apply to graphs
        """
        self.graphs = graphs or self._load_graphs()
        self.transform = transform
        logger.info(f"Initialized dataset with {len(self.graphs)} graphs")

    def _load_graphs(self) -> list:
        """Load graphs from saved file."""
        try:
            graphs = torch.load(
                training_config.graphs_save_path,
                weights_only=False,  # ✅ ДОБАВЬ ЭТО
            )
            logger.info(
                f"Loaded {len(graphs)} graphs from {training_config.graphs_save_path}"
            )
            return graphs
        except Exception as e:
            logger.error(f"Error loading graphs: {e}")
            raise

    def __len__(self) -> int:
        """Return number of graphs in dataset."""
        return len(self.graphs)

    def __getitem__(self, idx: int):
        """
        Get graph by index.

        Args:
            idx: Index of graph to retrieve

        Returns:
            PyG Data object
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self)}")

        graph = self.graphs[idx]

        if self.transform:
            graph = self.transform(graph)

        return graph

    def get_data_loader(
        self, batch_size: int = None, shuffle: bool = True, num_workers: int = 0
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
        )


def split_dataset(
    dataset: HydraulicGraphDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple:
    """
    Split dataset into train/validation/test sets.

    Args:
        dataset: Complete dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split indices
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create subsets
    train_graphs = [dataset.graphs[i] for i in train_indices]
    val_graphs = [dataset.graphs[i] for i in val_indices]
    test_graphs = [dataset.graphs[i] for i in test_indices]

    train_dataset = HydraulicGraphDataset(train_graphs)
    val_dataset = HydraulicGraphDataset(val_graphs)
    test_dataset = HydraulicGraphDataset(test_graphs)

    logger.info(
        f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Example usage
    dataset = HydraulicGraphDataset()
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample graph: {dataset[0]}")
