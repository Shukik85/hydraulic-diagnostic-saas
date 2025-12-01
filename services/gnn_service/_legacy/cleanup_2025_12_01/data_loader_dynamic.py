# services/gnn_service/data_loader_dynamic.py
"""
DataLoader utilities (dynamic) for batch temporal graphs with arbitrary component counts.
"""
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Any
import logging
from dataset_dynamic import DynamicTemporalGraphDataset

logger = logging.getLogger(__name__)

def dynamic_graph_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate batch работает с dynamic component/features.
    """
    component_keys = batch[0]["component_features"].keys()
    B = len(batch)
    batch_features = {}
    for comp in component_keys:
        # [B, T, F_comp]
        comp_feats = [item["component_features"][comp] for item in batch]
        batch_features[comp] = torch.stack(comp_feats, dim=0)
    # Batch metadata
    meta = [item["metadata"] for item in batch]
    return {"component_features": batch_features, "metadata": meta}

def create_dynamic_dataloaders(
    csv_path: str,
    metadata_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    sequence_length: int = 5,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dynamic train/val/test dataloaders.
    """
    train_dataset = DynamicTemporalGraphDataset(
        csv_path, metadata_path, split="train", sequence_length=sequence_length, **kwargs)
    val_dataset = DynamicTemporalGraphDataset(
        csv_path, metadata_path, split="val", sequence_length=sequence_length, **kwargs)
    test_dataset = DynamicTemporalGraphDataset(
        csv_path, metadata_path, split="test", sequence_length=sequence_length, **kwargs)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=dynamic_graph_collate_fn, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=dynamic_graph_collate_fn, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=dynamic_graph_collate_fn, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader
