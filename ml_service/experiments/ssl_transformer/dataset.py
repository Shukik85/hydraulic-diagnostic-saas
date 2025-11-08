"""
Component Dataset Loader
PyTorch Dataset for component-level training
"""
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class ComponentDataset(Dataset):
    """
    Dataset for component-level models (cylinder or pump)
    """
    
    def __init__(self, csv_path: str, component_type: str = 'cylinder'):
        """
        Args:
            csv_path: Path to component CSV
            component_type: 'cylinder' or 'pump'
        """
        self.df = pd.read_csv(csv_path)
        self.component_type = component_type
        
        # Define feature columns
        if component_type == 'cylinder':
            self.feature_cols = [
                'pressure_extend', 'pressure_retract', 
                'position', 'velocity', 'force',
                'pressure_diff', 'load_ratio'
            ]
        elif component_type == 'pump':
            self.feature_cols = [
                'pressure_outlet', 'speed_rpm', 
                'temperature', 'vibration', 'power'
            ]
        
        # Target column
        self.target_col = 'fault_any'
        
        # Extract features and labels
        self.X = self.df[self.feature_cols].values.astype('float32')
        self.y = self.df[self.target_col].values.astype('int64')
        
        print(f"✅ Loaded {len(self)} samples")
        print(f"   Features: {self.X.shape}")
        print(f"   Classes: {self.y.sum()} faults / {len(self) - self.y.sum()} normal")
        print(f"   Fault rate: {self.y.mean()*100:.2f}%")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def get_dataloaders(
    csv_path: str,
    component_type: str = 'cylinder',
    batch_size: int = 64,
    train_split: float = 0.7,
    val_split: float = 0.15,
    num_workers: int = 4
):
    """
    Create train/val/test dataloaders with stratified split
    """
    # Load full dataset
    df = pd.read_csv(csv_path)
    
    # Stratified split
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_split, 
        stratify=df['fault_any'],
        random_state=42
    )
    
    val_size = val_split / (1 - train_split)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        stratify=temp_df['fault_any'],
        random_state=42
    )
    
    # Save splits
    data_dir = Path(csv_path).parent
    train_df.to_csv(data_dir / f'{component_type}_train.csv', index=False)
    val_df.to_csv(data_dir / f'{component_type}_val.csv', index=False)
    test_df.to_csv(data_dir / f'{component_type}_test.csv', index=False)
    
    print("\n📊 Dataset splits:")
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val:   {len(val_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")
    
    # Create datasets
    train_dataset = ComponentDataset(data_dir / f'{component_type}_train.csv', component_type)
    val_dataset = ComponentDataset(data_dir / f'{component_type}_val.csv', component_type)
    test_dataset = ComponentDataset(data_dir / f'{component_type}_test.csv', component_type)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test cylinder loader
    print("Testing CylinderDataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        'data/component_cylinder.csv',
        component_type='cylinder',
        batch_size=64
    )
    
    # Test one batch
    x, y = next(iter(train_loader))
    print("\n✅ Batch shapes:")
    print(f"   X: {x.shape}")
    print(f"   y: {y.shape}")
