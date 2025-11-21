"""Feature engineering configuration.

Configuration classes для:
- Feature extraction parameters
- DataLoader settings
- Normalization methods

Python 3.14 Features:
    - Deferred annotations
    - dataclass with slots
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True, frozen=True)
class FeatureConfig:
    """Configuration для feature engineering.
    
    Attributes:
        use_statistical: Использовать statistical features (mean, std, etc.)
        percentiles: Percentiles для вычисления
        use_frequency: Использовать frequency domain features (FFT)
        num_frequencies: Количество frequency components
        use_temporal: Использовать temporal features (rolling windows)
        window_sizes: Размеры окон для rolling statistics
        use_hydraulic: Использовать hydraulic-specific features
        normalization: Метод нормализации (standardize, minmax, robust)
        handle_missing: Метод обработки пропусков (ffill, bfill, interpolate, drop)
        outlier_threshold: Threshold для outlier detection (в стандартных отклонениях)
    
    Examples:
        >>> config = FeatureConfig(
        ...     use_statistical=True,
        ...     percentiles=[5, 25, 50, 75, 95],
        ...     num_frequencies=10
        ... )
        >>> config.total_features_per_sensor
        42  # Statistical(8) + Frequency(10) + Temporal(20) + Hydraulic(4)
    """
    
    # Statistical features
    use_statistical: bool = True
    percentiles: list[int] = field(default_factory=lambda: [5, 25, 50, 75, 95])
    
    # Frequency domain features
    use_frequency: bool = True
    num_frequencies: int = 10
    
    # Temporal features
    use_temporal: bool = True
    window_sizes: list[int] = field(default_factory=lambda: [5, 10, 30])
    
    # Hydraulic-specific features
    use_hydraulic: bool = True
    
    # Preprocessing
    normalization: Literal["standardize", "minmax", "robust", "none"] = "standardize"
    handle_missing: Literal["ffill", "bfill", "interpolate", "drop"] = "ffill"
    outlier_threshold: float = 3.0  # Standard deviations
    
    @property
    def statistical_features_count(self) -> int:
        """Количество statistical features.
        
        Returns:
            count: mean(1) + std(1) + min(1) + max(1) + percentiles(5) + skew(1) + kurt(1) = 11
        """
        if not self.use_statistical:
            return 0
        return 11  # mean, std, min, max, 5 percentiles, skew, kurtosis
    
    @property
    def frequency_features_count(self) -> int:
        """Количество frequency features.
        
        Returns:
            count: FFT magnitudes + dominant freq + spectral entropy = num_frequencies + 2
        """
        if not self.use_frequency:
            return 0
        return self.num_frequencies + 2
    
    @property
    def temporal_features_count(self) -> int:
        """Количество temporal features.
        
        Returns:
            count: (rolling_mean + rolling_std) * windows + EMA + autocorr(3) + trend = 2*3 + 5
        """
        if not self.use_temporal:
            return 0
        return len(self.window_sizes) * 2 + 5  # Rolling mean/std + EMA + autocorr + trend
    
    @property
    def hydraulic_features_count(self) -> int:
        """Количество hydraulic-specific features.
        
        Returns:
            count: pressure_ratio + temp_delta + flow_efficiency + cavitation_index = 4
        """
        if not self.use_hydraulic:
            return 0
        return 4
    
    @property
    def total_features_per_sensor(self) -> int:
        """Общее количество features per sensor.
        
        Returns:
            count: Statistical + Frequency + Temporal + Hydraulic
        """
        return (
            self.statistical_features_count
            + self.frequency_features_count
            + self.temporal_features_count
            + self.hydraulic_features_count
        )


@dataclass(slots=True, frozen=True)
class DataLoaderConfig:
    """Configuration для PyTorch DataLoader.
    
    Attributes:
        batch_size: Размер batch (количество graphs)
        num_workers: Количество worker processes для data loading
        pin_memory: Использовать pinned memory для GPU transfer
        persistent_workers: Сохранять workers между epochs
        prefetch_factor: Количество batches для prefetch per worker
        shuffle_train: Shuffle training data
        shuffle_val: Shuffle validation data
        shuffle_test: Shuffle test data
        drop_last_train: Drop last incomplete batch в training
        drop_last_val: Drop last incomplete batch в validation
    
    Examples:
        >>> config = DataLoaderConfig(batch_size=32, num_workers=4)
        >>> train_loader = create_dataloader(dataset, config=config, split="train")
    """
    
    # Batch configuration
    batch_size: int = 32
    
    # Worker configuration
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Shuffle configuration
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False
    
    # Drop last batch
    drop_last_train: bool = False
    drop_last_val: bool = False
    drop_last_test: bool = False
    
    def get_loader_kwargs(self, split: Literal["train", "val", "test"]) -> dict:
        """Получить kwargs для DataLoader в зависимости от split.
        
        Args:
            split: Dataset split (train, val, test)
        
        Returns:
            kwargs: Dictionary с параметрами для DataLoader
        
        Examples:
            >>> config = DataLoaderConfig()
            >>> train_kwargs = config.get_loader_kwargs("train")
            >>> train_loader = DataLoader(dataset, **train_kwargs)
        """
        shuffle_map = {
            "train": self.shuffle_train,
            "val": self.shuffle_val,
            "test": self.shuffle_test,
        }
        
        drop_last_map = {
            "train": self.drop_last_train,
            "val": self.drop_last_val,
            "test": self.drop_last_test,
        }
        
        return {
            "batch_size": self.batch_size,
            "shuffle": shuffle_map[split],
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
            "prefetch_factor": self.prefetch_factor if self.num_workers > 0 else None,
            "drop_last": drop_last_map[split],
        }
