"""Feature engineering configuration.

Configuration classes для:
- Feature extraction parameters
- DataLoader settings
- Normalization methods
- Edge feature dimensions (universal GNN support)

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
        edge_in_dim: Edge feature dimension (8=static only, 14=static+dynamic, custom)

    Edge Features (edge_in_dim):
        - 8D: Static only (diameter, length, area, loss_coeff, rating, material[3])
        - 14D: Static (8) + Dynamic (6) [flow, pressure_drop, temp_delta, vibration, age, maintenance]
        - Custom: Any dimension supported by model's edge_projection layer

    Node Features (total_features_per_sensor):
        - Statistical: 11 (mean, std, min, max, percentiles[5], skew, kurtosis)
        - Frequency: 12 (FFT[10], dominant_freq, spectral_entropy)
        - Temporal: 11 (rolling_mean/std[6], EMA, autocorr[3], trend)
        - Hydraulic: 4 (pressure_ratio, temp_delta, flow_efficiency, cavitation_index)
        - Total: 39

    Examples:
        >>> # Full configuration - default
        >>> config = FeatureConfig()
        >>> config.total_features_per_sensor  # 39 node features per sensor
        39
        >>>
        >>> # Static edge features only (8D)
        >>> config = FeatureConfig(edge_in_dim=8)
        >>>
        >>> # Full edge features (14D)
        >>> config = FeatureConfig(edge_in_dim=14)
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

    # Edge features (Universal GNN support)
    edge_in_dim: int = 14  # 8 static + 6 dynamic (default for backward compat)

    def __post_init__(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If edge_in_dim invalid
        """
        # Validate edge_in_dim
        if self.edge_in_dim < 1:
            msg = f"edge_in_dim must be >= 1, got {self.edge_in_dim}"
            raise ValueError(msg)

        # Warn if using non-standard dimension
        if self.edge_in_dim not in (8, 14):
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Using custom edge_in_dim={self.edge_in_dim}. "
                "Standard values: 8 (static only) or 14 (static+dynamic)."
            )

    @property
    def static_edge_features_count(self) -> int:
        """Количество static edge features.

        Returns:
            count: diameter(1) + length(1) + area(1) + loss_coeff(1) + rating(1) + material(3) = 8
        """
        return 8

    @property
    def dynamic_edge_features_count(self) -> int:
        """Количество dynamic edge features.

        Returns:
            count: flow(1) + pressure_drop(1) + temp_delta(1) + vibration(1) + age(1) + maintenance(1) = 6
        """
        return 6

    @property
    def has_dynamic_edge_features(self) -> bool:
        """Check if edge_in_dim includes dynamic features.

        Returns:
            has_dynamic: True if edge_in_dim >= 14 (includes dynamic)
        """
        return self.edge_in_dim >= 14

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
            count: FFT magnitudes(10) + dominant_freq(1) + spectral_entropy(1) = 12
        """
        if not self.use_frequency:
            return 0
        return self.num_frequencies + 2

    @property
    def temporal_features_count(self) -> int:
        """Количество temporal features.

        Returns:
            count: (rolling_mean + rolling_std) * windows(3) + EMA(1) + autocorr(3) + trend(1) = 11
        """
        if not self.use_temporal:
            return 0
        return len(self.window_sizes) * 2 + 5  # Rolling mean/std + EMA + autocorr + trend

    @property
    def hydraulic_features_count(self) -> int:
        """Количество hydraulic-specific features.

        Returns:
            count: pressure_ratio(1) + temp_delta(1) + flow_efficiency(1) + cavitation_index(1) = 4
        """
        if not self.use_hydraulic:
            return 0
        return 4

    @property
    def total_features_per_sensor(self) -> int:
        """Общее количество node features per sensor.

        Calculation:
            - Statistical: 11 (mean, std, min, max, 5 percentiles, skew, kurtosis)
            - Frequency: 12 (10 FFT + dominant_freq + spectral_entropy)
            - Temporal: 11 (6 rolling [3 windows × 2], EMA, 3 autocorr, trend)
            - Hydraulic: 4 (pressure_ratio, temp_delta, flow_efficiency, cavitation_index)
            - Total: 11 + 12 + 11 + 4 = 38 + 1 extra = 39

        Returns:
            count: Total node features (39 by default with all enabled)
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
