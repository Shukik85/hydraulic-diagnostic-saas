"""Tests for EdgeFeatureNormalizer module.

Tests mixed normalization strategy:
- Log + z-score for flow
- Z-score for pressure/temperature
- Min-max for vibration/age
- Pass-through for maintenance score

Author: GNN Service Team
Python: 3.14+
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.normalization import (
    EdgeFeatureNormalizer,
    NormalizationStatistics,
    create_edge_feature_normalizer,
)


class TestNormalizationStatistics:
    """Test NormalizationStatistics model."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = NormalizationStatistics()

        # Check defaults
        assert stats.flow_log_mean == 4.0
        assert stats.flow_log_std == 1.0
        assert stats.pressure_drop_mean == 2.0
        assert stats.pressure_drop_std == 1.0
        assert stats.temp_delta_mean == 5.0
        assert stats.temp_delta_std == 3.0
        assert stats.vibration_max == 10.0
        assert stats.age_max == 100000.0

    def test_custom_values(self):
        """Test custom statistics values."""
        stats = NormalizationStatistics(
            flow_log_mean=4.5,
            flow_log_std=1.2,
            pressure_drop_mean=2.5,
            pressure_drop_std=1.5
        )

        assert stats.flow_log_mean == 4.5
        assert stats.flow_log_std == 1.2
        assert stats.pressure_drop_mean == 2.5
        assert stats.pressure_drop_std == 1.5


class TestFlowNormalization:
    """Test flow rate normalization (log + z-score)."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer with known statistics."""
        stats = NormalizationStatistics(
            flow_log_mean=4.0,
            flow_log_std=1.0
        )
        return EdgeFeatureNormalizer(stats=stats)

    def test_normalize_flow_typical(self, normalizer):
        """Test flow normalization with typical value."""
        flow_lpm = 100.0

        flow_norm = normalizer.normalize_flow(flow_lpm)

        # Should be normalized (not extreme)
        assert -5 < flow_norm < 5

    def test_normalize_flow_zero(self, normalizer):
        """Test flow normalization with zero flow."""
        flow_norm = normalizer.normalize_flow(0.0)

        # log(1 + 0) = 0, then (0 - 4.0) / 1.0 = -4.0
        assert abs(flow_norm - (-4.0)) < 0.01

    def test_normalize_flow_high(self, normalizer):
        """Test flow normalization with high flow."""
        flow_lpm = 500.0

        flow_norm = normalizer.normalize_flow(flow_lpm)

        # Should be clipped to [-5, 5]
        assert -5 <= flow_norm <= 5

    def test_denormalize_flow(self, normalizer):
        """Test flow denormalization (inverse transform)."""
        flow_original = 100.0

        # Normalize then denormalize
        flow_norm = normalizer.normalize_flow(flow_original)
        flow_back = normalizer.denormalize_flow(flow_norm)

        # Should recover original value (within tolerance)
        assert abs(flow_back - flow_original) < 1.0

    def test_denormalize_flow_negative(self, normalizer):
        """Test denormalization ensures non-negative flow."""
        flow_norm = -10.0  # Extreme normalized value

        flow_back = normalizer.denormalize_flow(flow_norm)

        # Should be non-negative
        assert flow_back >= 0.0


class TestPressureNormalization:
    """Test pressure drop normalization (z-score)."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer with known statistics."""
        stats = NormalizationStatistics(
            pressure_drop_mean=2.0,
            pressure_drop_std=1.0
        )
        return EdgeFeatureNormalizer(stats=stats)

    def test_normalize_pressure_typical(self, normalizer):
        """Test pressure normalization with typical value."""
        dp_bar = 2.5

        dp_norm = normalizer.normalize_pressure_drop(dp_bar)

        # (2.5 - 2.0) / 1.0 = 0.5
        assert abs(dp_norm - 0.5) < 0.01

    def test_normalize_pressure_negative(self, normalizer):
        """Test pressure normalization allows negative (backflow)."""
        dp_bar = -1.0

        dp_norm = normalizer.normalize_pressure_drop(dp_bar)

        # Should handle negative values
        assert dp_norm < 0

    def test_normalize_pressure_clipping(self, normalizer):
        """Test pressure normalization clips outliers."""
        dp_bar = 100.0  # Extreme value

        dp_norm = normalizer.normalize_pressure_drop(dp_bar)

        # Should clip to [-5, 5]
        assert abs(dp_norm) <= 5.0

    def test_denormalize_pressure(self, normalizer):
        """Test pressure denormalization."""
        dp_original = 2.5

        dp_norm = normalizer.normalize_pressure_drop(dp_original)
        dp_back = normalizer.denormalize_pressure_drop(dp_norm)

        assert abs(dp_back - dp_original) < 0.01


class TestTemperatureNormalization:
    """Test temperature delta normalization (z-score)."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer with known statistics."""
        stats = NormalizationStatistics(
            temp_delta_mean=5.0,
            temp_delta_std=3.0
        )
        return EdgeFeatureNormalizer(stats=stats)

    def test_normalize_temp_typical(self, normalizer):
        """Test temperature normalization with typical value."""
        dt_c = 8.0

        dt_norm = normalizer.normalize_temp_delta(dt_c)

        # (8.0 - 5.0) / 3.0 = 1.0
        assert abs(dt_norm - 1.0) < 0.01

    def test_normalize_temp_negative(self, normalizer):
        """Test temperature normalization allows negative (cooling)."""
        dt_c = -2.0

        dt_norm = normalizer.normalize_temp_delta(dt_c)

        # Should handle negative values
        assert dt_norm < 0

    def test_denormalize_temp(self, normalizer):
        """Test temperature denormalization."""
        dt_original = 8.0

        dt_norm = normalizer.normalize_temp_delta(dt_original)
        dt_back = normalizer.denormalize_temp_delta(dt_norm)

        assert abs(dt_back - dt_original) < 0.01


class TestVibrationNormalization:
    """Test vibration normalization (min-max [0, 1])."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer with known statistics."""
        stats = NormalizationStatistics(
            vibration_max=10.0
        )
        return EdgeFeatureNormalizer(stats=stats)

    def test_normalize_vibration_typical(self, normalizer):
        """Test vibration normalization with typical value."""
        vib_g = 5.0

        vib_norm = normalizer.normalize_vibration(vib_g)

        # 5.0 / 10.0 = 0.5
        assert abs(vib_norm - 0.5) < 0.01

    def test_normalize_vibration_zero(self, normalizer):
        """Test vibration normalization with zero."""
        vib_norm = normalizer.normalize_vibration(0.0)

        assert vib_norm == 0.0

    def test_normalize_vibration_max(self, normalizer):
        """Test vibration normalization at maximum."""
        vib_norm = normalizer.normalize_vibration(10.0)

        assert abs(vib_norm - 1.0) < 0.01

    def test_normalize_vibration_clipping(self, normalizer):
        """Test vibration normalization clips to [0, 1]."""
        vib_norm = normalizer.normalize_vibration(20.0)  # Exceeds max

        assert vib_norm == 1.0

    def test_denormalize_vibration(self, normalizer):
        """Test vibration denormalization."""
        vib_original = 5.0

        vib_norm = normalizer.normalize_vibration(vib_original)
        vib_back = normalizer.denormalize_vibration(vib_norm)

        assert abs(vib_back - vib_original) < 0.01


class TestAgeNormalization:
    """Test age normalization (min-max [0, 1])."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer with known statistics."""
        stats = NormalizationStatistics(
            age_max=100000.0
        )
        return EdgeFeatureNormalizer(stats=stats)

    def test_normalize_age_typical(self, normalizer):
        """Test age normalization with typical value."""
        age_hours = 50000.0

        age_norm = normalizer.normalize_age(age_hours)

        # 50000 / 100000 = 0.5
        assert abs(age_norm - 0.5) < 0.01

    def test_normalize_age_zero(self, normalizer):
        """Test age normalization with zero (new component)."""
        age_norm = normalizer.normalize_age(0.0)

        assert age_norm == 0.0

    def test_normalize_age_max(self, normalizer):
        """Test age normalization at maximum."""
        age_norm = normalizer.normalize_age(100000.0)

        assert abs(age_norm - 1.0) < 0.01

    def test_normalize_age_clipping(self, normalizer):
        """Test age normalization clips to [0, 1]."""
        age_norm = normalizer.normalize_age(200000.0)  # Exceeds max

        assert age_norm == 1.0

    def test_denormalize_age(self, normalizer):
        """Test age denormalization."""
        age_original = 50000.0

        age_norm = normalizer.normalize_age(age_original)
        age_back = normalizer.denormalize_age(age_norm)

        assert abs(age_back - age_original) < 1.0


class TestMaintenanceScore:
    """Test maintenance score (pass-through)."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer."""
        return EdgeFeatureNormalizer()

    def test_normalize_maintenance_score(self, normalizer):
        """Test maintenance score is pass-through."""
        score = 0.7

        score_norm = normalizer.normalize_maintenance_score(score)

        # Should be unchanged
        assert score_norm == score

    def test_normalize_maintenance_score_clipping(self, normalizer):
        """Test maintenance score clips to [0, 1]."""
        score_low = normalizer.normalize_maintenance_score(-0.5)
        score_high = normalizer.normalize_maintenance_score(1.5)

        assert score_low == 0.0
        assert score_high == 1.0


class TestBatchNormalization:
    """Test batch normalization (normalize_all)."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer with default stats."""
        return EdgeFeatureNormalizer()

    @pytest.fixture
    def features(self):
        """Create sample features."""
        return {
            "flow_rate_lpm": 100.0,
            "pressure_drop_bar": 2.0,
            "temperature_delta_c": 5.0,
            "vibration_level_g": 1.0,
            "age_hours": 10000.0,
            "maintenance_score": 0.8
        }

    def test_normalize_all(self, normalizer, features):
        """Test normalizing all features at once."""
        normalized = normalizer.normalize_all(features)

        # Check all keys present
        assert set(normalized.keys()) == set(features.keys())

        # Check reasonable ranges
        assert -5 <= normalized["flow_rate_lpm"] <= 5
        assert -5 <= normalized["pressure_drop_bar"] <= 5
        assert -5 <= normalized["temperature_delta_c"] <= 5
        assert 0 <= normalized["vibration_level_g"] <= 1
        assert 0 <= normalized["age_hours"] <= 1
        assert 0 <= normalized["maintenance_score"] <= 1


class TestFitFromTrainingData:
    """Test fitting normalizer from training data."""

    @pytest.fixture
    def training_data(self):
        """Create synthetic training data."""
        np.random.seed(42)

        n_samples = 1000
        data = []

        for _ in range(n_samples):
            data.append({
                "flow_rate_lpm": np.random.lognormal(4.0, 1.0),
                "pressure_drop_bar": np.random.normal(2.0, 1.0),
                "temperature_delta_c": np.random.normal(5.0, 3.0),
                "vibration_level_g": np.random.uniform(0, 5),
                "age_hours": np.random.uniform(0, 50000),
                "maintenance_score": np.random.uniform(0, 1)
            })

        return data

    def test_fit(self, training_data):
        """Test fitting normalizer from training data."""
        normalizer = EdgeFeatureNormalizer()

        normalizer.fit(training_data)

        # Check statistics updated
        assert normalizer.stats.flow_log_mean != 4.0  # Changed from default
        assert normalizer.stats.pressure_drop_mean != 2.0

    def test_fit_updates_all_stats(self, training_data):
        """Test fit updates all statistics."""
        normalizer = EdgeFeatureNormalizer()

        # Store defaults
        defaults = normalizer.stats.model_dump()

        normalizer.fit(training_data)

        # Check all updated
        new_stats = normalizer.stats.model_dump()

        # At least some stats should change
        changes = sum(
            1 for k in defaults
            if abs(defaults[k] - new_stats[k]) > 0.1
        )
        assert changes >= 4  # At least 4 stats changed significantly


class TestPersistence:
    """Test save/load functionality."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer with custom stats."""
        stats = NormalizationStatistics(
            flow_log_mean=4.5,
            flow_log_std=1.2,
            pressure_drop_mean=2.5,
            pressure_drop_std=1.5
        )
        return EdgeFeatureNormalizer(stats=stats)

    def test_save_and_load(self, normalizer):
        """Test saving and loading normalizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "normalizer.json"

            # Save
            normalizer.save(path)

            # Check file exists
            assert path.exists()

            # Load
            loaded = EdgeFeatureNormalizer.load(path)

            # Check stats match
            assert loaded.stats.flow_log_mean == normalizer.stats.flow_log_mean
            assert loaded.stats.flow_log_std == normalizer.stats.flow_log_std
            assert loaded.stats.pressure_drop_mean == normalizer.stats.pressure_drop_mean

    def test_get_stats(self, normalizer):
        """Test getting stats as dictionary."""
        stats_dict = normalizer.get_stats()

        assert isinstance(stats_dict, dict)
        assert "flow_log_mean" in stats_dict
        assert stats_dict["flow_log_mean"] == 4.5

    def test_load_stats(self):
        """Test loading stats from dictionary."""
        normalizer = EdgeFeatureNormalizer()

        stats_dict = {
            "flow_log_mean": 4.5,
            "flow_log_std": 1.2,
            "pressure_drop_mean": 2.5,
            "pressure_drop_std": 1.5,
            "temp_delta_mean": 5.0,
            "temp_delta_std": 3.0,
            "vibration_max": 10.0,
            "age_max": 100000.0
        }

        normalizer.load_stats(stats_dict)

        assert normalizer.stats.flow_log_mean == 4.5
        assert normalizer.stats.pressure_drop_mean == 2.5


class TestFactoryFunction:
    """Test factory function."""

    def test_create_normalizer_default(self):
        """Test factory with defaults."""
        normalizer = create_edge_feature_normalizer()

        assert isinstance(normalizer, EdgeFeatureNormalizer)
        assert normalizer.stats.flow_log_mean == 4.0

    def test_create_normalizer_custom_stats(self):
        """Test factory with custom stats."""
        stats = NormalizationStatistics(flow_log_mean=4.5)
        normalizer = create_edge_feature_normalizer(stats=stats)

        assert normalizer.stats.flow_log_mean == 4.5
