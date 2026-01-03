# 📊 Data Preparation Guide: Production-Grade ML Systems

**Based on**:
- Chip Huyen: *Designing Machine Learning Systems* (Arch + pipelines)
- Lakshmanan et al.: *ML Design Patterns* (Reproducibility + no skew)
- Aurélien Géron: *Hands-On ML* (Practical preprocessing)
- Kuhn & Johnson: *Feature Engineering and Selection* (Signal representation)

**Context**: UCI Hydraulic Dataset (UC Irvine archive) → GNN-based Diagnostics  
**Timeline**: 4 weeks, 8-10 hrs/week  
**Goal**: Production-ready train/val/test splits with no data leakage, repeatable, with drift detection

---

## Week 1: Understanding the System (8-10 hours)

### Part 1.1: ML Systems Thinking (Chip Huyen)

**Read**: Chapters 1-3 from *Designing Machine Learning Systems*
- Chapter 1: ML System design lifecycle
- Chapter 2: Business objectives → ML objectives
- Chapter 3: Data collection, labeling, and quality

**Key concepts**:
- ML systems are NOT just models; they're pipelines with feedback loops
- Data quality >> model complexity
- Training-serving skew is THE problem in production

**Questions to answer for YOUR hydraulic system**:

```markdown
1. **Business objective**: What does "accurate diagnostics" mean?
   - Latency requirement? (real-time vs batch)
   - Accuracy requirement? (recall vs precision trade-off)
   - Cost of false positives/negatives?

2. **Data objective**: 
   - What are we predicting? (9 fault types)
   - What's the label space? (multi-label or single?)
   - Is this imbalanced? (yes, normal >> anomalies)

3. **System requirements**:
   - Reproducibility: same random seed → same train/val/test
   - Monitoring: can we detect when live data drifts from training distribution?
   - Versioning: can we track which data version trained which model?
```

**Assignment**: Write down answers for YOUR system in `DATA_REQUIREMENTS.md`

---

### Part 1.2: UCI Dataset Specification

**Read**: `services/gnn_service/data/raw_real_dataset/documentation.txt` + `profile.txt`

**What we have**:

```
UCI Hydraulic Test Rig Dataset
├─ Sensors: 11 channels (PS, FS, TS, EPS, CE, CP, SE, VS)
├─ Duration: 60 hours continuous
├─ Sample rate: ~100 Hz (varies by sensor)
├─ Conditions: 8 fault types + normal
│  ├─ FS: Cooler condition
│  ├─ PS: Pump leakage  
│  ├─ HE: Hydroelectric condition (hydro-erosion)
│  ├─ CE: Contamination/Erosion level
│  ├─ CP: Cooler Power
│  └─ Various degradation levels
└─ Labels: In profile.txt (condition states over time)
```

**Key insight** (Huyen): This is NOT a standard supervised dataset. We have:
- Multivariate time series ✅
- Labels that change over time ✅
- Continuous monitoring data (NOT isolated samples) ✅
- High class imbalance (mostly normal) ✅

**Assignment**: Parse `description.txt` and `profile.txt`, answer:
- How many complete cycles (from normal → fault → recovery) do we have?
- What's the label distribution?
- Are there overlapping faults (multi-label)?

---

## Week 2: Data Ingestion & Cleaning (8-10 hours)

### Part 2.1: Raw Data Loading Pipeline (Géron + Huyen)

**Read**: 
- Géron, Chapter 2: *End-to-end project* (data loading)
- Huyen, Chapter 3: *Data collection quality* (handling raw formats)

**Core principle**: Reproducible data → Same git commit → Same behavior forever

**Step 1: Load raw `.txt` files**

```python
# services/gnn_service/src/data/loaders.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class UCIHydraulicLoader:
    """
    Load UCI Hydraulic dataset from raw .txt files.
    
    Principle (Huyen): Each data loading step should be:
    - Deterministic (same input → same output)
    - Logged (what was loaded, why)
    - Versioned (file hash included in metadata)
    - Testable (unit tests for each transformation)
    """
    
    SENSOR_FILES = {
        'PS1': 'PS1.txt',  # Pressure sensor 1
        'PS2': 'PS2.txt',  # Pressure sensor 2  
        'PS3': 'PS3.txt',  # Pressure sensor 3
        'PS4': 'PS4.txt',  # Pressure sensor 4
        'PS5': 'PS5.txt',  # Pressure sensor 5
        'PS6': 'PS6.txt',  # Pressure sensor 6
        'FS1': 'FS1.txt',  # Flow sensor 1
        'FS2': 'FS2.txt',  # Flow sensor 2
        'TS1': 'TS1.txt',  # Temperature sensor 1
        'TS2': 'TS2.txt',  # Temperature sensor 2
        'TS3': 'TS3.txt',  # Temperature sensor 3
        'TS4': 'TS4.txt',  # Temperature sensor 4
        'EPS1': 'EPS1.txt', # Electrical power sensor
        'CE': 'CE.txt',     # Contamination/Erosion
        'CP': 'CP.txt',     # Cooler Power
        'SE': 'SE.txt',     # Specific Energy
        'VS1': 'VS1.txt',   # Vibration sensor
    }
    
    SAMPLE_RATES = {  # Hz (from documentation)
        'PS1': 100, 'PS2': 100, 'PS3': 100, 'PS4': 100, 'PS5': 100, 'PS6': 100,
        'FS1': 10, 'FS2': 10,
        'TS1': 1, 'TS2': 1, 'TS3': 1, 'TS4': 1,
        'EPS1': 100,
        'CE': 10, 'CP': 10, 'SE': 10,
        'VS1': 100,
    }
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_data = {}
        self.metadata = {
            'loaded_at': pd.Timestamp.now().isoformat(),
            'source_dir': str(data_dir),
            'file_hashes': {},  # For reproducibility
        }
    
    def load_all_sensors(self) -> Dict[str, np.ndarray]:
        """
        Load all sensor files into memory.
        
        Principle (Géron): Handle each file independently first,
        then combine. Easier to debug.
        
        Returns:
            dict: sensor_name → raw values (1D array)
        """
        for sensor_name, filename in self.SENSOR_FILES.items():
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue
            
            # Load: simple .txt with one value per line
            data = np.loadtxt(filepath, dtype=np.float32)
            
            # Compute hash for reproducibility tracking
            file_hash = self._compute_hash(data)
            self.metadata['file_hashes'][sensor_name] = file_hash
            
            self.raw_data[sensor_name] = data
            logger.info(f"Loaded {sensor_name}: {len(data)} samples, hash={file_hash[:8]}...")
        
        return self.raw_data
    
    @staticmethod
    def _compute_hash(data: np.ndarray) -> str:
        """SHA256 hash of data for reproducibility tracking."""
        import hashlib
        return hashlib.sha256(data.tobytes()).hexdigest()
    
    def load_labels(self) -> pd.DataFrame:
        """
        Parse profile.txt to get condition labels over time.
        
        Output: DataFrame with columns:
        - timestamp (or index)
        - condition (fault type)
        - severity (degradation level if available)
        """
        profile_path = self.data_dir / 'profile.txt'
        
        # Parse profile.txt manually (format may vary)
        # Typical format: lines with condition codes
        labels = self._parse_profile(profile_path)
        logger.info(f"Loaded labels: {len(labels)} state transitions")
        return labels
    
    @staticmethod
    def _parse_profile(filepath: Path) -> pd.DataFrame:
        """
        Parse profile.txt with condition information.
        Format (from documentation):
        - Each line: state code, duration, or similar
        """
        # This is dataset-specific; adjust based on actual format
        with open(filepath) as f:
            content = f.read()
        
        # Extract condition transitions
        # Placeholder: you'll need to parse actual format
        logger.warning("Profile parsing is dataset-specific; customize this")
        
        return pd.DataFrame()  # Implement based on actual format
```

**Step 2: Synchronize time axes**

```python
def align_to_common_timeline(raw_data: Dict[str, np.ndarray], 
                           sample_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Key problem (Huyen): Different sensors have different sample rates.
    Solution: Interpolate or resample to common grid.
    
    Principle: Choose a "slow" rate (most conservative, no extrapolation).
    Here: 1 Hz (every 1 second) to capture all dynamics.
    
    Args:
        raw_data: sensor_name → values
        sample_rates: sensor_name → Hz
    
    Returns:
        DataFrame with aligned index (datetime/sample number)
        Columns: sensor values
    """
    # Assume each file has samples at their native rate
    # Create timeline for slowest sensor
    
    slowest_rate = min(sample_rates.values())
    total_samples = len(next(iter(raw_data.values())))
    total_duration_s = total_samples / max(sample_rates.values())
    
    # Create common timeline at 1 Hz
    common_timeline = np.arange(0, total_duration_s, 1.0)  # 1 second intervals
    
    aligned_data = {}
    for sensor_name, values in raw_data.items():
        rate = sample_rates[sensor_name]
        sensor_timeline = np.arange(len(values)) / rate
        
        # Interpolate to common timeline
        aligned = np.interp(common_timeline, sensor_timeline, values, 
                           left=np.nan, right=np.nan)
        aligned_data[sensor_name] = aligned
    
    df = pd.DataFrame(aligned_data, 
                     index=pd.TimedeltaIndex(common_timeline, unit='s'))
    return df
```

**Step 3: Handle missing values & outliers**

```python
class DataCleaner:
    """
    Clean data following Géron's approach:
    1. Identify missing
    2. Identify outliers (NOT remove, mark!)
    3. Log decisions (reproducibility)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_log = []
    
    def handle_missing(self, method='forward_fill'):
        """
        Strategy (Huyen): Don't just drop missing;
        think about what it means in your domain.
        
        For hydraulic sensors:
        - Missing → sensor error?
        - Forward fill? (assumes steady state)
        - Interpolate? (assumes smooth change)
        
        Choose one consistently.
        """
        missing_counts = self.df.isnull().sum()
        
        for col in missing_counts[missing_counts > 0].index:
            if method == 'forward_fill':
                self.df[col].fillna(method='ffill', inplace=True)
                self.cleaning_log.append(
                    f"Column {col}: {missing_counts[col]} missing values filled (forward fill)"
                )
            elif method == 'interpolate':
                self.df[col].interpolate(method='linear', inplace=True)
                self.cleaning_log.append(
                    f"Column {col}: {missing_counts[col]} missing values interpolated"
                )
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        Detect but don't remove (you might want them for anomaly detection).
        Mark them for later decision.
        """
        outlier_mask = pd.DataFrame(False, index=self.df.index, columns=self.df.columns)
        
        for col in self.df.columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            
            outliers = (self.df[col] < lower) | (self.df[col] > upper)
            outlier_mask[col] = outliers
            
            n_outliers = outliers.sum()
            if n_outliers > 0:
                self.cleaning_log.append(
                    f"Column {col}: {n_outliers} outliers detected (IQR method)"
                )
        
        return outlier_mask
```

**Assignment**:
1. Implement `UCIHydraulicLoader.load_all_sensors()`
2. Implement `align_to_common_timeline()`
3. Log all decisions to a cleaning report
4. Verify: reproducibility check (run twice, get same result)

---

### Part 2.2: Exploratory Data Analysis (EDA)

**Read**: Géron Chapter 2 (EDA section)

**Goal**: Understand what you're working with BEFORE feature engineering

```python
def eda_report(df: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """
    Generate reproducible EDA report (not just plots).
    """
    report = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'statistics': df.describe().T.to_dict(),
        'correlations': df.corr().to_dict(),
        'label_distribution': labels.value_counts().to_dict() if 'labels' in locals() else None,
        'temporal_coverage': {
            'start': df.index[0],
            'end': df.index[-1],
            'duration_hours': (df.index[-1] - df.index[0]).total_seconds() / 3600,
        }
    }
    return report

# Save as JSON (reproducible, version-controllable)
import json
with open('eda_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
```

**Assignment**: Generate and commit `eda_report.json`

---

## Week 3: Feature Engineering & Windowing (8-10 hours)

### Part 3.1: Time Series → Windowed Features (Kuhn & Johnson)

**Read**:
- Kuhn & Johnson, Part 2: *Feature Engineering* (Ch. 3-5)
- Géron, Ch. 6: *Decision Trees* section on temporal features

**Core concept**: Time series → static feature vectors

**Problem**: We have continuous 60-hour trace. GNN expects:
- Nodes (components): 5-10 nodes
- Edges (connections): 10-20 edges
- Features (per node, per edge): 5-20 floats
- Labels (per node): fault type

**Solution**: Create sliding windows

```python
class TemporalFeatureExtractor:
    """
    Extract features from time windows (Kuhn & Johnson approach).
    
    Key: Choose window size based on domain knowledge.
    For hydraulics: 1-5 minutes captures system dynamics.
    """
    
    def __init__(self, window_size_s: int = 300, step_size_s: int = 60):
        """
        Args:
            window_size_s: 300s = 5 min (typical hydraulic cycle)
            step_size_s: 60s overlap (sliding window)
        """
        self.window_size_s = window_size_s
        self.step_size_s = step_size_s
    
    def create_windows(self, df: pd.DataFrame) -> list:
        """
        Split time series into windows.
        
        Returns: list of (start_idx, end_idx, window_df)
        """
        windows = []
        
        for start_idx in range(0, len(df) - self.window_size_s, self.step_size_s):
            end_idx = start_idx + self.window_size_s
            window_df = df.iloc[start_idx:end_idx].copy()
            
            windows.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'data': window_df,
                'timestamp': df.index[start_idx],
            })
        
        return windows
    
    def extract_features(self, window_df: pd.DataFrame) -> dict:
        """
        For each window, compute features (Kuhn & Johnson).
        
        Principle: Use domain knowledge, not just stat summaries.
        """
        features = {}
        
        # Statistical features
        for col in window_df.columns:
            features[f"{col}_mean"] = window_df[col].mean()
            features[f"{col}_std"] = window_df[col].std()
            features[f"{col}_min"] = window_df[col].min()
            features[f"{col}_max"] = window_df[col].max()
            features[f"{col}_median"] = window_df[col].median()
            
            # Domain-specific: slope (degradation indicator)
            x = np.arange(len(window_df))
            y = window_df[col].values
            slope, intercept = np.polyfit(x, y, 1)
            features[f"{col}_trend"] = slope
            
            # Energy (power × time, for hydraulics)
            if 'PS' in col or 'FS' in col:
                # Assume PS = pressure, FS = flow
                # Power ≈ P × Q
                power = window_df.get('PS1', window_df[col]) * window_df[col]
                features[f"{col}_energy"] = power.sum()
        
        # Inter-sensor features (correlations)
        for col1 in window_df.columns:
            for col2 in window_df.columns:
                if col1 < col2:
                    features[f"corr_{col1}_{col2}"] = window_df[col1].corr(window_df[col2])
        
        return features
```

**Assignment**:
1. Determine optimal window size (1-5 min) by domain reasoning
2. Create windowed features
3. Save to `processed_data/windowed_features.pkl` or `.parquet`

---

### Part 3.2: Normalize & Scale (ML Design Patterns)

**Read**: ML Design Patterns, Chapter 2: *Representation* (Normalization)

**Principle** (Critical!): 
- Fit scaler on TRAIN only
- Apply same scaler to VAL/TEST
- Save scaler with model (training-serving consistency)

```python
from sklearn.preprocessing import StandardScaler
import pickle

class ScalingPipeline:
    """
    Design Pattern: Fit once, transform multiple times.
    NEVER fit on full data (data leakage).
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_on_train(self, X_train: np.ndarray):
        """Fit scaler on TRAIN data only."""
        self.scaler.fit(X_train)
        self.is_fitted = True
        
        # Log scaling parameters for reproducibility
        print(f"Scaler fitted:")
        print(f"  Mean: {self.scaler.mean_}")
        print(f"  Std: {self.scaler.scale_}")
    
    def transform_all(self, X_train, X_val, X_test):
        """Apply SAME scaler to all splits."""
        assert self.is_fitted, "Fit on train first!"
        
        return (
            self.scaler.transform(X_train),
            self.scaler.transform(X_val),
            self.scaler.transform(X_test),
        )
    
    def save(self, filepath):
        """Save scaler with train/val/test split."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved: {filepath}")
```

**Assignment**:
1. Fit scaler on train
2. Transform all splits
3. Save scaler alongside data

---

## Week 4: Reproducible Data Splits (8-10 hours)

### Part 4.1: Train/Val/Test Strategy (ML Design Patterns)

**Read**: ML Design Patterns, Chapter 2: *Splits* + Chapter 4: *Serving*

**Critical principle** (Huyen + Design Patterns):
- NO RANDOM SHUFFLE of time series (breaks temporal dependencies)
- Use temporal split: past → train, next → val, last → test
- NO overlap between splits
- Save split indices (reproducibility)

```python
class TemporalDataSplitter:
    """
    Split time series respecting temporal order.
    NO future data leaks to train/val.
    """
    
    def __init__(self, df: pd.DataFrame, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        self.df = df
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_indices = {}
    
    def split(self):
        """Create temporal splits."""
        n = len(self.df)
        
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        # NO OVERLAP
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)
        
        self.split_indices = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
        }
        
        return {
            'train': self.df.iloc[train_idx],
            'val': self.df.iloc[val_idx],
            'test': self.df.iloc[test_idx],
        }
    
    def save_indices(self, filepath: Path):
        """Save split indices for reproducibility."""
        np.savez(
            filepath,
            train_idx=self.split_indices['train'],
            val_idx=self.split_indices['val'],
            test_idx=self.split_indices['test'],
        )
        print(f"Split indices saved: {filepath}")
    
    def verify_no_overlap(self):
        """Check splits don't overlap."""
        train = set(self.split_indices['train'])
        val = set(self.split_indices['val'])
        test = set(self.split_indices['test'])
        
        assert len(train & val) == 0, "Train-Val overlap!"
        assert len(train & test) == 0, "Train-Test overlap!"
        assert len(val & test) == 0, "Val-Test overlap!"
        print("✓ No overlaps detected")
```

**Assignment**:
1. Implement temporal split (70/15/15)
2. Verify no overlap
3. Save indices to file

---

### Part 4.2: Data Leakage Detection (Huyen + Design Patterns)

**Read**: Huyen, Chapter 3: *Data leakage*

**Types to check**:

```python
class DataLeakageDetector:
    
    @staticmethod
    def check_feature_target_correlation(X_train, X_val, y_train, y_val):
        """
        Are features on train & val from same distribution?
        (If not, there might be temporal drift or leakage)
        """
        mean_train = X_train.mean(axis=0)
        mean_val = X_val.mean(axis=0)
        
        drift = np.abs((mean_train - mean_val) / (mean_train + 1e-8)).max()
        print(f"Max feature drift (train → val): {drift:.4f}")
        
        if drift > 0.5:  # 50% change is suspicious
            print("⚠️  WARNING: Significant feature drift detected!")
            return False
        return True
    
    @staticmethod
    def check_temporal_order(split_indices):
        """
        Verify train < val < test chronologically.
        """
        train_max = split_indices['train'].max()
        val_min = split_indices['val'].min()
        val_max = split_indices['val'].max()
        test_min = split_indices['test'].min()
        
        assert train_max < val_min, "Train after Val!"
        assert val_max < test_min, "Val after Test!"
        print("✓ Temporal order verified")
    
    @staticmethod
    def check_label_distribution(y_train, y_val, y_test):
        """
        Are class distributions similar across splits?
        (Large difference → train-test skew)
        """
        train_dist = np.bincount(y_train) / len(y_train)
        val_dist = np.bincount(y_val, minlength=len(train_dist)) / len(y_val)
        test_dist = np.bincount(y_test, minlength=len(train_dist)) / len(y_test)
        
        # KL divergence
        from scipy.stats import entropy
        kl_train_val = entropy(train_dist, val_dist + 1e-10)
        kl_val_test = entropy(val_dist, test_dist + 1e-10)
        
        print(f"KL(train || val): {kl_train_val:.4f}")
        print(f"KL(val || test): {kl_val_test:.4f}")
        
        if kl_train_val > 0.5 or kl_val_test > 0.5:
            print("⚠️  WARNING: Class distribution shifts detected!")
```

**Assignment**: Run all checks, document findings

---

### Part 4.3: Metadata & Versioning

**Read**: ML Design Patterns, Chapter 5: *Data versioning*

**Create comprehensive metadata**:

```python
import json
from datetime import datetime
import hashlib

def create_data_manifest(df_train, df_val, df_test, split_info):
    """
    Document EVERYTHING about your data splits.
    This goes to git alongside the data.
    """
    manifest = {
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'source': 'UCI Hydraulic Dataset',
        'source_url': 'https://archive.ics.uci.edu/ml/datasets/Hydraulic+Systems+Predictive+Maintenance+and+Condition+Monitoring',
        
        'preprocessing': {
            'window_size_s': 300,
            'window_step_s': 60,
            'normalization': 'StandardScaler (fit on train)',
            'missing_handling': 'forward_fill',
            'outlier_handling': 'marked (not removed)',
        },
        
        'splits': {
            'train': {
                'n_samples': len(df_train),
                'date_range': f"{df_train.index[0]} to {df_train.index[-1]}",
                'hash': hashlib.sha256(df_train.to_string().encode()).hexdigest()[:16],
            },
            'val': {
                'n_samples': len(df_val),
                'date_range': f"{df_val.index[0]} to {df_val.index[-1]}",
                'hash': hashlib.sha256(df_val.to_string().encode()).hexdigest()[:16],
            },
            'test': {
                'n_samples': len(df_test),
                'date_range': f"{df_test.index[0]} to {df_test.index[-1]}",
                'hash': hashlib.sha256(df_test.to_string().encode()).hexdigest()[:16],
            },
        },
        
        'quality_checks': {
            'no_data_leakage': True,
            'temporal_order_verified': True,
            'missing_values_handled': True,
            'scaling_fit_on_train_only': True,
        },
        
        'columns': list(df_train.columns),
        'n_features': df_train.shape[1],
    }
    
    return manifest

# Save manifest
manifest = create_data_manifest(df_train, df_val, df_test, split_info)
with open('services/gnn_service/data/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
```

**Assignment**: Create and commit manifest.json

---

## Directory Structure (After Completion)

```
services/gnn_service/data/
├── raw_real_dataset/              # Original UC Irvine files
│   ├── PS*.txt
│   ├── FS*.txt
│   ├── TS*.txt
│   ├── EPS1.txt
│   ├── CE.txt
│   ├── CP.txt
│   ├── SE.txt
│   ├── VS1.txt
│   ├── profile.txt
│   ├── documentation.txt
│   └── description.txt
│
├── processed/
│   ├── train.pkl                  # Train split (windowed + scaled)
│   ├── val.pkl                    # Val split
│   ├── test.pkl                   # Test split
│   │
│   ├── scaler.pkl                 # Fitted StandardScaler
│   ├── split_indices.npz          # train_idx, val_idx, test_idx
│   │
│   ├── manifest.json              # Data provenance & versioning
│   ├── eda_report.json            # Exploratory analysis
│   └── cleaning_log.txt           # All preprocessing decisions
│
└── README.md                        # Data documentation
```

---

## Reproducibility Checklist

✅ **Deterministic pipeline**:
- [ ] No random shuffling of time series
- [ ] Explicit random seeds (if used)
- [ ] All transformations logged

✅ **No data leakage**:
- [ ] Train/val/test don't overlap
- [ ] Scaler fit on train only
- [ ] Labels from train don't influence feature engineering

✅ **Versioning**:
- [ ] Data hashes saved
- [ ] Manifest with full provenance
- [ ] All code changes tracked in git

✅ **Training-serving consistency**:
- [ ] Same scaler used for training and serving
- [ ] Same window size/step size
- [ ] Same missing value strategy

---

## Key Takeaways (For Your Project)

**Chip Huyen**:
- ML systems = pipelines, not just models
- Data quality >> model complexity
- Design for reproducibility from day 1

**ML Design Patterns**:
- Separate concerns: raw data → processed → features
- Fit transformations on train, apply to val/test
- Version everything (data, scaler, split indices)

**Géron**:
- EDA before modeling
- Handle missing values explicitly
- Document decisions

**Kuhn & Johnson**:
- Features encode domain knowledge
- Time series → windowed features respects temporal structure
- Normalize independently for train/val/test

---

## References

1. Huyen, Chip. *Designing Machine Learning Systems*. O'Reilly, 2022. Ch. 1-3.
2. Lakshmanan et al. *Machine Learning Design Patterns*. O'Reilly, 2021. Ch. 2, 4-5.
3. Géron, Aurélien. *Hands-On ML with Scikit-Learn, Keras, TensorFlow*. O'Reilly, 2019. Ch. 2.
4. Kuhn & Johnson. *Feature Engineering and Selection*. Routledge, 2019. Part 2.
5. UCI Hydraulic Dataset: https://archive.ics.uci.edu/ml/datasets/Hydraulic+Systems+Predictive+Maintenance+and+Condition+Monitoring

---

**Status**: Hands-on guide ready  
**Next**: Implement Week 1-4 with actual code  
**Output**: `data/processed/{train,val,test}.pkl` + reproducible pipeline  
