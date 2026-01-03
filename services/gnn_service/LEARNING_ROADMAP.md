# 🚫 4-Week Learning Roadmap: ML Systems & Data Preparation

**Goal**: Transform UCI Hydraulic raw data → production-ready train/val/test splits  
**Time**: 4 weeks, 8-10 hrs/week (32-40 hours total)  
**Prerequisites**: Python, pandas, numpy, git  
**Target**: Reproducible, leak-free data pipeline for GNN training  

---

## Week 1: ML Systems Thinking + Data Understanding

### Day 1-2: Read & Understand (Chip Huyen)

**Reading**:
- Huyen: *Designing ML Systems*, Chapters 1-3 (~60 min read)
- Huyen blog: [ML System Design](https://huyenchip.com/machine-learning-systems-design/)

**Key concepts**:
- ML systems are NOT just models
- Data quality >> model complexity
- Data leakage is the #1 production killer
- Reproducibility = reliability

**Quiz yourself**:
1. What's a "training-serving skew"? (Give 2 examples)
2. Why is data distribution shift dangerous?
3. How would you detect if your model is leaking?

---

### Day 3-4: UCI Dataset Specification (Géron + Huyen)

**Files to read**:
- `services/gnn_service/data/raw_real_dataset/documentation.txt`
- `services/gnn_service/data/raw_real_dataset/description.txt`
- `services/gnn_service/data/raw_real_dataset/profile.txt`

**Assignment 1.1: Data Specification Document**

Create file: `services/gnn_service/DATA_REQUIREMENTS.md`

Fill in:
```markdown
# Data Specification

## Business Objective
- What are we predicting?
- Latency requirement?
- Accuracy requirement? (recall vs precision)

## Data Objective
- Label space: (9 fault types + normal)
- Is it imbalanced?
- Training objective: (multi-label or single-label?)

## Dataset Details
- Source: UCI Hydraulic Test Rig
- Duration: [X hours]
- Sensors: [list with rates]
- Conditions: [list with descriptions]

## Quality Requirements
- Max missing data tolerance: [%]
- Outlier handling strategy: [description]
- Data leakage prevention: [checklist]
```

**Assignment 1.2: Parse profile.txt**

Code file: `services/gnn_service/scripts/week1_parse_profile.py`

```python
# TODO: Parse profile.txt and answer:
# 1. How many different condition states?
# 2. How long does each condition last?
# 3. Overlapping faults or sequential?
# 4. Missing data periods?

from pathlib import Path

profile_path = Path("services/gnn_service/data/raw_real_dataset/profile.txt")
with open(profile_path) as f:
    content = f.read()

print("Profile content (first 500 chars):")
print(content[:500])

# Your parsing code here
```

**Deliverable**: Commit both files with analysis

---

### Day 5: Load & Explore Data (Practical)

**Read**: Géron Chapter 2 (End-to-end project, data loading)

**Assignment 1.3: Test the loader**

```bash
# In Python shell or script:
from pathlib import Path
from src.data.loaders import UCIHydraulicLoader

data_dir = Path("services/gnn_service/data/raw_real_dataset")
loader = UCIHydraulicLoader(data_dir)

# Load
raw_data = loader.load_all_sensors(verbose=True)

# Align to 1 Hz
df_aligned = loader.align_to_common_timeline(target_rate_hz=1.0)

print(f"Shape: {df_aligned.shape}")
print(f"Missing: {df_aligned.isnull().sum()}")
print(df_aligned.describe())

# Save as CSV for exploration
df_aligned.to_csv("aligned_sensors_1hz.csv")
```

**Questions to answer**:
- How many samples total?
- How many missing per sensor?
- What's the value range per sensor?
- Any obvious anomalies?

**Deliverable**: Run script, save aligned CSV, document findings in issue/PR

---

## Week 2: Raw Data Loading & Cleaning

### Day 1-2: Deep dive into loaders.py (Huyen + Géron)

**Read**: 
- Géron Chapter 2 (full)
- ML Design Patterns Chapter 2 (Representation)

**Assignment 2.1: Understand loaders.py**

File: `services/gnn_service/src/data/loaders.py` (already created)

Tasks:
1. Run the `if __name__ == "__main__":` block
2. Verify all sensors load correctly
3. Check alignment (timeline should be 1 Hz, continuous)
4. Answer:
   - Why use `np.interp()` instead of `resample()`?
   - What does `left=np.nan, right=np.nan` do?
   - Why track file hashes?

**Assignment 2.2: Extend loader for your specific needs**

```python
# Add to loaders.py:

class UCIHydraulicLoader:
    def load_profile_structured(self) -> pd.DataFrame:
        """
        Parse profile.txt into structured DataFrame.
        
        TODO: Implement based on actual profile format
        Output should have columns:
        - timestamp (or index)
        - fault_type
        - severity (if available)
        - confidence (if available)
        """
        pass  # Implement
    
    def get_condition_timeline(self) -> Dict[str, Tuple[int, int]]:
        """
        Return {condition_name: (start_idx, end_idx)} for each condition.
        
        Example output:
        {
            'FS_NEW': (0, 1000),
            'FS_OLD': (1000, 2500),
            'PS_NEW': (2500, 4000),
            ...
        }
        """
        pass  # Implement
```

---

### Day 3-4: Data Cleaning

**Read**: ML Design Patterns Chapter 2 (Quality)

**Assignment 2.3: Test DataCleaner**

```python
from src.data.loaders import DataCleaner

# Load and align
loader = UCIHydraulicLoader(data_dir)
raw = loader.load_all_sensors()
df_aligned = loader.align_to_common_timeline()

# Clean
cleaner = DataCleaner(df_aligned)

# Test: forward fill
cleaner.handle_missing(method='forward_fill', max_gap=10)
print(cleaner.get_report())

# Test: detect outliers
outlier_mask = cleaner.detect_outliers(method='iqr', threshold=1.5)
print(f"Outliers detected: {outlier_mask.sum().sum()} points")

# Save cleaned data
df_cleaned = cleaner.df
df_cleaned.to_pickle("week2_cleaned_data.pkl")
```

**Questions**:
- How many missing values per sensor?
- What % are outliers?
- Reasonable to forward-fill or interpolate?

**Deliverable**: Cleaned data saved, summary stats

---

### Day 5: EDA Report

**Assignment 2.4: Create EDA report**

File: `services/gnn_service/scripts/week2_eda.py`

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

df_cleaned = pd.read_pickle("week2_cleaned_data.pkl")

# Generate EDA
eda_report = {
    'shape': df_cleaned.shape,
    'dtypes': df_cleaned.dtypes.astype(str).to_dict(),
    'missing_pct': (df_cleaned.isnull().sum() / len(df_cleaned) * 100).to_dict(),
    'statistics': df_cleaned.describe().T.to_dict(),
    'correlations': df_cleaned.corr().to_dict(),
    'temporal_coverage_hours': (df_cleaned.index[-1] - df_cleaned.index[0]).total_seconds() / 3600,
}

# Save
with open('eda_report.json', 'w') as f:
    json.dump(eda_report, f, indent=2, default=str)

print(json.dumps(eda_report, indent=2, default=str))
```

**Deliverable**: `eda_report.json` committed

---

## Week 3: Feature Engineering & Windowing

### Day 1-2: Time Series → Windows (Kuhn & Johnson)

**Read**: Kuhn & Johnson, Chapters 3-5 (~2 hours)

**Key concept**: Sliding windows extract features from continuous time series

**Assignment 3.1: Implement TemporalFeatureExtractor**

File: `services/gnn_service/src/data/feature_engineering.py` (create new)

```python
import numpy as np
import pandas as pd
from typing import List, Dict

class TemporalFeatureExtractor:
    """
    Extract features from time windows.
    
    Domain knowledge: Hydraulic cycles ~1-5 minutes.
    Choose window size: 300 seconds (5 min)
    Choose step size: 60 seconds (1 min overlap)
    """
    
    def __init__(self, window_size_s: int = 300, step_size_s: int = 60):
        self.window_size_s = window_size_s
        self.step_size_s = step_size_s
    
    def create_windows(self, df: pd.DataFrame) -> List[Dict]:
        """
        Split time series into sliding windows.
        
        Returns: list of dicts with 'start_idx', 'end_idx', 'data'
        """
        windows = []
        
        # Assume index is 1 Hz (1 sample per second)
        n_samples_per_window = self.window_size_s
        n_samples_per_step = self.step_size_s
        
        for start_idx in range(0, len(df) - n_samples_per_window, n_samples_per_step):
            end_idx = start_idx + n_samples_per_window
            window_data = df.iloc[start_idx:end_idx].copy()
            
            windows.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'data': window_data,
                'timestamp': df.index[start_idx],
            })
        
        return windows
    
    def extract_features(self, window_df: pd.DataFrame) -> Dict[str, float]:
        """
        For each window, compute statistical + domain features.
        """
        features = {}
        
        # Statistical features per sensor
        for col in window_df.columns:
            features[f"{col}_mean"] = window_df[col].mean()
            features[f"{col}_std"] = window_df[col].std()
            features[f"{col}_min"] = window_df[col].min()
            features[f"{col}_max"] = window_df[col].max()
            features[f"{col}_median"] = window_df[col].median()
            
            # Trend (degradation indicator)
            x = np.arange(len(window_df))
            y = window_df[col].values
            if len(x) > 1:
                slope, _ = np.polyfit(x, y, 1)
                features[f"{col}_trend"] = slope
            else:
                features[f"{col}_trend"] = 0.0
        
        # Inter-sensor correlations
        corr_matrix = window_df.corr()
        for col1 in window_df.columns:
            for col2 in window_df.columns:
                if col1 < col2:
                    features[f"corr_{col1}_{col2}"] = corr_matrix.loc[col1, col2]
        
        return features

# TODO: Implement! Then test:
# extractor = TemporalFeatureExtractor(window_size_s=300, step_size_s=60)
# windows = extractor.create_windows(df_cleaned)
# features_list = [extractor.extract_features(w['data']) for w in windows]
```

---

### Day 3-4: Normalization (ML Design Patterns)

**Read**: ML Design Patterns Chapter 2 (Representation - Normalization)

**Critical principle**: 
- Fit scaler on TRAIN only
- Apply same to VAL/TEST
- Save scaler with model

**Assignment 3.2: Implement ScalingPipeline**

File: `services/gnn_service/src/data/preprocessing.py`

```python
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

class ScalingPipeline:
    """
    Fit-once, transform-many pattern (ML Design Patterns).
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_on_train(self, X_train: np.ndarray):
        """
        FIT on TRAIN data only.
        """
        self.scaler.fit(X_train)
        self.is_fitted = True
        print(f"Scaler fitted on {len(X_train)} samples")
        print(f"  Mean: {self.scaler.mean_[:5]}...")
        print(f"  Std:  {self.scaler.scale_[:5]}...")
    
    def transform_all(self, X_train, X_val, X_test):
        """
        Apply SAME scaler to all splits.
        """
        assert self.is_fitted, "Fit on train first!"
        return (
            self.scaler.transform(X_train),
            self.scaler.transform(X_val),
            self.scaler.transform(X_test),
        )
    
    def save(self, filepath):
        """Save for production serving."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved: {filepath}")
```

---

### Day 5: Integration

**Assignment 3.3: Full pipeline integration**

File: `services/gnn_service/scripts/week3_full_pipeline.py`

```python
# Load → Clean → Window → Scale

from pathlib import Path
from src.data.loaders import UCIHydraulicLoader, DataCleaner
from src.data.feature_engineering import TemporalFeatureExtractor
from src.data.preprocessing import ScalingPipeline

# 1. Load
loader = UCIHydraulicLoader(Path("services/gnn_service/data/raw_real_dataset"))
raw = loader.load_all_sensors()
df = loader.align_to_common_timeline()

# 2. Clean
cleaner = DataCleaner(df)
cleaner.handle_missing(method='forward_fill')
df_cleaned = cleaner.df

# 3. Window
extractor = TemporalFeatureExtractor()
windows = extractor.create_windows(df_cleaned)
features_list = [extractor.extract_features(w['data']) for w in windows]

# Convert to DataFrame
df_features = pd.DataFrame(features_list)
print(f"Features shape: {df_features.shape}")
print(f"Columns: {df_features.columns.tolist()[:10]}...")  # first 10

# 4. Scale (later, after split)
print("\n✅ Week 3 pipeline complete")
df_features.to_pickle("week3_features.pkl")
```

**Deliverable**: `week3_features.pkl`, verified shape/columns

---

## Week 4: Reproducible Splits & Versioning

### Day 1-2: Train/Val/Test Splitting (ML Design Patterns + Huyen)

**Read**: ML Design Patterns Chapter 2-4, Huyen Chapter 3 (Data leakage)

**Assignment 4.1: Implement TemporalDataSplitter**

File: `services/gnn_service/src/data/splitting.py`

```python
import numpy as np
import pandas as pd

class TemporalDataSplitter:
    """
    Split time series respecting temporal order.
    
    NO RANDOM SHUFFLE (breaks temporal dependencies)
    NO FUTURE DATA TO TRAIN
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
        """Create temporal splits (past → train, next → val, last → test)."""
        n = len(self.df)
        
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)
        
        self.split_indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        
        return {
            'train': self.df.iloc[train_idx],
            'val': self.df.iloc[val_idx],
            'test': self.df.iloc[test_idx],
        }
    
    def verify_no_overlap(self):
        """Check no overlap between splits."""
        train = set(self.split_indices['train'])
        val = set(self.split_indices['val'])
        test = set(self.split_indices['test'])
        
        assert len(train & val) == 0
        assert len(train & test) == 0
        assert len(val & test) == 0
        print("✅ No overlaps")
    
    def save_indices(self, filepath):
        """Save split indices (reproducibility)."""
        np.savez(
            filepath,
            train_idx=self.split_indices['train'],
            val_idx=self.split_indices['val'],
            test_idx=self.split_indices['test'],
        )
        print(f"Saved to {filepath}")
```

---

### Day 3: Data Leakage Detection

**Assignment 4.2: Implement leak detection**

File: `services/gnn_service/src/data/validation.py`

```python
import numpy as np
from scipy.stats import entropy

class DataLeakageDetector:
    
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
        print("✅ Temporal order verified")
    
    @staticmethod
    def check_label_distribution(y_train, y_val, y_test):
        """
        Are class distributions similar?
        """
        train_dist = np.bincount(y_train) / len(y_train)
        val_dist = np.bincount(y_val, minlength=len(train_dist)) / len(y_val)
        test_dist = np.bincount(y_test, minlength=len(train_dist)) / len(y_test)
        
        kl_train_val = entropy(train_dist + 1e-10, val_dist + 1e-10)
        kl_val_test = entropy(val_dist + 1e-10, test_dist + 1e-10)
        
        print(f"KL(train || val): {kl_train_val:.4f}")
        print(f"KL(val || test): {kl_val_test:.4f}")
        
        if kl_train_val > 0.5 or kl_val_test > 0.5:
            print("⚠️ WARNING: Class distribution shift")
```

---

### Day 4: Metadata & Versioning

**Read**: ML Design Patterns Chapter 5 (Data versioning)

**Assignment 4.3: Create data manifest**

File: `services/gnn_service/scripts/week4_manifest.py`

```python
import json
from datetime import datetime
import hashlib

def create_data_manifest(X_train, X_val, X_test, split_info):
    """
    Document EVERYTHING about data splits.
    """
    manifest = {
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'source': 'UCI Hydraulic Dataset',
        
        'preprocessing': {
            'window_size_s': 300,
            'window_step_s': 60,
            'scaling': 'StandardScaler (fit on train)',
            'missing_handling': 'forward_fill',
        },
        
        'splits': {
            'train': {'n_samples': len(X_train)},
            'val': {'n_samples': len(X_val)},
            'test': {'n_samples': len(X_test)},
        },
        
        'quality_checks': {
            'no_data_leakage': True,
            'temporal_order_verified': True,
            'scaling_fit_on_train': True,
        },
    }
    
    return manifest

# Save
manifest = create_data_manifest(X_train, X_val, X_test, split_info)
with open('manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
```

---

### Day 5: Final Integration & Review

**Assignment 4.4: Full week 4 pipeline**

File: `services/gnn_service/scripts/week4_full_pipeline.py`

```python
# Load → Clean → Window → Scale → SPLIT → Verify → Save

from pathlib import Path
from src.data.loaders import UCIHydraulicLoader, DataCleaner
from src.data.feature_engineering import TemporalFeatureExtractor
from src.data.preprocessing import ScalingPipeline
from src.data.splitting import TemporalDataSplitter
from src.data.validation import DataLeakageDetector
import pandas as pd
import numpy as np

# ... (steps 1-3 same as week 3) ...

# 4. SPLIT (NO LEAKAGE)
splitter = TemporalDataSplitter(df_features, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
splits = splitter.split()
X_train, X_val, X_test = splits['train'], splits['val'], splits['test']

splitter.verify_no_overlap()

# 5. SCALE (fit on train only)
scaler = ScalingPipeline()
scaler.fit_on_train(X_train.values)
X_train_scaled, X_val_scaled, X_test_scaled = scaler.transform_all(
    X_train.values, X_val.values, X_test.values
)

# 6. VERIFY
detector = DataLeakageDetector()
detector.check_temporal_order(splitter.split_indices)

# 7. SAVE
scaler.save('data/processed/scaler.pkl')
np.savez('data/processed/split_indices.npz', 
         train_idx=splitter.split_indices['train'],
         val_idx=splitter.split_indices['val'],
         test_idx=splitter.split_indices['test'])

np.savez('data/processed/train_val_test.npz',
         X_train=X_train_scaled,
         X_val=X_val_scaled,
         X_test=X_test_scaled)

print("✅ All splits saved and verified")
```

**Deliverable**: `data/processed/{train_val_test.npz, scaler.pkl, split_indices.npz, manifest.json}`

---

## Final Checklist

### Code Quality
- [ ] All code has docstrings
- [ ] All transformations logged
- [ ] No magic numbers (use constants)
- [ ] Unit tests for loaders, cleaners, splitters

### Data Quality
- [ ] Missing data handled
- [ ] Outliers marked (not removed)
- [ ] No data leakage
- [ ] Train/val/test don't overlap
- [ ] Same scaler applied to all

### Reproducibility
- [ ] File hashes saved
- [ ] Split indices saved
- [ ] Scaler saved
- [ ] Manifest.json created
- [ ] All code in git

### Documentation
- [ ] README in `data/processed/`
- [ ] Manifest with provenance
- [ ] Weekly notes with findings
- [ ] Known issues documented

---

## Time Budget (32-40 hours)

| Week | Task | Hours |
|------|------|-------|
| 1 | Reading + data exploration | 8-10 |
| 2 | Loading + cleaning | 8-10 |
| 3 | Windowing + scaling | 8-10 |
| 4 | Splitting + versioning | 8-10 |
| Total | | 32-40 |

---

## Success Criteria

✅ **Functional**: Data pipeline loads, cleans, windows, scales, splits  
✅ **Reproducible**: Same result every run (no randomness)  
✅ **No leakage**: Train/val/test independent, scaler fit once  
✅ **Documented**: Every step logged, manifest created  
✅ **Committed**: All code and data versions in git  

---

## Resources

- **Chip Huyen**: [Designing ML Systems](https://huyenchip.com/machine-learning-systems-design/)
- **ML Design Patterns**: [GitHub repo](https://github.com/GoogleCloudPlatform/ml-design-patterns)
- **UCI Dataset**: [Hydraulic Systems](https://archive.ics.uci.edu/ml/datasets/Hydraulic+Systems+Predictive+Maintenance+and+Condition+Monitoring)
- **Géron**: Hands-On ML with Scikit-Learn, Keras, TensorFlow (O'Reilly)
- **Kuhn & Johnson**: Feature Engineering and Selection (Routledge)

---

**Status**: Ready for implementation  
**Start**: Now (Jan 3-31, 2026)  
**Output**: Production-ready train/val/test splits for GNN training
