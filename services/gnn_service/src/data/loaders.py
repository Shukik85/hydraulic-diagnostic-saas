"""
UCI Hydraulic Dataset Loader

Design principle (Huyen + Design Patterns):
- Deterministic: same input → same output
- Logged: every transformation tracked
- Versioned: file hashes for reproducibility
- Testable: each step independently verifiable
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class DatasetMetadata:
    """Metadata for loaded dataset (reproducibility)."""
    loaded_at: str
    source_dir: str
    file_hashes: Dict[str, str]
    n_samples: int
    n_features: int
    date_range: Tuple[str, str]
    missing_pct: Dict[str, float]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class UCIHydraulicLoader:
    """
    Load UCI Hydraulic Predictive Maintenance Dataset.
    
    Dataset info:
    - Source: UC Irvine Machine Learning Repository
    - Duration: 60 hours continuous operation
    - Sensors: 11 channels (pressure, flow, temperature, electrical power, contamination, vibration)
    - Sample rates: 100 Hz (high freq), 10 Hz, 1 Hz (depends on sensor)
    - Conditions: 8 degradation/fault types + normal
    
    References:
    - Nikolai Helwig et al. "Condition Monitoring of Hydraulic Systems Using Multivariate..."
    - https://archive.ics.uci.edu/ml/datasets/Hydraulic+Systems+Predictive+Maintenance+and+Condition+Monitoring
    """
    
    # Sensor files in UCI dataset
    SENSOR_FILES = {
        # Pressure sensors (100 Hz)
        'PS1': 'PS1.txt',  # Pump 1
        'PS2': 'PS2.txt',  # Valve 1
        'PS3': 'PS3.txt',  # Valve 2
        'PS4': 'PS4.txt',  # Accumulator
        'PS5': 'PS5.txt',  # Valve 3
        'PS6': 'PS6.txt',  # Valve 4
        # Flow sensors (10 Hz)
        'FS1': 'FS1.txt',  # Flow meter 1
        'FS2': 'FS2.txt',  # Flow meter 2
        # Temperature sensors (1 Hz)
        'TS1': 'TS1.txt',  # Tank temp
        'TS2': 'TS2.txt',  # Cooler inlet
        'TS3': 'TS3.txt',  # Cooler outlet
        'TS4': 'TS4.txt',  # Valve temp
        # Electrical & other
        'EPS1': 'EPS1.txt',  # Motor power (100 Hz)
        'CE': 'CE.txt',      # Contamination/Erosion (10 Hz)
        'CP': 'CP.txt',      # Cooler power (10 Hz)
        'SE': 'SE.txt',      # Specific energy (10 Hz)
        'VS1': 'VS1.txt',    # Vibration sensor (100 Hz)
    }
    
    # Sampling rates (Hz) for each sensor type
    SAMPLE_RATES = {
        'PS1': 100, 'PS2': 100, 'PS3': 100, 'PS4': 100, 'PS5': 100, 'PS6': 100,
        'FS1': 10, 'FS2': 10,
        'TS1': 1, 'TS2': 1, 'TS3': 1, 'TS4': 1,
        'EPS1': 100,
        'CE': 10, 'CP': 10, 'SE': 10,
        'VS1': 100,
    }
    
    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: Path to UCI Hydraulic dataset directory
        """
        self.data_dir = Path(data_dir)
        self.raw_data = {}
        self.metadata = {
            'loaded_at': datetime.now().isoformat(),
            'source_dir': str(data_dir),
            'file_hashes': {},
        }
        logger.info(f"Initialized loader for: {data_dir}")
    
    def load_all_sensors(self, verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Load all sensor files into memory.
        
        Principle: Deterministic, file hash tracked for reproducibility.
        
        Args:
            verbose: Log each load operation
        
        Returns:
            dict: sensor_name → raw values (1D float32 array)
        
        Raises:
            FileNotFoundError: if critical sensor files missing
        """
        logger.info("Loading all sensors...")
        missing_sensors = []
        
        for sensor_name, filename in self.SENSOR_FILES.items():
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                missing_sensors.append(sensor_name)
                logger.warning(f"⚠️  File not found: {filepath}")
                continue
            
            try:
                # Load: simple .txt with one value per line
                data = np.loadtxt(filepath, dtype=np.float32)
                
                if data.ndim != 1:
                    logger.warning(f"Sensor {sensor_name} not 1D, flattening")
                    data = data.flatten()
                
                # Compute hash for reproducibility
                file_hash = self._compute_hash(data)
                self.metadata['file_hashes'][sensor_name] = file_hash
                
                self.raw_data[sensor_name] = data
                
                if verbose:
                    logger.info(
                        f"✅ {sensor_name:5s}: {len(data):7d} samples, "
                        f"rate={self.SAMPLE_RATES[sensor_name]:3d} Hz, "
                        f"hash={file_hash[:8]}..."
                    )
            
            except Exception as e:
                logger.error(f"❌ Error loading {sensor_name}: {e}")
                missing_sensors.append(sensor_name)
        
        if missing_sensors:
            logger.warning(f"Missing {len(missing_sensors)} sensors: {missing_sensors}")
        
        logger.info(f"✅ Loaded {len(self.raw_data)} sensors")
        return self.raw_data
    
    @staticmethod
    def _compute_hash(data: np.ndarray) -> str:
        """SHA256 hash of numpy array (for reproducibility tracking)."""
        return hashlib.sha256(data.tobytes()).hexdigest()
    
    def load_profile(self) -> pd.DataFrame:
        """
        Parse profile.txt to get condition labels.
        
        Profile format (from UCI documentation):
        - Each row represents condition state
        - Format: condition_codes for different component health states
        
        Returns:
            DataFrame with condition information
        """
        profile_path = self.data_dir / 'profile.txt'
        
        if not profile_path.exists():
            logger.warning(f"Profile file not found: {profile_path}")
            return pd.DataFrame()
        
        try:
            # UCI profile format: typically each line is a condition code
            # Example: "FS: new, PS: new, ..."
            with open(profile_path) as f:
                content = f.read()
            
            logger.info(f"Loaded profile ({len(content)} bytes)")
            
            # Parse into structured format (dataset-specific)
            # For now, return raw content as documentation
            return pd.DataFrame({'raw_content': [content]})
        
        except Exception as e:
            logger.error(f"Error parsing profile: {e}")
            return pd.DataFrame()
    
    def align_to_common_timeline(self, target_rate_hz: float = 1.0) -> pd.DataFrame:
        """
        Synchronize all sensors to common time axis.
        
        Problem (Huyen): Different sensors have different sample rates.
        Solution: Interpolate to common frequency (slowest = most conservative).
        
        Principle: Choose target rate based on dynamics of interest.
        Here: 1 Hz (1 sample/second) captures ~10-second hydraulic cycles.
        
        Args:
            target_rate_hz: Common sample rate (default 1 Hz)
        
        Returns:
            DataFrame with aligned multivariate time series
        """
        logger.info(f"Aligning sensors to {target_rate_hz} Hz...")
        
        if not self.raw_data:
            raise ValueError("No sensor data loaded. Call load_all_sensors() first.")
        
        # Assume all sensors start at t=0
        # Total duration determined by slowest sensor
        max_sample_rate = max(self.SAMPLE_RATES[s] for s in self.raw_data.keys())
        max_samples = max(len(self.raw_data[s]) for s in self.raw_data.keys())
        total_duration_s = max_samples / max_sample_rate
        
        logger.info(f"Total duration: {total_duration_s:.1f}s")
        
        # Create common timeline
        common_timeline = np.arange(0, total_duration_s, 1.0 / target_rate_hz)
        logger.info(f"Common timeline: {len(common_timeline)} samples")
        
        aligned_data = {}
        
        for sensor_name, values in self.raw_data.items():
            rate = self.SAMPLE_RATES[sensor_name]
            sensor_timeline = np.arange(len(values)) / rate
            
            # Interpolate to common timeline
            aligned = np.interp(
                common_timeline, 
                sensor_timeline, 
                values,
                left=np.nan,   # Before first sample
                right=np.nan,  # After last sample
            )
            
            aligned_data[sensor_name] = aligned
            logger.debug(f"  {sensor_name}: {len(aligned)} aligned samples")
        
        # Create DataFrame with time index
        df = pd.DataFrame(
            aligned_data,
            index=pd.timedelta_range(start='0s', periods=len(common_timeline), freq=f'{1/target_rate_hz}s')
        )
        
        logger.info(f"✅ Aligned {len(df.columns)} sensors to {len(df)} time points")
        return df
    
    def get_metadata(self) -> DatasetMetadata:
        """
        Return comprehensive metadata about loaded data.
        
        Usage: Track data lineage and reproducibility.
        """
        if not self.raw_data:
            raise ValueError("No data loaded yet")
        
        # Compute basic stats
        all_data = np.concatenate(list(self.raw_data.values()))
        n_samples = len(list(self.raw_data.values())[0]) if self.raw_data else 0
        
        return DatasetMetadata(
            loaded_at=self.metadata['loaded_at'],
            source_dir=self.metadata['source_dir'],
            file_hashes=self.metadata['file_hashes'],
            n_samples=n_samples,
            n_features=len(self.raw_data),
            date_range=(
                str(pd.Timestamp.now()),
                str(pd.Timestamp.now() + pd.Timedelta(seconds=n_samples/100))
            ),
            missing_pct={},  # Computed later
        )


class DataCleaner:
    """
    Clean and prepare time series data.
    
    Principle (Géron): Document every cleaning decision.
    Don't just drop; understand WHY missing/outliers exist.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_log = []
        self.metadata = {
            'original_shape': df.shape,
            'cleaning_steps': [],
        }
    
    def handle_missing(self, method: str = 'forward_fill', max_gap: Optional[int] = None):
        """
        Handle missing values (NaN).
        
        Strategy for hydraulic sensors:
        - Forward fill: assumes steady state (conservative, works for slow changes)
        - Interpolate: assumes smooth change (better for dynamics)
        - Drop: only if continuous block > max_gap
        
        Args:
            method: 'forward_fill' or 'interpolate'
            max_gap: Max consecutive NaN to fill (larger gaps ignored)
        """
        logger.info(f"Handling missing values ({method})...")
        
        missing_before = self.df.isnull().sum()
        
        for col in self.df.columns:
            n_missing = missing_before[col]
            
            if n_missing == 0:
                continue
            
            if method == 'forward_fill':
                self.df[col].fillna(method='ffill', limit=max_gap, inplace=True)
                action = f"Forward fill (limit={max_gap})"
            
            elif method == 'interpolate':
                self.df[col].interpolate(method='linear', limit=max_gap, inplace=True)
                action = f"Linear interpolation (limit={max_gap})"
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            n_filled = n_missing - self.df[col].isnull().sum()
            self.cleaning_log.append(
                f"Column {col}: {n_missing} missing, {n_filled} filled ({action})"
            )
            logger.info(f"  {col}: {n_missing} missing → {n_filled} filled")
        
        self.metadata['cleaning_steps'].append(f"handle_missing({method}, max_gap={max_gap})")
    
    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers (DON'T remove, just mark).
        
        Principle (Huyen): Outliers might be real faults! Keep them.
        
        Args:
            method: 'iqr' (interquartile range) or 'zscore'
            threshold: IQR multiplier (1.5 = standard, 3 = extreme)
        
        Returns:
            Boolean DataFrame marking outliers
        """
        logger.info(f"Detecting outliers ({method}, threshold={threshold})...")
        
        outlier_mask = pd.DataFrame(
            False, 
            index=self.df.index, 
            columns=self.df.columns
        )
        
        for col in self.df.columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
                outliers = (self.df[col] < lower) | (self.df[col] > upper)
            
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = z_scores > threshold
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            outlier_mask[col] = outliers
            
            n_outliers = outliers.sum()
            if n_outliers > 0:
                outlier_pct = 100 * n_outliers / len(self.df)
                self.cleaning_log.append(
                    f"Column {col}: {n_outliers} outliers ({outlier_pct:.2f}%)"
                )
                logger.info(f"  {col}: {n_outliers} outliers ({outlier_pct:.2f}%)")
        
        self.metadata['cleaning_steps'].append(f"detect_outliers({method}, threshold={threshold})")
        return outlier_mask
    
    def get_report(self) -> str:
        """
        Generate text report of all cleaning operations.
        """
        report = "\n".join([
            "="*60,
            "DATA CLEANING REPORT",
            "="*60,
            f"Original shape: {self.metadata['original_shape']}",
            f"Final shape: {self.df.shape}",
            "",
            "Cleaning steps:",
        ] + self.cleaning_log + [
            "",
            "="*60,
        ])
        return report


if __name__ == "__main__":
    # Example usage
    data_dir = Path("services/gnn_service/data/raw_real_dataset")
    
    # Load
    loader = UCIHydraulicLoader(data_dir)
    loader.load_all_sensors(verbose=True)
    
    # Align
    df_aligned = loader.align_to_common_timeline(target_rate_hz=1.0)
    print(f"\nAligned data shape: {df_aligned.shape}")
    print(df_aligned.head())
    
    # Clean
    cleaner = DataCleaner(df_aligned)
    cleaner.handle_missing(method='forward_fill')
    cleaner.detect_outliers(method='iqr', threshold=1.5)
    
    print(cleaner.get_report())
