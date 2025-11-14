"""Model Drift Detection"""

from typing import List, Dict
from collections import deque
from datetime import datetime
import numpy as np
from scipy import stats
from prometheus_client import Gauge
import logging

logger = logging.getLogger(__name__)

model_drift_score = Gauge(
    'model_drift_score',
    'Model drift score (0-1)',
    ['model_type', 'model_version']
)

class DriftDetector:
    """Детектор drift для моделей"""
    
    def __init__(
        self,
        model_type: str,
        model_version: str,
        reference_window: int = 10000,
        detection_window: int = 1000
    ):
        self.model_type = model_type
        self.model_version = model_version
        self.reference_window = reference_window
        self.detection_window = detection_window
        
        self.reference_predictions = deque(maxlen=reference_window)
        self.reference_confidences = deque(maxlen=reference_window)
        self.current_predictions = deque(maxlen=detection_window)
        self.current_confidences = deque(maxlen=detection_window)
        self.drift_history = deque(maxlen=1000)
        
        self._initialized = False
    
    def add_reference_sample(self, prediction: float, confidence: float):
        """Добавить sample в reference distribution"""
        self.reference_predictions.append(prediction)
        self.reference_confidences.append(confidence)
        
        if len(self.reference_predictions) >= self.reference_window:
            self._initialized = True
            logger.info(f"Drift detector initialized for {self.model_type}/{self.model_version}")
    
    def add_production_sample(self, prediction: float, confidence: float):
        """Добавить production sample"""
        if not self._initialized:
            self.add_reference_sample(prediction, confidence)
            return
        
        self.current_predictions.append(prediction)
        self.current_confidences.append(confidence)
        
        if len(self.current_predictions) >= self.detection_window:
            drift_score = self.detect_drift()
            if drift_score > 0.3:
                logger.warning(f"Drift detected for {self.model_type}/{self.model_version}: {drift_score:.3f}")
    
    def detect_drift(self) -> float:
        """Детектировать drift"""
        if not self._initialized or len(self.current_predictions) < self.detection_window:
            return 0.0
        
        ref_preds = list(self.reference_predictions)
        cur_preds = list(self.current_predictions)
        
        # Statistical drift: KS test
        ks_statistic, ks_pvalue = stats.ks_2samp(ref_preds, cur_preds)
        statistical_drift = 1 - ks_pvalue
        
        # Performance drift
        ref_conf_mean = np.mean(list(self.reference_confidences))
        cur_conf_mean = np.mean(list(self.current_confidences))
        confidence_drop = max(0, ref_conf_mean - cur_conf_mean)
        performance_drift = min(confidence_drop / 0.2, 1.0)
        
        # Distribution shift
        ref_mean = np.mean(ref_preds)
        cur_mean = np.mean(cur_preds)
        mean_shift = abs(cur_mean - ref_mean)
        
        ref_std = np.std(ref_preds)
        cur_std = np.std(cur_preds)
        variance_shift = abs(cur_std - ref_std)
        
        distribution_drift = min((mean_shift + variance_shift) / 0.5, 1.0)
        
        # Combined score
        drift_score = (
            statistical_drift * 0.4 +
            performance_drift * 0.3 +
            distribution_drift * 0.3
        )
        
        model_drift_score.labels(
            model_type=self.model_type,
            model_version=self.model_version
        ).set(drift_score)
        
        self.drift_history.append({
            'timestamp': datetime.utcnow(),
            'drift_score': drift_score,
            'statistical_drift': statistical_drift,
            'performance_drift': performance_drift,
            'distribution_drift': distribution_drift,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue
        })
        
        return drift_score
    
    def get_drift_report(self) -> Dict:
        """Получить отчёт по drift"""
        if not self.drift_history:
            return {'status': 'no_data'}
        
        recent_drifts = list(self.drift_history)[-100:]
        current_drift = recent_drifts[-1]['drift_score']
        
        return {
            'status': 'active',
            'model_type': self.model_type,
            'model_version': self.model_version,
            'current_drift_score': current_drift,
            'avg_drift_7d': np.mean([d['drift_score'] for d in recent_drifts]),
            'max_drift_7d': max([d['drift_score'] for d in recent_drifts]),
            'alert_threshold': 0.3,
            'alert_triggered': current_drift > 0.3,
            'recommendation': self._get_recommendation(current_drift)
        }
    
    def _get_recommendation(self, drift_score: float) -> str:
        if drift_score < 0.1:
            return "Модель работает стабильно."
        elif drift_score < 0.3:
            return "Небольшой drift. Мониторим."
        elif drift_score < 0.5:
            return "Умеренный drift. Рекомендуется переобучение."
        else:
            return "Критический drift! Требуется срочное переобучение."

_drift_detectors: Dict[str, DriftDetector] = {}

def get_drift_detector(model_type: str, model_version: str) -> DriftDetector:
    """Получить drift detector"""
    key = f"{model_type}_{model_version}"
    if key not in _drift_detectors:
        _drift_detectors[key] = DriftDetector(model_type, model_version)
    return _drift_detectors[key]
