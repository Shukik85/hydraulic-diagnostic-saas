"""Model Versioning для A/B testing и rollback"""

from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Model version metadata"""
    model_type: str  # 'gnn' or 'rag'
    version: str
    is_champion: bool = False
    traffic_percentage: int = 0
    model_path: str = ""
    config: Dict = field(default_factory=dict)
    
    # Performance metrics
    avg_inference_time_ms: Optional[float] = None
    avg_confidence: Optional[float] = None
    error_rate: float = 0.0
    
    deployed_at: Optional[datetime] = None

class ModelRegistry:
    """Централизованный реестр моделей"""
    
    def __init__(self):
        self._versions: Dict[str, List[ModelVersion]] = {
            'gnn': [],
            'rag': []
        }
        self._init_default_models()
    
    def _init_default_models(self):
        """Регистрация текущих production моделей"""
        
        # GNN model
        self.register_version(ModelVersion(
            model_type='gnn',
            version='v2.1.0',
            is_champion=True,
            traffic_percentage=100,
            model_path='/models/gnn_v2.1.0.pt',
            config={
                'architecture': 'GraphSAGE',
                'hidden_dim': 128,
                'num_layers': 3,
                'torch_compile': True
            }
        ))
        
        # RAG model
        self.register_version(ModelVersion(
            model_type='rag',
            version='gpt-4-turbo-2024-04-09',
            is_champion=True,
            traffic_percentage=100,
            model_path='openai://gpt-4-turbo-2024-04-09',
            config={
                'temperature': 0.3,
                'max_tokens': 2000,
                'language': 'russian'
            }
        ))
    
    def register_version(self, version: ModelVersion):
        """Зарегистрировать новую версию модели"""
        version.deployed_at = datetime.utcnow()
        self._versions[version.model_type].append(version)
        logger.info(f"Registered {version.model_type} model version {version.version}")
    
    def get_champion(self, model_type: str) -> ModelVersion:
        """Получить текущую production модель"""
        versions = self._versions.get(model_type, [])
        for v in versions:
            if v.is_champion:
                return v
        return versions[-1] if versions else None
    
    def get_version_for_request(self, model_type: str, user_id: str = None) -> ModelVersion:
        """A/B testing: выбрать версию модели для запроса"""
        versions = self._versions.get(model_type, [])
        active_variants = [v for v in versions if v.traffic_percentage > 0]
        
        if len(active_variants) == 1:
            return active_variants[0]
        
        # A/B testing: распределение по traffic_percentage
        import random
        if user_id:
            random.seed(hash(user_id))
        
        roll = random.randint(1, 100)
        cumulative = 0
        
        for variant in active_variants:
            cumulative += variant.traffic_percentage
            if roll <= cumulative:
                return variant
        
        return self.get_champion(model_type)
    
    def update_metrics(
        self,
        model_type: str,
        version: str,
        inference_time_ms: float,
        confidence: float,
        error: bool = False
    ):
        """Обновить метрики модели"""
        versions = self._versions.get(model_type, [])
        
        for v in versions:
            if v.version == version:
                if v.avg_inference_time_ms is None:
                    v.avg_inference_time_ms = inference_time_ms
                else:
                    v.avg_inference_time_ms = v.avg_inference_time_ms * 0.9 + inference_time_ms * 0.1
                
                if v.avg_confidence is None:
                    v.avg_confidence = confidence
                else:
                    v.avg_confidence = v.avg_confidence * 0.9 + confidence * 0.1
                
                if error:
                    v.error_rate = v.error_rate * 0.99 + 0.01
                else:
                    v.error_rate = v.error_rate * 0.99
                break
    
    def promote_to_champion(self, model_type: str, version: str):
        """Промоутить версию в production"""
        versions = self._versions.get(model_type, [])
        
        for v in versions:
            v.is_champion = False
        
        for v in versions:
            if v.version == version:
                v.is_champion = True
                v.traffic_percentage = 100
                logger.info(f"Promoted {model_type}/{version} to champion")
                return True
        
        return False
    
    def list_versions(self, model_type: str) -> List[ModelVersion]:
        """Список всех версий модели"""
        return self._versions.get(model_type, [])

# Singleton
model_registry = ModelRegistry()
