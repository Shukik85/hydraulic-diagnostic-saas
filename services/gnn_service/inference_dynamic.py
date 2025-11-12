# services/gnn_service/inference_dynamic.py
"""
Dynamic inference engine для UniversalDynamicGNN.
"""
import torch
import logging
from typing import Dict, List, Any
import json
from model_dynamic_gnn import UniversalDynamicGNN, create_model
from schemas import EquipmentMetadata

logger = logging.getLogger(__name__)

class DynamicGNNInference:
    def __init__(self, model_path: str, metadata_path: str, device: str = "cuda"):
        self.device = device
        with open(metadata_path) as f:
            metadata = EquipmentMetadata(**json.load(f))
        self.metadata = metadata
        self.model = create_model(metadata, device=device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        logger.info(f"Model loaded from {model_path}, device: {device}")

    def predict(self, component_features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # Assumes features are in shape [B, T, F] per component
        health, degradation, attn = self.model(component_features, return_attention=True)
        return {
            "health": health.detach().cpu().numpy().tolist(),
            "degradation": degradation.detach().cpu().numpy().tolist(),
            "attention": attn.detach().cpu().numpy() if attn is not None else None,
        }
