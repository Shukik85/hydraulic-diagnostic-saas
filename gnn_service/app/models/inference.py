"""Inference engine с explainability."""

import torch
import structlog
from torch_geometric.data import Data, Batch

from app.config import get_settings
from app.models.tgat import TGAT
from app.models.explainer import AttentionExplainer

logger = structlog.get_logger(__name__)
settings = get_settings()


class InferenceEngine:
    """Inference engine для T-GAT model."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.model: TGAT | None = None
        self.explainer: AttentionExplainer | None = None
        self.model_loaded = False
        
        logger.info("InferenceEngine initialized", device=self.device)
    
    async def load_model(self) -> None:
        """Load T-GAT model from checkpoint."""
        try:
            # Initialize model
            self.model = TGAT(
                input_dim=10,  # TODO: Из config или checkpoint metadata
                hidden_dim=settings.hidden_dim,
                num_layers=settings.num_layers,
                num_heads=settings.num_heads,
                dropout=settings.dropout,
                num_classes=2,
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize explainer
            self.explainer = AttentionExplainer(
                attention_threshold=settings.attention_threshold,
            )
            
            self.model_loaded = True
            logger.info("Model loaded successfully", device=self.device)
            
        except FileNotFoundError:
            logger.warning(
                "Model checkpoint not found, using random weights for dev",
                path=self.model_path,
            )
            # Dev mode: random weights
            self.model = TGAT(
                input_dim=10,
                hidden_dim=settings.hidden_dim,
                num_layers=settings.num_layers,
                num_heads=settings.num_heads,
                dropout=settings.dropout,
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.explainer = AttentionExplainer()
            self.model_loaded = True
        
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise
    
    async def predict(
        self,
        node_features: list[list[float]],
        edge_index: list[list[int]],
        edge_attr: list[list[float]] | None = None,
        component_names: list[str] | None = None,
    ) -> dict:
        """Single equipment prediction.
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge connectivity
            edge_attr: Edge attributes
            component_names: Component names для explainability
        
        Returns:
            Prediction dict with explainability
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Convert to PyG Data
        data = self._build_graph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data = data.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits, attention_weights_list = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
            )
            
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred].item()
            anomaly_score = probs[0, 1].item()  # Probability of anomaly class
        
        # Explainability (if anomaly detected)
        explanation = None
        if pred == 1 and self.explainer is not None:
            explanation = self.explainer.explain(
                attention_weights_list=attention_weights_list,
                node_features=node_features,
                edge_index=edge_index,
                component_names=component_names or [f"node_{i}" for i in range(len(node_features))],
            )
        
        return {
            "prediction": pred,
            "probability": confidence,
            "anomaly_score": anomaly_score,
            "explanation": explanation,
        }
    
    async def batch_predict(
        self,
        graphs: list[dict],
    ) -> list[dict]:
        """Batch inference для fleet.
        
        Args:
            graphs: List of graph dicts
        
        Returns:
            List of predictions
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Build batch
        data_list = [
            self._build_graph(
                node_features=g["node_features"],
                edge_index=g["edge_index"],
                edge_attr=g.get("edge_attr"),
            )
            for g in graphs
        ]
        
        batch = Batch.from_data_list(data_list).to(self.device)
        
        # Batch inference
        with torch.no_grad():
            logits, _ = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr if hasattr(batch, "edge_attr") else None,
                batch=batch.batch,
            )
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        
        # Build results
        results = []
        for i, g in enumerate(graphs):
            results.append({
                "equipment_id": g.get("equipment_id", f"equipment_{i}"),
                "prediction": preds[i].item(),
                "probability": probs[i, preds[i]].item(),
                "anomaly_score": probs[i, 1].item(),
            })
        
        return results
    
    def _build_graph(
        self,
        node_features: list[list[float]],
        edge_index: list[list[int]],
        edge_attr: list[list[float]] | None = None,
    ) -> Data:
        """Build PyG Data object."""
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        if edge_attr is not None:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            return Data(x=x, edge_index=edge_index)
