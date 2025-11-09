"""Production inference for GNN anomaly detection."""
import torch
import numpy as np
from typing import Dict, List, Tuple
from torch_geometric.data import Data

from .model import GNNClassifier
from .config import config


class GNNInference:
    """Production inference wrapper."""
    
    def __init__(self, model_path: str):
        """Load trained model.
        
        Args:
            model_path: Path to trained model checkpoint
        """
        self.device = torch.device(config.device)
        
        # Load model
        self.model = GNNClassifier()
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle PyTorch Lightning checkpoint
        if "state_dict" in checkpoint:
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
    
    @torch.no_grad()
    def predict(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: np.ndarray = None,
    ) -> Dict[str, any]:
        """Run inference on a single graph.
        
        Args:
            node_features: Node feature matrix [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
        
        Returns:
            Dictionary with:
                - prediction: 0 (normal) or 1 (anomaly)
                - probability: Confidence score [0, 1]
                - node_embeddings: Node embeddings for explainability
                - attention_weights: Attention weights for each layer
        """
        # Convert to PyG Data
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        
        if edge_attr is not None:
            edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32).to(self.device)
        else:
            edge_attr_t = None
        
        data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
        
        # Forward pass
        logits, embeddings, attention_weights = self.model(data)
        
        # Get prediction
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()
        
        return {
            "prediction": pred_class,
            "probability": confidence,
            "anomaly_score": probs[0, 1].item(),  # P(anomaly)
            "node_embeddings": embeddings.cpu().numpy(),
            "attention_weights": [
                (edge_idx.cpu().numpy(), alpha.cpu().numpy())
                for edge_idx, alpha in attention_weights
            ],
        }
    
    def batch_predict(
        self,
        graphs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> List[Dict[str, any]]:
        """Run inference on multiple graphs.
        
        Args:
            graphs: List of (node_features, edge_index, edge_attr) tuples
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for node_features, edge_index, edge_attr in graphs:
            result = self.predict(node_features, edge_index, edge_attr)
            results.append(result)
        return results
    
    def explain_prediction(
        self,
        result: Dict[str, any],
        component_names: List[str],
    ) -> Dict[str, any]:
        """Generate explainability insights.
        
        Args:
            result: Prediction result from predict()
            component_names: List of component names (ordered by node index)
        
        Returns:
            Explanation dictionary with:
                - critical_components: Components with highest attention
                - suspicious_connections: Edges with high attention
        """
        # Analyze attention weights (last layer)
        edge_index, attention = result["attention_weights"][-1]
        
        # Find top-k critical edges
        top_k = 5
        top_indices = np.argsort(attention.flatten())[-top_k:][::-1]
        
        critical_connections = []
        for idx in top_indices:
            src, dst = edge_index[:, idx]
            weight = attention[idx]
            critical_connections.append({
                "from": component_names[src],
                "to": component_names[dst],
                "attention_weight": float(weight),
            })
        
        # Find critical nodes (sum of incoming attention)
        node_attention = np.zeros(len(component_names))
        for i, (src, dst) in enumerate(edge_index.T):
            node_attention[dst] += attention[i]
        
        top_nodes = np.argsort(node_attention)[-top_k:][::-1]
        critical_components = [
            {
                "component": component_names[i],
                "attention_score": float(node_attention[i]),
            }
            for i in top_nodes
        ]
        
        return {
            "critical_components": critical_components,
            "critical_connections": critical_connections,
        }
