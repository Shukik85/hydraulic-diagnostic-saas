"""
Inference class for hydraulic diagnostics GNN model.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from config import api_config, model_config, physical_norms, training_config
from model import HydraulicGNN, create_model

logger = logging.getLogger(__name__)


class GNNInference:
    """
    Hydraulic diagnostics inference engine.

    Provides:
    - Single and batch predictions
    - Fault probability estimation
    - Status classification (normal/warning/critical)
    - Deviation analysis from nominal values
    - Attention-based explainability
    """

    def __init__(self, model_path: str = None, metadata_path: str = None):
        self.model_path = model_path or training_config.model_save_path
        self.metadata_path = metadata_path or training_config.metadata_path
        self.device = torch.device(model_config.device)

        # Load metadata
        self.metadata = self._load_metadata()
        self.component_names = model_config.component_names
        self.physical_norms = physical_norms

        # Load model
        self.model = self._load_model()

        # Thresholds
        self.warning_threshold = api_config.warning_threshold
        self.critical_threshold = api_config.critical_threshold

        logger.info("GNNInference initialized successfully")

    def _load_metadata(self) -> dict:
        """Load equipment metadata."""
        try:
            with open(self.metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded from {self.metadata_path}")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}

    def _load_model(self) -> HydraulicGNN:
        """Load trained model from checkpoint."""
        try:
            model = create_model(self.device)

            if Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.warning(
                    f"Model checkpoint not found at {self.model_path}, using initialized model"
                )

            model.eval()
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _create_edge_index(self) -> torch.Tensor:
        """Create standard graph topology."""
        edge_list = []

        # Pump (node 0) connected to all other nodes (1-6)
        for target_node in range(1, 7):
            edge_list.append([0, target_node])
            edge_list.append([target_node, 0])

        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def _calculate_deviations(
        self, node_features: torch.Tensor
    ) -> dict[str, dict[str, float]]:
        """
        Calculate deviations from nominal values for each component.

        Args:
            node_features: Tensor of node features [7, 15]

        Returns:
            Dictionary of deviations per component and feature
        """
        deviations = {}

        for i, component in enumerate(self.component_names):
            component_features = node_features[i]
            deviations[component] = {}

            # Extract raw features (first 5 of 15 features)
            raw_features = component_features[:5].cpu().numpy()

            # Get nominal values for this component
            try:
                norms = getattr(self.physical_norms, component.upper())
                feature_names = list(norms.keys())

                for j, feature_name in enumerate(feature_names):
                    if j < len(raw_features):
                        nominal = norms[feature_name]["nominal"]
                        actual_value = raw_features[j]
                        deviation = actual_value - nominal
                        deviation_pct = (
                            (deviation / nominal) * 100 if nominal > 0 else 0
                        )

                        deviations[component][feature_name] = {
                            "actual": float(actual_value),
                            "nominal": float(nominal),
                            "deviation": float(deviation),
                            "deviation_pct": float(deviation_pct),
                        }

            except (AttributeError, IndexError) as e:
                logger.warning(f"Error calculating deviations for {component}: {e}")
                deviations[component] = {}

        return deviations

    def _classify_status(self, probability: float) -> str:
        """Classify component status based on fault probability."""
        if probability >= self.critical_threshold:
            return "critical"
        elif probability >= self.warning_threshold:
            return "warning"
        else:
            return "normal"

    def _analyze_attention(self, attention_weights: dict) -> dict[str, Any]:
        """Analyze attention weights for explainability."""
        analysis = {
            "critical_paths": [],
            "most_attended_components": [],
            "attention_scores": {},
        }

        try:
            # Analyze last GAT layer attention
            last_layer_key = f"gat_layer_{model_config.num_gat_layers - 1}"
            if last_layer_key in attention_weights:
                edge_index, attention_scores = attention_weights[last_layer_key]

                # Convert to numpy for analysis
                attention_scores = attention_scores.cpu().numpy()
                edge_index = edge_index.cpu().numpy()

                # Find most important connections
                for i in range(len(attention_scores)):
                    source_node = edge_index[0, i] % model_config.num_nodes
                    target_node = edge_index[1, i] % model_config.num_nodes
                    score = attention_scores[i]

                    analysis["attention_scores"][f"{source_node}-{target_node}"] = (
                        float(score)
                    )

                # Get top attention paths
                top_indices = np.argsort(attention_scores)[-5:][::-1]  # Top 5
                for idx in top_indices:
                    source = edge_index[0, idx] % model_config.num_nodes
                    target = edge_index[1, idx] % model_config.num_nodes
                    analysis["critical_paths"].append(
                        {
                            "from": self.component_names[source],
                            "to": self.component_names[target],
                            "attention_score": float(attention_scores[idx]),
                        }
                    )

                # Most attended components
                component_attention = {}
                for i, component in enumerate(self.component_names):
                    # Average attention to/from this component
                    incoming = attention_scores[
                        edge_index[1] % model_config.num_nodes == i
                    ]
                    outgoing = attention_scores[
                        edge_index[0] % model_config.num_nodes == i
                    ]

                    if len(incoming) > 0 or len(outgoing) > 0:
                        total_attention = np.concatenate([incoming, outgoing]).mean()
                        component_attention[component] = float(total_attention)

                # Sort by attention score
                sorted_components = sorted(
                    component_attention.items(), key=lambda x: x[1], reverse=True
                )
                analysis["most_attended_components"] = sorted_components[:3]

        except Exception as e:
            logger.warning(f"Error analyzing attention: {e}")

        return analysis

    def predict(self, node_features: list[list[float]]) -> dict[str, Any]:
        """
        Perform single graph inference.

        Args:
            node_features: List of 7 lists, each containing 15 features per component

        Returns:
            Dictionary with comprehensive diagnostics results
        """
        try:
            # Convert to tensor
            x = torch.tensor(node_features, dtype=torch.float).to(self.device)

            # Add batch dimension
            x = x.view(-1, model_config.num_node_features)  # [7, 15] -> [7, 15]
            batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)

            # Create edge index
            edge_index = self._create_edge_index().to(self.device)

            # Model prediction
            with torch.no_grad():
                logits, embeddings, attention_weights = self.model(x, edge_index, batch)
                probabilities = torch.sigmoid(logits)[0]  # First (and only) batch

            # Calculate deviations
            deviations = self._calculate_deviations(x)

            # Analyze attention
            attention_analysis = self._analyze_attention(attention_weights)

            # Prepare results
            results = {
                "system_health": float(1.0 - probabilities.mean().item()),
                "components": {},
                "attention_analysis": attention_analysis,
                "root_cause": self._identify_root_cause(
                    probabilities, deviations, attention_analysis
                ),
                "timestamp": torch.cuda.Event(enable_timing=True)
                if self.device.type == "cuda"
                else None,
            }

            # Component-wise results
            for i, component in enumerate(self.component_names):
                prob = probabilities[i].item()

                results["components"][component] = {
                    "fault_probability": prob,
                    "status": self._classify_status(prob),
                    "deviations": deviations.get(component, {}),
                    "expected_values": self._get_expected_values(component),
                    "attention_score": next(
                        (
                            score
                            for comp, score in attention_analysis[
                                "most_attended_components"
                            ]
                            if comp == component
                        ),
                        0.0,
                    ),
                }

            logger.info(
                f"Prediction completed. System health: {results['system_health']:.3f}"
            )
            return results

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def batch_predict(
        self, batch_node_features: list[list[list[float]]]
    ) -> list[dict[str, Any]]:
        """
        Perform batch inference for multiple graphs.

        Args:
            batch_node_features: List of graphs, each containing 7 nodes with 15 features

        Returns:
            List of prediction results for each graph
        """
        results = []

        for i, node_features in enumerate(batch_node_features):
            try:
                result = self.predict(node_features)
                result["graph_id"] = i
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1} graphs...")

            except Exception as e:
                logger.error(f"Error processing graph {i}: {e}")
                # Add error result
                results.append(
                    {
                        "graph_id": i,
                        "error": str(e),
                        "system_health": 0.0,
                        "components": {},
                    }
                )

        return results

    def _identify_root_cause(
        self, probabilities: torch.Tensor, deviations: dict, attention_analysis: dict
    ) -> dict[str, Any]:
        """Identify potential root cause of faults."""
        root_cause = {
            "primary_suspect": None,
            "confidence": 0.0,
            "supporting_evidence": [],
            "propagation_path": [],
        }

        try:
            # Find component with highest fault probability
            max_prob_idx = probabilities.argmax().item()
            primary_component = self.component_names[max_prob_idx]
            max_prob = probabilities[max_prob_idx].item()

            root_cause["primary_suspect"] = primary_component
            root_cause["confidence"] = max_prob

            # Supporting evidence from deviations
            if primary_component in deviations:
                comp_deviations = deviations[primary_component]
                for feature, data in comp_deviations.items():
                    if abs(data["deviation_pct"]) > 10:  # Significant deviation
                        root_cause["supporting_evidence"].append(
                            {
                                "feature": feature,
                                "deviation": data["deviation_pct"],
                                "actual": data["actual"],
                                "expected": data["nominal"],
                            }
                        )

            # Propagation path from attention analysis
            if "critical_paths" in attention_analysis:
                relevant_paths = [
                    path
                    for path in attention_analysis["critical_paths"]
                    if path["to"] == primary_component
                    or path["from"] == primary_component
                ]
                root_cause["propagation_path"] = relevant_paths[:3]  # Top 3 paths

        except Exception as e:
            logger.warning(f"Error identifying root cause: {e}")

        return root_cause

    def _get_expected_values(self, component: str) -> dict[str, float]:
        """Get expected nominal values for a component."""
        try:
            norms = getattr(self.physical_norms, component.upper())
            expected = {}

            for feature, values in norms.items():
                expected[feature] = values["nominal"]

            return expected
        except AttributeError:
            return {}

    def update_thresholds(
        self, warning_threshold: float = None, critical_threshold: float = None
    ):
        """Update classification thresholds."""
        if warning_threshold is not None:
            self.warning_threshold = warning_threshold
        if critical_threshold is not None:
            self.critical_threshold = critical_threshold

        logger.info(
            f"Thresholds updated: warning={self.warning_threshold}, critical={self.critical_threshold}"
        )


# Singleton instance for API use
_inference_engine = None


def get_inference_engine() -> GNNInference:
    """Get singleton inference engine instance."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = GNNInference()
    return _inference_engine


if __name__ == "__main__":
    # Test inference
    engine = GNNInference()

    # Create test data
    test_features = [
        [250.0, 0.8, 0.1, 65.0, 0.6, 0.05, 2.5, 0.5, 0.1, 75.0, 0.7, 0.2]
        + [0.0] * 3,  # Pump
        [180.0, 0.7, 0.2, 60.0, 0.5, 0.1, 50.0, 0.5, 0.0, 120.0, 0.8, 0.1]
        + [0.0] * 3,  # Boom
        [160.0, 0.6, 0.3, 55.0, 0.4, 0.2, 45.0, 0.4, 0.1, 105.0, 0.7, 0.2]
        + [0.0] * 3,  # Stick
        [140.0, 0.5, 0.4, 50.0, 0.3, 0.3, 40.0, 0.4, 0.2, 90.0, 0.6, 0.3]
        + [0.0] * 3,  # Bucket
        [175.0, 0.7, 0.1, 500.0, 0.6, 0.1, 120.0, 0.5, 0.1, 70.0, 0.6, 0.1]
        + [0.0] * 3,  # Swing
        [170.0, 0.6, 0.2, 480.0, 0.5, 0.2, 110.0, 0.4, 0.2, 68.0, 0.5, 0.2]
        + [0.0] * 3,  # Left
        [170.0, 0.6, 0.2, 480.0, 0.5, 0.2, 110.0, 0.4, 0.2, 68.0, 0.5, 0.2]
        + [0.0] * 3,  # Right
    ]

    result = engine.predict(test_features)
    print("Test prediction completed!")
    print(f"System health: {result['system_health']:.3f}")
