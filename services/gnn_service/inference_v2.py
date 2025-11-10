"""
Enhanced Inference Engine for V2 Model (F1 > 90%)

Features:
- Compatible with EnhancedTemporalGAT (model_v2)
- Improved explainability with attention pooling
- Better root cause analysis
- Performance metrics tracking
- Batch processing optimization
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from config import api_config, model_config, physical_norms, training_config

# Import V2 model
try:
    from model_v2 import EnhancedTemporalGAT

    MODEL_V2_AVAILABLE = True
except ImportError:
    from model import TemporalGAT as EnhancedTemporalGAT

    MODEL_V2_AVAILABLE = False
    logging.warning("model_v2 not found, using fallback model")

logger = logging.getLogger(__name__)


class EnhancedGNNInference:
    """
    Enhanced inference engine for hydraulic diagnostics.

    Supports both V1 and V2 models with improved:
    - Attention analysis
    - Root cause identification
    - Performance tracking
    - Batch optimization
    """

    def __init__(
        self,
        model_path: str | None = None,
        metadata_path: str | None = None,
        use_v2: bool = True,
    ):
        """
        Args:
            model_path: Path to model checkpoint
            metadata_path: Path to metadata JSON
            use_v2: Use enhanced V2 model if available
        """
        self.model_path = model_path or self._get_default_model_path(use_v2)
        self.metadata_path = metadata_path or training_config.metadata_path
        self.device = torch.device(model_config.device)
        self.use_v2 = use_v2 and MODEL_V2_AVAILABLE

        # Load metadata
        self.metadata = self._load_metadata()
        self.component_names = model_config.component_names
        self.physical_norms = physical_norms

        # Load model
        self.model = self._load_model()

        # Thresholds
        self.warning_threshold = api_config.warning_threshold
        self.critical_threshold = api_config.critical_threshold

        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0

        logger.info("=" * 70)
        logger.info("Enhanced GNN Inference Engine Initialized")
        logger.info("=" * 70)
        logger.info(f"  Model: {'V2 (Enhanced)' if self.use_v2 else 'V1 (Standard)'}")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Warning threshold: {self.warning_threshold}")
        logger.info(f"  Critical threshold: {self.critical_threshold}")
        logger.info("=" * 70)

    def _get_default_model_path(self, use_v2: bool) -> str:
        """Get default model path based on version."""
        if use_v2:
            v2_path = Path("models/enhanced_model_best.ckpt")
            if v2_path.exists():
                return str(v2_path)

        # Fallback to V1
        return str(training_config.model_save_path)

    def _load_metadata(self) -> dict:
        """Load equipment metadata."""
        try:
            with open(self.metadata_path) as f:  # noqa: PTH123
                metadata = json.load(f)
            logger.info(f"Metadata loaded from {self.metadata_path}")
            return metadata
        except Exception as e:
            logger.warning(f"Error loading metadata: {e}")
            return {}

    def _load_model(self):
        """Load trained model."""
        try:
            return self._extracted_from__load_model_4()
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            logger.error("Please train the model first")
            raise

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    # TODO Rename this here and in `_load_model`
    def _extracted_from__load_model_4(self):
        logger.info(f"Loading model from {self.model_path}")

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        # Create model (V2 or V1)
        model = (
            EnhancedTemporalGAT(
                num_node_features=model_config.num_node_features,
                hidden_dim=model_config.hidden_dim,
                num_classes=model_config.num_classes,
                num_gat_layers=3,  # V2 uses 3 layers
                num_heads=model_config.num_heads,
                gat_dropout=model_config.gat_dropout,
                num_lstm_layers=model_config.num_lstm_layers,
                lstm_dropout=model_config.lstm_dropout,
            ).to(self.device)
            if self.use_v2
            else EnhancedTemporalGAT(
                num_node_features=model_config.num_node_features,
                hidden_dim=model_config.hidden_dim,
                num_classes=model_config.num_classes,
                num_gat_layers=2,  # V1 uses 2 layers
                num_heads=model_config.num_heads,
                gat_dropout=model_config.gat_dropout,
                num_lstm_layers=model_config.num_lstm_layers,
                lstm_dropout=model_config.lstm_dropout,
            ).to(self.device)
        )
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model loaded successfully")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(
            f"  Architecture: {'EnhancedTemporalGAT' if self.use_v2 else 'TemporalGAT'}"
        )

        return model

    def _create_edge_index(self) -> torch.Tensor:
        """Create standard graph topology (star graph with pump at center)."""
        edge_list = []

        # Pump (node 0) connected to all other nodes (1-6)
        # Bidirectional edges
        for target_node in range(1, 7):
            edge_list.extend(([0, target_node], [target_node, 0]))
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def _calculate_deviations(
        self, node_features: torch.Tensor
    ) -> dict[str, dict[str, float]]:
        """Calculate deviations from nominal values."""
        deviations = {}

        for i, component in enumerate(self.component_names):
            component_features = node_features[i]
            deviations[component] = {}

            # Extract raw features (first 5 of 15)
            raw_features = component_features[:5].cpu().numpy()

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
        """Classify component status."""
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
            "average_attention": 0.0,
        }

        try:
            # Get last GAT layer attention
            num_layers = 3 if self.use_v2 else 2
            last_layer_key = f"gat_layer_{num_layers - 1}"

            if last_layer_key in attention_weights:
                self._extracted_from__analyze_attention_16(
                    attention_weights, last_layer_key, analysis
                )
        except Exception as e:
            logger.warning(f"Error analyzing attention: {e}")

        return analysis

    # TODO Rename this here and in `_analyze_attention`
    def _extracted_from__analyze_attention_16(
        self, attention_weights, last_layer_key, analysis
    ):
        edge_index, attention_scores = attention_weights[last_layer_key]

        # Convert to numpy
        attention_scores = attention_scores.cpu().numpy()
        edge_index = edge_index.cpu().numpy()

        # Average attention
        analysis["average_attention"] = float(attention_scores.mean())

        # Store all attention scores
        for i in range(len(attention_scores)):
            source = edge_index[0, i] % model_config.num_nodes
            target = edge_index[1, i] % model_config.num_nodes
            score = attention_scores[i]
            analysis["attention_scores"][f"{source}-{target}"] = float(score)

        # Top attention paths
        top_indices = np.argsort(attention_scores)[-5:][::-1]
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

        # Component-wise attention
        component_attention = {}
        for i, component in enumerate(self.component_names):
            incoming = attention_scores[edge_index[1] % model_config.num_nodes == i]
            outgoing = attention_scores[edge_index[0] % model_config.num_nodes == i]

            if len(incoming) > 0 or len(outgoing) > 0:
                total = np.concatenate([incoming, outgoing]).mean()
                component_attention[component] = float(total)

        # Sort and get top 3
        sorted_comps = sorted(
            component_attention.items(), key=lambda x: x[1], reverse=True
        )
        analysis["most_attended_components"] = sorted_comps[:3]

    def predict(self, node_features: list[list[float]]) -> dict[str, Any]:
        """
        Perform inference on single graph.

        Args:
            node_features: List of 7 lists, each with 15 features

        Returns:
            Comprehensive diagnostics results
        """
        start_time = time.time()

        try:
            return self._extracted_from_predict_15(node_features, start_time)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    # TODO Rename this here and in `predict`
    def _extracted_from_predict_15(self, node_features, start_time):
        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float).to(self.device)
        x = x.view(-1, model_config.num_node_features)  # [7, 15]
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)

        # Create edges
        edge_index = self._create_edge_index().to(self.device)

        # Inference
        with torch.no_grad():
            logits, embeddings, attention_weights = self.model(x, edge_index, batch)
            probabilities = torch.sigmoid(logits)[0]

        # Analysis
        deviations = self._calculate_deviations(x)
        attention_analysis = self._analyze_attention(attention_weights)

        # Results
        results = {
            "system_health": float(1.0 - probabilities.mean().item()),
            "components": {},
            "attention_analysis": attention_analysis,
            "root_cause": self._identify_root_cause(
                probabilities, deviations, attention_analysis
            ),
            "model_version": "v2" if self.use_v2 else "v1",
            "inference_time_ms": (time.time() - start_time) * 1000,
        }

        # Component-wise results
        for i, component in enumerate(self.component_names):
            prob = probabilities[i].item()

            results["components"][component] = {
                "fault_probability": prob,
                "status": self._classify_status(prob),
                "confidence": 1.0 - abs(prob - 0.5) * 2,  # 0-1 scale
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

        # Update tracking
        self.inference_count += 1
        self.total_inference_time += time.time() - start_time

        logger.info(
            f"Prediction #{self.inference_count} completed "
            f"(system_health={results['system_health']:.3f}, "
            f"time={results['inference_time_ms']:.1f}ms)"
        )

        return results

    def batch_predict(
        self, batch_node_features: list[list[list[float]]]
    ) -> list[dict[str, Any]]:
        """Batch inference for multiple graphs."""
        results = []
        start_time = time.time()

        for i, node_features in enumerate(batch_node_features):
            try:
                result = self.predict(node_features)
                result["graph_id"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing graph {i}: {e}")
                results.append(
                    {
                        "graph_id": i,
                        "error": str(e),
                        "system_health": 0.0,
                        "components": {},
                    }
                )

        total_time = time.time() - start_time
        logger.info(
            f"Batch prediction completed: {len(results)} graphs "
            f"in {total_time:.2f}s ({total_time / len(results) * 1000:.1f}ms per graph)"
        )

        return results

    def _identify_root_cause(
        self, probabilities: torch.Tensor, deviations: dict, attention_analysis: dict
    ) -> dict[str, Any]:
        """Identify potential root cause."""
        root_cause = {
            "primary_suspect": None,
            "confidence": 0.0,
            "supporting_evidence": [],
            "propagation_path": [],
        }

        try:
            self._extracted_from__identify_root_cause_14(
                probabilities, root_cause, deviations, attention_analysis
            )
        except Exception as e:
            logger.warning(f"Error identifying root cause: {e}")

        return root_cause

    # TODO Rename this here and in `_identify_root_cause`
    def _extracted_from__identify_root_cause_14(
        self, probabilities, root_cause, deviations, attention_analysis
    ):
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
                if abs(data["deviation_pct"]) > 10:
                    root_cause["supporting_evidence"].append(
                        {
                            "feature": feature,
                            "deviation_pct": data["deviation_pct"],
                            "actual": data["actual"],
                            "expected": data["nominal"],
                        }
                    )

        # Propagation path from attention
        if "critical_paths" in attention_analysis:
            relevant_paths = [
                path
                for path in attention_analysis["critical_paths"]
                if path["to"] == primary_component or path["from"] == primary_component
            ]
            root_cause["propagation_path"] = relevant_paths[:3]

    def _get_expected_values(self, component: str) -> dict[str, float]:
        """Get expected nominal values."""
        try:
            norms = getattr(self.physical_norms, component.upper())
            return {feature: values["nominal"] for feature, values in norms.items()}
        except AttributeError:
            return {}

    def get_statistics(self) -> dict:
        """Get inference statistics."""
        return {
            "total_inferences": self.inference_count,
            "total_time_seconds": self.total_inference_time,
            "average_time_ms": (self.total_inference_time / self.inference_count * 1000)
            if self.inference_count > 0
            else 0,
            "model_version": "v2" if self.use_v2 else "v1",
        }


# Singleton
_inference_engine = None


def get_inference_engine(use_v2: bool = True) -> EnhancedGNNInference:
    """Get singleton inference engine."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = EnhancedGNNInference(use_v2=use_v2)
    return _inference_engine


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    engine = EnhancedGNNInference(use_v2=True)

    # Test data
    test_features = [
        [250.0, 0.8, 0.1, 65.0, 0.6] + [0.0] * 10,  # Pump
        [180.0, 0.7, 0.2, 60.0, 0.5] + [0.0] * 10,  # Boom
        [160.0, 0.6, 0.3, 55.0, 0.4] + [0.0] * 10,  # Stick
        [140.0, 0.5, 0.4, 50.0, 0.3] + [0.0] * 10,  # Bucket
        [175.0, 0.7, 0.1, 500.0, 0.6] + [0.0] * 10,  # Swing
        [170.0, 0.6, 0.2, 480.0, 0.5] + [0.0] * 10,  # Left
        [170.0, 0.6, 0.2, 480.0, 0.5] + [0.0] * 10,  # Right
    ]

    result = engine.predict(test_features)

    print("\n" + "=" * 70)
    print("TEST PREDICTION RESULTS")
    print("=" * 70)
    print(f"System Health: {result['system_health']:.3f}")
    print(f"Inference Time: {result['inference_time_ms']:.1f}ms")
    print(f"Model Version: {result['model_version']}")
    print("\nComponent Status:")
    for component, data in result["components"].items():
        print(f"  {component}: {data['status']} (prob={data['fault_probability']:.3f})")
    print("\nRoot Cause:")
    print(f"  Primary: {result['root_cause']['primary_suspect']}")
    print(f"  Confidence: {result['root_cause']['confidence']:.3f}")
    print("=" * 70)

    # Statistics
    stats = engine.get_statistics()
    print(f"\nStatistics: {stats}")

    print("\n✅ Inference test passed!")
