#!/usr/bin/env python3
"""
Comprehensive Model Testing Script
Validates all models for production readiness
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add ml_service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    AdaptiveModel,
    CatBoostModel,
    EnsembleModel,
    RandomForestModel,
    XGBoostModel,
    check_model_availability,
)

logger = structlog.get_logger()
console = Console()


class ModelTester:
    """Comprehensive model testing suite."""

    def __init__(self):
        self.console = console
        self.test_results = {}
        self.test_features = self._generate_test_data()

    def _generate_test_data(self, n_samples: int = 100) -> np.ndarray:
        """Generate realistic test data."""
        np.random.seed(42)  # Reproducible tests

        # Generate diverse test cases
        test_data = []

        # Normal operating conditions (70%)
        normal_samples = int(n_samples * 0.7)
        normal_data = np.random.normal(0, 1, (normal_samples, 25))
        test_data.extend(normal_data)

        # Mild anomalies (20%)
        mild_anomaly_samples = int(n_samples * 0.2)
        mild_anomalies = np.random.normal(0, 2, (mild_anomaly_samples, 25))
        mild_anomalies[:, 0] += 1.5  # Pressure variation
        test_data.extend(mild_anomalies)

        # Severe anomalies (10%)
        severe_anomaly_samples = n_samples - normal_samples - mild_anomaly_samples
        severe_anomalies = np.random.normal(0, 1, (severe_anomaly_samples, 25))
        severe_anomalies[:, 0] += 4  # High pressure
        severe_anomalies[:, 1] -= 3  # Low flow
        severe_anomalies[:, 2] += 3  # High temperature
        test_data.extend(severe_anomalies)

        return np.array(test_data)

    async def test_individual_model(self, model_class, model_name: str) -> dict[str, Any]:
        """Test individual model comprehensively."""
        results = {
            "model_name": model_name,
            "load_success": False,
            "load_time_ms": 0,
            "prediction_success": False,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "errors": [],
            "predictions_made": 0,
            "feature_compatibility": True,
            "confidence_range": (0, 0),
            "score_range": (0, 0),
        }

        try:
            # Test model loading
            load_start = time.time()
            model = model_class()
            await model.load()
            load_time = (time.time() - load_start) * 1000

            results["load_success"] = model.is_loaded
            results["load_time_ms"] = load_time

            if not model.is_loaded:
                results["errors"].append("Model failed to load")
                return results

            # Test predictions
            latencies = []
            confidences = []
            scores = []
            successful_predictions = 0

            for i, features in enumerate(self.test_features[:50]):  # Test with 50 samples
                try:
                    pred_start = time.time()
                    prediction = await model.predict(features)
                    pred_time = (time.time() - pred_start) * 1000

                    latencies.append(pred_time)
                    confidences.append(prediction.get("confidence", 0))
                    scores.append(prediction.get("score", 0))
                    successful_predictions += 1

                except Exception as e:
                    results["errors"].append(f"Prediction {i + 1} failed: {str(e)}")

            if latencies:
                results["prediction_success"] = True
                results["avg_latency_ms"] = np.mean(latencies)
                results["p95_latency_ms"] = np.percentile(latencies, 95)
                results["predictions_made"] = successful_predictions
                results["confidence_range"] = (np.min(confidences), np.max(confidences))
                results["score_range"] = (np.min(scores), np.max(scores))

            # Test feature compatibility
            try:
                # Test with different feature sizes
                await model.predict(np.random.rand(10))  # Wrong size
                results["errors"].append("Model should reject wrong feature size")
            except (ValueError, RuntimeError):
                pass  # Expected behavior

            # Cleanup
            await model.cleanup()

        except Exception as e:
            results["errors"].append(f"Model test failed: {str(e)}")

        return results

    async def test_ensemble_model(self) -> dict[str, Any]:
        """Test ensemble model specifically."""
        results = {
            "model_name": "ensemble",
            "load_success": False,
            "models_loaded": [],
            "load_time_ms": 0,
            "prediction_success": False,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "consensus_strength": 0,
            "fallback_strategies": [],
            "errors": [],
            "weight_adjustments": False,
        }

        try:
            # Test ensemble loading
            load_start = time.time()
            ensemble = EnsembleModel()
            await ensemble.load_models()
            load_time = (time.time() - load_start) * 1000

            results["load_success"] = ensemble.is_loaded
            results["load_time_ms"] = load_time
            results["models_loaded"] = ensemble.get_loaded_models()

            if not ensemble.is_loaded:
                results["errors"].append("Ensemble failed to load")
                return results

            # Test ensemble predictions
            latencies = []
            consensus_scores = []

            for i, features in enumerate(self.test_features[:20]):  # Test with 20 samples
                try:
                    pred_start = time.time()
                    prediction = await ensemble.predict(features)
                    pred_time = (time.time() - pred_start) * 1000

                    latencies.append(pred_time)
                    consensus_scores.append(prediction.get("consensus_strength", 0))

                except Exception as e:
                    results["errors"].append(f"Ensemble prediction {i + 1} failed: {str(e)}")

            if latencies:
                results["prediction_success"] = True
                results["avg_latency_ms"] = np.mean(latencies)
                results["p95_latency_ms"] = np.percentile(latencies, 95)
                results["consensus_strength"] = np.mean(consensus_scores)

            # Test warmup
            await ensemble.warmup(warmup_samples=5)

            # Check performance metrics
            perf_metrics = ensemble.get_performance_metrics()
            results["weight_adjustments"] = perf_metrics.get("weight_adjustments", 0) > 0
            results["fallback_strategies"] = list(ensemble.fallback_strategies.keys())

            # Cleanup
            await ensemble.cleanup()

        except Exception as e:
            results["errors"].append(f"Ensemble test failed: {str(e)}")

        return results

    async def run_all_tests(self) -> dict[str, Any]:
        """Run comprehensive test suite."""
        self.console.print(Panel.fit("🧪 Starting Comprehensive Model Testing", style="bold blue"))

        all_results = {
            "test_timestamp": time.time(),
            "individual_models": {},
            "ensemble_results": {},
            "summary": {"total_models_tested": 0, "models_passed": 0, "models_failed": 0, "overall_success": False},
        }

        # Test individual models
        model_classes = {
            "catboost": CatBoostModel,
            "xgboost": XGBoostModel,
            "random_forest": RandomForestModel,
            "adaptive": AdaptiveModel,
        }

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console
        ) as progress:
            for model_name, model_class in model_classes.items():
                task = progress.add_task(f"Testing {model_name}...", total=None)

                results = await self.test_individual_model(model_class, model_name)
                all_results["individual_models"][model_name] = results

                if results["load_success"] and results["prediction_success"]:
                    all_results["summary"]["models_passed"] += 1
                    progress.update(task, description=f"✅ {model_name} - PASSED")
                else:
                    all_results["summary"]["models_failed"] += 1
                    progress.update(task, description=f"❌ {model_name} - FAILED")

                all_results["summary"]["total_models_tested"] += 1
                progress.remove_task(task)

            # Test ensemble
            ensemble_task = progress.add_task("Testing ensemble...", total=None)
            ensemble_results = await self.test_ensemble_model()
            all_results["ensemble_results"] = ensemble_results

            if ensemble_results["load_success"] and ensemble_results["prediction_success"]:
                progress.update(ensemble_task, description="✅ ensemble - PASSED")
            else:
                progress.update(ensemble_task, description="❌ ensemble - FAILED")

        # Overall success assessment
        all_results["summary"]["overall_success"] = (
            all_results["summary"]["models_passed"] >= 2  # At least 2 models work
            and ensemble_results["load_success"]  # Ensemble loads
        )

        return all_results

    def display_results(self, results: dict[str, Any]) -> None:
        """Display test results in a formatted table."""

        # Individual Models Results Table
        individual_table = Table(title="Individual Model Test Results")
        individual_table.add_column("Model", style="cyan")
        individual_table.add_column("Load", style="green")
        individual_table.add_column("Predict", style="green")
        individual_table.add_column("Avg Latency (ms)", style="yellow")
        individual_table.add_column("P95 Latency (ms)", style="yellow")
        individual_table.add_column("Predictions", style="blue")
        individual_table.add_column("Errors", style="red")

        for model_name, model_results in results["individual_models"].items():
            individual_table.add_row(
                model_name,
                "✅" if model_results["load_success"] else "❌",
                "✅" if model_results["prediction_success"] else "❌",
                f"{model_results['avg_latency_ms']:.1f}",
                f"{model_results['p95_latency_ms']:.1f}",
                str(model_results["predictions_made"]),
                str(len(model_results["errors"])),
            )

        self.console.print(individual_table)

        # Ensemble Results
        ensemble_results = results["ensemble_results"]
        ensemble_table = Table(title="Ensemble Model Test Results")
        ensemble_table.add_column("Metric", style="cyan")
        ensemble_table.add_column("Value", style="green")

        ensemble_table.add_row("Load Success", "✅" if ensemble_results["load_success"] else "❌")
        ensemble_table.add_row("Models Loaded", str(len(ensemble_results["models_loaded"])))
        ensemble_table.add_row("Loaded Models", ", ".join(ensemble_results["models_loaded"]))
        ensemble_table.add_row("Avg Latency (ms)", f"{ensemble_results['avg_latency_ms']:.1f}")
        ensemble_table.add_row("P95 Latency (ms)", f"{ensemble_results['p95_latency_ms']:.1f}")
        ensemble_table.add_row("Consensus Strength", f"{ensemble_results['consensus_strength']:.3f}")
        ensemble_table.add_row("Errors", str(len(ensemble_results["errors"])))

        self.console.print(ensemble_table)

        # Summary
        summary = results["summary"]
        summary_style = "green" if summary["overall_success"] else "red"
        summary_text = "✅ PRODUCTION READY" if summary["overall_success"] else "❌ NOT READY FOR PRODUCTION"

        summary_panel = Panel(
            f"""Models Tested: {summary["total_models_tested"]}
Passed: {summary["models_passed"]}
Failed: {summary["models_failed"]}

Status: {summary_text}""",
            title="Test Summary",
            style=summary_style,
        )

        self.console.print(summary_panel)

        # Error Details
        if (
            any(len(model_results["errors"]) > 0 for model_results in results["individual_models"].values())
            or len(ensemble_results["errors"]) > 0
        ):
            error_table = Table(title="Error Details")
            error_table.add_column("Model", style="cyan")
            error_table.add_column("Errors", style="red")

            for model_name, model_results in results["individual_models"].items():
                if model_results["errors"]:
                    error_table.add_row(model_name, "\n".join(model_results["errors"]))

            if ensemble_results["errors"]:
                error_table.add_row("ensemble", "\n".join(ensemble_results["errors"]))

            self.console.print(error_table)

    def save_results(self, results: dict[str, Any], filename: str = "model_test_results.json") -> None:
        """Save test results to file."""
        import json

        output_path = Path(__file__).parent.parent / "reports" / filename
        output_path.parent.mkdir(exist_ok=True)

        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Deep convert the results
        json_results = json.loads(json.dumps(results, default=convert_numpy))

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)

        self.console.print(f"📄 Results saved to: {output_path}")


async def main():
    """Main test runner."""
    tester = ModelTester()

    # Check model availability first
    console.print("\n🔍 Checking model availability...")
    availability = check_model_availability()

    avail_table = Table(title="Model Availability Check")
    avail_table.add_column("Model", style="cyan")
    avail_table.add_column("Available", style="green")

    for model_name, available in availability.items():
        avail_table.add_row(model_name, "✅" if available else "❌")

    console.print(avail_table)

    # Run comprehensive tests
    console.print("\n🚀 Running comprehensive tests...")
    results = await tester.run_all_tests()

    # Display results
    console.print("\n📊 Test Results")
    tester.display_results(results)

    # Save results
    tester.save_results(results)

    # Return exit code based on success
    return 0 if results["summary"]["overall_success"] else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)
