"""Mock inference engine for rapid API prototyping.

Provides realistic-looking predictions without requiring:
- Model checkpoint
- GPU
- Full GNN setup

Will be replaced with actual UniversalTemporalGNN when model is trained.

Features:
    - Async inference simulation
    - Realistic latency (50-100ms)
    - Multi-component support
    - Severity gradients

TODO:
    - Replace with src.inference.InferenceEngine
    - Load actual model checkpoints
    - Implement real GNN forward pass
"""

from __future__ import annotations

import asyncio
import random

import numpy as np


class MockInferenceEngine:
    """Mock inference engine for testing API without model.
    
    Generates realistic but synthetic predictions for all hydraulic components.
    Simulates latency and provides component-specific reliability scores.
    
    Attributes:
        model_version: Version identifier for tracking
        components: List of supported hydraulic components
        severity_thresholds: Health score thresholds for severity grades
    
    Examples:
        >>> engine = MockInferenceEngine()
        >>> predictions = await engine.predict(
        ...     equipment_id="pump_001",
        ...     sensor_readings={"PS1": [100.5, 101.2], ...}
        ... )
    """

    def __init__(self):
        """Initialize mock engine."""
        self.model_version = "mock-v0.1.0"
        self.components = ["cooler", "valve", "pump", "accumulator"]
        self.severity_thresholds = {
            "failure": (0.0, 0.70),
            "degraded": (0.70, 0.85),
            "optimal": (0.85, 1.0)
        }

    async def predict(
        self,
        equipment_id: str,
        sensor_readings: dict[str, list[float]],
        topology_id: str | None = None
    ) -> dict:
        """Generate mock predictions for equipment.
        
        Args:
            equipment_id: Equipment identifier
            sensor_readings: Sensor time series data
            topology_id: Optional topology identifier
        
        Returns:
            Dictionary with predictions for all components
        """
        # Simulate processing delay (50-100ms)
        await asyncio.sleep(random.uniform(0.05, 0.10))

        # Generate predictions
        components = []
        health_scores = []

        for component in self.components:
            # Generate realistic health score
            # Slightly biased toward healthier systems
            health_score = random.gauss(0.82, 0.12)  # μ=0.82, σ=0.12
            health_score = max(0.0, min(1.0, health_score))  # Clamp to [0, 1]

            # Determine severity grade
            severity_grade = self._get_severity_grade(health_score)

            # Confidence (slightly higher for degraded/failure states)
            if severity_grade == "failure":
                confidence = random.uniform(0.85, 0.98)
            elif severity_grade == "degraded":
                confidence = random.uniform(0.80, 0.95)
            else:
                confidence = random.uniform(0.75, 0.90)

            # Contributing sensors (top 1-3 that influenced prediction)
            num_contributors = random.randint(1, 3)
            all_sensors = list(sensor_readings.keys())
            contributors = random.sample(
                all_sensors,
                min(num_contributors, len(all_sensors))
            ) if all_sensors else []

            components.append({
                "component_name": component,
                "health_score": round(health_score, 3),
                "severity_grade": severity_grade,
                "confidence": round(confidence, 3),
                "contributing_sensors": sorted(contributors)
            })

            health_scores.append(health_score)

        # Overall health (weighted average)
        overall_health = float(np.mean(health_scores))

        # Generate recommendations
        recommendations = self._generate_recommendations(components)

        return {
            "overall_health": round(overall_health, 3),
            "components": components,
            "recommendations": recommendations
        }

    def _get_severity_grade(self, health_score: float) -> str:
        """Determine severity grade from health score.
        
        Args:
            health_score: Health score (0-1)
        
        Returns:
            Severity grade: 'optimal', 'degraded', or 'failure'
        """
        for grade, (min_score, max_score) in self.severity_thresholds.items():
            if min_score <= health_score <= max_score:
                return grade
        return "optimal"  # Default

    def _generate_recommendations(self, components: list[dict]) -> list[str]:
        """Generate maintenance recommendations from predictions.
        
        Args:
            components: Component predictions
        
        Returns:
            List of maintenance recommendations
        """
        recommendations = []

        # Check for critical failures
        failures = [c for c in components if c["severity_grade"] == "failure"]
        if failures:
            for comp in failures:
                recommendations.append(
                    f"🔴 CRITICAL: {comp['component_name']} requires immediate maintenance"
                )

        # Check for degraded components
        degraded = [c for c in components if c["severity_grade"] == "degraded"]
        if degraded:
            for comp in degraded:
                recommendations.append(
                    f"⚠️ WARNING: {comp['component_name']} - schedule maintenance within 7 days"
                )

        # If all optimal
        if not failures and not degraded:
            recommendations.append("✅ All components operating optimally - no action needed")

        return recommendations

    def get_stats(self) -> dict:
        """Get inference engine statistics.
        
        Returns:
            Dictionary with engine stats
        """
        return {
            "model_version": self.model_version,
            "supported_components": self.components,
            "inference_type": "mock",
            "average_latency_ms": 75,  # Mock average latency
            "status": "ready"
        }
