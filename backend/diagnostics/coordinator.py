"""Diagnostic Coordinator для orchestration GNN + component models.

Оркестрирует hybrid diagnostics:
1. System-level (GNN) анализ
2. Component-level (ml_service) анализ critical компонентов
3. Aggregation и recommendation generation
"""

import structlog
from django.conf import settings

from services.gnn_client import GNNClient
from services.ml_client import MLClient
from diagnostics.graph_builder import GraphBuilder
from diagnostics.models import DiagnosticResult

logger = structlog.get_logger(__name__)


class DiagnosticCoordinator:
    """Coordinator для hybrid diagnostics."""
    
    def __init__(self):
        self.gnn_client = GNNClient(
            base_url=settings.GNN_SERVICE_URL,
            api_key=settings.GNN_INTERNAL_API_KEY,
        )
        self.ml_client = MLClient(
            base_url=settings.ML_SERVICE_URL,
            api_key=settings.ML_INTERNAL_API_KEY,
        )
        self.graph_builder = GraphBuilder()
    
    async def run_diagnostics(
        self,
        equipment_id: int,
        mode: str = "hybrid",
    ) -> dict:
        """Run diagnostic analysis.
        
        Args:
            equipment_id: Equipment ID
            mode: Diagnostic mode ("gnn_only", "component_only", "hybrid")
        
        Returns:
            Diagnostic result dict
        """
        logger.info(
            "Starting diagnostics",
            equipment_id=equipment_id,
            mode=mode,
        )
        
        # Build equipment graph
        graph_data = await self.graph_builder.build_graph(
            equipment_id=equipment_id,
            window_seconds=20,
        )
        
        result = {
            "equipment_id": equipment_id,
            "system": None,
            "components": [],
            "recommendation": None,
        }
        
        # System-level (GNN)
        if mode in ["gnn_only", "hybrid"]:
            try:
                gnn_result = await self.gnn_client.predict(
                    node_features=graph_data["node_features"],
                    edge_index=graph_data["edge_index"],
                    edge_attr=graph_data.get("edge_attr"),
                    component_names=graph_data["component_names"],
                )
                result["system"] = gnn_result
                
                logger.info(
                    "GNN analysis complete",
                    prediction=gnn_result["prediction"],
                    score=gnn_result["anomaly_score"],
                )
            except Exception as e:
                logger.error("GNN analysis failed", error=str(e))
                result["system"] = {"error": str(e)}
        
        # Component-level (if GNN detected anomaly)
        if mode == "hybrid" and result["system"] and result["system"].get("prediction") == 1:
            explanation = result["system"].get("explanation", {})
            critical_components = explanation.get("critical_components", [])
            
            for component in critical_components:
                try:
                    comp_result = await self._analyze_component(
                        equipment_id=equipment_id,
                        component_type=component,
                    )
                    result["components"].append(comp_result)
                except Exception as e:
                    logger.error(
                        "Component analysis failed",
                        component=component,
                        error=str(e),
                    )
        
        # Generate recommendation
        result["recommendation"] = self._generate_recommendation(
            system_result=result["system"],
            component_results=result["components"],
        )
        
        # Save to DB
        await self._save_result(equipment_id, result)
        
        return result
    
    async def _analyze_component(
        self,
        equipment_id: int,
        component_type: str,
    ) -> dict:
        """Analyze specific component."""
        # Map component type to ml_service endpoint
        endpoint_map = {
            "pump": "predict_pump",
            "boom": "predict_cylinder_boom",
            "stick": "predict_cylinder_stick",
            "bucket": "predict_cylinder_bucket",
        }
        
        endpoint = endpoint_map.get(component_type)
        if not endpoint:
            return {
                "component": component_type,
                "error": f"No model for {component_type}",
            }
        
        # Call ml_service
        result = await self.ml_client.predict(
            equipment_id=equipment_id,
            model_type=component_type,
        )
        
        return {
            "component": component_type,
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "diagnosis": result.get("diagnosis"),
            "action": result.get("recommended_action"),
        }
    
    def _generate_recommendation(
        self,
        system_result: dict | None,
        component_results: list[dict],
    ) -> dict:
        """Generate actionable recommendation."""
        if not system_result or system_result.get("prediction") == 0:
            return {
                "priority": "low",
                "action": "Continue monitoring",
                "timeframe": "routine",
            }
        
        # Anomaly detected
        anomaly_score = system_result.get("anomaly_score", 0)
        
        if anomaly_score > 0.9:
            priority = "critical"
            timeframe = "immediate"
            action = "Stop operation, inspect immediately"
        elif anomaly_score > 0.7:
            priority = "high"
            timeframe = "24h"
            action = "Schedule urgent inspection"
        else:
            priority = "medium"
            timeframe = "48-72h"
            action = "Plan maintenance check"
        
        # Add component-specific actions
        component_actions = []
        for comp in component_results:
            if comp.get("action"):
                component_actions.append(f"{comp['component']}: {comp['action']}")
        
        return {
            "priority": priority,
            "action": action,
            "timeframe": timeframe,
            "component_actions": component_actions,
            "system_reasoning": system_result.get("explanation", {}).get("reasoning"),
        }
    
    async def _save_result(
        self,
        equipment_id: int,
        result: dict,
    ) -> None:
        """Save diagnostic result to DB."""
        try:
            DiagnosticResult.objects.create(
                equipment_id=equipment_id,
                diagnostic_type="hybrid",
                gnn_prediction=result["system"].get("prediction") if result["system"] else None,
                gnn_score=result["system"].get("anomaly_score") if result["system"] else None,
                recommendation=result["recommendation"],
                raw_data=result,
            )
        except Exception as e:
            logger.error("Failed to save diagnostic result", error=str(e))
