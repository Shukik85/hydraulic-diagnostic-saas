# services/gnn_service/post_processor.py
"""
Post-processing utilities for GNN prediction results.
"""
def build_prediction_response(raw_result, metadata):
    # TODO: interpret raw health/degradation, decode to component_ids etc.
    # This stub will need expansion for RAG service and UI/frontend
    return {
        "system_health": float(sum(raw_result["health"][0])/len(raw_result["health"][0])),
        "components": {c: {"health": h, "degradation": d} for c, h, d in zip(metadata.component_ids, raw_result["health"][0], raw_result["degradation"][0])},
        "attention_analysis": {},
        "root_cause": {},
        "performance_metrics": {}
    }
