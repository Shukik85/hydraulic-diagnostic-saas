"""Интеграционный AB-пайплайн: сравнение reasoning RAG для prod и test GNN моделей."""

# Пример (может быть оформлен как management команды или CI job)
import requests

from apps.gnn_config.gnn_service_client import GNNAdminClient

GNN_SERVICE_URL = "http://gnn-service:8002"
RAG_SERVICE_URL = "http://rag-service:8003"
ADMIN_TOKEN = "your_token"
client = GNNAdminClient(base_url=GNN_SERVICE_URL, admin_token=ADMIN_TOKEN)

equipment_id = "exc_001"
time_window = {"start_time": "2025-11-01T00:00:00Z", "end_time": "2025-11-13T00:00:00Z"}
prod_result = client.get_inference(equipment_id, time_window)  # реализовать этот метод, если нужен

# Тестируем экспериментальную модель
test_model_path = "/app/models/test/experiment_v20251115.onnx"
test_result = client.test_inference(equipment_id, time_window, test_model_path)


def call_rag_diagnosis(gnn_result: dict):
    url = f"{RAG_SERVICE_URL}/interpret/diagnosis"
    payload = {
        "gnn_result": gnn_result,
        "equipment_context": {"equipment_id": equipment_id},
        "historical_context": None,
    }
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


prod_diag = call_rag_diagnosis(prod_result)
test_diag = call_rag_diagnosis(test_result)

# AB анализ (выборка summary, reasoning, recommendations)
print("=== PROD ===\n", prod_diag["summary"], prod_diag["reasoning"], prod_diag["recommendations"])
print("=== TEST ===\n", test_diag["summary"], test_diag["reasoning"], test_diag["recommendations"])

# Можно сохранять в артефакты CI или делать сравнение/trace по метрикам
