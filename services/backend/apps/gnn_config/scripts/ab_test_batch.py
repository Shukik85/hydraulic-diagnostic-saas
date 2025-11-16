"""
Batch AB-тестовый пайплайн для GNN + RAG reasoning моделей

Для каждой записи:
- Получает prod и test inference
- Запускает RAG-interpretation для каждого результата
- Сохраняет side-by-side сравнение (summary, reasoning, метрики)
- Сохраняет отчёт в .csv и .md
"""

import csv
import json
from datetime import datetime

import requests

from apps.gnn_config.gnn_service_client import GNNAdminClient

# --- CONFIG ---
GNN_SERVICE_URL = "http://gnn-service:8002"
RAG_SERVICE_URL = "http://rag-service:8003"
ADMIN_TOKEN = "your_admin_token"
TEST_MODEL_PATH = "/app/models/test/experiment_v20251116.onnx"
DATASET_PATH = "test_dataset.json"
REPORT_CSV = "ab_report.csv"
REPORT_MD = "ab_report.md"
BATCH_LIMIT = 50


def call_rag_diagnosis(gnn_result, equipment_id):
    url = f"{RAG_SERVICE_URL}/interpret/diagnosis"
    payload = {
        "gnn_result": gnn_result,
        "equipment_context": {"equipment_id": equipment_id},
        "historical_context": None,
    }
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


client = GNNAdminClient(base_url=GNN_SERVICE_URL, admin_token=ADMIN_TOKEN)

with open(DATASET_PATH) as f:
    dataset = json.load(f)

rows = []


def safe_inference(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except Exception as e:
        return {"error": str(e)}


for i, record in enumerate(dataset[:BATCH_LIMIT]):
    eq_id = record["equipment_id"]
    t_win = record["time_window"]
    print(f"[{i + 1}/{len(dataset)}] {eq_id}")
    prod_inf = safe_inference(client.get_inference, eq_id, t_win)
    test_inf = safe_inference(client.test_inference, eq_id, t_win, TEST_MODEL_PATH)
    prod_diag = safe_inference(call_rag_diagnosis, prod_inf, eq_id)
    test_diag = safe_inference(call_rag_diagnosis, test_inf, eq_id)
    rows.append(
        {
            "equipment_id": eq_id,
            "prod_score": prod_inf.get("overall_health_score", None),
            "test_score": test_inf.get("overall_health_score", None),
            "prod_anomaly_cnt": len(prod_inf.get("anomalies", []))
            if isinstance(prod_inf.get("anomalies", []), list)
            else None,
            "test_anomaly_cnt": len(test_inf.get("anomalies", []))
            if isinstance(test_inf.get("anomalies", []), list)
            else None,
            "prod_summary": prod_diag.get("summary")
            if isinstance(prod_diag, dict)
            else str(prod_diag),
            "test_summary": test_diag.get("summary")
            if isinstance(test_diag, dict)
            else str(test_diag),
            "prod_reasoning": prod_diag.get("reasoning")
            if isinstance(prod_diag, dict)
            else str(prod_diag),
            "test_reasoning": test_diag.get("reasoning")
            if isinstance(test_diag, dict)
            else str(test_diag),
        }
    )

with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write("# Batch AB-тест моделей GNN + RAG\n")
    f.write(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(
        "\n| Equipment | ProdScore | TestScore | ProdAnom | TestAnom | ProdSummary | TestSummary |\n"
    )
    f.write(
        "|-----------|-----------|-----------|----------|----------|-------------|-------------|\n"
    )
    for row in rows:
        f.write(
            f"| {row['equipment_id']} | {row['prod_score']} | {row['test_score']} | {row['prod_anomaly_cnt']} | {row['test_anomaly_cnt']} | {row['prod_summary'][:60]} | {row['test_summary'][:60]} |\n"
        )
    f.write(f"\nВсего протестировано: {len(rows)} записей\n")

print(f"\nОтчет готов: {REPORT_CSV}, {REPORT_MD}")
