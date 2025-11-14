#!/bin/bash
# run_all_monitoring_mlops_tests.sh
# Запускает все тесты по мониторингу и MLOps (pytest, curl, результаты)
set -e

PYTEST=${PYTEST:-pytest}

echo "[1/3] Pytest: Full Monitoring/MLOps + RAG endpoints..."
$PYTEST tests/test_monitoring_and_mlops.py
$PYTEST services/rag_service/tests/test_rag_endpoints.py

echo "[2/3] Smoke test endpoints via curl..."
for svc in 8002 8003 8004; do
  curl -s "http://localhost:$svc/health" | tee /tmp/health_$svc.json
  curl -s "http://localhost:$svc/ready" | tee /tmp/ready_$svc.json
  curl -s "http://localhost:$svc/metrics" | grep model_ | tee /tmp/metrics_$svc.txt
  echo "------"
done

# 3. Summarize results
echo '[3/3] All monitoring, drift & admin endpoint tests executed.'
