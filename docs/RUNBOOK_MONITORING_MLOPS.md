# Runbook: Monitoring & MLOps Endpoints (hydraulic-diagnostic-saas)

## Назначение
Подробное руководство по CI/CD, проверке и мониторингу production-ready MLops инфраструктуры.

---

## Как использовать

1. Запустите сервисы diagnosis_service (8003), gnn_service (8002), rag_service (8004)
2. Выполните:
```bash
bash scripts/run_all_monitoring_mlops_tests.sh
```
3. Проверяйте результаты в stdout и папках /tmp (curl-результаты)
4. Для CI: подключите этот скрипт в тестовый этап

---

## Покрытие скрипта и тестов

- Health/ready/prometheus endpoints на diagnosis/gnn/rag
- Drift & AB test runner (фоновые задачи)
- Проверка админ endpoint’ов с JWT
- Проверка экспорта ML метрик и status code

---

## Экспорт артефактов (для CI)

Добавьте upload (github/gitlab):
```yaml
- name: Archive endpoint test results
  uses: actions/upload-artifact@v3
  with:
    name: endpoint-logs
    path: /tmp/health_*.json,/tmp/ready_*.json,/tmp/metrics_*.txt
```

---

## Критические точки и реакции

- Endpoint  Health/Ready не работает → Разблокировать сервис, перезапустить деплой
- Drift >0.3 → Команда информируется, retrain/rollback через /admin endpoints
- Ошибка в CI — лог, статус, publish артефактов, ручная верификация

---

## Контакты
- Все вопросы по эксплуатации — #mlops или #devops-support канал
- Для расширения/новых сценариев см. примеры в `tests/` и `scripts/`

---

Runbook актуален для hydraulic-diagnostic-saas v1.0+ и всех production деплоев. SLA — full infra transparency.
