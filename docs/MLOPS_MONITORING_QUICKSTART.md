# Production Monitoring & MLOps: Быстрый старт

## 1. Prometheus

### Подключить Alert Rules
- Добавьте в prometheus.yml:
  ```yaml
  rule_files:
    - '/etc/prometheus/alerts/prometheus-alerts.yml'
  ```
- Поместите файл `prometheus-alerts.yml` в директорию alerts (или путь выше)

## 2. Grafana

### Импортировать Dashboard
- Войдите в Grafana (`http://grafana:3000/`)
- "Dashboards" → "+ Import"
- Загрузите файл `grafana-mlops-dashboard.json`
- Назначьте DataSource: Prometheus

## 3. Проверка/Smoke test
```bash
bash scripts/run_all_monitoring_mlops_tests.sh
```
- Проверит: health, metrics, drift, admin endpoints — результаты в /tmp

## 4. Alerting (Email/Slack)
- В Alertmanager настройте route для критичных alert (см. документацию Prometheus)
- Пример:
  ```yaml
  receivers:
    - name: 'slack-notifications'
      slack_configs:
        - api_url: 'https://hooks.slack.com/services/...'  # ваш url
          channel: '#mlops-alerts'
          send_resolved: true
  ```
- Реагировать на: ModelDriftDetected, HighErrorRate, NotReady alerts

## 5. Troubleshooting
- При ошибках сервисов: логи + /tmp/* из smoke test
- Для проверки алертов — вручную повысьте drift/inject error через diagnosis_service/mlops

## 6. SLA/OKR Метрики
- Графики SLA и Error Rate/Downtime доступны по dashboard панели
- Экспорт dashboard возможен через меню Grafana

---
**Продакшн мониторинг готов к Go-Live**
- Контакты DevOps/MLops указаны в runbook
