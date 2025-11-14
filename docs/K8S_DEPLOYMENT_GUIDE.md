# Hydraulic Diagnostic SaaS — Kubernetes Production Deployment Guide

## 1. Namespace preparation
```bash
kubectl create namespace hydraulic-prod || true
```

## 2. Secrets and ConfigMap
```bash
kubectl apply -f infrastructure/k8s/secrets-config.yaml
```
*Отредактируйте секреты/конфиг, заточив под ваши пароли и endpoints!*

## 3. Deploy core services
```bash
kubectl apply -f infrastructure/k8s/diagnosis-service.yaml
kubectl apply -f infrastructure/k8s/gnn-service.yaml
kubectl apply -f infrastructure/k8s/rag-service.yaml
```

## 4. Проверка и мониторинг
```bash
kubectl get pods,svc,hpa -n hydraulic-prod
kubectl describe deployment diagnosis-service -n hydraulic-prod
```
- `kubectl logs POD -n hydraulic-prod` — логи
- Ждать состояния READY для всех pods

## 5. Smoke test (локально/CI)
Проброс порта:
```bash
kubectl port-forward svc/diagnosis-service 8003:8003 -n hydraulic-prod
kubectl port-forward svc/gnn-service 8002:8002 -n hydraulic-prod
kubectl port-forward svc/rag-service 8004:8004 -n hydraulic-prod
```
Затем:
```bash
curl http://localhost:8003/health
curl http://localhost:8002/health
curl http://localhost:8004/health
```

## 6. Rollout & zero-downtime обновление
- Обновляйте образ через `kubectl set image ...`
- Kubernetes гарантирует rolling update и сохранение minReplicas
- Проверяйте состояние деплоя: `kubectl rollout status deployment/diagnosis-service -n hydraulic-prod`

## 7. Troubleshooting & rollback
- Если pod не READY: проверьте `kubectl describe pod ...` и логи
- Rollback: `kubectl rollout undo deployment/diagnosis-service -n hydraulic-prod`

## 8. Мониторинг и HPA
- Всё авто-скейлится на основе CPU/memory.
- Метрики/alert читаются Prometheus (см. monitoring quickstart)

---
**Успешного деплоя!** Для продвинутого GitOps/Helm/backup/restore и disaster recovery — см. отдельный гайд.
