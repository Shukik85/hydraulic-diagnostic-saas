# Deployment Guide: Enterprise++ Architecture

## 🎯 Prerequisites

### Infrastructure Requirements

- **Kubernetes Cluster**: v1.28+
- **GPU Nodes**: NVIDIA T4/V100/A100
- **Storage**: EFS/NFS for model storage
- **Database**: TimescaleDB 2.13+
- **Cache**: Redis 7.0+

### Tools Required

```bash
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# istioctl
curl -L https://istio.io/downloadIstio | sh -
```

## Step 1: Install Istio Service Mesh

```bash
# 1. Install Istio
istioctl install --set profile=production -y

# 2. Enable sidecar injection
kubectl label namespace hydraulic-prod istio-injection=enabled

# 3. Apply namespace
kubectl apply -f infrastructure/istio/namespace.yaml

# 4. Apply mTLS policies
kubectl apply -f infrastructure/istio/peer-authentication.yaml

# 5. Apply authorization policies
kubectl apply -f infrastructure/istio/authorization-policies.yaml

# 6. Apply virtual services
kubectl apply -f infrastructure/istio/virtual-services.yaml

# 7. Apply destination rules
kubectl apply -f infrastructure/istio/destination-rules.yaml
```

### Verify Istio

```bash
# Check Istio components
kubectl get pods -n istio-system

# Verify mTLS
istioctl x describe pod <pod-name> -n hydraulic-prod
```

## Step 2: Deploy TimescaleDB

```bash
# 1. Add TimescaleDB Helm repo
helm repo add timescaledb https://charts.timescale.com

# 2. Install TimescaleDB
helm install timescaledb timescaledb/timescaledb-single \
  --namespace hydraulic-prod \
  --set replicaCount=3 \
  --set persistentVolume.size=500Gi \
  --set resources.requests.memory=16Gi \
  --set resources.requests.cpu=4

# 3. Create hypertables
kubectl exec -it timescaledb-0 -n hydraulic-prod -- psql -U postgres -d hydraulic_db
```

```sql
-- Create sensor_data hypertable
CREATE TABLE sensor_data (
    timestamp TIMESTAMPTZ NOT NULL,
    equipment_id TEXT NOT NULL,
    sensor_type TEXT NOT NULL,
    component_id TEXT,
    value DOUBLE PRECISION,
    unit TEXT,
    quality_flag INTEGER
);

SELECT create_hypertable(
    'sensor_data',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day'
);

-- Add space partitioning
SELECT add_dimension(
    'sensor_data',
    'equipment_id',
    number_partitions => 4
);

-- Create indexes
CREATE INDEX idx_sensor_data_equipment_time
ON sensor_data (equipment_id, timestamp DESC);

-- Enable compression
ALTER TABLE sensor_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'equipment_id, sensor_type',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('sensor_data', INTERVAL '7 days');
SELECT add_retention_policy('sensor_data', INTERVAL '365 days');
```

## Step 3: Deploy Redis Cluster

```bash
# 1. Add Bitnami repo
helm repo add bitnami https://charts.bitnami.com/bitnami

# 2. Install Redis Cluster
helm install redis-cluster bitnami/redis-cluster \
  --namespace hydraulic-prod \
  --set cluster.nodes=6 \
  --set cluster.replicas=1 \
  --set persistence.size=10Gi \
  --set password=<secure-password>
```

## Step 4: Deploy Kong API Gateway

```bash
# 1. Create Kong PostgreSQL database
kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: kong-postgres-secret
  namespace: hydraulic-prod
type: Opaque
stringData:
  password: <secure-password>
EOF

# 2. Deploy Kong
kubectl apply -f infrastructure/kong/kong-deployment.yaml

# 3. Apply Kong configuration
kubectl apply -f infrastructure/kong/kong-config.yaml

# 4. Verify Kong
kubectl get pods -n hydraulic-prod -l app=api-gateway
```

## Step 5: Build and Push GNN Service Image

```bash
# 1. Build Docker image
cd services/gnn_service
docker build -f Dockerfile.production -t ghcr.io/shukik85/gnn-service:1.0.0-cuda12.8 .

# 2. Push to registry
docker push ghcr.io/shukik85/gnn-service:1.0.0-cuda12.8
```

## Step 6: Deploy GNN Service

```bash
# 1. Create secrets
kubectl create secret generic timescaledb-credentials \
  --from-literal=host=timescaledb.hydraulic-prod.svc.cluster.local \
  --from-literal=port=5432 \
  --from-literal=username=postgres \
  --from-literal=password=<db-password> \
  -n hydraulic-prod

kubectl create secret generic jwt-secret \
  --from-literal=secret-key=<jwt-secret> \
  -n hydraulic-prod

# 2. Deploy GNN service
kubectl apply -f services/gnn_service/kubernetes/deployment.yaml

# 3. Deploy HPA
kubectl apply -f services/gnn_service/kubernetes/hpa.yaml

# 4. Verify deployment
kubectl get pods -n hydraulic-prod -l app=gnn-service
kubectl logs -f deployment/gnn-service -n hydraulic-prod
```

## Step 7: Deploy Monitoring Stack

```bash
# 1. Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# 2. Install Grafana dashboards
kubectl apply -f infrastructure/monitoring/dashboards/

# 3. Install Jaeger
helm install jaeger jaegertracing/jaeger \
  --namespace monitoring
```

## Step 8: Configure DNS and TLS

```bash
# 1. Create TLS certificate
kubectl create secret tls hydraulic-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n hydraulic-prod

# 2. Configure Ingress
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hydraulic-ingress
  namespace: hydraulic-prod
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - api.hydraulic-diagnostics.com
      secretName: hydraulic-tls
  rules:
    - host: api.hydraulic-diagnostics.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: kong-proxy
                port:
                  number: 80
EOF
```

## Step 9: Verify Deployment

```bash
# 1. Check all pods
kubectl get pods -n hydraulic-prod

# 2. Check services
kubectl get svc -n hydraulic-prod

# 3. Test API Gateway
curl -k https://api.hydraulic-diagnostics.com/health

# 4. Test GNN service
kubectl port-forward svc/gnn-service 8002:8002 -n hydraulic-prod
curl http://localhost:8002/health

# 5. Check Istio mTLS
istioctl x describe pod gnn-service-<pod-id> -n hydraulic-prod
```

## Step 10: Production Checklist

- [ ] Istio service mesh deployed
- [ ] mTLS enabled for all services
- [ ] Authorization policies configured
- [ ] TimescaleDB deployed with hypertables
- [ ] Redis cluster deployed
- [ ] Kong API Gateway configured
- [ ] GNN service deployed with GPU support
- [ ] HPA configured for auto-scaling
- [ ] Monitoring stack deployed
- [ ] TLS certificates configured
- [ ] DNS records created
- [ ] Backup policies configured
- [ ] Disaster recovery tested
- [ ] Load testing completed
- [ ] Security audit passed

## 🔧 Troubleshooting

### Pod Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n hydraulic-prod

# Check logs
kubectl logs <pod-name> -n hydraulic-prod

# Check sidecar logs
kubectl logs <pod-name> -c istio-proxy -n hydraulic-prod
```

### mTLS Issues

```bash
# Verify certificates
istioctl proxy-config secret <pod-name> -n hydraulic-prod

# Check mTLS status
istioctl authn tls-check <pod-name> <target-service> -n hydraulic-prod
```

### Database Connection Issues

```bash
# Test connection
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h timescaledb.hydraulic-prod.svc.cluster.local -U postgres
```

## 📊 Monitoring URLs

- **Grafana**: http://grafana.monitoring.svc.cluster.local:3000
- **Prometheus**: http://prometheus.monitoring.svc.cluster.local:9090
- **Jaeger**: http://jaeger.monitoring.svc.cluster.local:16686
- **Kiali** (Istio): http://kiali.istio-system.svc.cluster.local:20001

## 🔒 Security Hardening

1. **Rotate certificates** каждые 90 дней
2. **Update secrets** через external secret manager
3. **Enable audit logs** для всех API calls
4. **Configure network policies** для pod-level isolation
5. **Enable Pod Security Standards**
