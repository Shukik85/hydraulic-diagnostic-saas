# Hydraulic Diagnostic Service - Production Ready

## Quick Start

### Local Development
```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r services/api/requirements.txt
cd services/api
uvicorn app.main:app --reload
```

### Docker
```bash
docker build -t hydraulic-service:1.0.0 .
docker run --gpus all -p 8000:8000 hydraulic-service:1.0.0
```

### Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get deployment hydraulic-service
```

## Features
- PyTorch 2.8.0 with CUDA 12.1
- Global exception handling
- Health checks for K8s
- TimescaleDB hypertables
- Docker multi-stage build
- Kubernetes auto-scaling
