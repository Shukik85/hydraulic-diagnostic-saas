# Enterprise++ Architecture Guide

## 🏗️ Обзор

Этот документ описывает enterprise-grade архитектуру для Hydraulic Diagnostic SaaS с:

- ✅ Zero-Trust Security (mTLS, continuous authentication)
- ✅ Service Mesh (Istio) для secure inter-service communication
- ✅ API Gateway (Kong) с rate limiting и JWT validation
- ✅ Multi-tenancy с data isolation
- ✅ Production-ready GNN service с GPU support
- ✅ Comprehensive observability (Prometheus, Grafana, Jaeger)
- ✅ Enterprise SSO (SAML, OIDC)
- ✅ Audit logging для compliance

## 🛡️ Архитектура

### Layers Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         EXTERNAL LAYER                          │
│                    (Public Internet / VPN)                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                        ┌─────────▼─────────┐
                        │   CloudFlare CDN  │
                        │   + WAF + DDoS    │
                        └─────────┬─────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  TLS Termination  │
                        │  (AWS ALB / NLB)  │
                        └─────────┬─────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│                          EDGE LAYER                                │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              API Gateway (Kong / Ambassador)                  │ │
│  │  • JWT Validation                                             │ │
│  │  • Rate Limiting (1000 req/min per tenant)                    │ │
│  │  • Request Routing                                            │ │
│  │  • Response Caching                                           │ │
│  │  • CORS / CSRF Protection                                     │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────┼───────────────┐
              │                   │                   │
┌─────────────▼─────────────────────────────────────────────────┐
│                    SERVICE MESH (Istio)                            │
│              Data Plane: Envoy Sidecars (mTLS)                    │
│              Control Plane: Istiod (Policy Distribution)          │
└───────────────────────────────────────────────────────────────┘
              │                   │                   │
    ┌─────────▼──────┐  ┌────────▼────────┐  ┌──────▼─────────┐
    │  Auth Service  │  │ Equipment       │  │  Diagnosis     │
    │                │  │ Service         │  │  Service       │
    │ • JWT Gen      │  │                 │  │                │
    │ • SSO          │  │ • CRUD Systems  │  │ • Orchestrator │
    │ • RBAC         │  │ • Metadata      │  │ • Queue Mgmt   │
    │ • Audit Log    │  │ • Multi-tenant  │  │ • Workflow     │
    └────────┬───────┘  └────────┬────────┘  └────────┬───────┘
             │                   │                     │
             │          ┌────────▼────────┐           │
             │          │  GNN Service    │◄──────────┘
             │          │                 │
             │          │ • Inference     │
             │          │ • GPU Pool      │
             │          │ • Model Versioning│
             │          └────────┬────────┘
             │                   │
┌────────────▼───────────────▼─────────────────────────────────┐
│                        DATA LAYER                                 │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐       │
│  │ TimescaleDB  │  │ Redis Cluster │  │  S3 / MinIO    │       │
│  │              │  │               │  │                │       │
│  │ • Time-series│  │ • Session     │  │ • Models       │       │
│  │ • Compression│  │ • Cache       │  │ • Reports      │       │
│  │ • Retention  │  │ • Rate Limit  │  │ • Backups      │       │
│  └──────────────┘  └───────────────┘  └────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Service Communication Matrix

| Source Service | Target Service | Protocol | Auth Method | Purpose |
|----------------|----------------|----------|-------------|---------|  
| **Web Browser** | `api-gateway` | HTTPS/WSS | JWT Bearer Token | External entry point |
| `api-gateway` | `auth-service` | gRPC/mTLS | Service Account | Token validation |
| `api-gateway` | `equipment-service` | gRPC/mTLS | Service Account | CRUD operations |
| `api-gateway` | `diagnosis-service` | gRPC/mTLS | Service Account | Orchestration |
| `diagnosis-service` | `gnn-service` | gRPC/mTLS | Service Account | ML inference |
| `gnn-service` | `timescaledb` | PostgreSQL/TLS | DB credentials | Data retrieval |
| `equipment-service` | `timescaledb` | PostgreSQL/TLS | DB credentials | Metadata storage |
| `auth-service` | `redis-cluster` | Redis/TLS | Password | Session cache |

## 🔐 Security Features

### 1. Zero-Trust Architecture

- **mTLS everywhere**: Все inter-service connections используют mutual TLS
- **Continuous authentication**: JWT tokens проверяются на каждом request
- **Device fingerprinting**: Отслеживание device changes
- **IP whitelisting**: Опциональное ограничение по IP

### 2. Enterprise SSO

- **SAML 2.0**: Для enterprise identity providers
- **OIDC**: Google, Azure AD, Okta support
- **Custom mapping**: Role и permission mapping

### 3. Audit Logging

- **Tamper-evident**: Cryptographic hashing
- **Compliance**: SOC 2, ISO 27001 ready
- **SIEM integration**: Splunk, ELK, Datadog

## 🚀 Deployment

См. [DEPLOYMENT.md](./DEPLOYMENT.md) для подробных инструкций.

## 📈 Performance

### SLA Targets

- **Availability**: 99.95% uptime
- **Latency**: p99 < 500ms
- **Throughput**: 10,000 req/s per region
- **Error Rate**: < 0.1%

### Auto-Scaling

- **HPA**: Основано на CPU, memory, custom metrics
- **Min replicas**: 3 per service
- **Max replicas**: 20 (API Gateway), 12 (GNN Service)

## 📊 Observability

### Metrics (Prometheus)

- Request latency histograms
- Error rate counters
- GPU utilization gauges
- Database connection pool metrics

### Tracing (Jaeger)

- Distributed request tracing
- Service dependency mapping
- Performance bottleneck identification

### Logging (Structured)

- JSON format
- Correlation IDs
- Log aggregation в ELK/Datadog

## 🛠️ Development

### Local Setup

```bash
# 1. Install dependencies
pip install -r requirements-prod.txt

# 2. Setup environment
cp .env.example .env

# 3. Start local services
docker-compose up -d

# 4. Run service
uvicorn main:app --reload
```

### Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Load testing
locust -f tests/load/locustfile.py
```

## 📝 License

Proprietary - Hydraulic Diagnostic SaaS
