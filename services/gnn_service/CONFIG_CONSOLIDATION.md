# Configuration Consolidation Guide

**Purpose**: Unified configuration management for GNN Service Phase 2  
**Last Updated**: January 3, 2026

---

## Quick Reference

### Development Environment

```bash
PORT=8000
HOST=0.0.0.0
DEVEL=true

# Model
MODEL_VERSION=v2.1.0
MODEL_PATH=models/universal_temporal_gnn_v2.1.0.ckpt
DEVICE=cpu  # or cuda

# Inference
BATCH_SIZE=32
INFERENCE_TIMEOUT_S=30

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=logs/gnn_service.log
```

### Production Environment

```bash
PORT=8000
HOST=0.0.0.0
DEVEL=false

# Model
MODEL_VERSION=v2.1.0
MODEL_PATH=/opt/models/universal_temporal_gnn_v2.1.0.ckpt
DEVICE=cuda

# Inference
BATCH_SIZE=128
INFERENCE_TIMEOUT_S=60

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/gnn_service.log

# Monitoring
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318
```

---

## Configuration Sources

### Priority Order

1. **Environment Variables** (highest priority)
2. **config.py** (default values)
3. **Docker/K8s ConfigMaps** (deployment-specific)

### Environment Files

**Development**:
```bash
cp .env.example .env
# Edit .env with local values
```

**Production**:
```bash
# Use K8s ConfigMap
kubectl create configmap gnn-service-config --from-file=.env
```

---

## Environment Variables

### Service Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PORT` | int | 8000 | HTTP server port |
| `HOST` | str | 0.0.0.0 | Bind address |
| `WORKERS` | int | 4 | Uvicorn worker count |
| `DEBUG` | bool | false | Debug mode (dev only) |

### Model Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL_VERSION` | str | v2.1.0 | Model version tag |
| `MODEL_PATH` | str | models/v2.1.0.ckpt | Checkpoint path |
| `DEVICE` | str | auto | cpu, cuda, or auto |
| `PRECISION` | str | fp32 | fp32, fp16, or bf16 |
| `NUM_WORKERS` | int | 4 | DataLoader workers |

### Inference Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BATCH_SIZE` | int | 32 | Batch size for inference |
| `INFERENCE_TIMEOUT_S` | int | 30 | Max inference time |
| `ENABLE_DYNAMIC_BATCHING` | bool | true | Auto batching |
| `MAX_BATCH_WAIT_MS` | int | 50 | Max wait for batching |
| `MAX_BATCH_SIZE` | int | 128 | Max batch size |

### Database Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TIMESCALEDB_URL` | str | - | PostgreSQL connection |
| `DB_POOL_SIZE` | int | 20 | Connection pool size |
| `DB_QUERY_TIMEOUT_S` | int | 10 | Query timeout |

### Caching Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TOPOLOGY_CACHE_SIZE` | int | 100 | LRU cache items |
| `TOPOLOGY_CACHE_TTL_S` | int | 300 | Cache TTL seconds |
| `ENABLE_TOPOLOGY_CACHE` | bool | true | Enable caching |

### OpenTelemetry Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OTEL_ENABLED` | bool | true | Enable tracing |
| `OTEL_SERVICE_NAME` | str | gnn-service-v2 | Service name |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | str | - | Collector endpoint |
| `OTEL_EXPORTER_OTLP_TIMEOUT_MS` | int | 10000 | Export timeout |

### Rate Limiting Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RATE_LIMIT_ENABLED` | bool | true | Enable rate limiting |
| `RATE_LIMIT_REQUESTS` | int | 100 | Requests per window |
| `RATE_LIMIT_WINDOW_S` | int | 60 | Time window seconds |
| `REDIS_URL` | str | - | Redis for distributed limits |

### Security Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_BODY_SIZE_MB` | int | 10 | Request body limit |
| `ALLOWED_ORIGINS` | str | localhost:3000 | CORS allowed origins |
| `API_KEY` | str | - | API authentication key |
| `API_KEY_HEADER` | str | X-API-Key | Key header name |

### Logging Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_LEVEL` | str | INFO | DEBUG, INFO, WARNING, ERROR |
| `LOG_FILE` | str | logs/gnn_service.log | Log file path |
| `LOG_FORMAT` | str | standard | Log format (standard, json) |
| `LOG_RETENTION_DAYS` | int | 30 | Log retention |

---

## Configuration Files

### .env (Local Development)

```bash
# Copy template
cp .env.example .env

# Edit with your values
vi .env
```

**Key sections**:
```bash
# Service
PORT=8000
HOST=0.0.0.0
DEBUG=true

# Model
MODEL_VERSION=v2.1.0
MODEL_PATH=models/v2.1.0.ckpt
DEVICE=cpu

# Development
LOG_LEVEL=DEBUG
OTEL_ENABLED=false
```

### configs/config.py (Application Defaults)

```python
class Config:
    """Default configuration."""
    PORT = int(os.getenv('PORT', '8000'))
    HOST = os.getenv('HOST', '0.0.0.0')
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # Model
    MODEL_VERSION = os.getenv('MODEL_VERSION', 'v2.1.0')
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/v2.1.0.ckpt')
    DEVICE = os.getenv('DEVICE', 'auto')
    
    # Inference
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    INFERENCE_TIMEOUT_S = int(os.getenv('INFERENCE_TIMEOUT_S', '30'))
```

### K8s ConfigMap (Production)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gnn-service-config
  namespace: production
data:
  .env: |
    PORT=8000
    HOST=0.0.0.0
    MODEL_VERSION=v2.1.0
    MODEL_PATH=/opt/models/v2.1.0.ckpt
    DEVICE=cuda
    OTEL_ENABLED=true
    OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318
    RATE_LIMIT_ENABLED=true
    TIMESCALEDB_URL=postgresql://user:pass@pg:5432/hydraulic
```

---

## Configuration by Environment

### Development Setup

**Goal**: Quick iteration, local debugging

```bash
# .env
PORT=8000
DEBUG=true
DEVICE=cpu
LOG_LEVEL=DEBUG
OTEL_ENABLED=false
RATE_LIMIT_ENABLED=false
TIMESCALEDB_URL=postgresql://localhost:5432/hydraulic_dev
```

**Run**:
```bash
uvicorn src.api.main:app --reload
```

### Staging Setup

**Goal**: Pre-production testing

```bash
# .env (from K8s ConfigMap)
PORT=8000
DEVICE=cuda
LOG_LEVEL=INFO
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger-staging:4318
RATE_LIMIT_ENABLED=true
TIMESCALEDB_URL=postgresql://user:pass@pg-staging:5432/hydraulic_staging
```

### Production Setup

**Goal**: Performance, reliability, monitoring

```bash
# .env (from K8s Secret)
PORT=8000
WORKERS=8
DEVICE=cuda
PRECISION=fp16
LOG_LEVEL=WARNING  # Less logging = faster
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger-prod:4318
RATE_LIMIT_ENABLED=true
TIMESCALEDB_URL=postgresql://prod_user:***@pg-prod.rds.amazonaws.com:5432/hydraulic_prod
```

---

## Model Configuration (Phase 2)

### Model Checkpoint Paths

```bash
# Development (local)
model/v2.1.0.ckpt

# Docker
/app/models/v2.1.0.ckpt

# K8s (mounted from ConfigMap)
/opt/models/v2.1.0.ckpt

# S3 (auto-download)
s3://hydraulic-models/v2.1.0.ckpt
```

### Model Initialization

```python
from src.models import UniversalTemporalGNNv2, ModelConfig

# Config from environment
config = ModelConfig.from_env()  # Reads DEVICE, PRECISION, etc.

# Load model
model = UniversalTemporalGNNv2(config)
model.load_state_dict(torch.load(os.getenv('MODEL_PATH')))
model.to(os.getenv('DEVICE', 'cpu'))
model.eval()
```

---

## Validation

### Verify Configuration

```bash
python << 'EOF'
import os
from pathlib import Path

print("Validating configuration...\n")

# Check environment variables
required = ['MODEL_PATH', 'TIMESCALEDB_URL']
for var in required:
    value = os.getenv(var)
    if value:
        print(f"\u2705 {var}: {value[:50]}..." if len(value) > 50 else f"\u2705 {var}: {value}")
    else:
        print(f"\u274c {var}: NOT SET")

# Check model file exists
model_path = os.getenv('MODEL_PATH')
if Path(model_path).exists():
    print(f"\u2705 Model file exists: {model_path}")
else:
    print(f"\u274c Model file NOT FOUND: {model_path}")

print("\n✅ Configuration validation complete")
EOF
```

---

## Troubleshooting

### Issue: Model not loading

**Symptom**: `FileNotFoundError: [Errno 2] No such file or directory: 'models/v2.1.0.ckpt'`

**Fix**:
```bash
# Check MODEL_PATH
echo $MODEL_PATH

# Verify file exists
ls -la models/v2.1.0.ckpt

# Update .env
vi .env  # Set MODEL_PATH=/full/path/to/v2.1.0.ckpt
```

### Issue: CUDA not available

**Symptom**: `RuntimeError: CUDA is not available`

**Fix**:
```bash
# Check DEVICE
echo $DEVICE

# Switch to CPU
export DEVICE=cpu
# or in .env
DEVICE=cpu
```

### Issue: Database connection fails

**Symptom**: `psycopg2.OperationalError: FATAL: password authentication failed`

**Fix**:
```bash
# Verify TIMESCALEDB_URL
echo $TIMESCALEDB_URL

# Check credentials
psql postgresql://user:pass@host:5432/db

# Update .env with correct credentials
vi .env
```

---

## Best Practices

1. **Never commit secrets** to git
   - Use .env.example (without values)
   - Use K8s Secrets in production

2. **Use environment-specific configs**
   - .env for development
   - ConfigMap for staging/prod

3. **Validate on startup**
   ```python
   # In main.py startup event
   @app.on_event('startup')
   async def validate_config():
       assert os.getenv('MODEL_PATH'), "MODEL_PATH not set"
       assert Path(os.getenv('MODEL_PATH')).exists()
       assert os.getenv('DEVICE') in ['cpu', 'cuda', 'auto']
   ```

4. **Log configuration (masked)**
   ```python
   logger.info(f"Service starting: {config.SERVICE_NAME}")
   logger.info(f"Model: {config.MODEL_VERSION}")
   logger.info(f"Device: {config.DEVICE}")
   # Don't log: passwords, API keys, tokens
   ```

---

## Reference

- **[README.md](README.md)** - Service overview
- **[.env.example](.env.example)** - Example configuration
- **[configs/config.py](configs/config.py)** - Default values
- **[PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md)** - Deployment info

---

**Last Updated**: January 3, 2026  
**Scope**: Phase 2 GNN Service Configuration
