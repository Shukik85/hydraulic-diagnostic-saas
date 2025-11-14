# 🎯 GNN Service Production Refactoring Plan

## Дата: 14 ноября 2025, 02:23 MSK
## Дедлайн Production: 15 ноября 2025

---

## 🔴 КРИТИЧНЫЕ ЗАДАЧИ (Сегодня)

### 1. Production Inference Engine
**Файл**: `services/gnn_service/inference/engine.py`
- ✅ Dynamic request batching (max 50ms latency)
- ✅ Model warmup на старте
- ✅ GPU memory management
- ✅ Async processing queue
- ✅ Circuit breaker pattern
- ✅ Request timeout handling

### 2. Modern Configuration System
**Файл**: `services/gnn_service/core/config.py`
- ✅ Pydantic Settings v2
- ✅ Environment variable validation
- ✅ Type-safe configuration
- ✅ .env file support
- ✅ Runtime validation

### 3. Database Lifecycle Management
**Файл**: `services/gnn_service/db/manager.py`
- ✅ AsyncPG pool с proper lifecycle
- ✅ FastAPI dependency injection
- ✅ Connection health monitoring
- ✅ Graceful shutdown
- ✅ Retry logic with exponential backoff

### 4. Observability Stack
**Файлы**:
- `services/gnn_service/core/logging.py` - Structured logging
- `services/gnn_service/core/metrics.py` - Prometheus metrics
- `services/gnn_service/core/tracing.py` - OpenTelemetry tracing
- ✅ JSON logging format
- ✅ Request correlation IDs
- ✅ Distributed tracing
- ✅ Custom metrics (inference latency, GPU usage, etc.)

### 5. Demo Systems Integration
**Файл**: `services/gnn_service/demo/systems.py`
- ✅ Excavator demo system
- ✅ Injection molding machine
- ✅ CNC machine
- ✅ Industrial robot
- ✅ Automated generation via CLI

### 6. Docker Optimization
**Файл**: `services/gnn_service/Dockerfile.production.v2`
- ✅ Multi-stage build optimization
- ✅ PyTorch 2.5.1 + CUDA 12.6
- ✅ Layer caching strategy
- ✅ Non-root user security
- ✅ Health check integration

### 7. Dependencies Update
**Файл**: `services/gnn_service/requirements-2025.txt`
- ✅ Python 3.13 compatibility
- ✅ PyTorch 2.5.1
- ✅ torch-geometric 2.6.0
- ✅ FastAPI 0.115.0
- ✅ Pydantic 2.9.0
- ✅ OpenTelemetry stack

---

## 📋 СТРУКТУРА НОВЫХ ФАЙЛОВ

```
services/gnn_service/
├── core/                           # NEW: Core utilities
│   ├── __init__.py
│   ├── config.py                   # Pydantic Settings v2
│   ├── logging.py                  # Structured logging
│   ├── metrics.py                  # Prometheus metrics
│   ├── tracing.py                  # OpenTelemetry
│   └── exceptions.py               # Custom exceptions
│
├── inference/                      # NEW: Production inference
│   ├── __init__.py
│   ├── engine.py                   # Batching inference engine
│   ├── preprocessor.py             # Data preprocessing
│   ├── postprocessor.py            # Result post-processing
│   └── cache.py                    # Response caching
│
├── db/                             # NEW: Database layer
│   ├── __init__.py
│   ├── manager.py                  # Connection pool manager
│   ├── repositories.py             # Data access layer
│   └── queries.py                  # Optimized SQL queries
│
├── demo/                           # NEW: Demo systems
│   ├── __init__.py
│   ├── systems.py                  # System definitions
│   ├── generator.py                # Data generator
│   └── cli.py                      # CLI commands
│
├── models/                         # REFACTORED: Model definitions
│   ├── __init__.py
│   ├── gnn.py                      # UniversalTemporalGNN
│   ├── loader.py                   # Model loading utilities
│   └── registry.py                 # Model version registry
│
├── api/                            # REFACTORED: API routes
│   ├── __init__.py
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── inference.py            # Inference endpoints
│   │   ├── admin.py                # Admin endpoints
│   │   └── monitoring.py           # Health/metrics
│   └── dependencies.py             # FastAPI dependencies
│
├── main.py                         # REFACTORED: Application entry
├── Dockerfile.production.v2        # NEW: Optimized Dockerfile
├── requirements-2025.txt           # NEW: Updated dependencies
├── pyproject.toml                  # NEW: Project configuration
└── docker-compose.production.yml   # NEW: Production compose
```

---

## 🔧 КЛЮЧЕВЫЕ УЛУЧШЕНИЯ

### Inference Engine
- **До**: Синхронная обработка, без batching
- **После**: Async batching с max 50ms latency, GPU memory pooling

### Configuration
- **До**: Dataclass с global singleton
- **После**: Pydantic Settings v2 с валидацией и env support

### Database
- **До**: Singleton без lifecycle management
- **После**: FastAPI dependency injection с proper shutdown

### Observability
- **До**: Простой print logging
- **После**: Structured JSON logs + Prometheus + OpenTelemetry

### Docker
- **До**: 2.5GB image, CUDA 12.8, security issues
- **После**: 1.2GB optimized image, CUDA 12.6, non-root user

### Dependencies
- **До**: PyTorch 2.2.0, Python 3.11
- **После**: PyTorch 2.5.1, Python 3.13, modern stack

---

## 📊 ОЖИДАЕМЫЕ МЕТРИКИ

### Performance
- Inference latency: **200ms → 80ms** (p95)
- Throughput: **10 req/s → 50 req/s**
- GPU utilization: **30% → 75%**
- Memory footprint: **4GB → 2.5GB**

### Reliability
- Uptime: **99.5% → 99.9%**
- Error rate: **5% → 0.1%**
- Recovery time: **Manual → Auto (30s)**

### Observability
- Log volume: **100 lines/min → 1000 events/min (structured)**
- Metrics: **0 → 50+ custom metrics**
- Traces: **None → Full distributed tracing**

---

## 🎯 IMPLEMENTATION PRIORITY

**Day 1 (Сегодня, 14 ноября)**
1. ✅ Core infrastructure (config, logging, metrics)
2. ✅ Production inference engine
3. ✅ Database lifecycle management
4. ✅ Demo systems integration

**Day 2 (15 ноября - дедлайн)**
1. ✅ Docker optimization
2. ✅ Integration testing
3. ✅ Production deployment
4. ✅ Monitoring setup

---

## 🚀 НАЧИНАЕМ ИМПЛЕМЕНТАЦИЮ!

**Статус**: 🟢 READY TO START
**ETA**: 24 часа до production
**Confidence**: 95%

Поехали! 🔥
