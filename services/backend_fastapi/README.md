# Backend FastAPI - Core Microservice

## 🎯 Overview

Production-ready FastAPI backend for Hydraulic Diagnostics SaaS. Replaces Django DRF with high-performance async architecture.

## 🚀 Features

- **Equipment Metadata Management** - Dynamic topology support for any hydraulic system
- **Sensor Data Ingestion** - High-throughput batch ingestion with validation
- **User Management** - Authentication, authorization, subscription tiers
- **GNN Integration** - Seamless connection to GNN inference service
- **TimescaleDB** - Optimized time-series storage with compression & retention
- **API-First** - Auto-generated OpenAPI documentation
- **Type Safety** - Pydantic schemas for validation
- **Async Native** - Non-blocking I/O for maximum performance

## 📦 Installation

```bash
cd services/backend_fastapi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

## 🔧 Configuration

Create `.env` file:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/hydraulic_db
REDIS_URL=redis://localhost:6379/0
GNN_SERVICE_URL=http://localhost:8001
SECRET_KEY=your-secret-key-change-in-production
DEBUG=True
ENV=development
```

## 🏃 Running

### Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker-compose up --build
```

## 📡 API Endpoints

### Health
- `GET /health/` - Basic health check
- `GET /health/ready` - Readiness probe (checks DB)
- `GET /health/live` - Liveness probe

### Users
- `POST /users/register` - Register new user
- `GET /users/me` - Get current user info
- `GET /users/{user_id}/usage` - Get usage statistics

### Metadata
- `POST /metadata/` - Create equipment
- `GET /metadata/{system_id}` - Get equipment
- `GET /metadata/` - List all equipment
- `PATCH /metadata/{system_id}` - Update equipment
- `DELETE /metadata/{system_id}` - Delete equipment

### Ingestion
- `POST /ingestion/ingest` - Ingest sensor data batch
- `GET /ingestion/systems/{system_id}/latest` - Get latest readings

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific test file
pytest tests/test_metadata.py -v
```

## 🗄️ Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## 📊 Performance

- **Throughput**: ~7,000 requests/sec (vs Django's ~1,500)
- **Latency (p95)**: <15ms (vs Django's ~80ms)
- **Memory**: ~50MB per worker
- **Concurrency**: Native async, non-blocking I/O

## 🔒 Security

- API key authentication
- Rate limiting per subscription tier
- Input validation (Pydantic)
- SQL injection prevention (SQLAlchemy)
- CORS configuration
- Environment-based secrets

## 📈 Monitoring

- Prometheus metrics at `/metrics`
- Structured logging (JSON format)
- OpenTelemetry tracing support
- Health check endpoints for orchestration

## 🛣️ Roadmap

- [x] Equipment metadata CRUD
- [x] Sensor data ingestion
- [x] User management
- [x] GNN service integration
- [ ] WebSocket real-time streaming
- [ ] Billing integration (Stripe)
- [ ] RAG service integration
- [ ] Advanced analytics API

## 🤝 Integration

### With GNN Service
```python
from services.gnn_client import GNNClient

client = GNNClient()
result = await client.infer(user_id="123", system_id="press_01")
# {"anomaly_scores": {"pump": 0.05, "valve": 0.87}, ...}
```

### With Frontend (Nuxt)
```typescript
// Auto-generated TypeScript client
import { MetadataApi } from '~/api/generated'

const api = new MetadataApi()
const equipment = await api.createEquipment({...})
```

## 📞 Support

Questions: shukik85@ya.ru
