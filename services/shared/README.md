# HDX Shared Package

Shared utilities, schemas, and clients for Hydraulic Diagnostics SaaS.

## Installation

```bash
# Development (editable)
pip install -e .

# Production
pip install .
Usage
GNN Client
text
from shared.clients import GNNClient, GNNPredictionRequest

client = GNNClient(base_url="http://gnn-service:8001")

request = GNNPredictionRequest(
    sensor_data=[...],
    timestamp="2025-11-13T22:00:00Z"
)

result = await client.predict(request)
print(result.predictions)
Equipment Schemas
python
from shared.schemas import SensorData, ComponentType, HydraulicSystem

sensor = SensorData(
    sensor_id="P001",
    sensor_type="pressure",
    timestamp=datetime.now(),
    value=150.5,
    unit="bar"
)
Validation
text
from shared.validation import validate_sensor_batch

result = validate_sensor_batch(sensors)
print(f"Valid: {result['valid_count']}")
print(f"Invalid: {result['invalid_count']}")
Structure
text
shared/
├── clients/          # API clients (GNN, RAG, etc.)
├── schemas/          # Pydantic models
├── validation/       # Data validation
├── utils/            # Utilities
└── middleware/       # FastAPI middleware
Development
text
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy shared/
