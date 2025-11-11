# 🔗 Level 6 Integration Guide

## Overview

Level 6 добавляет **Sensor Mapping** и **Data Source Setup** к equipment initialization workflow.

---

## 📊 Architecture

```
Level 5: Validation
  ↓
Level 6A: Sensor Mapping ← NEW
  ↓
Level 6B: Data Source Setup ← NEW
  ├─→ CSV Upload
  ├─→ Simulator
  └─→ (Future: IoT Gateway, API Polling)
  ↓
System Ready ✓
```

---

## 🗄️ Database Migrations

### 1. Create sensor_mappings table

```bash
cd services/backend_fastapi
alembic revision --autogenerate -m "Add sensor_mappings table"
alembic upgrade head
```

### 2. Create data_sources table

Already included in migration above.

---

## 🔌 Backend Integration

### 1. Register new API routes

```python
# services/backend_fastapi/main.py
from api import sensor_mapping, csv_upload, simulator

app.include_router(sensor_mapping.router)
app.include_router(csv_upload.router)
app.include_router(simulator.router)
```

### 2. Update Equipment model

```python
# services/backend_fastapi/models/equipment.py
from sqlalchemy.orm import relationship

class Equipment(Base):
    # ... existing fields ...

    # Add relationship
    sensor_mappings = relationship("SensorMapping", back_populates="equipment")
```

---

## 🎨 Frontend Integration

### 1. Add to metadata wizard flow

```typescript
// services/frontend/pages/equipment/wizard.vue
const steps = [
  { name: 'Basic Info', component: 'Level1BasicInfo' },
  { name: 'Graph Builder', component: 'Level2GraphBuilder' },
  { name: 'Component Details', component: 'Level3ComponentDetails' },
  { name: 'Duty Cycle', component: 'Level4DutyCycle' },
  { name: 'Validation', component: 'Level5Validation' },
  { name: 'Sensor Mapping', component: 'Level6SensorMapping' },  // NEW
  { name: 'Data Source', component: 'Level6DataSource' }         // NEW
]
```

### 2. Handle completion

```typescript
const handleLevel6AComplete = async (data) => {
  // Sensor mappings saved
  console.log('Mapped sensors:', data.mappings)

  // Show data source modal
  showDataSourceSetup.value = true
}

const handleLevel6BComplete = async (method) => {
  if (method === 'skip') {
    // Navigate to dashboard
    navigateTo(`/equipment/${equipmentId}`)
  } else {
    // Navigate to specific setup
    if (method === 'csv') {
      navigateTo(`/equipment/${equipmentId}/upload-csv`)
    } else if (method === 'simulator') {
      navigateTo(`/equipment/${equipmentId}/simulator`)
    }
  }
}
```

---

## 🧪 Testing

### 1. Test sensor mapping auto-detection

```bash
curl -X POST http://localhost:8100/api/sensor-mappings/equipment/{equipment_id}/auto-detect \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "available_sensors": ["pump_P1", "pump_T1", "valve_P1"]
  }'
```

### 2. Test CSV upload

```bash
# Create test CSV
echo "timestamp,pump_P1,pump_T1
2025-11-11T20:00:00,220.5,65.2
2025-11-11T20:00:01,221.0,65.3" > test.csv

# Validate
curl -X POST http://localhost:8100/api/csv-upload/validate \
  -F "file=@test.csv" \
  -F "equipment_id=YOUR_EQUIPMENT_ID"

# Import
curl -X POST http://localhost:8100/api/csv-upload/import \
  -F "file=@test.csv" \
  -F "equipment_id=YOUR_EQUIPMENT_ID"
```

### 3. Test simulator

```bash
curl -X POST http://localhost:8100/api/simulator/start \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "YOUR_EQUIPMENT_ID",
    "scenario": "normal",
    "duration": 300,
    "noise_level": 0.1,
    "sampling_rate": 10
  }'
```

---

## 📈 GNN Integration

### How sensor mappings enable GNN inference:

```python
# services/gnn_service/data_prep.py
async def prepare_graph_data(equipment_id: str):
    # 1. Get equipment metadata
    equipment = await get_equipment(equipment_id)

    # 2. Get sensor mappings
    mappings = await get_sensor_mappings(equipment_id)

    # 3. Get latest sensor readings
    readings = await get_latest_readings(equipment_id, limit=100)

    # 4. Build node features
    node_features = []
    for component_idx in range(len(equipment.components)):
        # Get sensors for this component
        component_sensors = [
            m for m in mappings 
            if m.component_index == component_idx
        ]

        features = []
        for mapping in component_sensors:
            # Get readings for this sensor
            sensor_data = [
                r for r in readings 
                if r.sensor_id == mapping.sensor_id
            ]

            if sensor_data:
                # Extract features
                latest = sensor_data[-1].value
                mean = np.mean([r.value for r in sensor_data])
                std = np.std([r.value for r in sensor_data])

                # Normalize using expected range
                normalized = (latest - mapping.expected_range_min) / \
                            (mapping.expected_range_max - mapping.expected_range_min)

                features.extend([normalized, mean, std])

        node_features.append(features)

    # 5. Create PyTorch Geometric data
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(equipment.adjacency_matrix).t()
    )

    return data
```

---

## ✅ Checklist

Before deploying Level 6:

- [ ] Database migrations applied
- [ ] API routes registered
- [ ] Frontend components integrated
- [ ] CSV template generation works
- [ ] Auto-detection tested
- [ ] Simulator produces valid data
- [ ] GNN data prep updated
- [ ] End-to-end flow tested

---

## 🚀 Deployment

```bash
# 1. Backend
cd services/backend_fastapi
alembic upgrade head
docker-compose restart backend_fastapi

# 2. Frontend
cd services/frontend
npm run build
docker-compose restart frontend

# 3. Verify
curl http://localhost:8100/health/
```

---

## 📞 Support

Questions? Check:
- `/docs/FRONTEND_DEVELOPER_GUIDE.md`
- `/docs/API_EXAMPLES.md`
- API docs: `http://localhost:8100/docs`
