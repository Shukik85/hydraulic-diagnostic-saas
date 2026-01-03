# GNN Service API Documentation

**Version:** 2.0.0  
**Base URL:** `http://localhost:8000`  
**Date:** 2025-12-03

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
   - [v2 Endpoints (NEW)](#v2-endpoints)
   - [v1 Endpoints (Legacy)](#v1-endpoints)
   - [Health Checks](#health-checks)
4. [Request/Response Schemas](#schemas)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Examples](#examples)
8. [Migration Guide](#migration-guide)

---

## Overview

The GNN Service provides RESTful API endpoints for hydraulic system diagnostics using Graph Neural Networks.

### Key Features

- **Real-time diagnostics** - Sub-200ms inference
- **Multi-label classification** - Health, degradation, 9 anomaly types
- **Topology templates** - Pre-configured system templates
- **Dynamic edge features** - Physics-based flow estimation
- **Backward compatible** - v1 API still supported

### Architecture

```
Client Request
     ↓
FastAPI (main.py)
     ↓
TopologyService (resolve template)
     ↓
InferenceEngine (prepare graph)
     ├→ EdgeFeatureComputer (dynamic features)
     ├→ EdgeFeatureNormalizer (normalize)
     └→ GraphBuilder (build PyG graph)
     ↓
UniversalTemporalGNN (predict)
     ↓
PredictionResponse
```

---

## Authentication

**Current:** No authentication required (development)  
**Future:** API key or JWT token

```http
Authorization: Bearer <token>
```

---

## API Endpoints

### v2 Endpoints (NEW)

#### 1. Minimal Inference

**POST** `/api/v2/inference/minimal`

Simplest API - auto-computes dynamic edge features.

**Request Body:**
```json
{
  "equipment_id": "string",
  "timestamp": "2025-12-03T23:00:00Z",
  "sensor_readings": {
    "<component_id>": {
      "pressure_bar": 150.0,
      "temperature_c": 65.0,
      "vibration_g": 0.8,
      "rpm": 1450  // optional
    }
  },
  "topology_id": "standard_pump_system"
}
```

**Response:** `200 OK`
```json
{
  "equipment_id": "pump_001",
  "health": {
    "score": 0.85,
    "status": "good"
  },
  "degradation": {
    "rate": 0.12,
    "severity": "low"
  },
  "anomaly": {
    "predictions": {
      "pressure_drop": 0.05,
      "overheating": 0.15,
      "cavitation": 0.02,
      "leakage": 0.08,
      "vibration_anomaly": 0.12,
      "flow_restriction": 0.04,
      "contamination": 0.10,
      "seal_degradation": 0.06,
      "valve_stiction": 0.03
    }
  },
  "inference_time_ms": 145.2
}
```

**Errors:**
- `400 Bad Request` - Invalid topology_id
- `422 Unprocessable Entity` - Validation error
- `503 Service Unavailable` - Engine not ready

---

#### 2. List Topologies

**GET** `/api/v2/topologies`

List all available topology templates.

**Response:** `200 OK`
```json
{
  "templates": [
    {
      "template_id": "standard_pump_system",
      "name": "Standard Pump System",
      "description": "Single pump with filter, valve, cylinder",
      "num_components": 4,
      "num_edges": 3
    },
    {
      "template_id": "dual_pump_system",
      "name": "Dual Pump System",
      "description": "Redundant pump configuration",
      "num_components": 7,
      "num_edges": 6
    }
  ]
}
```

---

#### 3. Get Topology by ID

**GET** `/api/v2/topologies/{topology_id}`

Get detailed topology template.

**Parameters:**
- `topology_id` (path) - Template identifier

**Response:** `200 OK`
```json
{
  "template_id": "standard_pump_system",
  "name": "Standard Pump System",
  "description": "...",
  "components": [
    {
      "component_id": "pump_main",
      "component_type": "pump",
      "manufacturer": "Bosch Rexroth",
      "model": "A10VSO"
    }
  ],
  "edges": [
    {
      "source_id": "pump_main",
      "target_id": "filter_main",
      "edge_type": "pipe",
      "diameter_mm": 25.0,
      "length_m": 2.0,
      "material": "steel"
    }
  ]
}
```

**Errors:**
- `404 Not Found` - Topology not found

---

#### 4. Validate Topology

**POST** `/api/v2/topologies/validate`

Validate custom topology before use.

**Request Body:**
```json
{
  "equipment_id": "custom_system",
  "components": {
    "pump_1": {...}
  },
  "edges": [{...}]
}
```

**Response:** `200 OK`
```json
{
  "is_valid": true,
  "errors": [],
  "num_components": 4,
  "num_edges": 3
}
```

---

### v1 Endpoints (Legacy)

#### 1. Single Prediction

**POST** `/api/v1/predict`

Legacy single prediction endpoint.

**Request Body:**
```json
{
  "equipment_id": "string",
  "sensor_data": {...},  // DataFrame-like
  "topology": {...}  // Full GraphTopology
}
```

**Response:** Same as v2

---

#### 2. Batch Prediction

**POST** `/api/v1/batch/predict`

Legacy batch prediction.

**Request Body:**
```json
{
  "requests": [{...}],
  "topology": {...}
}
```

**Response:** Array of predictions

---

### Health Checks

#### GET /health

Basic health check.

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "service": "gnn-service",
  "version": "2.0.0"
}
```

#### GET /healthz

Alias for `/health`.

#### GET /ready

Readiness probe (checks model loaded).

**Response:** `200 OK` or `503 Service Unavailable`
```json
{
  "status": "ready",
  "model_loaded": true,
  "topology_service_ready": true,
  "stats": {...}
}
```

---

## Schemas

### MinimalInferenceRequest

```json
{
  "equipment_id": "string (required)",
  "timestamp": "datetime (required)",
  "sensor_readings": {
    "<component_id>": {
      "pressure_bar": "float (required, >0)",
      "temperature_c": "float (required, >-273.15)",
      "vibration_g": "float (optional, ≥0)",
      "rpm": "int (optional, ≥0)"
    }
  },
  "topology_id": "string (required)"
}
```

### PredictionResponse

```json
{
  "equipment_id": "string",
  "health": {
    "score": "float [0, 1]",
    "status": "string (good/warning/critical)"
  },
  "degradation": {
    "rate": "float [0, 1]",
    "severity": "string (low/medium/high)"
  },
  "anomaly": {
    "predictions": {
      "<anomaly_type>": "float [0, 1]"
    }
  },
  "inference_time_ms": "float"
}
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message",
  "errors": [...]  // Optional validation errors
}
```

### Status Codes

- `200 OK` - Success
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service not ready

---

## Rate Limiting

**Current:** No rate limiting  
**Future:** 100 requests/minute per API key

**Headers (Future):**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1638518400
```

---

## Examples

### cURL

```bash
# Minimal inference
curl -X POST http://localhost:8000/api/v2/inference/minimal \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "pump_001",
    "timestamp": "2025-12-03T23:00:00Z",
    "sensor_readings": {
      "pump_main": {
        "pressure_bar": 150.0,
        "temperature_c": 65.0,
        "vibration_g": 0.8
      }
    },
    "topology_id": "standard_pump_system"
  }'

# List topologies
curl http://localhost:8000/api/v2/topologies
```

### Python

```python
import requests
from datetime import datetime

# Client
class GNNClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict(self, equipment_id, sensor_readings, topology_id):
        url = f"{self.base_url}/api/v2/inference/minimal"
        
        payload = {
            "equipment_id": equipment_id,
            "timestamp": datetime.now().isoformat(),
            "sensor_readings": sensor_readings,
            "topology_id": topology_id
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()

# Usage
client = GNNClient()

result = client.predict(
    equipment_id="pump_001",
    sensor_readings={
        "pump_main": {
            "pressure_bar": 150.0,
            "temperature_c": 65.0,
            "vibration_g": 0.8
        }
    },
    topology_id="standard_pump_system"
)

print(f"Health: {result['health']['score']:.2f}")
print(f"Degradation: {result['degradation']['rate']:.2f}")
```

### JavaScript

```javascript
// Fetch API
async function predict(equipmentId, sensorReadings, topologyId) {
  const response = await fetch('http://localhost:8000/api/v2/inference/minimal', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      equipment_id: equipmentId,
      timestamp: new Date().toISOString(),
      sensor_readings: sensorReadings,
      topology_id: topologyId
    })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  
  return await response.json();
}

// Usage
const result = await predict(
  'pump_001',
  {
    pump_main: {
      pressure_bar: 150.0,
      temperature_c: 65.0,
      vibration_g: 0.8
    }
  },
  'standard_pump_system'
);

console.log(`Health: ${result.health.score}`);
```

---

## Migration Guide

### v1 → v2

**v1 (Legacy):**
```json
{
  "equipment_id": "pump_001",
  "sensor_data": {...},  // Complex DataFrame
  "topology": {...}      // Full topology object
}
```

**v2 (NEW):**
```json
{
  "equipment_id": "pump_001",
  "timestamp": "2025-12-03T23:00:00Z",
  "sensor_readings": {   // Per-component readings
    "pump_main": {...}
  },
  "topology_id": "standard_pump_system"  // Template ID
}
```

**Benefits:**
- Simpler request format
- Auto-compute dynamic features
- Topology templates (no need to send full graph)
- Better error messages
- Faster inference

**Backward Compatibility:**  
v1 endpoints still work! No rush to migrate.

---

## OpenAPI Documentation

**Swagger UI:** http://localhost:8000/docs  
**ReDoc:** http://localhost:8000/redoc  
**OpenAPI Schema:** http://localhost:8000/openapi.json

---

## Support

**Issues:** https://github.com/Shukik85/hydraulic-diagnostic-saas/issues  
**Email:** support@example.com  
**Slack:** #gnn-service

---

**Last Updated:** 2025-12-03  
**Version:** 2.0.0
