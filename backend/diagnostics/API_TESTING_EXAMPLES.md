# 🧪 API Testing Examples

**Ingestion API Quick Test Guide**

---

## 🔑 Authentication

### Get JWT Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

**Response:**
```json
{
  "access": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Set token:**
```bash
export TOKEN="<your-access-token>"
```

---

## 📥 POST /api/v1/data/ingest

### Example 1: Single Reading (Success)

```bash
curl -X POST http://localhost:8000/api/v1/data/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "system_id": "550e8400-e29b-41d4-a716-446655440000",
    "readings": [
      {
        "sensor_id": "550e8400-e29b-41d4-a716-446655440001",
        "timestamp": "2025-11-07T22:50:00Z",
        "value": 125.5,
        "unit": "bar",
        "quality": 95
      }
    ]
  }'
```

**Expected Response (202 Accepted):**
```json
{
  "job_id": "a8f5f167-0e07-4b0c-8e6a-3c3c3e3c3e3c",
  "status": "queued"
}
```

### Example 2: Multiple Readings (Batch)

```bash
curl -X POST http://localhost:8000/api/v1/data/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "system_id": "550e8400-e29b-41d4-a716-446655440000",
    "readings": [
      {
        "sensor_id": "550e8400-e29b-41d4-a716-446655440001",
        "timestamp": "2025-11-07T22:50:00Z",
        "value": 125.5,
        "unit": "bar",
        "quality": 95
      },
      {
        "sensor_id": "550e8400-e29b-41d4-a716-446655440002",
        "timestamp": "2025-11-07T22:50:01Z",
        "value": 65.2,
        "unit": "celsius",
        "quality": 98
      },
      {
        "sensor_id": "550e8400-e29b-41d4-a716-446655440003",
        "timestamp": "2025-11-07T22:50:02Z",
        "value": 1450,
        "unit": "rpm",
        "quality": 92
      },
      {
        "sensor_id": "550e8400-e29b-41d4-a716-446655440004",
        "timestamp": "2025-11-07T22:50:03Z",
        "value": 245.8,
        "unit": "lpm",
        "quality": 97
      }
    ]
  }'
```

### Example 3: Invalid Data (Quarantine Test)

```bash
# Out of range value (800 bar > max 700)
curl -X POST http://localhost:8000/api/v1/data/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "system_id": "550e8400-e29b-41d4-a716-446655440000",
    "readings": [
      {
        "sensor_id": "550e8400-e29b-41d4-a716-446655440005",
        "timestamp": "2025-11-07T22:50:00Z",
        "value": 800.0,
        "unit": "bar",
        "quality": 95
      }
    ]
  }'
```

**Expected:** Job completes with quarantined_readings: 1

### Example 4: Validation Error (400 Bad Request)

```bash
# Empty readings array
curl -X POST http://localhost:8000/api/v1/data/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "system_id": "550e8400-e29b-41d4-a716-446655440000",
    "readings": []
  }'
```

**Expected Response (400 Bad Request):**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": {
      "readings": ["Ensure this field has at least 1 elements."]
    },
    "timestamp": "2025-11-07T22:50:00Z"
  }
}
```

### Example 5: Rate Limit Test (429)

```bash
# Send 16 requests rapidly (rate limit is 15/min)
for i in {1..16}; do
  curl -X POST http://localhost:8000/api/v1/data/ingest \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{...}' &
done
wait
```

**Expected:** 16th request returns 429 Too Many Requests

---

## 📋 GET /api/v1/jobs/{job_id}/

### Example 1: Check Job Status

```bash
# Use job_id from POST /data/ingest response
export JOB_ID="a8f5f167-0e07-4b0c-8e6a-3c3c3e3c3e3c"

curl -X GET http://localhost:8000/api/v1/jobs/$JOB_ID/ \
  -H "Authorization: Bearer $TOKEN"
```

**Response (Completed Job):**
```json
{
  "job_id": "a8f5f167-0e07-4b0c-8e6a-3c3c3e3c3e3c",
  "status": "completed",
  "total_readings": 100,
  "inserted_readings": 97,
  "quarantined_readings": 3,
  "created_at": "2025-11-07T22:50:00Z",
  "completed_at": "2025-11-07T22:50:05Z",
  "error_message": null,
  "processing_time_ms": 4523,
  "success_rate": 97.0,
  "system_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response (Processing Job):**
```json
{
  "job_id": "a8f5f167-0e07-4b0c-8e6a-3c3c3e3c3e3c",
  "status": "processing",
  "total_readings": 1000,
  "inserted_readings": 0,
  "quarantined_readings": 0,
  "created_at": "2025-11-07T22:50:00Z",
  "completed_at": null,
  "error_message": null,
  "processing_time_ms": null,
  "success_rate": 0.0,
  "system_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response (Failed Job):**
```json
{
  "job_id": "a8f5f167-0e07-4b0c-8e6a-3c3c3e3c3e3c",
  "status": "failed",
  "total_readings": 100,
  "inserted_readings": 0,
  "quarantined_readings": 100,
  "created_at": "2025-11-07T22:50:00Z",
  "completed_at": "2025-11-07T22:50:02Z",
  "error_message": "System 550e8400-e29b-41d4-a716-446655440000 not found",
  "processing_time_ms": 1234,
  "success_rate": 0.0,
  "system_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Example 2: Job Not Found (404)

```bash
curl -X GET http://localhost:8000/api/v1/jobs/00000000-0000-0000-0000-000000000000/ \
  -H "Authorization: Bearer $TOKEN"
```

**Expected Response (404 Not Found):**
```json
{
  "detail": "Not found."
}
```

### Example 3: Invalid UUID (400)

```bash
curl -X GET http://localhost:8000/api/v1/jobs/invalid-uuid/ \
  -H "Authorization: Bearer $TOKEN"
```

**Expected Response (400 Bad Request):**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid job_id format",
    "details": {
      "job_id": "Must be a valid UUID"
    },
    "timestamp": "2025-11-07T22:50:00Z"
  }
}
```

---

## 📊 E2E Test Script

### Complete Flow Test

```bash
#!/bin/bash
# e2e_test.sh - End-to-end ingestion API test

set -e

BASE_URL="http://localhost:8000/api/v1"

echo "=== E2E Ingestion API Test ==="
echo ""

# Step 1: Login
echo "[1/4] Getting JWT token..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}')

TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access')
echo "✅ Token obtained"

# Step 2: POST ingestion
echo ""
echo "[2/4] Posting bulk ingestion..."
INGEST_RESPONSE=$(curl -s -X POST "$BASE_URL/data/ingest" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "system_id": "550e8400-e29b-41d4-a716-446655440000",
    "readings": [
      {
        "sensor_id": "550e8400-e29b-41d4-a716-446655440001",
        "timestamp": "2025-11-07T22:50:00Z",
        "value": 125.5,
        "unit": "bar",
        "quality": 95
      },
      {
        "sensor_id": "550e8400-e29b-41d4-a716-446655440002",
        "timestamp": "2025-11-07T22:50:01Z",
        "value": 65.2,
        "unit": "celsius",
        "quality": 98
      }
    ]
  }')

JOB_ID=$(echo $INGEST_RESPONSE | jq -r '.job_id')
echo "✅ Job created: $JOB_ID"

# Step 3: Wait for processing
echo ""
echo "[3/4] Waiting for job to complete..."
sleep 5

# Step 4: Check job status
echo ""
echo "[4/4] Checking job status..."
STATUS_RESPONSE=$(curl -s -X GET "$BASE_URL/jobs/$JOB_ID/" \
  -H "Authorization: Bearer $TOKEN")

echo "$STATUS_RESPONSE" | jq .

JOB_STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')
INSERTED=$(echo $STATUS_RESPONSE | jq -r '.inserted_readings')
QUARANTINED=$(echo $STATUS_RESPONSE | jq -r '.quarantined_readings')

echo ""
echo "=== Test Results ==="
echo "Status: $JOB_STATUS"
echo "Inserted: $INSERTED"
echo "Quarantined: $QUARANTINED"

if [ "$JOB_STATUS" = "completed" ] && [ "$INSERTED" -gt 0 ]; then
  echo ""
  echo "✅ E2E TEST PASSED"
  exit 0
else
  echo ""
  echo "❌ E2E TEST FAILED"
  exit 1
fi
```

**Run:**
```bash
chmod +x e2e_test.sh
./e2e_test.sh
```

---

## 💍 Python Test Script

### Using `requests` library

```python
import requests
import time
from datetime import datetime, timezone

BASE_URL = "http://localhost:8000/api/v1"

# Step 1: Login
login_response = requests.post(
    f"{BASE_URL}/auth/login",
    json={"username": "admin", "password": "admin123"}
)
token = login_response.json()["access"]
headers = {"Authorization": f"Bearer {token}"}

print("✅ Authenticated")

# Step 2: POST ingestion
ingestion_data = {
    "system_id": "550e8400-e29b-41d4-a716-446655440000",
    "readings": [
        {
            "sensor_id": "550e8400-e29b-41d4-a716-446655440001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "value": 125.5,
            "unit": "bar",
            "quality": 95,
        }
    ],
}

ingestion_response = requests.post(
    f"{BASE_URL}/data/ingest",
    json=ingestion_data,
    headers=headers
)

assert ingestion_response.status_code == 202
job_id = ingestion_response.json()["job_id"]
print(f"✅ Job created: {job_id}")

# Step 3: Poll job status
max_attempts = 10
for attempt in range(max_attempts):
    time.sleep(1)
    
    status_response = requests.get(
        f"{BASE_URL}/jobs/{job_id}/",
        headers=headers
    )
    
    job_data = status_response.json()
    print(f"Attempt {attempt + 1}: Status = {job_data['status']}")
    
    if job_data["status"] in ["completed", "failed"]:
        break

# Step 4: Verify results
if job_data["status"] == "completed":
    print(f"✅ E2E Test PASSED")
    print(f"   Inserted: {job_data['inserted_readings']}")
    print(f"   Quarantined: {job_data['quarantined_readings']}")
    print(f"   Success rate: {job_data['success_rate']}%")
else:
    print(f"❌ E2E Test FAILED")
    print(f"   Error: {job_data.get('error_message')}")
```

---

## 📁 Postman Collection

### Import OpenAPI Spec

1. Open Postman
2. Import → Link
3. Paste: `https://raw.githubusercontent.com/Shukik85/hydraulic-diagnostic-saas/master/backend/docs/openapi_v3.1.yaml`
4. Generate Collection

### Environment Variables

```json
{
  "base_url": "http://localhost:8000/api/v1",
  "username": "admin",
  "password": "admin123",
  "token": "<set-after-login>",
  "system_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_id": "<set-after-ingestion>"
}
```

### Test Scenarios

1. **Happy Path:**
   - Login → Get token
   - POST /data/ingest with valid data
   - GET /jobs/{job_id}/ until completed
   - Verify success_rate = 100%

2. **Validation Errors:**
   - POST with empty readings
   - POST with invalid UUID
   - POST with out-of-range values
   - Verify 400 responses

3. **Quarantine Workflow:**
   - POST with mix of valid/invalid data
   - GET job status
   - Verify quarantined_readings > 0
   - Check admin panel for quarantine records

4. **Rate Limiting:**
   - Send 20 requests in 1 minute
   - Verify 429 after 15th request

---

## 🐞 Common Issues

### Issue 1: "System not found" Error

**Cause:** System UUID doesn't exist in database

**Solution:**
```bash
# Create test system
docker-compose exec backend python manage.py shell
```

```python
from diagnostics.models import HydraulicSystem
from users.models import User

user = User.objects.first()
system = HydraulicSystem.objects.create(
    id="550e8400-e29b-41d4-a716-446655440000",
    name="Test System",
    system_type="industrial",
    status="active",
    owner=user,
)
print(f"Created system: {system.id}")
```

### Issue 2: Celery Task Not Processing

**Cause:** Celery workers not running

**Solution:**
```bash
# Check Celery status
docker-compose exec celery_worker celery -A project inspect active

# Restart workers
docker-compose restart celery_worker celery_beat
```

### Issue 3: JWT Token Expired

**Cause:** Token TTL exceeded (default 30 minutes)

**Solution:**
```bash
# Get new token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

---

## 📊 Performance Benchmarks

### Load Test with Apache Bench

```bash
# 1000 requests, 10 concurrent
ab -n 1000 -c 10 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -p test_payload.json \
  http://localhost:8000/api/v1/data/ingest
```

**Expected:**
- Requests/sec: >100
- Mean response time: <50ms (p95)
- Failed requests: 0

### Load Test with Locust

```python
# locustfile.py
from locust import HttpUser, task, between
import json

class IngestionUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Login
        response = self.client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        self.token = response.json()["access"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task
    def bulk_ingest(self):
        payload = {
            "system_id": "550e8400-e29b-41d4-a716-446655440000",
            "readings": [
                {
                    "sensor_id": "550e8400-e29b-41d4-a716-446655440001",
                    "timestamp": "2025-11-07T22:50:00Z",
                    "value": 125.5,
                    "unit": "bar",
                    "quality": 95,
                }
            ],
        }
        self.client.post(
            "/api/v1/data/ingest",
            json=payload,
            headers=self.headers
        )
```

**Run:**
```bash
locust -f locustfile.py --host=http://localhost:8000
```

---

**Last Updated:** 2025-11-07 23:45 MSK  
**Status:** ✅ Ready for testing
