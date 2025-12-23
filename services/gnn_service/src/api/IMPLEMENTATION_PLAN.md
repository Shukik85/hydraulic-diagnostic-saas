# 🛠️ API, Database & Auth Implementation Plan

**Status**: 📋 Planning Phase  
**Priority**: P1 (High) - Required for Production  
**Dependencies**: TimescaleDB, PostgreSQL, JWT library  
**Estimated Effort**: 3-4 days

---

## 🎯 Objectives

### 1. **Production TimescaleDB Integration**
- Replace mock `TimescaleConnector` with real asyncpg-based implementation
- Define hypertable schema for sensor readings
- Implement efficient time-series queries
- Add connection pooling and health checks

### 2. **JWT Authentication System**
- Implement JWT token validation middleware
- Add user extraction from Bearer tokens
- Create role-based access control (RBAC)
- Add equipment access control per user

### 3. **Data Access Layer**
- Map TimescaleDB schema → EdgeSensorReading/ComponentSensorReading
- Handle missing data, outliers, timestamp gaps
- Implement Row-Level Security (RLS) for multi-tenant isolation

---

## 📊 Database Schema Design

### **A. Sensor Readings Hypertable** (TimescaleDB)

```sql
-- ============================================================================
-- SENSOR READINGS HYPERTABLE
-- ============================================================================

CREATE TABLE sensor_readings (
    -- Primary Keys
    timestamp TIMESTAMPTZ NOT NULL,           -- Measurement timestamp
    equipment_id VARCHAR(100) NOT NULL,       -- Equipment identifier
    component_id VARCHAR(100) NOT NULL,       -- Component or edge ID
    
    -- Sensor Metadata
    sensor_type VARCHAR(50) NOT NULL,         -- 'pressure_inlet', 'pressure_outlet', 'flow', 'temperature', 'vibration', 'rpm', 'position', 'current', 'voltage'
    
    -- Sensor Value
    value DOUBLE PRECISION NOT NULL,          -- Measurement value
    unit VARCHAR(20),                         -- 'bar', 'lpm', '°C', 'g', 'RPM', '%', 'A', 'V'
    
    -- Quality Control
    quality_flag SMALLINT DEFAULT 0,          -- 0=OK, 1=Warning, 2=Bad, 3=Missing
    source VARCHAR(50) DEFAULT 'sensor',      -- 'sensor', 'computed', 'interpolated'
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable (TimescaleDB extension)
SELECT create_hypertable('sensor_readings', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day'
);

-- Compression policy (save disk space for old data)
ALTER TABLE sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'equipment_id, component_id'
);

SELECT add_compression_policy('sensor_readings', INTERVAL '7 days');

-- Retention policy (delete data older than 1 year)
SELECT add_retention_policy('sensor_readings', INTERVAL '1 year');

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Equipment-centric queries (most common)
CREATE INDEX idx_equipment_time ON sensor_readings (equipment_id, timestamp DESC);

-- Component-centric queries
CREATE INDEX idx_component_time ON sensor_readings (component_id, timestamp DESC);

-- Sensor type queries (for specific sensor types)
CREATE INDEX idx_sensor_type ON sensor_readings (sensor_type, timestamp DESC);

-- Composite index for edge sensor queries
CREATE INDEX idx_edge_sensors ON sensor_readings (equipment_id, component_id, sensor_type, timestamp DESC)
WHERE component_id LIKE '%__%';  -- Only for edge sensors (contain '__')

-- ============================================================================
-- CONTINUOUS AGGREGATES (Pre-computed statistics)
-- ============================================================================

-- 5-minute aggregates (for real-time monitoring)
CREATE MATERIALIZED VIEW sensor_readings_5min
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes', timestamp) AS bucket,
    equipment_id,
    component_id,
    sensor_type,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    STDDEV(value) as stddev_value,
    COUNT(*) as sample_count
FROM sensor_readings
GROUP BY bucket, equipment_id, component_id, sensor_type;

SELECT add_continuous_aggregate_policy('sensor_readings_5min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '5 minutes'
);
```

### **B. User Authentication** (PostgreSQL)

```sql
-- ============================================================================
-- USERS TABLE
-- ============================================================================

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,     -- bcrypt hash
    
    -- Roles: 'admin', 'engineer', 'operator', 'viewer'
    roles TEXT[] DEFAULT ARRAY['viewer']::TEXT[],
    
    -- Account Status
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- ============================================================================
-- EQUIPMENT ACCESS CONTROL
-- ============================================================================

CREATE TABLE equipment_access (
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    equipment_pattern VARCHAR(100) NOT NULL,  -- 'excavator_001', 'pump_*', '*'
    access_level VARCHAR(20) DEFAULT 'read',  -- 'read', 'write', 'admin'
    granted_by UUID REFERENCES users(user_id),
    granted_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,                   -- NULL = never expires
    
    PRIMARY KEY (user_id, equipment_pattern)
);

CREATE INDEX idx_equipment_access_user ON equipment_access(user_id);

-- ============================================================================
-- AUDIT LOG (Track all actions)
-- ============================================================================

CREATE TABLE audit_log (
    log_id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    action VARCHAR(50) NOT NULL,              -- 'diagnose', 'predict', 'train', 'access_denied'
    equipment_id VARCHAR(100),
    request_id UUID,                          -- From X-Request-ID header
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    metadata JSONB,                           -- Additional context
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_user_time ON audit_log(user_id, timestamp DESC);
CREATE INDEX idx_audit_action ON audit_log(action, timestamp DESC);

-- Convert to hypertable for efficient time-series storage
SELECT create_hypertable('audit_log', 'timestamp', 
    chunk_time_interval => INTERVAL '1 month'
);

-- Retention policy (keep audit logs for 2 years)
SELECT add_retention_policy('audit_log', INTERVAL '2 years');
```

### **C. Row-Level Security (RLS)** for Multi-Tenant Isolation

```sql
-- ============================================================================
-- ROW-LEVEL SECURITY ON SENSOR_READINGS
-- ============================================================================

ALTER TABLE sensor_readings ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see equipment they have access to
CREATE POLICY equipment_access_policy ON sensor_readings
FOR SELECT
USING (
    equipment_id IN (
        SELECT DISTINCT 
            CASE 
                -- Wildcard: 'pump_*' matches 'pump_001', 'pump_002', etc.
                WHEN ea.equipment_pattern LIKE '%*' THEN
                    sensor_readings.equipment_id LIKE REPLACE(ea.equipment_pattern, '*', '%')
                -- Exact match
                ELSE
                    sensor_readings.equipment_id = ea.equipment_pattern
            END
        FROM equipment_access ea
        JOIN users u ON ea.user_id = u.user_id
        WHERE u.username = current_user
          AND (ea.expires_at IS NULL OR ea.expires_at > NOW())
    )
    OR EXISTS (
        -- Admins see everything
        SELECT 1 FROM users 
        WHERE username = current_user 
        AND 'admin' = ANY(roles)
    )
);

-- Policy: Only admins can insert data
CREATE POLICY sensor_insert_policy ON sensor_readings
FOR INSERT
WITH CHECK (
    EXISTS (
        SELECT 1 FROM users
        WHERE username = current_user
        AND 'admin' = ANY(roles)
    )
);
```

---

## 🔧 Implementation Tasks

### **Phase 1: Database Setup** (Day 1)

#### Task 1.1: Create SQL Schema Files
- [ ] `services/gnn_service/migrations/001_sensor_readings.sql`
- [ ] `services/gnn_service/migrations/002_users_auth.sql`
- [ ] `services/gnn_service/migrations/003_equipment_access.sql`
- [ ] `services/gnn_service/migrations/004_row_level_security.sql`
- [ ] `services/gnn_service/migrations/005_continuous_aggregates.sql`

#### Task 1.2: Setup Migration Tool
```bash
# Add to requirements.txt
alembic==1.13.1  # Database migrations
asyncpg==0.29.0  # PostgreSQL async driver

# Initialize Alembic
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Create sensor_readings hypertable"

# Apply migrations
alembic upgrade head
```

#### Task 1.3: Update docker-compose.yml
```yaml
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: hydraulic_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d  # Auto-run on first start
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d hydraulic_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  gnn_service:
    depends_on:
      timescaledb:
        condition: service_healthy
```

---

### **Phase 2: TimescaleDB Connector** (Day 1-2)

#### Task 2.1: Create Production Connector

**File**: `services/gnn_service/src/data/timescale_connector.py`

```python
"""Production TimescaleDB connector for sensor data retrieval.

Replaces mock connector with real asyncpg-based implementation.

Features:
    - Connection pooling for high throughput
    - Prepared statements for query optimization
    - Automatic retry on transient failures
    - Query result caching (optional)
    - Metrics collection (query time, row count)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import asyncpg
import polars as pl
from asyncpg import Pool

logger = logging.getLogger(__name__)


class TimescaleConnector:
    """Production connector for TimescaleDB sensor data.
    
    Examples:
        >>> connector = TimescaleConnector(
        ...     host="localhost",
        ...     port=5432,
        ...     database="hydraulic_db",
        ...     user="user",
        ...     password="password"
        ... )
        >>> await connector.connect()
        >>> df = await connector.fetch_sensor_data(
        ...     equipment_id="excavator_001",
        ...     start_time="2025-12-23T00:00:00Z",
        ...     end_time="2025-12-23T01:00:00Z"
        ... )
        >>> print(df.shape)
        (3600, 5)  # 1 hour, 10s intervals, 10 sensors
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        min_pool_size: int = 5,
        max_pool_size: int = 20,
    ):
        """Initialize connector with connection parameters.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_pool_size: Minimum connection pool size
            max_pool_size: Maximum connection pool size
        """
        self.pool: Optional[Pool] = None
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "min_size": min_pool_size,
            "max_size": max_pool_size,
        }
        logger.info(
            "Initialized TimescaleConnector",
            extra={
                "host": host,
                "port": port,
                "database": database,
                "pool_size": f"{min_pool_size}-{max_pool_size}",
            },
        )

    async def connect(self) -> None:
        """Create connection pool.
        
        Raises:
            asyncpg.PostgresError: If connection fails
        """
        try:
            self.pool = await asyncpg.create_pool(**self.conn_params)
            logger.info("✅ Connection pool created")
        except Exception as e:
            logger.error(f"❌ Failed to create connection pool: {e}", exc_info=True)
            raise

    async def fetch_sensor_data(
        self,
        equipment_id: str,
        start_time: str,
        end_time: str,
        component_ids: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """Fetch sensor data from TimescaleDB hypertable.
        
        Args:
            equipment_id: Equipment identifier
            start_time: Start timestamp (ISO 8601)
            end_time: End timestamp (ISO 8601)
            component_ids: Optional list of component IDs to filter
            
        Returns:
            Polars DataFrame with columns:
                [timestamp, component_id, sensor_type, value, unit]
                
        Raises:
            ValueError: If time range invalid
            asyncpg.PostgresError: If query fails
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized. Call connect() first.")

        # Parse timestamps
        start_dt = datetime.fromisoformat(start_time.replace("Z", ""))
        end_dt = datetime.fromisoformat(end_time.replace("Z", ""))

        if end_dt <= start_dt:
            raise ValueError(f"Invalid time range: {start_time} to {end_time}")

        # Build query
        query = """
        SELECT 
            timestamp,
            component_id,
            sensor_type,
            value,
            unit,
            quality_flag
        FROM sensor_readings
        WHERE equipment_id = $1
          AND timestamp BETWEEN $2 AND $3
        """
        
        params = [equipment_id, start_dt, end_dt]
        
        if component_ids:
            query += " AND component_id = ANY($4)"
            params.append(component_ids)
        
        query += " ORDER BY timestamp, component_id, sensor_type"

        # Execute query
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            logger.info(
                f"✅ Fetched {len(rows)} sensor readings",
                extra={
                    "equipment_id": equipment_id,
                    "time_range": f"{start_time} to {end_time}",
                    "row_count": len(rows),
                },
            )

            # Convert to Polars DataFrame
            if not rows:
                # Return empty DataFrame with correct schema
                return pl.DataFrame(
                    schema={
                        "timestamp": pl.Datetime,
                        "component_id": pl.Utf8,
                        "sensor_type": pl.Utf8,
                        "value": pl.Float64,
                        "unit": pl.Utf8,
                        "quality_flag": pl.Int16,
                    }
                )

            df = pl.DataFrame(
                [
                    {
                        "timestamp": row["timestamp"],
                        "component_id": row["component_id"],
                        "sensor_type": row["sensor_type"],
                        "value": float(row["value"]),
                        "unit": row["unit"],
                        "quality_flag": row["quality_flag"],
                    }
                    for row in rows
                ]
            )

            return df

        except Exception as e:
            logger.error(
                f"❌ Query failed: {e}",
                extra={
                    "equipment_id": equipment_id,
                    "time_range": f"{start_time} to {end_time}",
                },
                exc_info=True,
            )
            raise

    async def health_check(self) -> bool:
        """Check database connection health.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.pool:
            return False

        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close connection pool and release resources."""
        if self.pool:
            await self.pool.close()
            logger.info("🔒 Connection pool closed")
```

#### Task 2.2: Data Mapping Layer

**File**: `services/gnn_service/src/data/sensor_mapper.py`

```python
"""Map TimescaleDB sensor data to EdgeSensorReading and ComponentSensorReading.

Handles:
    - Edge sensor aggregation (pressure_inlet + pressure_outlet + flow + temp + vibration)
    - Component sensor aggregation (rpm + position + current + voltage)
    - Missing data handling
    - Timestamp alignment
"""

from datetime import datetime
from typing import Optional

import polars as pl

from src.schemas.requests import ComponentSensorReading, EdgeSensorReading


class SensorDataMapper:
    """Map TimescaleDB flat format to structured sensor readings."""

    def map_to_edge_readings(
        self,
        df: pl.DataFrame,
        timestamp: datetime,
    ) -> dict[str, EdgeSensorReading]:
        """Map flat sensor data to EdgeSensorReading objects.
        
        Args:
            df: Polars DataFrame from TimescaleDB
            timestamp: Target timestamp
            
        Returns:
            Dict mapping edge_id → EdgeSensorReading
        """
        # Filter edge sensors (component_id contains '__')
        edge_df = df.filter(pl.col("component_id").str.contains("__"))
        
        # Group by edge_id
        edge_readings = {}
        
        for edge_id in edge_df["component_id"].unique():
            edge_data = edge_df.filter(pl.col("component_id") == edge_id)
            
            # Extract sensor values
            pressure_inlet = self._get_sensor_value(
                edge_data, "pressure_inlet", required=True
            )
            pressure_outlet = self._get_sensor_value(
                edge_data, "pressure_outlet", required=True
            )
            flow_rate = self._get_sensor_value(edge_data, "flow")
            temperature = self._get_sensor_value(edge_data, "temperature")
            vibration = self._get_sensor_value(edge_data, "vibration")
            
            edge_readings[edge_id] = EdgeSensorReading(
                edge_id=edge_id,
                pressure_inlet_bar=pressure_inlet,
                pressure_outlet_bar=pressure_outlet,
                flow_rate_lpm=flow_rate,
                temperature_c=temperature,
                vibration_g=vibration,
                timestamp=timestamp,
            )
        
        return edge_readings

    def map_to_component_readings(
        self,
        df: pl.DataFrame,
        timestamp: datetime,
    ) -> dict[str, ComponentSensorReading]:
        """Map flat sensor data to ComponentSensorReading objects.
        
        Args:
            df: Polars DataFrame from TimescaleDB
            timestamp: Target timestamp
            
        Returns:
            Dict mapping component_id → ComponentSensorReading
        """
        # Filter component sensors (component_id does NOT contain '__')
        comp_df = df.filter(~pl.col("component_id").str.contains("__"))
        
        component_readings = {}
        
        for comp_id in comp_df["component_id"].unique():
            comp_data = comp_df.filter(pl.col("component_id") == comp_id)
            
            # Extract sensor values
            rpm = self._get_sensor_value(comp_data, "rpm")
            position = self._get_sensor_value(comp_data, "position")
            current = self._get_sensor_value(comp_data, "current")
            voltage = self._get_sensor_value(comp_data, "voltage")
            
            # Only create reading if at least one sensor exists
            if any(v is not None for v in [rpm, position, current, voltage]):
                component_readings[comp_id] = ComponentSensorReading(
                    component_id=comp_id,
                    rpm=rpm,
                    position_percent=position,
                    current_a=current,
                    voltage_v=voltage,
                    timestamp=timestamp,
                )
        
        return component_readings

    def _get_sensor_value(
        self,
        df: pl.DataFrame,
        sensor_type: str,
        required: bool = False,
    ) -> Optional[float]:
        """Extract sensor value from DataFrame.
        
        Args:
            df: Filtered DataFrame for specific component/edge
            sensor_type: Sensor type to extract
            required: If True, raise error if missing
            
        Returns:
            Sensor value or None if not found
            
        Raises:
            ValueError: If required sensor is missing
        """
        sensor_data = df.filter(pl.col("sensor_type") == sensor_type)
        
        if sensor_data.is_empty():
            if required:
                raise ValueError(
                    f"Required sensor '{sensor_type}' not found in data"
                )
            return None
        
        # Get latest value (in case multiple timestamps)
        return float(sensor_data.sort("timestamp", descending=True)["value"][0])
```

---

### **Phase 3: JWT Authentication** (Day 2-3)

#### Task 3.1: JWT Service

**File**: `services/gnn_service/src/middleware/auth.py`

```python
# See detailed implementation in previous response section "B. JWT Authentication Middleware"
```

#### Task 3.2: Update main.py

```python
from src.middleware.auth import get_current_user, require_role, User, JWTService

# Initialize JWT service in lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing startup code ...
    
    # Initialize JWT service
    import os
    jwt_secret = os.getenv("JWT_SECRET_KEY")
    if not jwt_secret:
        raise RuntimeError("JWT_SECRET_KEY not set in environment")
    
    app.state.jwt_service = JWTService(secret_key=jwt_secret)
    logger.info("✅ JWT service initialized")
    
    yield
    
    # ... existing shutdown code ...

# Update endpoints
@app.post("/v1/diagnose", dependencies=[Depends(get_current_user)])
async def run_diagnosis(
    request: MinimalInferenceRequest,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    # ... with access control ...
```

---

### **Phase 4: Testing & Validation** (Day 3-4)

#### Task 4.1: Integration Tests

```python
# tests/integration/test_timescale_connector.py

import pytest
import asyncio
from src.data.timescale_connector import TimescaleConnector

@pytest.mark.asyncio
async def test_fetch_sensor_data():
    connector = TimescaleConnector(
        host="localhost",
        port=5432,
        database="hydraulic_db_test",
        user="test_user",
        password="test_password",
    )
    
    await connector.connect()
    
    df = await connector.fetch_sensor_data(
        equipment_id="test_equipment",
        start_time="2025-12-23T00:00:00Z",
        end_time="2025-12-23T01:00:00Z",
    )
    
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "component_id" in df.columns
    
    await connector.close()
```

#### Task 4.2: Auth Tests

```python
# tests/integration/test_auth.py

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

def test_diagnose_requires_auth():
    client = TestClient(app)
    
    response = client.post("/v1/diagnose", json={...})
    
    # Should return 401 Unauthorized without token
    assert response.status_code == 401

def test_diagnose_with_valid_token():
    client = TestClient(app)
    
    # Create valid JWT token
    token = create_test_token(user_id="test_user", roles=["engineer"])
    
    response = client.post(
        "/v1/diagnose",
        json={...},
        headers={"Authorization": f"Bearer {token}"},
    )
    
    assert response.status_code == 200
```

---

## 📈 Success Metrics

- [ ] TimescaleDB connector passes all integration tests
- [ ] Query performance: <100ms for 1-hour time windows
- [ ] JWT authentication blocks unauthorized requests
- [ ] Equipment access control correctly enforces permissions
- [ ] RLS policies prevent data leakage between users
- [ ] Audit log captures all diagnostic requests
- [ ] Health checks pass in K8s environment

---

## 🔄 Migration Strategy

### Development → Production

1. **Week 1**: Implement on feature branch with mock data
2. **Week 2**: Deploy to staging with synthetic test database
3. **Week 3**: Load real sensor data, validate data quality
4. **Week 4**: Production deployment with phased rollout

### Backward Compatibility

- Keep mock connector available via env flag: `USE_MOCK_CONNECTOR=true`
- Allow unauthenticated access in dev: `REQUIRE_AUTH=false`
- Gradual migration: Old endpoints coexist with new auth endpoints

---

## 📚 References

- TimescaleDB Docs: https://docs.timescale.com/
- asyncpg Documentation: https://magicstack.github.io/asyncpg/
- FastAPI Security: https://fastapi.tiangolo.com/tutorial/security/
- PyJWT: https://pyjwt.readthedocs.io/

---

**Next Steps**: Proceed to GNN Model improvements (GraphBuilderV2, training pipeline)
