# Wizard ↔️ Diagnosis Integration

## 🎯 Concept

**Wizard creates foundation metadata → Diagnosis uses it for sensor analysis**

Without wizard metadata, diagnosis cannot function. The wizard defines:
- What sensors exist
- What their normal ranges are
- What operating modes are valid
- What the system topology looks like

## 📦 5 Levels of Metadata

### Level 1: P&ID Schema (📝 Required)
```typescript
level1: {
  schemaFormat: 'pdf' | 'svg' | 'png' | 'jpg',
  schemaUrl: string,        // URL to uploaded schema
  schemaFileName: string,
  uploadedAt: Date
}
```
**Purpose:** Visual reference for system topology

---

### Level 2: Sensor Placement (📍 Required)
```typescript
level2: {
  sensors: [{
    id: string,
    name: string,
    type: 'pressure' | 'temperature' | 'flow' | 'vibration' | 'position',
    coordinates?: { x: number, y: number }  // Position on schema
  }]
}
```
**Purpose:** Defines what sensors to monitor in diagnosis

---

### Level 3: Nominal Values (⚠️ Critical for Anomaly Detection)
```typescript
level3: {
  nominalValues: {
    [sensorId]: {
      min: number,      // Minimum acceptable value
      max: number,      // Maximum acceptable value
      nominal: number,  // Expected normal value
      unit: string      // e.g., 'bar', '°C', 'L/min'
    }
  }
}
```
**Purpose:** Thresholds for anomaly detection algorithms

---

### Level 4: Operating Modes (🔧 Context for Analysis)
```typescript
level4: {
  operatingModes: [{
    name: string,          // e.g., 'Normal', 'Startup', 'Shutdown'
    description?: string,
    expectedValues?: Record<string, number>  // Expected sensor values in this mode
  }]
}
```
**Purpose:** Different normal behaviors for context-aware diagnosis

---

### Level 5: AI Configuration (🤖 ML Setup)
```typescript
level5: {
  aiReadinessScore: number,  // 0-100%, calculated from levels 1-4
  modelType?: 'gnn' | 'lstm' | 'transformer',
  trainingStatus?: 'pending' | 'in_progress' | 'completed' | 'failed',
  lastTrainedAt?: Date
}
```
**Purpose:** AI/ML model configuration and status

---

## 🔄 Data Flow

### 1. User Creates System Metadata (Wizard)
```
User fills wizard (5 levels)
  ↓
wizard saves metadata
  ↓
POST /api/v1/systems/{systemId}/metadata
  ↓
Backend stores in database
```

### 2. Diagnosis Uses Metadata
```
User opens diagnosis page
  ↓
GET /api/v1/systems/{systemId}/metadata
  ↓
Diagnosis loads:
  - Sensors to display (level2)
  - Thresholds for alerts (level3)
  - Operating mode context (level4)
  - AI model config (level5)
  ↓
Real-time sensor data combined with metadata
  ↓
Anomalies detected based on thresholds
```

---

## 📚 API Endpoints

### Save Metadata
```http
POST /api/v1/systems/{systemId}/metadata
Content-Type: application/json

{
  "systemId": "system-123",
  "systemName": "Hydraulic Press #5",
  "level1": { ... },
  "level2": { ... },
  "level3": { ... },
  "level4": { ... },
  "level5": { ... }
}
```

### Load Metadata
```http
GET /api/v1/systems/{systemId}/metadata

Response:
{
  "systemId": "system-123",
  "systemName": "Hydraulic Press #5",
  "createdAt": "2025-11-26T19:00:00Z",
  "updatedAt": "2025-11-26T19:30:00Z",
  "completionLevel": 5,
  "aiReadinessScore": 95,
  "level1": { ... },
  "level2": { ... },
  "level3": { ... },
  "level4": { ... },
  "level5": { ... }
}
```

### Check Readiness
```http
GET /api/v1/systems/{systemId}/metadata/readiness

Response:
{
  "isReady": true,
  "completionLevel": 5,
  "aiReadinessScore": 95,
  "missingLevels": []
}
```

---

## 🛠️ Frontend Usage

### In Wizard (Save)
```typescript
import { useSystemMetadata } from '~/composables/useSystemMetadata';

const { saveMetadata } = useSystemMetadata();

const saveWizardData = async () => {
  const success = await saveMetadata({
    systemId: 'system-123',
    systemName: 'Hydraulic Press #5',
    level1: { ... },
    level2: { ... },
    level3: { ... },
    level4: { ... },
    level5: { ... }
  });
  
  if (success) {
    // Navigate to diagnosis
    router.push(`/diagnosis/system-123`);
  }
};
```

### In Diagnosis (Load)
```typescript
import { useSystemMetadata } from '~/composables/useSystemMetadata';

const { loadMetadata, isReadyForDiagnosis, getSensors, getNominalValues } = useSystemMetadata();

onMounted(async () => {
  const systemId = route.params.id;
  
  // Check if metadata exists
  if (!isReadyForDiagnosis(systemId)) {
    toast.warning('System not configured', 'Please complete wizard first');
    router.push(`/wizard/metadata?systemId=${systemId}`);
    return;
  }
  
  // Load metadata
  const metadata = await loadMetadata(systemId);
  
  // Get sensors from metadata
  const sensors = getSensors(systemId);
  
  // Get thresholds from metadata
  const thresholds = getNominalValues(systemId);
  
  // Now diagnosis can function!
});
```

---

## ✅ Validation Flow

### Minimum Requirements for Diagnosis
```typescript
function canStartDiagnosis(metadata: SystemMetadata): boolean {
  return (
    metadata.level1 !== undefined &&  // Schema uploaded
    metadata.level2 !== undefined &&  // Sensors defined
    metadata.level2.sensors.length > 0  // At least 1 sensor
  );
}
```

### Recommended for Full Features
```typescript
function isFullyConfigured(metadata: SystemMetadata): boolean {
  return (
    metadata.completionLevel === 5 &&
    metadata.aiReadinessScore >= 80
  );
}
```

---

## 📊 AI Readiness Score Calculation

```typescript
function calculateReadinessScore(metadata: SystemMetadata): number {
  let score = 0;
  
  // Level 1: Schema (+20%)
  if (metadata.level1?.schemaUrl) score += 20;
  
  // Level 2: Sensors (+10% each, max 30%)
  if (metadata.level2?.sensors) {
    score += Math.min(metadata.level2.sensors.length * 10, 30);
  }
  
  // Level 3: Nominal values (+25%)
  if (metadata.level3?.nominalValues) {
    const count = Object.keys(metadata.level3.nominalValues).length;
    if (count > 0) score += 25;
  }
  
  // Level 4: Operating modes (+5% each, max 25%)
  if (metadata.level4?.operatingModes) {
    score += Math.min(metadata.level4.operatingModes.length * 5, 25);
  }
  
  return Math.min(score, 100);
}
```

**Score Interpretation:**
- **0-49%**: Not ready for diagnosis
- **50-79%**: Ready for basic diagnosis
- **80-100%**: Ready for full AI-powered diagnosis

---

## 🔗 Database Schema (Backend)

```sql
CREATE TABLE system_metadata (
  system_id VARCHAR(255) PRIMARY KEY,
  system_name VARCHAR(255) NOT NULL,
  completion_level INT CHECK (completion_level BETWEEN 1 AND 5),
  ai_readiness_score INT CHECK (ai_readiness_score BETWEEN 0 AND 100),
  
  -- Level 1: P&ID Schema
  schema_format VARCHAR(10),
  schema_url TEXT,
  schema_file_name VARCHAR(255),
  
  -- Level 2-5: Store as JSONB for flexibility
  level2_data JSONB,
  level3_data JSONB,
  level4_data JSONB,
  level5_data JSONB,
  
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_system_metadata_readiness 
  ON system_metadata(ai_readiness_score);

CREATE INDEX idx_system_metadata_completion 
  ON system_metadata(completion_level);
```

---

## 🚦 Error Handling

### Diagnosis Page Guard
```typescript
// middleware/diagnosis-ready.ts
export default defineNuxtRouteMiddleware(async (to) => {
  const systemId = to.params.id as string;
  const { isReadyForDiagnosis } = useSystemMetadata();
  
  if (!isReadyForDiagnosis(systemId)) {
    return navigateTo({
      path: '/wizard/metadata',
      query: { systemId, returnTo: to.fullPath }
    });
  }
});
```

### Usage in Diagnosis Page
```typescript
definePageMeta({
  middleware: ['auth', 'diagnosis-ready']
});
```

---

## 🌐 Production Implementation Checklist

- [ ] Backend API endpoints implemented
- [ ] Database schema created
- [ ] File upload to S3/storage for schemas
- [ ] Metadata validation on backend
- [ ] Metadata versioning (for updates)
- [ ] Metadata migration tools
- [ ] Frontend localStorage replaced with API calls
- [ ] Error handling for missing metadata
- [ ] Metadata export/import functionality
- [ ] Audit log for metadata changes

---

## 💡 Benefits of This Architecture

1. **Separation of Concerns**
   - Wizard = Data entry/configuration
   - Diagnosis = Real-time monitoring

2. **Reusability**
   - Same metadata can be used by:
     - Diagnosis page
     - Reports generator
     - ML training pipeline
     - API documentation

3. **Incremental Setup**
   - Basic diagnosis works with Level 1+2
   - Advanced features require Level 3+4+5
   - Users can complete wizard incrementally

4. **Type Safety**
   - Full TypeScript definitions
   - Frontend-backend contract
   - Validation at compile time

5. **Extensibility**
   - Easy to add Level 6, 7, etc.
   - JSONB storage allows flexible schemas
   - No breaking changes to existing data

---

**Created:** 2025-11-26  
**Last Updated:** 2025-11-26  
**Version:** 1.0.0
