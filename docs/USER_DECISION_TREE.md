# User Decision Tree: Complete Journey

## 👤 Обзор

Этот документ описывает полный user journey в Hydraulic Diagnostic SaaS, включая:

- Authentication flow
- System management (CRUD)
- Diagnosis workflow
- Report generation
- Settings management

## 🔑 Authentication Flow

### 1. Login Process

```
User opens app
    ↓
[Authenticated?]
    │
    ├── No → Login Page
    │         │
    │         ├── Email/Password
    │         │       ↓
    │         │   [Credentials Valid?]
    │         │       │
    │         │       ├── Yes → Generate JWT + Device Fingerprint
    │         │       └── No → Show Error → Retry
    │         │
    │         ├── Enterprise SSO (SAML/OIDC)
    │         │       ↓
    │         │   Redirect to IdP
    │         │       ↓
    │         │   Callback with assertion
    │         │       ↓
    │         │   Validate & map roles
    │         │       ↓
    │         │   Generate JWT
    │         │
    │         └── Multi-Factor Auth (MFA)
    │                 ↓
    │             Email/SMS code
    │                 ↓
    │             Verify code
    │                 ↓
    │             Generate JWT
    │
    └── Yes → [Session Valid?]
                │
                ├── Yes → Load Dashboard
                │
                └── No → Refresh Token
                          │
                          ├── Success → Dashboard
                          └── Fail → Login Page
```

## 📊 Dashboard

### Main Actions

```
Dashboard
    │
    ├── View Systems
    │       ↓
    │   List all hydraulic systems
    │       ↓
    │   [Select System]
    │       ↓
    │   System Details + Real-time Data
    │       │
    │       ├── Edit System
    │       ├── Delete System
    │       └── Run Diagnosis
    │
    ├── Create System
    │       ↓
    │   [Has Permission?]
    │       │
    │       ├── Yes → System Form
    │       │           ↓
    │       │       Fill Details
    │       │           ↓
    │       │       [Validation]
    │       │           │
    │       │           ├── Pass → Save to DB
    │       │           │           ↓
    │       │           │       Optimistic UI Update
    │       │           │           ↓
    │       │           │       [API Success?]
    │       │           │           │
    │       │           │           ├── Yes → Refresh List
    │       │           │           └── No → Rollback UI
    │       │           │
    │       │           └── Fail → Show Errors
    │       │
    │       └── No → 403 Forbidden
    │
    ├── Run Diagnosis
    │       ↓
    │   Select System
    │       ↓
    │   Choose Time Range
    │       ↓
    │   Select Sensors
    │       ↓
    │   Submit Request
    │       ↓
    │   [Data Available?]
    │       │
    │       ├── Yes → Queue for GNN Inference
    │       │           ↓
    │       │       WebSocket Updates (Progress)
    │       │           ↓
    │       │       [Inference Status]
    │       │           │
    │       │           ├── Processing → Show Progress
    │       │           ├── Completed → Show Results
    │       │           │               │
    │       │           │               ├── Export PDF
    │       │           │               ├── Share Link
    │       │           │               ├── Save Report
    │       │           │               └── New Diagnosis
    │       │           │
    │       │           └── Failed → Show Error
    │       │
    │       └── No → 404: No Data Error
    │
    ├── View Reports
    │       ↓
    │   Apply Filters (Date, System, Type)
    │       ↓
    │   Generate Charts
    │       ↓
    │   [Export?]
    │       │
    │       ├── CSV
    │       ├── Excel
    │       └── PDF
    │
    └── Settings
            │
            ├── Profile
            │       ↓
            │   Edit Name, Email, Avatar
            │
            ├── Security
            │       │
            │       ├── Change Password
            │       ├── Enable MFA
            │       └── View Active Sessions
            │
            ├── Notifications
            │       │
            │       ├── Email Preferences
            │       ├── Push Notifications
            │       └── Alerts Threshold
            │
            └── API Keys
                    │
                    ├── Generate New Key
                    ├── Revoke Key
                    └── View Usage Stats
```

## 🔍 Diagnosis Workflow Detail

### Step-by-Step Process

1. **System Selection**
   - User selects equipment from dropdown
   - System metadata loaded from DB
   - Real-time status displayed

2. **Time Range Selection**
   - Last hour / 24 hours / 7 days / Custom
   - Validates data availability
   - Shows sensor coverage

3. **Sensor Selection**
   - Multi-select: Pressure, Temperature, Flow, Vibration
   - Validates sensor availability for time range
   - Shows sample count

4. **Submission**
   - Validate all inputs
   - Generate request ID
   - Submit to Diagnosis Service

5. **Queue Processing**
   - Diagnosis Service queries TimescaleDB
   - Transforms data to graph format
   - Calls GNN Service via gRPC

6. **GNN Inference**
   - Load model on GPU
   - Run forward pass
   - Compute health scores
   - Detect anomalies
   - Generate recommendations

7. **Results Display**
   - Component health scores (0-1)
   - Anomaly list with severity
   - Maintenance recommendations
   - Predicted time to failure
   - Confidence intervals

## ⚠️ Error Handling

### Common Errors

| Error | User Message | Action |
|-------|--------------|--------|
| 401 Unauthorized | "Session expired. Please log in again." | Redirect to login |
| 403 Forbidden | "You don't have permission for this action." | Show error, disable action |
| 404 Not Found | "No data available for selected time range." | Suggest different range |
| 429 Rate Limit | "Too many requests. Please wait." | Show cooldown timer |
| 500 Server Error | "Something went wrong. Please try again." | Retry button, contact support |
| 503 Service Unavailable | "Service temporarily unavailable." | Maintenance message |

## 📊 User Metrics

### Tracked Events

- User login/logout
- System CRUD operations
- Diagnosis requests
- Report exports
- Settings changes
- API key usage

### Analytics

- Session duration
- Feature usage frequency
- Diagnosis success rate
- Average response time
- User satisfaction (NPS)

## 🔔 Notifications

### Real-time Alerts

- System health critical
- Anomaly detected
- Diagnosis completed
- Report ready
- System down

### Delivery Channels

- In-app notifications
- Email alerts
- Push notifications (mobile)
- WebSocket updates
- SMS (critical only)

## 📝 Best Practices

### For Users

1. **Regular Monitoring**: Check systems daily
2. **Preventive Diagnostics**: Run weekly scans
3. **Act on Recommendations**: Address high-priority alerts
4. **Review Reports**: Analyze trends monthly
5. **Update Metadata**: Keep system info current

### For Administrators

1. **User Management**: Regular access reviews
2. **Audit Logs**: Monitor unusual activity
3. **Performance**: Track response times
4. **Data Quality**: Ensure sensor coverage
5. **Backup**: Regular data exports
