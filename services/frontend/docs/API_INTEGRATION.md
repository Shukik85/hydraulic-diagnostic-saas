# 🔌 API Integration Guide

> Полное руководство по интеграции с backend API

---

## 🎯 Overview

**API Type:** RESTful + WebSocket  
**Authentication:** JWT Bearer tokens  
**Format:** JSON  
**OpenAPI Version:** 3.1.0

### Services

- **Django Backend** (Port 8000) - Auth, Equipment, Core
- **GNN Service** (Port 8002) - Anomaly Detection
- **RAG Service** (Port 8004) - AI Interpretation
- **API Gateway** (Kong) - Unified entry point

---

## 🚀 Quick Start

### 1. Generate API Client

```bash
npm run generate:api
```

This generates TypeScript client from OpenAPI spec:
```
generated/api/
├── services/
│   ├── DiagnosisService.ts
│   ├── EquipmentService.ts
│   ├── GNNService.ts
│   └── RAGService.ts
├── models/ (TypeScript types)
└── core/
```

### 2. Use in Component

```vue
<script setup lang="ts">
const api = useGeneratedApi()

// Fully typed!
const systems = await api.equipment.listSystems()
</script>
```

---

## 📚 API Reference

See: `../../specs/combined-api.json` для полной OpenAPI spec.

---

**Last Updated:** November 15, 2025