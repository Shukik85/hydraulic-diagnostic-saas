# 🏗️ Frontend Architecture

> Hydraulic Diagnostic SaaS - Архитектура frontend приложения

**Version:** 1.0.0  
**Last Updated:** November 15, 2025  
**Author:** Plotnikov Aleksandr

---

## 📊 High-Level Overview

```
┌───────────────────────────────────────────────────────┐
│                    USER (Browser)                          │
└───────────────────────┬──────────────────────────────┘
                         │
                         │ HTTPS/WSS
                         ↓
┌───────────────────────────────────────────────────────┐
│                 NUXT 3 SSR SERVER                           │
│                                                             │
│  ┌────────────────────────────────────────────┐  │
│  │          VUE 3 APPLICATION                      │  │
│  │                                                 │  │
│  │  ┌─────────────┐   ┌────────────────┐  │  │
│  │  │   PAGES     │   │  COMPONENTS   │  │  │
│  │  │  (Routing)  │   │   (UI/Logic)   │  │  │
│  │  └───────┬──────┘   └───────┬────────┘  │  │
│  │         │                  │             │  │
│  │         └────────┬─────────┘             │  │
│  │                  │                       │  │
│  │         ┌──────┴──────────┐           │  │
│  │         │  COMPOSABLES    │           │  │
│  │         │  (Business      │           │  │
│  │         │   Logic)        │           │  │
│  │         └───────┬────────┘           │  │
│  │                  │                       │  │
│  │         ┌──────┴──────────┐           │  │
│  │         │  PINIA STORES   │           │  │
│  │         │  (Global State) │           │  │
│  │         └───────┬────────┘           │  │
│  │                  │                       │  │
│  └───────────────┴───────────────────────────┘  │
│                     │                                │
│            ┌──────┴───────┐                         │
│            │  GENERATED   │                         │
│            │  API CLIENT  │ (OpenAPI Codegen)      │
│            └──────┬──────┘                         │
└────────────────────┴───────────────────────────────────┘
                       │
                       │ REST + WebSocket
                       ↓
┌───────────────────────────────────────────────────────┐
│                  BACKEND SERVICES                           │
│  Django + GNN Service + RAG Service + TimescaleDB          │
└───────────────────────────────────────────────────────┘
```

---

## 🎯 Core Principles

### 1. **Type Safety First**
- **100% TypeScript** - нет any, только strict mode
- **OpenAPI Codegen** - auto-generated API client
- **Zod schemas** - runtime validation
- **Compile-time checks** - ошибки ловятся до deploy

### 2. **Performance Optimization**
- **SSR** - fast initial load
- **Code splitting** - lazy loading
- **Image optimization** - WebP, responsive
- **Bundle size** - < 200KB initial
- **Caching** - aggressive HTTP cache

### 3. **Developer Experience**
- **Auto-imports** - Vue, Nuxt, composables
- **File-based routing** - нет ручной конфигурации
- **Hot reload** - instant feedback
- **ESLint + Prettier** - консистентный code style

### 4. **Production Ready**
- **Error boundaries** - graceful degradation
- **Loading states** - skeleton loaders
- **Offline support** - service worker (future)
- **Monitoring** - Sentry integration ready

---

## 🛠️ Tech Stack

### Core Framework

**Nuxt 3.12.4**
- **Why Nuxt?**
  - ✅ SSR/SSG out of the box
  - ✅ File-based routing
  - ✅ Auto-imports
  - ✅ SEO optimization
  - ✅ Production-ready defaults

**Vue 3.4**
- **Why Vue 3?**
  - ✅ Composition API - лучшая типизация
  - ✅ `<script setup>` - меньше boilerplate
  - ✅ Reactivity system - перформанс
  - ✅ Ecosystem - huge community

**TypeScript 5.5**
- **Why TypeScript?**
  - ✅ Catch bugs at compile time
  - ✅ Better IDE support
  - ✅ Self-documenting code
  - ✅ Refactoring confidence

---

### State Management

**Pinia** (Vuex 5)
- **Stores:**
  - `auth.store.ts` - Authentication state
  - `systems.store.ts` - Equipment state
  - `metadata.ts` - System metadata

**Why Pinia?**
- ✅ TypeScript-first
- ✅ No mutations boilerplate
- ✅ Devtools support
- ✅ Composition API style

**State Architecture:**
```typescript
// Global state (Pinia)
- Authentication (user, token, permissions)
- Equipment list (cached, reactive)
- Metadata (system info, user preferences)

// Local state (composables)
- Page-specific data (diagnostics, reports)
- Form state (reactive, validated)
- UI state (modals, tabs, filters)
```

---

### UI Layer

**Tailwind CSS 3.x**
- **Why Tailwind?**
  - ✅ Utility-first - fast development
  - ✅ Small bundle (tree-shaking)
  - ✅ Consistent design system
  - ✅ Responsive by default

**Design System:**
- **Custom components** with `u-*` prefix
- **Consistent spacing** - 4px grid
- **Color palette** - blue (primary), red (error), green (success)
- **Typography** - Inter font, responsive scale

**Component Library:**
```
components/
├── ui/              # Design system (buttons, cards, inputs)
├── dashboard/      # Dashboard widgets (metrics, charts)
├── rag/            # RAG interpretation UI
├── digital-twin/   # 3D visualization (future)
└── metadata/       # System metadata forms
```

---

### API Integration

**OpenAPI TypeScript Codegen**

**Architecture:**
```
Backend OpenAPI Spec (combined-api.json)
         ↓
openapi-typescript-codegen
         ↓
generated/api/ (auto-generated)
  ├── services/
  │   ├── DiagnosisService.ts
  │   ├── EquipmentService.ts
  │   ├── GNNService.ts
  │   └── RAGService.ts
  ├── models/ (TypeScript types)
  └── core/
         ↓
composables/useGeneratedApi.ts (wrapper)
         ↓
Components use typed API
```

**Benefits:**
- ✅ **Full type safety** - backend changes → compile errors
- ✅ **Auto-sync** - regenerates on every build
- ✅ **No manual types** - никогда не outdated
- ✅ **IDE autocomplete** - знает все endpoints

**Example:**
```typescript
const api = useGeneratedApi()

// Полностью типизировано!
const result: DiagnosisResponse = await api.diagnosis.runDiagnosis({
  equipmentId: 'exc_001',  // string
  diagnosisRequest: {      // DiagnosisRequest type
    timeWindow: {          // TimeWindow type
      startTime: '',       // ISO 8601 string
      endTime: ''
    }
  }
})

// TypeScript проверит все поля!
```

---

### Real-Time Updates

**WebSocket Integration**

**Architecture:**
```typescript
// composables/useWebSocket.ts
export function useWebSocket(channel: string) {
  const ws = ref<WebSocket | null>(null)
  const connected = ref(false)
  
  const connect = () => {
    ws.value = new WebSocket(`${wsBase}/${channel}`)
    ws.value.onopen = () => connected.value = true
    ws.value.onmessage = (event) => handleMessage(event)
  }
  
  return { connect, connected, send }
}
```

**Use Cases:**
- ✅ Real-time sensor data streaming
- ✅ Diagnostic progress updates
- ✅ Alert notifications
- ✅ System status changes

---

## 📁 Directory Structure

### Pages (File-based Routing)

```
pages/
├── index.vue                    # / (Landing)
├── dashboard.vue                # /dashboard
├── diagnostics.vue              # /diagnostics
├── diagnostics/
│   └── [id]/
│       ├── index.vue            # /diagnostics/:id
│       └── interpretation.vue   # /diagnostics/:id/interpretation (RAG)
├── systems/
│   ├── index.vue                # /systems
│   └── [id]/
│       ├── index.vue            # /systems/:id
│       ├── sensors.vue          # /systems/:id/sensors
│       └── equipments.vue       # /systems/:id/equipments
├── reports/
│   ├── index.vue                # /reports
│   └── [id].vue                 # /reports/:id
├── settings/
│   ├── index.vue                # /settings
│   ├── profile.vue              # /settings/profile
│   └── security.vue             # /settings/security
└── auth/
    ├── login.vue                # /auth/login
    └── register.vue             # /auth/register
```

### Components Organization

```
components/
├── ui/                      # Базовые UI компоненты
│   ├── UButton.vue
│   ├── UCard.vue
│   ├── UInput.vue
│   ├── UModal.vue
│   └── UBadge.vue
│
├── dashboard/               # Dashboard-specific
│   ├── MetricCard.vue
│   ├── AlertList.vue
│   └── SystemStatus.vue
│
├── rag/                     # RAG интеграция
│   ├── InterpretationPanel.vue  # Основной UI
│   ├── ReasoningSteps.vue       # Reasoning viz
│   └── KnowledgeContext.vue     # KB context
│
├── digital-twin/            # 3D visualization
│   └── ThreeCanvas.vue
│
└── metadata/                # System metadata
    ├── MetadataForm.vue
    └── MetadataViewer.vue
```

### Composables (Business Logic)

```
composables/
├── useGeneratedApi.ts       # API client wrapper
├── useRAG.ts                # RAG integration
├── useWebSocket.ts          # Real-time updates
├── useDigitalTwin.ts        # Digital twin state
├── useAnomalies.ts          # Anomaly detection
├── useSystemStatus.ts       # System health
├── useMockData.ts           # Demo data
└── usePasswordStrength.ts   # Password validation
```

---

## 🔄 Data Flow

### 1. **User Action → API Request**

```typescript
// Page/Component
const api = useGeneratedApi()
const { data, loading, error } = await api.diagnosis.runDiagnosis({...})
  ↓
// Composable (useGeneratedApi)
const authStore = useAuthStore()
headers['Authorization'] = `Bearer ${authStore.token}`
  ↓
// Generated API Client
axios.post(`${apiBase}/diagnosis/run`, {...}, { headers })
  ↓
// Backend (Django + GNN)
Process request, return typed response
  ↓
// Component updates UI
reactively update DOM
```

### 2. **Real-time Updates (WebSocket)**

```typescript
// Connect to channel
const { connect, on } = useWebSocket('diagnostics')
connect()
  ↓
// Listen for events
on('diagnosis_progress', (data) => {
  updateProgress(data.progress)
})
  ↓
// Backend pushes updates
WebSocket → Frontend (reactive)
  ↓
// UI updates automatically
Vue reactivity system
```

---

## 🎯 Component Hierarchy

### Dashboard Page Example

```
dashboard.vue (Page)
│
├── DashboardLayout (Layout)
│   ├── Navbar
│   ├── Sidebar
│   └── <slot> (page content)
│
├── MetricCard (x4)
│   ├── MetricIcon
│   ├── MetricValue
│   └── MetricTrend
│
├── AlertList
│   └── AlertItem (x5)
│       ├── AlertIcon
│       ├── AlertTitle
│       └── AlertActions
│
└── SystemStatus
    ├── StatusChart
    └── StatusTable
```

**Composables Used:**
- `useGeneratedApi()` - API calls
- `useWebSocket('dashboard')` - real-time
- `useSystemStatus()` - system health

---

## 🔐 Security

### Authentication Flow

```
1. User enters credentials
   ↓
2. POST /auth/login
   ↓
3. Backend validates, returns JWT
   ↓
4. Frontend stores token in localStorage
   ↓
5. All API requests include: Authorization: Bearer <token>
   ↓
6. Middleware checks auth on protected routes
   ↓
7. Redirect to /auth/login if not authenticated
```

### Security Measures

- ✅ **JWT tokens** - безопасное хранение
- ✅ **HTTPS only** (production)
- ✅ **CORS configured** - только доверенные домены
- ✅ **XSS protection** - Vue auto-escaping
- ✅ **CSRF tokens** - для mutation requests
- ✅ **Rate limiting** - backend защита

---

## ⚡ Performance

### Optimization Strategies

**1. Code Splitting**
```typescript
// nuxt.config.ts
vite: {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'api-client': ['./generated/api'],
          'charts': ['chart.js'],
          'utils': ['@vueuse/core']
        }
      }
    }
  }
}
```

**2. Lazy Loading**
```vue
<!-- Component lazy load -->
<script setup>
const HeavyChart = defineAsyncComponent(
  () => import('~/components/HeavyChart.vue')
)
</script>
```

**3. Image Optimization**
```vue
<NuxtImg
  src="/image.jpg"
  width="800"
  height="600"
  format="webp"
  loading="lazy"
/>
```

**4. Caching Strategy**
- **Static assets** - 1 year cache
- **API responses** - composable-level cache (5 min)
- **Page routes** - SSR cache (1 min)

**Target Metrics:**
- First Contentful Paint: **< 1.5s**
- Time to Interactive: **< 3s**
- Lighthouse Score: **> 90**
- Bundle Size (initial): **< 200KB**

---

## 🌍 Internationalization

### i18n Architecture

```
i18n/
├── ru.json    # Русский (основной)
└── en.json    # English (вторичный)
```

**Structure:**
```json
{
  "dashboard": {
    "title": "Панель управления",
    "welcome": "Добро пожаловать, {name}!"
  },
  "diagnostics": {...},
  "ui": {...}
}
```

**Usage:**
```vue
<script setup>
const { t, locale } = useI18n()
</script>

<template>
  <h1>{{ t('dashboard.title') }}</h1>
  <p>{{ t('dashboard.welcome', { name: userName }) }}</p>
  
  <!-- Language switcher -->
  <button @click="locale = 'en'">EN</button>
  <button @click="locale = 'ru'">RU</button>
</template>
```

---

## 🧪 Testing Strategy

### Unit Tests (Vitest)

**What to test:**
- ✅ Composables logic (useRAG, useApi)
- ✅ Utility functions
- ✅ Pinia stores
- ✅ Component logic

**Example:**
```typescript
import { describe, it, expect } from 'vitest'
import { useRAG } from '~/composables/useRAG'

describe('useRAG', () => {
  it('should parse reasoning tags', () => {
    const { parseRAGResponse } = useRAG()
    const result = parseRAGResponse('<думает>Test</думает>')
    expect(result.reasoning).toBe('Test')
  })
})
```

### E2E Tests (Playwright)

**Critical Flows:**
1. ✅ Login → Dashboard
2. ✅ Run Diagnosis → View Results
3. ✅ Open Interpretation → See RAG analysis
4. ✅ Add Equipment → View in list

**Example:**
```typescript
import { test, expect } from '@playwright/test'

test('run diagnosis flow', async ({ page }) => {
  await page.goto('/dashboard')
  await page.click('text="Запустить диагностику"')
  await expect(page.locator('.u-modal')).toBeVisible()
  // ...
})
```

---

## 🚀 Deployment

### Build Process

```bash
# 1. Install dependencies
npm install

# 2. Generate API client
npm run generate:api

# 3. Type check
npm run typecheck

# 4. Lint
npm run lint

# 5. Build
npm run build

# Output: .output/ directory
```

### Docker Deployment

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["node", ".output/server/index.mjs"]
```

### Environment Setup

**Development:**
```bash
NUXT_PUBLIC_ENVIRONMENT=development
NUXT_PUBLIC_API_BASE=http://localhost:8000/api/v1
```

**Production:**
```bash
NUXT_PUBLIC_ENVIRONMENT=production
NUXT_PUBLIC_API_BASE=https://api.hydraulic-diagnostics.com/api/v1
NUXT_PUBLIC_FORCE_HTTPS=true
```

---

## 📚 Best Practices

### Code Organization

1. **One responsibility per file**
2. **Composables for logic, components for UI**
3. **No business logic in components**
4. **Types in separate files**
5. **Constants in config/constants.ts**

### Naming Conventions

- **Files:** `kebab-case.vue`, `camelCase.ts`
- **Components:** `PascalCase` (UButton, MetricCard)
- **Composables:** `useCamelCase` (useApi, useRAG)
- **Stores:** `camelCase.store.ts` (auth.store.ts)
- **Types:** `PascalCase` interfaces

### Error Handling

```typescript
// ✅ Good
try {
  const result = await api.diagnosis.run({...})
  // handle success
} catch (error) {
  console.error('Diagnosis failed:', error)
  showNotification({ type: 'error', message: error.message })
}

// ❌ Bad
const result = await api.diagnosis.run({...})  // нет error handling
```

---

## 🔮 Future Roadmap

### Phase 2 (Q1 2026)
- [ ] **Mobile App** - React Native
- [ ] **Offline Mode** - Service Worker + IndexedDB
- [ ] **Advanced Analytics** - Custom charts library
- [ ] **3D Digital Twin** - Three.js integration

### Phase 3 (Q2 2026)
- [ ] **Multi-tenant** - Organization isolation
- [ ] **White-label** - Customizable branding
- [ ] **Plugin System** - Extensible architecture
- [ ] **GraphQL** - Alternative to REST

---

## 📞 Support & Maintenance

### Code Ownership

- **Lead Developer:** Plotnikov Aleksandr
- **Repository:** github.com/Shukik85/hydraulic-diagnostic-saas
- **Contact:** shukik85@ya.ru

### Contributing

See main README.md for contribution guidelines.

---

**Last Updated:** November 15, 2025  
**Document Version:** 1.0.0