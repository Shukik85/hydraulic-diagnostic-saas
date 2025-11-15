# 🔧 Hydraulic Diagnostic SaaS - Frontend

> AI-powered predictive maintenance platform for industrial hydraulic systems

[![Nuxt 3](https://img.shields.io/badge/Nuxt-3.12.4-00DC82?logo=nuxt.js)](https://nuxt.com)
[![Vue 3](https://img.shields.io/badge/Vue-3.4-4FC08D?logo=vue.js)](https://vuejs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6?logo=typescript)](https://www.typescriptlang.org)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

---

## 🌟 Key Features

- **🤖 AI-Powered Diagnostics** - GNN-based anomaly detection + RAG interpretation
- **📊 Real-time Monitoring** - WebSocket streaming with live updates
- **🌍 Multi-language** - Full i18n support (RU/EN)
- **📱 Responsive Design** - Mobile-first, works on all devices
- **🔒 Enterprise Security** - JWT auth, role-based access control
- **⚡ High Performance** - SSR/SSG, optimized bundle, lazy loading
- **🎨 Modern UI** - Tailwind CSS, unified design system
- **🔌 Type-Safe API** - OpenAPI auto-generated client

---

## 🏗️ Architecture

### Tech Stack

**Frontend Framework:**
- **Nuxt 3.12** - SSR/SSG, auto-imports, file-based routing
- **Vue 3.4** - Composition API, `<script setup>`, reactivity
- **TypeScript 5.5** - Full type safety

**State Management:**
- **Pinia** - Modern Vuex alternative
- **VueUse** - Composable utilities

**UI/Styling:**
- **Tailwind CSS** - Utility-first styling
- **Heroicons** - Icon system
- **Custom Design System** - Unified components (u-card, u-btn, etc.)

**API Integration:**
- **OpenAPI TypeScript Codegen** - Auto-generated type-safe client
- **Axios** - HTTP client
- **WebSocket** - Real-time updates

**Development Tools:**
- **ESLint 9** - Linting with Nuxt config
- **Playwright** - E2E testing
- **Vitest** - Unit testing
- **TypeScript** - Static type checking

---

## 📂 Project Structure

```
services/frontend/
├── pages/                    # File-based routing
│   ├── index.vue            # Landing page
│   ├── dashboard.vue        # Main dashboard
│   ├── diagnostics.vue      # Diagnostics monitoring
│   ├── diagnostics/         # Diagnostics sub-pages
│   │   └── [id]/
│   │       └── interpretation.vue  # RAG interpretation
│   ├── systems/             # Equipment management
│   ├── reports/             # Reporting
│   └── settings/            # User settings
│
├── components/              # Vue components
│   ├── ui/                  # Design system (U* components)
│   ├── dashboard/           # Dashboard widgets
│   ├── digital-twin/        # Digital twin visualization
│   ├── metadata/            # System metadata
│   └── rag/                 # RAG interpretation UI
│
├── composables/             # Composition API logic
│   ├── useGeneratedApi.ts   # Type-safe API client
│   ├── useRAG.ts            # RAG integration
│   ├── useWebSocket.ts      # Real-time updates
│   ├── useDigitalTwin.ts    # Digital twin state
│   └── useMockData.ts       # Mock data (dev/demo)
│
├── stores/                  # Pinia stores
│   ├── auth.store.ts        # Authentication
│   ├── systems.store.ts     # Equipment state
│   └── metadata.ts          # System metadata
│
├── layouts/                 # Page layouts
│   ├── default.vue          # Default layout
│   └── dashboard.vue        # Dashboard layout
│
├── middleware/              # Route middleware
│   └── auth.ts              # Auth guard
│
├── types/                   # TypeScript types
│   ├── api.ts               # API types
│   ├── rag.ts               # RAG types
│   └── features.ts          # Feature flags
│
├── generated/               # Auto-generated (OpenAPI)
│   └── api/                 # Type-safe API client
│
├── i18n/                    # Internationalization
│   ├── en.json              # English translations
│   └── ru.json              # Russian translations
│
├── styles/                  # Global styles
│   └── main.css             # Tailwind + custom CSS
│
├── public/                  # Static assets
│   ├── favicon.ico
│   └── images/
│
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # Architecture overview
│   ├── API_INTEGRATION.md   # API integration guide
│   ├── RAG_INTEGRATION.md   # RAG usage guide
│   └── DEPLOYMENT.md        # Deployment instructions
│
├── nuxt.config.ts           # Nuxt configuration
├── tailwind.config.ts       # Tailwind configuration
├── tsconfig.json            # TypeScript configuration
├── eslint.config.mjs        # ESLint configuration
├── .env.example             # Environment variables template
└── package.json             # Dependencies and scripts
```

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** >= 20.x
- **npm** >= 10.x
- **Backend services** running (Django + GNN + RAG)

### Installation

```bash
# Clone repository
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas/services/frontend

# Install dependencies
npm install

# Setup environment
cp .env.example .env
# Edit .env with your backend URLs

# Generate API client from OpenAPI spec
npm run generate:api

# Start development server
npm run dev
```

**Development server:** http://localhost:3000

---

## 🔧 Development

### Available Scripts

```bash
# Development
npm run dev              # Start dev server (http://localhost:3000)
npm run generate:api     # Generate API client from OpenAPI spec
npm run validate:api     # Validate OpenAPI spec

# Build
npm run build            # Production build
npm run generate         # Static site generation (SSG)
npm run preview          # Preview production build

# Quality
npm run lint             # Run ESLint
npm run lint:fix         # Fix ESLint errors
npm run typecheck        # TypeScript type checking
npm test                 # Run unit tests (Vitest)
npm run test:e2e         # Run E2E tests (Playwright)
```

### Environment Variables

Create `.env` file (see `.env.example`):

```bash
# API Configuration
NUXT_PUBLIC_API_BASE=http://localhost:8000/api/v1
NUXT_PUBLIC_WS_BASE=ws://localhost:8000/ws

# Feature Flags
NUXT_PUBLIC_ENABLE_RAG=true
NUXT_PUBLIC_ENABLE_WEBSOCKET=true
NUXT_PUBLIC_ENABLE_CHARTS=true

# Environment
NUXT_PUBLIC_ENVIRONMENT=development
```

### API Integration

The frontend uses **OpenAPI TypeScript Codegen** for type-safe API integration:

```typescript
// Auto-generated from backend OpenAPI spec
import { useGeneratedApi } from '~/composables/useGeneratedApi'

const api = useGeneratedApi()

// Fully typed!
const result = await api.diagnosis.runDiagnosis({
  equipmentId: 'exc_001',
  diagnosisRequest: { /* ... */ }
})

// RAG interpretation
const interpretation = await api.rag.interpretDiagnosis({
  gnnResult: result,
  equipmentContext: { /* ... */ }
})
```

**API client regeneration:**
- Auto-runs on `npm run dev` and `npm run build`
- Manual: `npm run generate:api`
- Requires: `../../specs/combined-api.json` (from backend)

---

## 🎨 UI Design System

### Custom Components (u-* prefix)

```vue
<template>
  <!-- Cards -->
  <div class="u-card">
    <h3 class="u-h3">Title</h3>
    <p class="u-body">Content</p>
  </div>

  <!-- Buttons -->
  <button class="u-btn u-btn-primary u-btn-md">
    Primary Action
  </button>

  <!-- Inputs -->
  <input class="u-input" type="text" placeholder="Enter value" />

  <!-- Badges -->
  <span class="u-badge u-badge-success">Active</span>

  <!-- Metrics -->
  <div class="u-metric-card">
    <div class="u-metric-value">92.5%</div>
    <div class="u-metric-label">Success Rate</div>
  </div>
</template>
```

**See:** `styles/main.css` for full design system

---

## 🌍 Internationalization (i18n)

### Supported Languages

- 🇷🇺 **Russian** (ru) - Primary
- 🇬🇧 **English** (en) - Secondary

### Usage

```vue
<script setup>
const { t } = useI18n()
</script>

<template>
  <h1>{{ t('dashboard.title') }}</h1>
  <p>{{ t('dashboard.welcome', { name: userName }) }}</p>
</template>
```

**Translation files:**
- `i18n/ru.json` - Russian translations
- `i18n/en.json` - English translations

---

## 🔌 Backend Integration

### Services Architecture

```
Frontend (Nuxt SSR)
    ↓
API Gateway (Kong)
    ↓
┌─────────────────────────────────┐
│  Backend Services               │
├─────────────────────────────────┤
│ Django (Port 8000)              │ ← Equipment, Users, Auth
│ GNN Service (Port 8002)         │ ← Anomaly Detection
│ RAG Service (Port 8004)         │ ← AI Interpretation
│ TimescaleDB (Port 5432)         │ ← Time-series data
└─────────────────────────────────┘
```

### API Endpoints

**Base URL:** `http://localhost:8000/api/v1` (dev)

**Services:**
- `/auth/*` - Authentication (Django)
- `/equipment/*` - Equipment management (Django)
- `/diagnosis/*` - Diagnostics (Django → GNN)
- `/gnn/*` - GNN direct access
- `/rag/*` - RAG interpretation (RAG Service)

**WebSocket:** `ws://localhost:8000/ws`

---

## 🚀 Deployment

### Production Build

```bash
# 1. Generate API client
npm run generate:api

# 2. Build for production
npm run build

# 3. Preview locally
npm run preview

# Output: .output/ directory (ready for Node.js hosting)
```

### Docker Deployment

```bash
# Build image
docker build -t hydraulic-frontend:latest .

# Run container
docker run -p 3000:3000 \
  -e NUXT_PUBLIC_API_BASE=https://api.hydraulic-diagnostics.com \
  hydraulic-frontend:latest
```

### Environment Setup

**Production `.env`:**
```bash
NUXT_PUBLIC_API_BASE=https://api.hydraulic-diagnostics.com
NUXT_PUBLIC_WS_BASE=wss://api.hydraulic-diagnostics.com/ws
NUXT_PUBLIC_ENABLE_RAG=true
NUXT_PUBLIC_ENABLE_WEBSOCKET=true
NUXT_PUBLIC_ENVIRONMENT=production
```

---

## 🧪 Testing

### Unit Tests (Vitest)

```bash
# Run all tests
npm test

# Watch mode
npm test -- --watch

# Coverage
npm test -- --coverage
```

### E2E Tests (Playwright)

```bash
# Install browsers
npx playwright install

# Run E2E tests
npm run test:e2e

# Interactive mode
npm run test:e2e -- --ui
```

---

## 📊 Key Pages

### User Flow

```
/ (Landing)
  ↓
/auth/login
  ↓
/dashboard (Main)
  ↓
/systems → /systems/[id] → /systems/[id]/sensors
  ↓
/diagnostics → /diagnostics/[id] → /diagnostics/[id]/interpretation
  ↓
/reports
```

### Main Pages

- **`/`** - Landing page (marketing)
- **`/dashboard`** - Main dashboard (metrics, alerts)
- **`/diagnostics`** - Diagnostics monitoring
- **`/diagnostics/[id]/interpretation`** - RAG AI interpretation
- **`/systems`** - Equipment management
- **`/reports`** - Reporting and analytics
- **`/settings`** - User settings and preferences
- **`/api-test`** - API testing tool (dev only)

---

## 🤖 RAG Integration

### AI-Powered Interpretation

The platform uses **DeepSeek-R1** (70B parameters) for intelligent diagnosis interpretation:

```typescript
import { useRAG } from '~/composables/useRAG'

const { interpretDiagnosis, loading, error } = useRAG()

// Get AI interpretation of GNN results
const interpretation = await interpretDiagnosis({
  gnnResults: diagnosticResults,
  equipmentId: 'exc_001',
  useKnowledgeBase: true  // Enable RAG retrieval
})

// Returns structured response:
// {
//   reasoning: string,      // Step-by-step analysis
//   summary: string,        // Executive summary
//   analysis: string,       // Detailed findings
//   recommendations: [...], // Action items
//   confidence: 0.92,       // AI confidence score
//   knowledgeUsed: [...]    // KB documents used
// }
```

**Features:**
- ✅ Structured reasoning process visualization
- ✅ Knowledge base context display
- ✅ Confidence scoring
- ✅ Real-time streaming responses
- ✅ Fallback to basic mode if RAG unavailable

**See:** `docs/RAG_INTEGRATION.md` for detailed guide

---

## 🔐 Authentication

### JWT-based Auth Flow

```typescript
import { useAuthStore } from '~/stores/auth.store'

const authStore = useAuthStore()

// Login
await authStore.login({
  email: 'user@example.com',
  password: 'password'
})

// Auto-injected in API calls
const api = useGeneratedApi()
const systems = await api.equipment.listSystems()
// ↑ Authorization header added automatically
```

**Protected routes:** Use `auth` middleware

```typescript
definePageMeta({
  middleware: ['auth']  // Redirects to /auth/login if not authenticated
})
```

---

## 📈 Performance

### Optimization Strategies

- **Code Splitting** - Automatic route-based chunks
- **Lazy Loading** - Components loaded on demand
- **Image Optimization** - `@nuxt/image` module
- **Bundle Analysis** - Manual chunks for large deps
- **SSR Caching** - Route caching enabled
- **API Response Caching** - Composables with cache

### Production Metrics (Target)

- **First Contentful Paint:** < 1.5s
- **Time to Interactive:** < 3s
- **Lighthouse Score:** > 90
- **Bundle Size:** < 200KB (initial)

---

## 🛠️ Development Guidelines

### Code Style

- **ESLint** - Follow Nuxt recommended config
- **TypeScript** - Strict mode enabled
- **Naming** - camelCase for variables, PascalCase for components
- **Components** - Use `<script setup>` composition API
- **Imports** - Auto-imports enabled (no need to import Vue, Nuxt composables)

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes, commit
git add .
git commit -m "feat(scope): description"

# Push and create PR
git push origin feature/your-feature
```

### Commit Convention

```
feat(scope): description     # New feature
fix(scope): description      # Bug fix
refactor(scope): description # Code refactoring
docs(scope): description     # Documentation
style(scope): description    # Formatting
test(scope): description     # Tests
chore(scope): description    # Maintenance
```

---

## 🐛 Troubleshooting

### Common Issues

**API client not generated:**
```bash
npm run generate:api
# Check ../../specs/combined-api.json exists
```

**TypeScript errors:**
```bash
npm run typecheck
# Fix any type errors before committing
```

**Port 3000 in use:**
```bash
# Change port
PORT=3001 npm run dev
```

**Module not found:**
```bash
# Clear cache and reinstall
rm -rf node_modules .nuxt
npm install
```

---

## 📚 Documentation

### Technical Docs

- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design and patterns
- **[API Integration Guide](docs/API_INTEGRATION.md)** - Backend integration
- **[RAG Integration Guide](docs/RAG_INTEGRATION.md)** - AI features usage
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment

### External Resources

- [Nuxt 3 Documentation](https://nuxt.com/docs)
- [Vue 3 Documentation](https://vuejs.org/guide)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Pinia Documentation](https://pinia.vuejs.org)

---

## 🤝 Contributing

### Development Process

1. **Create feature branch** from `master`
2. **Implement changes** following code style
3. **Write tests** for new features
4. **Update documentation** if needed
5. **Run linting** and type checking
6. **Create Pull Request** with description
7. **Request review** from team

### Quality Checklist

- [ ] Code follows ESLint rules
- [ ] TypeScript types are correct (`npm run typecheck`)
- [ ] Components are responsive (mobile-first)
- [ ] i18n keys added for new text
- [ ] Tests written and passing
- [ ] No console.log in production code
- [ ] Documentation updated

---

## 📝 License

Proprietary - All rights reserved

---

## 🎯 Project Status

**Version:** 1.0.0  
**Status:** 🟢 Production Ready (MVP)  
**Last Updated:** November 15, 2025

### Roadmap

- [x] Core diagnostics UI
- [x] Real-time monitoring
- [x] Equipment management
- [x] RAG AI interpretation
- [x] Multi-language support
- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard
- [ ] Predictive maintenance ML models
- [ ] Multi-tenant support

---

## 💼 For Investors

### Technology Highlights

**Why This Stack?**

- **Nuxt 3** - SEO-optimized, fast TTI, enterprise-proven
- **TypeScript** - Reduces bugs by 40%, better DX
- **OpenAPI Codegen** - Type-safe API, automatic sync
- **Microservices** - Scalable, independent deployment
- **AI/ML Integration** - GNN + RAG = cutting-edge diagnostics

**Production Ready:**
- ✅ Docker deployment
- ✅ CI/CD pipeline
- ✅ Monitoring and observability
- ✅ Security best practices
- ✅ Scalable architecture

### Market Fit

- **TAM:** $5.2B (industrial predictive maintenance)
- **Target:** Manufacturing, mining, oil & gas
- **Advantage:** AI-powered interpretation (unique differentiator)

---

## 📞 Support

**Developer:** Plotnikov Aleksandr  
**Email:** shukik85@ya.ru  
**GitHub:** [@Shukik85](https://github.com/Shukik85)

---

**Built with ❤️ and AI in Russia** 🇷🇺