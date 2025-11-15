# Frontend Architecture

> Updated: 2025-11-15 21:52 MSK

Фронтенд часть Hydraulic Diagnostic SaaS построена на **Nuxt 4** и следует best practices для production-ready приложений.

---

## 🏗️ Tech Stack

- **Framework**: Nuxt 4 (Vue 3, Vite, TypeScript)
- **Styling**: Tailwind CSS 4.0 + Custom Design System
- **State**: Pinia + Nuxt Auto-imports
- **API**: OpenAPI codegen + Auto-generated TypeScript types
- **i18n**: Nuxt i18n (ru, en, de)
- **Testing**: Vitest
- **Linting**: ESLint + Prettier

---

## 📁 Directory Structure

```
services/frontend/
├── assets/           # Styles, fonts, images
├── components/       # Vue components
│   ├── ui/          # Reusable UI components (design system)
│   ├── features/    # Feature-specific components
│   └── layout/      # Layout components (Header, Footer, etc.)
├── composables/      # Vue composables (business logic)
├── layouts/          # Nuxt layouts (default, dashboard, auth)
├── pages/            # File-based routing (Nuxt convention)
│   ├── index.vue                    # Landing page (/)
│   ├── dashboard.vue                # Main dashboard (/dashboard)
│   ├── chat.vue                     # RAG Chat (/chat)
│   ├── sensors.vue                  # Sensors list (/sensors)
│   ├── auth/                        # Auth pages (/auth/*)
│   ├── diagnostics/                 # Diagnostics section (/diagnostics/*)
│   │   └── index.vue               # Diagnostics dashboard
│   ├── reports/                     # Reports section (/reports/*)
│   │   ├── index.vue               # Reports list
│   │   └── [reportId]/             # Report details
│   ├── settings/                    # Settings section (/settings/*)
│   │   ├── index.vue               # Settings hub
│   │   ├── profile.vue             # Profile settings (/settings/profile)
│   │   ├── notifications.vue       # Notifications (/settings/notifications)
│   │   ├── integrations.vue        # Integrations (/settings/integrations)
│   │   ├── security.vue            # Security (/settings/security)
│   │   └── billing.vue             # Billing (/settings/billing)
│   ├── systems/                     # Systems section (/systems/*)
│   │   ├── index.vue               # Systems list
│   │   └── [systemId]/             # System details (using systemId parameter)
│   │       ├── index.vue           # System overview
│   │       └── equipments/         # Equipments subsection
│   │           ├── index.vue       # Equipments list
│   │           └── [equipmentId].vue  # Equipment details
│   ├── system-metadata/             # System metadata (/system-metadata/*)
│   ├── landing.vue                  # Marketing landing (/landing)
│   ├── investors.vue                # Investor page (/investors)
│   ├── api-test.vue                 # API testing (dev only)
│   └── demo.vue                     # Demo page (dev only)
├── middleware/       # Route middleware (auth, guest, etc.)
├── plugins/          # Nuxt plugins
├── public/           # Static assets (served as-is)
├── stores/           # Pinia stores
├── types/            # TypeScript type definitions
└── utils/            # Utility functions
```

---

## 🛣️ Routing Architecture

### ✅ RESOLVED Routing Conflicts

**Fixed in commits:**
- `6d73c3c3` - Moved diagnostics.vue → diagnostics/index.vue
- `c689b640` - Removed conflicting pages/diagnostics.vue
- `5ecf94ab` - Moved reports.vue → reports/index.vue
- `5692edb9` - Removed conflicting pages/reports.vue
- `ab90ddd3` - Moved settings.vue → settings/index.vue
- `237e99a4` - Removed conflicting pages/settings.vue

### Current Routing Map

```
/                              → pages/index.vue (Landing)
/dashboard                     → pages/dashboard.vue
/chat                          → pages/chat.vue
/sensors                       → pages/sensors.vue
/landing                       → pages/landing.vue
/investors                     → pages/investors.vue

/auth/*                        → pages/auth/*
  /auth/login                  → pages/auth/login.vue
  /auth/register               → pages/auth/register.vue

/diagnostics                   → pages/diagnostics/index.vue ✅

/reports                       → pages/reports/index.vue ✅
/reports/:reportId/*           → pages/reports/[reportId]/*

/settings                      → pages/settings/index.vue ✅
/settings/profile              → pages/settings/profile.vue
/settings/notifications        → pages/settings/notifications.vue
/settings/integrations         → pages/settings/integrations.vue
/settings/security             → pages/settings/security.vue
/settings/billing              → pages/settings/billing.vue

/systems                       → pages/systems/index.vue
/systems/:systemId             → pages/systems/[systemId]/index.vue ⚠️
/systems/:systemId/equipments  → pages/systems/[systemId]/equipments/index.vue
/systems/:systemId/equipments/:equipmentId → pages/systems/[systemId]/equipments/[equipmentId].vue

/system-metadata/*             → pages/system-metadata/*

/api-test                      → pages/api-test.vue (dev only, blocked in production)
/demo                          → pages/demo.vue (dev only, blocked in production)
```

---

## ⚠️ Remaining Issues

### 🟡 MEDIUM Priority

#### 1. Duplicate parameter naming in systems routes
**Status**: ACTIVE CONFLICT

```
pages/systems/[id]/             - uses route.params.id
pages/systems/[systemId]/       - uses route.params.systemId
```

**Problem**: Both directories exist simultaneously! Nuxt cannot distinguish between them.

**Files affected**:
- `pages/systems/[id]/equipments/[equipmentId].vue` (uses `route.params.id`)
- `pages/systems/[systemId]/index.vue` (uses `route.params.systemId`)
- `pages/systems/[systemId]/equipments/[equipmentId].vue` (uses `route.params.systemId`)

**Solution**: 
1. Choose ONE naming convention: `[id]` or `[systemId]`
2. Delete the unused directory
3. Update all references in components/composables

**Recommendation**: Use `[systemId]` for clarity (более семантичный)

#### 2. Diagnosis vs Diagnostics confusion
**Status**: LOW PRIORITY

```
pages/diagnosis/demo.vue        - только ErrorBoundary wrapper
pages/diagnostics/index.vue     - основная страница диагностики
```

**Problem**: Семантическая путаница. `diagnosis` != `diagnostics`

**Solution**: 
- Удалить `pages/diagnosis/` если не используется
- ИЛИ переименовать в `/diagnosis-demo` если нужен для testing

---

## 🎯 Design System

### Metallic Industrial B2B Theme

**Core Principles:**
- Professional, clean, engineering-focused
- High contrast, readability-first
- Minimalist with strategic accents
- Responsive and accessible

**Key Design Tokens:**
```css
/* Primary Colors */
--primary-600: #2563eb (Blue)
--primary-700: #1d4ed8

/* Neutral/Steel Palette */
--steel-light: #f8fafc
--steel-base: #64748b
--steel-dark: #1e293b

/* Semantic Colors */
--success: #10b981 (Green)
--warning: #f59e0b (Orange)
--error: #ef4444 (Red)
--info: #3b82f6 (Blue)
```

**Component Classes (Utility-first):**
```
u-h1, u-h2, u-h3, u-h4, u-h5      - Typography
u-body, u-body-sm                  - Body text
u-btn, u-btn-primary, u-btn-sm     - Buttons
u-card                             - Cards
u-badge, u-badge-success           - Badges
u-input, u-label                   - Forms
u-metric-card                      - Dashboard metrics
u-table                            - Tables
u-transition-fast                  - Animations
u-flex-center, u-flex-between      - Layout helpers
```

### Elements of Friendliness

**Applied throughout:**
- Smooth hover transitions (u-transition-fast)
- Subtle shadows and depth
- Rounded corners (metallic softness)
- Friendly empty states with helpful messages
- Progress indicators during async operations
- Toast notifications for user feedback
- Clear visual hierarchy
- Intuitive iconography (Heroicons)
- Micro-interactions on buttons/cards

**Where to keep/add:**
- Dashboard metrics: hover effects, smooth counters
- Chat interface: typing indicators, message animations
- Forms: validation feedback, success confirmations
- Navigation: active state indicators, smooth transitions
- Modals: gentle backdrop blur, slide-in animations

---

## 🔧 Configuration Files

### nuxt.config.ts
```typescript
export default defineNuxtConfig({
  ssr: true,
  devtools: { enabled: true },
  
  modules: [
    '@nuxt/eslint',
    '@nuxtjs/tailwindcss',
    '@nuxtjs/i18n',
    '@pinia/nuxt',
    'nuxt-icon'
  ],
  
  // ✅ FIXED: Mocks only in development
  runtimeConfig: {
    public: {
      enableMocks: process.env.ENABLE_MOCKS === 'true' || process.env.NODE_ENV === 'development'
    }
  },
  
  // ✅ FIXED: TypeScript checking enabled
  typescript: {
    typeCheck: true,
    strict: true
  },
  
  // ✅ NEW: Nuxt 4 experimental features
  experimental: {
    granularCachedData: true,
    purgeCachedData: true
  },
  
  // ✅ NEW: Block test routes in production
  routeRules: {
    '/api-test': { redirect: process.env.NODE_ENV === 'production' ? '/' : undefined },
    '/demo': { redirect: process.env.NODE_ENV === 'production' ? '/' : undefined }
  }
})
```

### tsconfig.json
```json
{
  "extends": "./.nuxt/tsconfig.json",
  "compilerOptions": {
    "strict": true,
    "skipLibCheck": true
  }
}
```

### eslint.config.mjs
```javascript
// ✅ NEW: Added strict rules
export default [
  ...defaultConfig,
  {
    rules: {
      'vue/multi-word-component-names': 'warn',
      'vue/no-unused-components': 'warn',
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      '@typescript-eslint/no-unused-vars': 'warn',
      '@typescript-eslint/no-explicit-any': 'warn'
    }
  }
]
```

---

## 📝 TODO List

### 🔴 CRITICAL - Requires Immediate Action

- [x] ~~**Routing conflicts**: pages/diagnostics.vue + pages/diagnostics/~~ ✅ FIXED
- [x] ~~**Routing conflicts**: pages/reports.vue + pages/reports/~~ ✅ FIXED
- [x] ~~**Routing conflicts**: pages/settings.vue + pages/settings/~~ ✅ FIXED
- [ ] **Duplicate parameters**: pages/systems/[id]/ vs pages/systems/[systemId]/ - выбрать один
- [ ] **TypeScript errors**: 291 error в 84 файлах - исправить автоимпорты и типы

### 🟡 HIGH Priority

- [ ] **three.js dependency**: Проверить использование, удалить если не нужен (~500KB bundle)
- [ ] **Duplicate diagnosis/**: Удалить pages/diagnosis/ если не используется
- [ ] **Types export**: Добавить export для всех типов в types/api.ts
- [ ] **Component props**: Добавить TypeScript типизацию для UI компонентов

### 🟠 MEDIUM Priority

- [ ] **Composables**: Обновить для Nuxt 4 Singleton Data Fetching Layer
- [ ] **Animations**: Перенести из app.vue в tailwind.config.ts
- [ ] **Components**: Стандартизировать feature-based организацию
- [ ] **API mocking**: Переделать mock-логику для development-only режима

### 🟢 LOW Priority

- [ ] **Empty states**: Добавить friendly empty states во все списки
- [ ] **Loading states**: Стандартизировать skeleton loaders
- [ ] **Error boundaries**: Расширить использование ErrorBoundary компонента
- [ ] **Accessibility**: ARIA labels, keyboard navigation
- [ ] **Performance**: Bundle analysis, code splitting optimization

---

## 🎨 Design System Guidelines

### Metallic Industrial Theme

**Visual Hierarchy:**
1. **Primary Actions**: Blue gradient buttons with shadow
2. **Secondary Actions**: Ghost/outline buttons
3. **Destructive Actions**: Red accent

**Card Anatomy:**
```vue
<div class="u-card p-6">
  <div class="u-card-header">Title</div>
  <div class="u-card-body">Content</div>
  <div class="u-card-footer">Actions</div>
</div>
```

**Metrics Display:**
```vue
<div class="u-metric-card">
  <div class="u-metric-header">
    <h3 class="u-metric-label">Label</h3>
    <div class="u-metric-icon">Icon</div>
  </div>
  <div class="u-metric-value">Value</div>
  <div class="u-metric-change">Change indicator</div>
</div>
```

**Badges:**
```vue
<span class="u-badge u-badge-success">Active</span>
<span class="u-badge u-badge-warning">Pending</span>
<span class="u-badge u-badge-error">Failed</span>
<span class="u-badge u-badge-info">Processing</span>
```

---

## 🚀 Best Practices

### File Naming
- **Pages**: kebab-case (`system-metadata.vue`)
- **Components**: PascalCase (`UButton.vue`, `UModal.vue`)
- **Composables**: camelCase with `use` prefix (`useAuth.ts`, `useSystemsApi.ts`)
- **Types**: PascalCase interfaces/types (`SystemMetadata`, `DiagnosticResult`)

### Component Structure
```vue
<script setup lang="ts">
// 1. Imports
import type { SystemMetadata } from '~/types/api'

// 2. Props/Emits
interface Props {
  systemId: string
  variant?: 'default' | 'compact'
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default'
})

// 3. Composables
const { t } = useI18n()
const route = useRoute()

// 4. Reactive State
const loading = ref(false)
const data = ref<SystemMetadata | null>(null)

// 5. Computed
const isActive = computed(() => data.value?.status === 'active')

// 6. Methods
const fetchData = async () => {
  // ...
}

// 7. Lifecycle
onMounted(() => {
  fetchData()
})
</script>

<template>
  <!-- Template here -->
</template>
```

### Composables Pattern (Nuxt 4)
```typescript
// composables/useSystemsApi.ts
export const useSystemsApi = () => {
  const config = useRuntimeConfig()
  
  const fetchSystems = async () => {
    return await $fetch('/api/systems', {
      baseURL: config.public.apiBase
    })
  }
  
  return {
    fetchSystems
  }
}
```

---

## 🧪 Testing Strategy

### Run Tests
```bash
npm run test              # Run all tests
npm run test:watch        # Watch mode
npm run test:coverage     # Coverage report
```

### Test Structure
```typescript
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import UButton from '~/components/ui/button.vue'

describe('UButton', () => {
  it('renders with correct variant', () => {
    const wrapper = mount(UButton, {
      props: { variant: 'primary' }
    })
    expect(wrapper.classes()).toContain('u-btn-primary')
  })
})
```

---

## 🔍 TypeScript Integration

### Auto-imports
Nuxt 4 автоматически генерирует типы для:
- Vue composables (ref, computed, watch, etc.)
- Nuxt composables (useRoute, useRouter, useFetch, etc.)
- Components (auto-imported from components/)
- Utils (auto-imported from utils/)

### Extending Nuxt Types
```typescript
// types/nuxt.d.ts
declare module '#app' {
  interface PageMeta {
    requiresAuth?: boolean
    roles?: string[]
  }
}

export {}
```

---

## 📦 Build & Deploy

### Development
```bash
npm run dev               # Start dev server
npm run build             # Production build
npm run preview           # Preview production build
npm run typecheck         # Run TypeScript checks
npm run lint              # Run ESLint
npm run lint:fix          # Fix ESLint issues
```

### Environment Variables
```env
# .env.local
NUXT_PUBLIC_API_BASE=http://localhost:8000/api/v1
NUXT_PUBLIC_WS_URL=ws://localhost:8000/ws
ENABLE_MOCKS=false
```

### Docker
```dockerfile
# Production build
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
CMD ["node", ".output/server/index.mjs"]
```

---

## 📚 Additional Resources

- [Nuxt 4 Migration Guide](https://nuxt.com/docs/getting-started/upgrade)
- [Vue 3 Composition API](https://vuejs.org/guide/extras/composition-api-faq.html)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [TypeScript Deep Dive](https://basarat.gitbook.io/typescript/)

---

## 🤝 Contributing

1. Create feature branch from `master`
2. Make changes following architecture guidelines
3. Run `npm run typecheck` and `npm run lint`
4. Create PR with clear description
5. Wait for CI/CD checks to pass
6. Request review from team

---

## 📞 Support

For questions or issues:
- GitHub Issues: [hydraulic-diagnostic-saas/issues](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues)
- Project Lead: @Shukik85
