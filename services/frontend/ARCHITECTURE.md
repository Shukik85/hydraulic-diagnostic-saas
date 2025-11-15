# Frontend Architecture Documentation

## Обзор проекта

Frontend часть Hydraulic Diagnostic SaaS построена на **Nuxt 4** с использованием Vue 3 Composition API, TypeScript и Tailwind CSS.

### Технологический стек

- **Framework**: Nuxt 4.2.1
- **Vue**: 3.5.24
- **TypeScript**: 5.9.3
- **Styling**: Tailwind CSS 6.14.0 + Custom Metallic Theme
- **State Management**: Pinia 3.0.4
- **Charts**: ECharts 6.0 + vue-echarts 8.0.1
- **HTTP Client**: Axios 1.13.2
- **i18n**: @nuxtjs/i18n 10.2.0

## Структура директорий

```
services/frontend/
├── app.vue                 # Root application component
├── nuxt.config.ts          # Nuxt configuration
├── tsconfig.json           # TypeScript configuration
├── tailwind.config.ts      # Tailwind CSS configuration
│
├── assets/                 # Static assets (images, fonts)
├── components/             # Vue components
│   ├── ui/                 # Reusable UI components
│   ├── dashboard/          # Dashboard-specific components
│   ├── digital-twin/       # Digital twin visualization
│   ├── rag/                # RAG chat components
│   ├── Diagnosis/          # Diagnostic components
│   ├── Error/              # Error handling components
│   └── Loading/            # Loading states
│
├── composables/            # Vue composables (reusable logic)
│   ├── useAnomalies.ts     # Anomaly detection logic
│   ├── useDigitalTwin.ts   # Digital twin state
│   ├── useRAG.ts           # RAG chat functionality
│   ├── useWebSocket.ts     # WebSocket connections
│   └── useMockData.ts      # Mock data for development
│
├── pages/                  # File-based routing
│   ├── index.vue           # Main landing page
│   ├── dashboard.vue       # Main dashboard
│   ├── chat.vue            # RAG chat page
│   ├── auth/               # Authentication pages
│   ├── diagnosis/          # Diagnostic pages (TODO: consolidate with diagnostics)
│   ├── diagnostics/        # Diagnostic pages (TODO: consolidate with diagnosis)
│   ├── reports/            # Reports pages
│   ├── settings/           # Settings pages
│   └── systems/            # System management
│
├── layouts/                # Layout templates
│   └── default.vue         # Default layout with navigation
│
├── middleware/             # Route middleware
│   └── auth.ts             # Authentication guard
│
├── stores/                 # Pinia stores
│   ├── auth.store.ts       # Authentication state
│   ├── systems.store.ts    # Systems management
│   └── metadata.ts         # Metadata management
│
├── generated/              # Auto-generated API client
│   └── api/                # OpenAPI TypeScript client
│
├── i18n/                   # Internationalization
│   └── locales/            # Translation files (ru.json, en.json)
│
├── public/                 # Public static files
├── server/                 # Server-side code
├── styles/                 # Global styles
│   └── metallic.css        # Custom metallic theme
│
├── tests/                  # Test files
├── types/                  # TypeScript type definitions
└── utils/                  # Utility functions
```

## Маршрутизация (File-based Routing)

Nuxt 4 использует файловую систему маршрутизации:

### Основные маршруты

| Route | File | Description |
|-------|------|-------------|
| `/` | `pages/index.vue` | Главная страница |
| `/dashboard` | `pages/dashboard.vue` | Панель управления |
| `/chat` | `pages/chat.vue` | RAG чат |
| `/auth/login` | `pages/auth/login.vue` | Авторизация |
| `/diagnostics` | `pages/diagnostics/` | Диагностика систем |
| `/reports` | `pages/reports/` | Отчёты |
| `/settings` | `pages/settings/` | Настройки |

### ⚠️ Known Issues

**TODO: Разрешить конфликты маршрутизации**:

1. `pages/diagnosis/` и `pages/diagnostics/` - две похожие директории
2. `pages/diagnostics.vue` + `pages/diagnostics/` - конфликт файл/директория
3. `pages/reports.vue` + `pages/reports/` - конфликт файл/директория
4. `pages/settings.vue` + `pages/settings/` - конфликт файл/директория

**Рекомендация**: Использовать только директории для масштабируемости.

## Composables Pattern

Проект использует Vue 3 Composition API с composables для переиспользуемой логики.

### Nuxt 4 Data Fetching

В Nuxt 4 все composables с `useAsyncData` автоматически share data с одинаковым key:

```typescript
// composables/useAnomalies.ts
export const useAnomalies = () => {
  const { data, error, refresh } = useAsyncData(
    'anomalies', // Singleton key - все компоненты share эти данные
    () => $fetch('/api/anomalies')
  )
  
  return { data, error, refresh }
}
```

**Преимущества**:
- Автоматический cleanup при unmount
- Reactive refs shared между компонентами
- Автоматическое кеширование

## State Management

### Pinia Stores

Используются для глобального состояния:

- **auth.store.ts**: Авторизация, токены, пользовательские данные
- **systems.store.ts**: Управление гидравлическими системами
- **metadata.ts**: Метаданные систем

## API Integration

### Auto-generated Client

Проект использует `openapi-typescript-codegen` для автоматической генерации API клиента:

```bash
npm run generate:api
```

Генерирует TypeScript клиент в `generated/api/` на основе OpenAPI spec.

### Mock Data

Для development используются моки:

```typescript
// nuxt.config.ts
runtimeConfig: {
  public: {
    enableMocks: process.env.ENABLE_MOCKS === 'true' || process.env.NODE_ENV === 'development'
  }
}
```

**Важно**: Моки автоматически отключены в production.

## Styling

### Tailwind CSS + Metallic Theme

Проект использует кастомную металлическую тему:

- `styles/metallic.css` - основная тема
- `tailwind.config.ts` - конфигурация Tailwind

**TODO**: Перенести animation utilities из `app.vue` в `tailwind.config.ts`.

## Internationalization (i18n)

Поддерживаемые языки:
- 🇷🇺 Русский (default)
- 🇬🇧 English

Переводы хранятся в `i18n/locales/`.

## Testing

- **Unit tests**: Vitest
- **E2E tests**: Playwright

```bash
npm run test        # Unit tests
npm run test:e2e    # E2E tests
```

## Development Workflow

### Commands

```bash
npm run dev         # Development server (port 3000)
npm run build       # Production build
npm run generate:api # Generate API client
npm run typecheck   # TypeScript type checking
```

### Environment Variables

```env
NUXT_PUBLIC_API_BASE=http://localhost:8000/api/v1
NUXT_PUBLIC_WS_BASE=ws://localhost:8000/ws
ENABLE_MOCKS=false  # true to enable mock data
```

## TODO List

### Критичное (CRITICAL)

- [ ] Разрешить routing conflicts (diagnostics.vue vs diagnostics/)
- [ ] Объединить diagnosis/ и diagnostics/ в одну директорию

### Высокий приоритет (HIGH)

- [ ] Проверить использование three.js - удалить если не нужен
- [ ] Консолидировать reports.vue и reports/
- [ ] Консолидировать settings.vue и settings/

### Средний приоритет (MEDIUM)

- [x] Удалить @nuxt/types из tsconfig.json
- [ ] Обновить composables для Nuxt 4 Singleton Data Fetching
- [ ] Перенести animations в tailwind.config.ts
- [ ] Стандартизировать организацию components/ (feature-based)

### Низкий приоритет (LOW)

- [x] Добавить custom ESLint правила
- [ ] Добавить pre-commit hooks (husky)
- [ ] Рассмотреть миграцию на app/ структуру (рекомендация Nuxt 4)

## Best Practices

### Component Naming

- PascalCase для компонентов: `DiagnosticCard.vue`
- Компоненты из нескольких слов: `vue/multi-word-component-names`

### Composables

- Префикс `use`: `useAnomalies`, `useRAG`
- Один composable = одна ответственность
- Использовать reactive refs

### TypeScript

- Strict mode включён
- Избегать `any` - использовать `unknown`
- Типы из generated API client

### State Management

- Pinia для глобального state
- Composables для локальной логики
- Nuxt 4 auto-import stores

## Resources

- [Nuxt 4 Documentation](https://nuxt.com)
- [Vue 3 Composition API](https://vuejs.org/guide/extras/composition-api-faq.html)
- [Pinia Documentation](https://pinia.vuejs.org)
- [Tailwind CSS](https://tailwindcss.com)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for detailed changes history.

## Migration Guide

See [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) for Nuxt 3 to Nuxt 4 migration details.
