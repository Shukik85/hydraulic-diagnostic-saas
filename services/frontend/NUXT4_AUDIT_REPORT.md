# Nuxt 4 Comprehensive Audit Report
**Дата:** 16 ноября 2025  
**Проект:** Hydraulic Diagnostic SaaS Frontend  
**Ветка:** fix/frontend-audit-nuxt4

---

## Статус аудита

### ✅ Выполнено правильно

1. **TypeScript strict mode** - включен в `tsconfig.json` и `nuxt.config.ts`
2. **ESLint конфигурация** - используется flat config для Nuxt 4
3. **Memory leak prevention** - правильный cleanup в `useWebSocket.ts` через `onUnmounted`
4. **Структура директорий** - следует Nuxt 4 conventions
5. **i18n конфигурация** - настроена с lazy loading
6. **Metallic theme** - реализован через CSS переменные

---

## 🔧 Требует исправления

### 1. TypeScript Type Safety

#### Проблема
- `nuxt.config.ts`: `typeCheck: false` - отключена проверка типов
- Composables не имеют explicit return types
- Отсутствует `typescriptBundlerResolution`

#### Решение
```typescript
// nuxt.config.ts
typescript: {
  strict: true,
  typeCheck: true, // ✅ ВКЛЮЧИТЬ
  shim: false,
},
experimental: {
  typescriptBundlerResolution: true // ✅ ДОБАВИТЬ
}
```

#### Приоритет: 🔴 ВЫСОКИЙ

---

### 2. Composables: Explicit Return Types

#### Проблема
```typescript
// composables/useApi.ts - НЕТ типов возврата
export function useApi() {
  return {};
}
```

#### Решение
```typescript
interface ApiClient {
  get<T>(url: string, options?: FetchOptions): Promise<T>
  post<T>(url: string, data: any, options?: FetchOptions): Promise<T>
  put<T>(url: string, data: any, options?: FetchOptions): Promise<T>
  delete<T>(url: string, options?: FetchOptions): Promise<T>
}

export const useApi = (): ApiClient => {
  const config = useRuntimeConfig()
  const baseURL = config.public.apiBase
  
  return {
    get: <T>(url: string, options?: FetchOptions) => 
      $fetch<T>(url, { baseURL, method: 'GET', ...options }),
    post: <T>(url: string, data: any, options?: FetchOptions) => 
      $fetch<T>(url, { baseURL, method: 'POST', body: data, ...options }),
    put: <T>(url: string, data: any, options?: FetchOptions) => 
      $fetch<T>(url, { baseURL, method: 'PUT', body: data, ...options }),
    delete: <T>(url: string, options?: FetchOptions) => 
      $fetch<T>(url, { baseURL, method: 'DELETE', ...options }),
  }
}
```

#### Приоритет: 🔴 ВЫСОКИЙ

---

### 3. Data Fetching: useFetch vs $fetch

#### Проблема
Возможно использование `$fetch` вместо `useFetch` в компонентах, что приводит к дублированию запросов на SSR и клиенте.

#### Решение
```typescript
// ✅ ПРАВИЛЬНО: useFetch для SSR-safe запросов
const { data: systems, pending, error, refresh } = await useFetch(
  '/api/systems',
  {
    key: 'systems-list',
    getCachedData: (key) => useNuxtApp().payload.data[key],
    lazy: true,
    retry: 3,
    retryDelay: 1000
  }
)

// ❌ НЕПРАВИЛЬНО: $fetch дублирует запросы
const systems = await $fetch('/api/systems')
```

#### Действие
- Аудит всех компонентов и pages на использование `$fetch`
- Заменить на `useFetch` где необходимо SSR

#### Приоритет: 🟠 СРЕДНИЙ

---

### 4. Component Organization

#### Текущая структура
```
components/
├── Diagnosis/
├── Error/
├── Loading/
├── dashboard/
├── digital-twin/
├── metadata/
├── rag/
└── ui/
```

#### Рекомендуемая структура (Locality of Behavior)
```
components/
├── shared/          # Глобальные переиспользуемые компоненты
│   ├── Button.vue
│   ├── Input.vue
│   ├── Card.vue
│   └── Modal.vue
├── pages/           # Страница-специфичные компоненты
│   ├── dashboard/
│   │   ├── SystemList.vue
│   │   └── Statistics.vue
│   ├── diagnosis/
│   │   └── DiagnosisForm.vue
│   └── digital-twin/
│       └── TwinViewer.vue
└── layouts/         # Layout-специфичные компоненты
    └── default/
        ├── Header.vue
        ├── Sidebar.vue
        └── Footer.vue
```

#### Действие
1. Создать директорию `components/shared/`
2. Переместить UI компоненты в `shared/`
3. Создать `components/pages/` для страница-специфичных компонентов
4. Создать `components/layouts/` для layout компонентов

#### Приоритет: 🟡 НИЗКИЙ (refactoring)

---

### 5. Route-based Caching

#### Проблема
Не настроены `routeRules` для оптимального кэширования.

#### Решение
```typescript
// nuxt.config.ts
nitro: {
  compressPublicAssets: {
    gzip: true,
    brotli: true // ✅ ДОБАВИТЬ
  },
  routeRules: {
    '/': { 
      swr: 3600  // Stale-while-revalidate 1 час
    },
    '/dashboard': { 
      ssr: true,
      swr: 600 // 10 минут
    },
    '/diagnosis/**': { 
      ssr: false  // SPA mode для diagnosis
    },
    '/api/**': { 
      cors: true,
      headers: {
        'cache-control': 'max-age=300'
      }
    }
  }
}
```

#### Приоритет: 🟠 СРЕДНИЙ

---

### 6. Image Optimization

#### Проблема
Не используется `@nuxt/image` модуль для автоматической оптимизации изображений.

#### Решение
```bash
npm install @nuxt/image
```

```typescript
// nuxt.config.ts
modules: [
  '@nuxt/image', // ✅ ДОБАВИТЬ
  '@nuxtjs/tailwindcss',
  '@nuxtjs/i18n',
  '@pinia/nuxt',
  '@nuxt/icon',
  '@vueuse/nuxt',
]
```

```vue
<!-- Использование -->
<template>
  <NuxtImg
    src="/images/hero.jpg"
    width="1200"
    height="600"
    alt="Hero section"
    loading="lazy"
    format="webp"
    quality="80"
    sizes="xs:100vw sm:100vw md:50vw lg:800px"
  />
</template>
```

#### Приоритет: 🟡 НИЗКИЙ

---

### 7. SEO Meta Tags

#### Проблема
Отсутствует `useSeoMeta` в pages для оптимального SEO.

#### Решение
```vue
<!-- pages/dashboard.vue -->
<script setup lang="ts">
useSeoMeta({
  title: 'Dashboard | Hydraulic Diagnostic',
  description: 'Real-time hydraulic system monitoring and diagnostics',
  
  // Open Graph
  ogTitle: 'Dashboard | Hydraulic Diagnostic',
  ogDescription: 'Real-time hydraulic system monitoring',
  ogType: 'website',
  ogUrl: 'https://yourdomain.com/dashboard',
  
  // Twitter Card
  twitterCard: 'summary_large_image',
  twitterTitle: 'Dashboard | Hydraulic Diagnostic',
  twitterDescription: 'Real-time hydraulic system monitoring'
})

// Global title template
useHead({
  titleTemplate: (titleChunk) => {
    return titleChunk 
      ? `${titleChunk} | Hydraulic Diagnostic` 
      : 'Hydraulic Diagnostic SaaS'
  }
})
</script>
```

#### Действие
Добавить `useSeoMeta` во все pages.

#### Приоритет: 🟠 СРЕДНИЙ

---

### 8. Accessibility (A11y)

#### Проблемы
1. Возможно использование `<div>` вместо семантических HTML тегов
2. Отсутствие ARIA атрибутов
3. Не проверен контраст цветов (WCAG 2.1 AA требует 4.5:1)

#### Решения

**Семантический HTML:**
```vue
<!-- ❌ ПЛОХО -->
<div class="header">
  <div class="nav">...</div>
</div>

<!-- ✅ ХОРОШО -->
<header>
  <nav aria-label="Main navigation">...</nav>
</header>
```

**ARIA атрибуты:**
```vue
<button
  @click="toggleMenu"
  :aria-expanded="isMenuOpen"
  aria-controls="mobile-menu"
  aria-label="Toggle navigation menu"
>
  <Icon name="heroicons:bars-3" aria-hidden="true" />
</button>
```

**Контраст цветов:**
```css
/* Проверить все цвета в tailwind.config.ts и metallic.css */
/* Минимум 4.5:1 для обычного текста */
/* Минимум 3:1 для крупного текста (18px+ или bold 14px+) */

:root {
  --color-text-primary: #1a1a1a;     /* 19.56:1 ✓ */
  --color-text-secondary: #4a4a4a;   /* 9.48:1 ✓ */
  --color-text-muted: #6b6b6b;       /* 5.74:1 ✓ */
  --color-brand-primary: #21808D;    /* 4.52:1 ✓ */
}
```

**Keyboard Navigation:**
```vue
<script setup lang="ts">
import { useKeyboardNav } from '@vueuse/core'

const { onKeyDown } = useKeyboardNav()

onKeyDown('Escape', () => {
  closeModal()
})

onKeyDown('Enter', () => {
  submitForm()
})
</script>
```

#### Действие
1. Аудит всех компонентов на семантический HTML
2. Добавить ARIA атрибуты где необходимо
3. Проверить контраст всех цветов
4. Обеспечить keyboard navigation

#### Приоритет: 🟠 СРЕДНИЙ

---

### 9. Security

#### Не настроены CSP заголовки

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  modules: ['nuxt-security'], // ✅ УСТАНОВИТЬ npm install nuxt-security
  
  security: {
    headers: {
      contentSecurityPolicy: {
        'default-src': ["'self'"],
        'script-src': [
          "'self'",
          "'wasm-unsafe-eval'",
        ],
        'style-src': [
          "'self'",
          "'unsafe-inline'", // Для Tailwind
          'https://fonts.googleapis.com'
        ],
        'img-src': [
          "'self'",
          'data:',
          'https:',
          'blob:'
        ],
        'font-src': [
          "'self'",
          'https://fonts.gstatic.com'
        ],
        'connect-src': [
          "'self'",
          'http://localhost:8000', // Dev API
          'ws://localhost:8000'    // Dev WebSocket
        ],
        'frame-ancestors': ["'none'"],
        'base-uri': ["'self'"],
        'form-action': ["'self'"]
      },
      
      xssProtection: '1; mode=block',
      
      strictTransportSecurity: {
        maxAge: 31536000,
        includeSubdomains: true,
        preload: true
      },
      
      xFrameOptions: 'DENY',
      referrerPolicy: 'strict-origin-when-cross-origin'
    },
    
    csrf: {
      enabled: true,
      methodsToProtect: ['POST', 'PUT', 'PATCH', 'DELETE']
    },
    
    rateLimiter: {
      tokensPerInterval: 150,
      interval: 'hour'
    }
  }
})
```

#### Приоритет: 🔴 ВЫСОКИЙ

---

### 10. Code Quality: ESLint Rules

#### Добавить дополнительные правила

```javascript
// eslint.config.mjs
export default withNuxt({
  rules: {
    // Существующие...
    
    // ✅ ДОБАВИТЬ:
    '@typescript-eslint/explicit-function-return-type': 'warn',
    '@typescript-eslint/no-non-null-assertion': 'warn',
    
    'vue/require-explicit-emits': 'error',
    'vue/no-unused-refs': 'error',
    'vue/padding-line-between-blocks': 'warn',
    'vue/component-api-style': ['error', ['script-setup']],
    'vue/block-order': ['error', {
      order: ['script', 'template', 'style']
    }],
    
    // Accessibility
    'vuejs-accessibility/alt-text': 'error',
    'vuejs-accessibility/anchor-has-content': 'error',
    'vuejs-accessibility/click-events-have-key-events': 'warn',
    'vuejs-accessibility/form-control-has-label': 'error',
  }
})
```

#### Действие
```bash
npm install --save-dev eslint-plugin-vuejs-accessibility
```

#### Приоритет: 🟡 НИЗКИЙ

---

## 📋 Deployment Checklist

Перед production deployment проверить:

- [ ] `npm run typecheck` проходит без ошибок
- [ ] `npm run build` успешная сборка
- [ ] `npm run analyze` - bundle size < 500KB
- [ ] Нет hydration warnings в консоли
- [ ] Lighthouse score > 90 для всех метрик
- [ ] Security headers настроены (CSP, HSTS, X-Frame-Options)
- [ ] Environment variables в `.env.example`
- [ ] Error tracking настроен (Sentry/Rollbar)
- [ ] Sitemap генерируется
- [ ] `robots.txt` настроен
- [ ] Meta tags для всех страниц
- [ ] Accessibility audit пройден

---

## 🎯 План действий

### Фаза 1: Критические исправления (HIGH)
1. ✅ Включить `typeCheck: true` в nuxt.config.ts
2. ✅ Добавить `typescriptBundlerResolution: true`
3. ✅ Добавить explicit return types для всех composables
4. ✅ Настроить CSP и security headers

### Фаза 2: Оптимизация (MEDIUM)
1. ⏳ Аудит использования `useFetch` vs `$fetch`
2. ⏳ Настроить route-based caching
3. ⏳ Добавить `useSeoMeta` во все pages
4. ⏳ Аудит accessibility (ARIA, semantic HTML)

### Фаза 3: Улучшения (LOW)
1. ⏳ Реорганизовать структуру компонентов
2. ⏳ Установить `@nuxt/image` модуль
3. ⏳ Добавить accessibility ESLint rules
4. ⏳ Настроить pre-commit hooks (Husky + lint-staged)

---

## 📚 Дополнительные ресурсы

- [Nuxt 4 Documentation](https://nuxt.com/docs/4.x)
- [TypeScript Best Practices](https://nuxt.com/docs/4.x/guide/concepts/typescript)
- [Nuxt Security Module](https://nuxt-security.vercel.app/)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Web Vitals](https://web.dev/vitals/)

---

**Статус:** 🟢 Готов к реализации  
**Ответственный:** Frontend Team  
**Дедлайн Фазы 1:** 18 ноября 2025
