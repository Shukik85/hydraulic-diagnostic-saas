# Hydraulic Diagnostic SaaS - Frontend

🚀 **Production-ready Nuxt 4 frontend** with TypeScript strict mode, comprehensive testing, and accessibility-first approach.

## ✨ Features

- **Nuxt 4** - Latest Nuxt with all modern features
- **TypeScript Strict Mode** - Full type safety, no `any` allowed
- **Pinia** - Type-safe state management
- **Tailwind CSS 4** - Utility-first styling
- **VeeValidate + Zod** - Form validation with schemas
- **ECharts** - Interactive data visualizations
- **Vitest** - Unit testing with >95% coverage target
- **Cypress** - E2E testing
- **WCAG 2.1 AA** - Accessibility compliance
- **i18n** - Multi-language support (ru/en)

## 📋 Prerequisites

- Node.js 20.x or 22.x
- npm 10.x+

## 🚀 Quick Start

### Installation

```bash
cd services/frontend
npm install
```

### Development

```bash
# Start dev server (http://localhost:3000)
npm run dev

# Type checking
npm run typecheck

# Linting
npm run lint
npm run lint:fix

# Formatting
npm run format
npm run format:check
```

### Testing

```bash
# Unit tests
npm run test:unit
npm run test:unit:watch
npm run test:unit:coverage

# E2E tests
npm run test:e2e
npm run test:e2e:open
```

### Build

```bash
# Production build
npm run build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
services/frontend/
├── components/          # Vue components (auto-imported)
│   ├── shared/         # Reusable UI components
│   ├── admin/          # Admin-specific components
│   └── diagnosis/      # Diagnosis-specific components
├── composables/        # Composition functions
├── pages/              # File-based routing
├── layouts/            # Layout components
├── stores/             # Pinia stores
├── types/              # TypeScript types
├── utils/              # Utility functions
├── middleware/         # Route middleware
├── plugins/            # Nuxt plugins
├── assets/             # Static assets (CSS, images)
├── public/             # Public files
├── tests/              # Test setup and utilities
└── cypress/            # E2E tests
```

## 🎯 Key Technologies

### Core

- **Nuxt 4.2+** - Vue meta-framework
- **Vue 3.5+** - Progressive JavaScript framework
- **TypeScript 5.9+** - Typed JavaScript
- **Pinia 3.0+** - State management

### UI & Styling

- **Tailwind CSS** - Utility-first CSS
- **Radix Vue** - Unstyled, accessible components
- **ECharts** - Data visualization
- **Heroicons** - Icon set

### Forms & Validation

- **VeeValidate 4** - Form validation
- **Zod** - Schema validation

### Testing

- **Vitest** - Unit testing
- **Cypress** - E2E testing
- **@vue/test-utils** - Vue component testing
- **happy-dom** - Lightweight DOM implementation

### Code Quality

- **ESLint** - Linting
- **Prettier** - Code formatting
- **TypeScript** - Static type checking

## 🔒 Security

- **nuxt-security** - Security headers and CSP
- **Rate limiting** - API request throttling
- **CORS** - Cross-origin resource sharing
- **mTLS** - Mutual TLS for service mesh

## 🌍 Internationalization

- **@nuxtjs/i18n** - i18n support
- Languages: Russian (ru), English (en)
- Locale files: `locales/ru.json`, `locales/en.json`

## 📊 Code Quality Targets

- ✅ **TypeScript**: Strict mode, no `any`
- ✅ **ESLint**: Zero errors
- ✅ **Prettier**: Consistent formatting
- ✅ **Test Coverage**: >95% (lines, statements, functions), >90% (branches)
- ✅ **Lighthouse**: >90 score
- ✅ **WCAG**: 2.1 AA compliance

## 🚢 Deployment

### Production Build

```bash
npm run build
```

### Environment Variables

Create `.env` file based on `.env.example`:

```env
NUXT_PUBLIC_API_BASE=https://api.hydraulic-diagnostics.com/api/v1
NUXT_PUBLIC_WS_BASE=wss://api.hydraulic-diagnostics.com/ws
```

## 📝 Development Guidelines

### TypeScript

- Use strict mode (no `any`)
- Explicit return types for functions
- Consistent type imports: `import type { ... }`

### Vue

- Use `<script setup>` syntax
- Explicit emits with types
- Component naming: PascalCase
- Props validation with TypeScript

### Testing

- Unit tests for all composables, stores, utilities
- E2E tests for critical user flows
- >95% coverage target

### Accessibility

- Semantic HTML
- ARIA attributes where needed
- Keyboard navigation
- Color contrast (WCAG AA)
- Focus management

## 🤝 Contributing

Before committing:

```bash
# Run all checks
npm run typecheck
npm run lint
npm run format:check
npm run test:unit:coverage
```

All checks must pass! ✅

## 📚 Documentation

- [Nuxt 4 Docs](https://nuxt.com/docs)
- [Vue 3 Docs](https://vuejs.org/)
- [Pinia Docs](https://pinia.vuejs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [VeeValidate](https://vee-validate.logaretm.com/v4/)
- [Zod](https://zod.dev/)

## 📧 Support

support@hydraulic-diagnostics.com
