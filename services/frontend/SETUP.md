# Frontend Setup Guide

## Prerequisites

Ensure you have the following installed:

- **Node.js**: 20.x or higher
- **npm**: 10.x or higher

Verify versions:

```bash
node --version  # Should be v20.x.x or higher
npm --version   # Should be 10.x.x or higher
```

## Installation

### 1. Install Dependencies

```bash
cd services/frontend
npm install
```

This will:
- Install all dependencies from `package.json`
- Generate `package-lock.json` for reproducible builds
- Create `.nuxt` directory with auto-generated TypeScript types

### 2. Verify Installation

Run these commands to verify everything is configured correctly:

```bash
# TypeScript type checking
npm run typecheck

# ESLint validation
npm run lint

# Prettier format check
npm run format:check
```

### 3. Development Server

Start the development server:

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Configuration Files

### Core Config

- **`nuxt.config.ts`**: Nuxt 4 configuration (modules, i18n, security, runtime)
- **`tailwind.config.ts`**: Tailwind CSS design system tokens
- **`tsconfig.json`**: TypeScript compiler options with path mappings
- **`eslint.config.js`**: ESLint flat config for TypeScript + Vue 3
- **`vitest.config.ts`**: Vitest unit test configuration
- **`cypress.config.ts`**: Cypress E2E test configuration

### CSS/Design System

- **`assets/css/main.css`**: Main CSS entry point
- **`assets/css/premium-tokens.css`**: Design system tokens (colors, spacing, typography)

### i18n

- **`i18n.config.ts`**: Vue I18n configuration
- **`i18n/locales/ru.json`**: Russian translations
- **`i18n/locales/en.json`**: English translations

## Build Commands

```bash
# Production build
npm run build

# Generate static site
npm run generate

# Preview production build
npm run preview
```

## Testing

```bash
# Unit tests
npm run test:unit

# Unit tests with coverage
npm run test:unit:coverage

# E2E tests
npm run test:e2e

# E2E tests interactive mode
npm run test:e2e:open
```

## Troubleshooting

### Issue: `.nuxt` directory not found

**Solution**: Run `npm run postinstall` to generate Nuxt type definitions:

```bash
npm run postinstall
```

### Issue: TypeScript errors about missing types

**Solution**: Ensure all `@types/*` packages are installed:

```bash
npm install --save-dev @types/node
```

### Issue: Tailwind CSS not applying styles

**Solution**: Verify these files exist:
- `tailwind.config.ts`
- `assets/css/main.css`
- `assets/css/premium-tokens.css`

And check `nuxt.config.ts` includes:

```typescript
modules: [
  '@nuxtjs/tailwindcss',
  // ...
],
css: ['~/assets/css/main.css'],
```

### Issue: i18n translations not loading

**Solution**: Verify `i18n.config.ts` has correct `langDir`:

```typescript
i18n: {
  langDir: 'locales',  // Not 'i18n/locales'
  // ...
}
```

### Issue: ESLint errors on valid code

**Solution**: Check `tsconfig.json` has correct `project` path in ESLint config:

```javascript
parserOptions: {
  project: './tsconfig.json',
}
```

## Environment Variables

Create `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Configure:

```bash
NUXT_PUBLIC_API_BASE=http://localhost:8000/api/v1
NUXT_PUBLIC_WS_BASE=ws://localhost:8000/ws
```

## CI/CD Verification

Run the full CI pipeline locally:

```bash
# Lint + Type check + Format check + Build
npm run typecheck && \
npm run lint && \
npm run format:check && \
npm run build
```

All commands should pass with exit code 0.

## Next Steps

1. ✅ Dependencies installed
2. ✅ Configuration verified
3. ✅ Development server running
4. 🚀 Start developing!

Refer to [README.md](./README.md) for architecture and development guidelines.
