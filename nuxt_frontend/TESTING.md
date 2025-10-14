# Тестирование фронтенда (nuxt_frontend)

Этот документ описывает настройку и запуск unit-тестов для Nuxt фронтенда с использованием Vitest и Vue Test Utils.

## Стек

- Vitest — фреймворк для unit-тестирования
- Vue Test Utils — утилиты для тестирования Vue 3
- jsdom — окружение браузера для тестов (headless)

## Установка

Зависимости уже прописаны в `nuxt_frontend/package.json`. Если нужно установить вручную:

```bash
npm install -D vitest @vitest/ui @vitest/coverage-v8 @vitejs/plugin-vue @vue/test-utils jsdom
```

## Конфигурация

Файл `nuxt_frontend/vitest.config.js`:

```js
import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
  plugins: [vue()],
  test: {
    globals: true,
    environment: 'jsdom',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/**',
        'tests/**',
        '*.config.js',
        '.nuxt/**',
        'dist/**'
      ]
    },
    include: ['tests/**/*.test.js']
  },
  resolve: {
    alias: {
      '~': path.resolve(__dirname, './'),
      '@': path.resolve(__dirname, './')
    }
  }
})
```

## Скрипты

В `package.json` доступны скрипты:

```json
{
  "scripts": {
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage"
  }
}
```

## Структура тестов

```
nuxt_frontend/
└── tests/
    └── composables/
        └── useSystems.test.js
```

## Пример: тест для composable useSystems

```js
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useSystems } from '../../composables/useSystems'

global.fetch = vi.fn()

describe('useSystems', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    fetch.mockClear()
  })

  it('fetches systems successfully', async () => {
    const mockSystems = [
      { id: 1, name: 'System 1', status: 'active' },
      { id: 2, name: 'System 2', status: 'inactive' }
    ]

    fetch.mockResolvedValueOnce({ ok: true, json: async () => mockSystems })

    const { systems, loading, error, fetchSystems } = useSystems()
    await fetchSystems()

    expect(systems.value).toEqual(mockSystems)
    expect(loading.value).toBe(false)
    expect(error.value).toBeNull()
  })
})
```

## Запуск тестов

```bash
# все тесты
npm test

# с UI
npm run test:ui

# с coverage
npm run test:coverage
```

## Coverage

После выполнения `npm run test:coverage` будут доступны:

- HTML отчет: `nuxt_frontend/coverage/index.html`
- JSON отчет: `nuxt_frontend/coverage/coverage-final.json`
- Текстовый summary в консоли

## Советы

- Для сетевых запросов мокайте `fetch`/`$fetch`.
- Если composable использует Nuxt runtimeConfig, мокайте через `vi.stubGlobal` или предоставляйте значения через обертки.
- Избегайте обращения к реальному DOM; полагайтесь на jsdom.
- Держите тесты независимыми: очищайте моки через `vi.clearAllMocks()` в `beforeEach`.
