import { defineVitestConfig } from '@nuxt/test-utils/config';
import { fileURLToPath } from 'node:url';

export default defineVitestConfig({
  test: {
    environment: 'happy-dom',
    globals: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      lines: 95,
      statements: 95,
      functions: 95,
      branches: 90,
      exclude: [
        'node_modules/',
        '.nuxt/',
        'dist/',
        '**/*.spec.ts',
        '**/*.test.ts',
        '**/types/**',
        '**/*.config.{ts,js}',
      ],
    },
    include: ['**/__tests__/**/*.spec.ts', '**/tests/**/*.spec.ts'],
    setupFiles: ['./tests/setup.ts'],
  },
  resolve: {
    alias: {
      '~': fileURLToPath(new URL('./', import.meta.url)),
      '@': fileURLToPath(new URL('./', import.meta.url)),
    },
  },
});
