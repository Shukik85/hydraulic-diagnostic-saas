import { defineConfig } from 'vitest/config';
import vue from '@vitejs/plugin-vue';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  plugins: [vue()],
  
  test: {
    // Environment
    environment: 'jsdom',
    globals: true,
    
    // Setup files
    setupFiles: ['./tests/setup.ts'],
    
    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        '.nuxt/',
        '.output/',
        'dist/',
        'coverage/',
        '**/*.config.*',
        '**/*.d.ts',
        '**/tests/**',
        '**/__tests__/**',
        '**/test/**',
      ],
      thresholds: {
        lines: 95,
        statements: 95,
        functions: 95,
        branches: 90,
      },
    },
    
    // Test files
    include: [
      'components/**/*.{test,spec}.{js,ts}',
      'composables/**/*.{test,spec}.{js,ts}',
      'stores/**/*.{test,spec}.{js,ts}',
      'utils/**/*.{test,spec}.{js,ts}',
      'types/**/*.{test,spec}.{js,ts}',
    ],
    
    // Exclude
    exclude: [
      'node_modules',
      '.nuxt',
      '.output',
      'dist',
      'cypress',
    ],
    
    // Reporters
    reporters: ['verbose'],
    
    // Mock settings
    mockReset: true,
    restoreMocks: true,
    clearMocks: true,
    
    // Timeout
    testTimeout: 10000,
  },
  
  resolve: {
    alias: {
      '~': fileURLToPath(new URL('./', import.meta.url)),
      '@': fileURLToPath(new URL('./', import.meta.url)),
    },
  },
});
