/**
 * Vitest test setup
 */

import { vi } from 'vitest';
import { config } from '@vue/test-utils';

// Mock Nuxt auto-imports
global.defineNuxtConfig = vi.fn();
global.useRuntimeConfig = vi.fn(() => ({
  public: {
    apiBase: 'http://localhost:8000/api/v1',
    wsBase: 'ws://localhost:8000/ws',
  },
}));

global.navigateTo = vi.fn();
global.useRouter = vi.fn(() => ({
  push: vi.fn(),
  replace: vi.fn(),
  go: vi.fn(),
  back: vi.fn(),
}));

global.useRoute = vi.fn(() => ({
  path: '/',
  params: {},
  query: {},
}));

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(global, 'localStorage', {
  value: localStorageMock,
});

// Configure Vue Test Utils
config.global.stubs = {
  Icon: true,
  NuxtLink: true,
};
