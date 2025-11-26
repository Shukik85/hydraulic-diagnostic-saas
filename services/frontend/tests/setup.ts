// Vitest global setup
import { vi } from 'vitest';
import { config } from '@vue/test-utils';

// Mock Nuxt auto-imports
global.defineNuxtComponent = vi.fn();
global.definePageMeta = vi.fn();
global.navigateTo = vi.fn();
global.useRuntimeConfig = vi.fn(() => ({
  public: {
    apiBase: 'http://localhost:8000',
  },
}));

// Mock composables
global.useRoute = vi.fn(() => ({
  params: {},
  query: {},
  path: '/',
}));

global.useRouter = vi.fn(() => ({
  push: vi.fn(),
  replace: vi.fn(),
  back: vi.fn(),
}));

global.useI18n = vi.fn(() => ({
  t: (key: string) => key,
  locale: { value: 'ru' },
}));

// Configure Vue Test Utils
config.global.mocks = {
  $t: (key: string) => key,
};

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  takeRecords() {
    return [];
  }
  unobserve() {}
};

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
};
