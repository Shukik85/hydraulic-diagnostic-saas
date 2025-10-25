import { defineNuxtConfig } from 'nuxt/config'

// Production-ready configuration for investor demo
export default defineNuxtConfig({
  // Enhanced devtools for development
  devtools: { enabled: true },

  // Performance optimizations
  nitro: {
    preset: 'node',
    compressPublicAssets: true,
    minify: true
  },

  // Future-ready Nuxt 4 compatibility
  future: {
    compatibilityVersion: 4
  },

  // Enhanced TypeScript support
  typescript: {
    strict: true,
    typeCheck: false // Disable for faster dev builds
  },

  // Core modules for enterprise SaaS
  modules: [
    '@nuxt/icon',
    '@pinia/nuxt',
    '@nuxtjs/color-mode',
    '@nuxt/content'
  ],

  // Premium styling system
  css: [
    '~/styles/premium-tokens.css'
  ],

  // Optimized app configuration
  app: {
    head: {
      charset: 'utf-8',
      viewport: 'width=device-width, initial-scale=1',
      htmlAttrs: {
        lang: 'ru'
      },
      meta: [
        { name: 'theme-color', content: '#2563eb' },
        { name: 'msapplication-TileColor', content: '#2563eb' }
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }
      ]
    },
    pageTransition: { name: 'page', mode: 'out-in' },
    layoutTransition: { name: 'layout', mode: 'out-in' }
  },

  // Component auto-import for clean code
  components: [
    {
      path: '~/components',
      pathPrefix: false
    }
  ],

  // Fast icon configuration (local-first)
  icon: {
    mode: 'local',
    provider: 'iconify',
    collections: ['heroicons'],
    serverBundle: {
      collections: ['heroicons']
    }
  },

  // Theme configuration
  colorMode: {
    preference: 'light',
    fallback: 'light',
    classSuffix: ''
  },

  // Content configuration (removed deprecated options)
  content: {
    documentDriven: false
  },

  // Production build optimizations
  build: {
    transpile: ['@headlessui/vue']
  },

  // Development performance
  vite: {
    define: {
      __DEV__: process.env.NODE_ENV !== 'production'
    },
    css: {
      devSourcemap: false
    },
    server: {
      hmr: {
        overlay: false
      }
    }
  },

  // Runtime configuration
  runtimeConfig: {
    apiSecret: process.env.NUXT_API_SECRET || 'dev-secret',

    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000/api',
      appName: 'Hydraulic Diagnostic SaaS',
      appVersion: '1.0.0',
      disableFontshare: true
    }
  },

  // Route rules for performance
  routeRules: {
    '/': { prerender: true },
    '/auth/**': { ssr: false },
    '/dashboard': { ssr: false },
    '/investors': { prerender: true },
    '/api/**': { cors: true }
  },

  ssr: true,

  // Removed deprecated experimental options
  experimental: {
    payloadExtraction: false,
    viewTransition: true
  }
})
