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
    typeCheck: true
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
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
        { rel: 'preload', as: 'style', href: '/_nuxt/styles/premium-tokens.css' }
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

  // Enhanced Tailwind configuration
  tailwindcss: {
    cssPath: '~/styles/premium-tokens.css',
    configPath: '~/tailwind.config.ts'
  },

  // Theme configuration
  colorMode: {
    preference: 'light',
    fallback: 'light',
    classSuffix: ''
  },

  // Content configuration
  content: {
    documentDriven: false,
    experimental: {
      clientDB: false
    }
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
      devSourcemap: false // Faster CSS in dev
    },
    server: {
      hmr: {
        overlay: false // Less intrusive HMR
      }
    }
  },

  // Runtime configuration
  runtimeConfig: {
    // Private keys
    apiSecret: process.env.NUXT_API_SECRET || 'dev-secret',

    // Public keys
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000/api',
      appName: 'Hydraulic Diagnostic SaaS',
      appVersion: '1.0.0',
      disableFontshare: true // Disable external font provider in dev
    }
  },

  // Route rules for performance
  routeRules: {
    '/': { prerender: true },
    '/auth/**': { ssr: false }, // Client-side auth pages for better UX
    '/dashboard': { ssr: false },
    '/investors': { prerender: true }, // Static investor page
    '/api/**': { cors: true }
  },

  // Enhanced SSR configuration
  ssr: true,

  // Experimental features for Nuxt 4
  experimental: {
    payloadExtraction: false,
    inlineSSRStyles: false,
    viewTransition: true
  }
})
