import { defineNuxtConfig } from 'nuxt/config'

export default defineNuxtConfig({
  // Development
  devtools: { enabled: true },
  
  // Core modules
  modules: [
    '@pinia/nuxt',
    '@nuxtjs/tailwindcss',
    '@nuxt/eslint',
    '@nuxt/image',
    '@nuxt/icon',
    '@nuxt/fonts',
    '@nuxtjs/seo'
  ],

  // CSS
  css: ['~/assets/css/globals.css'],

  // TypeScript
  typescript: {
    strict: true,
    typeCheck: false // Use vue-tsc in CI for better performance
  },

  // ESLint integration
  eslint: {
    checker: {
      lintOnStart: false, // Use CI for strict checking
      configType: 'flat'
    }
  },

  // SEO and meta
  site: {
    url: process.env.NUXT_PUBLIC_SITE_URL || 'http://localhost:3000',
    name: 'Hydraulic Diagnostic SaaS',
    description: 'Professional hydraulic system diagnostics and monitoring platform'
  },

  app: {
    head: {
      title: 'Hydraulic Diagnostic SaaS',
      htmlAttrs: {
        lang: 'ru'
      },
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'format-detection', content: 'telephone=no' },
        { name: 'theme-color', content: '#ffffff' }
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }
      ]
    }
  },

  // Image optimization
  image: {
    formats: ['webp', 'avif', 'png', 'jpg'],
    quality: 80,
    densities: [1, 2],
    screens: {
      xs: 320,
      sm: 640,
      md: 768,
      lg: 1024,
      xl: 1280,
      xxl: 1536
    }
  },

  // Runtime config for API integration
  runtimeConfig: {
    // Private keys (only available on server-side)
    apiSecret: '',
    
    // Public keys (exposed to client-side)
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000/api',
      siteUrl: process.env.NUXT_PUBLIC_SITE_URL || 'http://localhost:3000'
    }
  },

  // Server-side rendering
  ssr: true,
  
  // Nitro server configuration
  nitro: {
    preset: 'node-server',
    compressPublicAssets: true
  },

  // Build optimization
  build: {
    transpile: []
  },

  // Auto imports
  imports: {
    autoImport: true
  },

  // Tailwind CSS
  tailwindcss: {
    cssPath: '~/assets/css/globals.css',
    configPath: 'tailwind.config.js'
  },

  // Development server
  devServer: {
    port: 3000
  },

  // Route rules for caching (production optimization)
  routeRules: {
    '/': { prerender: true },
    '/api/**': { cors: true },
    '/admin/**': { index: false }
  },

  // Fonts optimization
  fonts: {
    defaults: {
      weights: [400, 500, 600, 700],
      styles: ['normal'],
      subsets: ['latin', 'cyrillic']
    }
  },

  // Security headers
  security: {
    headers: {
      crossOriginEmbedderPolicy: process.env.NODE_ENV === 'development' ? 'unsafe-none' : 'require-corp'
    }
  },

  // Experimental features
  experimental: {
    payloadExtraction: false
  },

  // Compatibility
  compatibilityDate: '2024-11-01'
})