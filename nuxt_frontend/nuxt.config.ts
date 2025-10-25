import { defineNuxtConfig } from 'nuxt/config'

export default defineNuxtConfig({
  compatibilityDate: '2025-10-25',
  devtools: { enabled: false },
  nitro: { preset: 'node', compressPublicAssets: true, minify: true },
  typescript: { strict: true, typeCheck: false },
  modules: ['@nuxt/icon', '@pinia/nuxt', '@nuxtjs/color-mode', '@nuxt/content'],
  css: ['~/styles/premium-tokens.css'],
  postcss: {
    plugins: {
      '@tailwindcss/postcss': {},
      autoprefixer: {}
    }
  },
  app: {
    head: {
      charset: 'utf-8',
      viewport: 'width=device-width, initial-scale=1',
      htmlAttrs: { lang: 'ru' },
      meta: [{ name: 'theme-color', content: '#0B1221' }]
    },
    pageTransition: { name: 'page', mode: 'out-in' },
    layoutTransition: { name: 'layout', mode: 'out-in' }
  },
  colorMode: { preference: 'light', fallback: 'light', classSuffix: '' },
  content: { documentDriven: false },
  build: { transpile: ['@headlessui/vue', 'echarts', 'vue-echarts'] },
  vite: {
    css: { devSourcemap: false },
    server: { hmr: { overlay: false } },
    optimizeDeps: { include: ['echarts', 'vue-echarts'] }
  },
  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000/api',
      appName: 'Гидравлика ИИ',
      appVersion: '1.0.0'
    }
  },
  routeRules: {
    '/': { prerender: true },
    '/**': { ssr: true }
  }
})