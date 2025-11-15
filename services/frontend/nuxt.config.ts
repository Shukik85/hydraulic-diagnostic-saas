export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },

  typescript: {
    strict: true,
    typeCheck: true, // ✅ Включена проверка типов для production
    shim: false,
  },

  modules: [
    '@nuxtjs/tailwindcss',
    '@nuxtjs/i18n',
    '@pinia/nuxt',
    '@nuxt/icon',
    '@vueuse/nuxt',
  ],

  css: [
    '~/styles/metallic.css',
  ],

  postcss: {
    plugins: {
      tailwindcss: {},
      autoprefixer: {},
    },
  },

  vite: {
    optimizeDeps: {
      include: [
        'axios',
        'echarts/core',
        'echarts/charts',
        'echarts/components',
        'vue-echarts',
        // TODO: Проверить использование three.js - если не используется, удалить
        // 'three',
        // 'three/examples/jsm/controls/OrbitControls',
      ],
    },
    ssr: {
      noExternal: ['vue-echarts', 'echarts'], // Удалён 'three' - проверить необходимость
    },
  },

  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000/api/v1',
      wsBase: process.env.NUXT_PUBLIC_WS_BASE || 'ws://localhost:8000/ws',
      // ✅ ИСПРАВЛЕНО: моки включены только в dev или через явную переменную
      enableMocks: process.env.ENABLE_MOCKS === 'true' || process.env.NODE_ENV === 'development',
    },
  },

  i18n: {
    locales: [
      { code: 'ru', iso: 'ru-RU', file: 'ru.json', name: 'Русский' },
      { code: 'en', iso: 'en-US', file: 'en.json', name: 'English' },
    ],
    defaultLocale: 'ru',
    strategy: 'no_prefix',
    langDir: 'locales/',
    lazy: true,
    detectBrowserLanguage: {
      useCookie: true,
      cookieKey: 'i18n_locale',
      redirectOn: 'root',
      alwaysRedirect: false,
      fallbackLocale: 'ru',
    },
    vueI18n: './i18n.config.ts',
  },

  app: {
    head: {
      title: 'Hydraulic Diagnostic SaaS',
      charset: 'utf-8',
      viewport: 'width=device-width, initial-scale=1',
      meta: [
        { name: 'description', content: 'AI-powered hydraulic diagnostics' },
        { name: 'theme-color', content: '#2b3340' },
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap' },
      ],
    },
  },

  nitro: {
    compressPublicAssets: true,
    routeRules: {
      '/api/**': { cors: true },
      '/diagnosis/demo': { ssr: false },
      // ✅ ДОБАВЛЕНО: блокировка тестовых страниц в production
      '/api-test': process.env.NODE_ENV === 'production' ? { redirect: '/' } : {},
      '/demo': process.env.NODE_ENV === 'production' ? { redirect: '/' } : {},
    },
  },

  build: {
    transpile: ['tslib'], // Удалён 'three' - проверить необходимость
  },

  devServer: {
    port: 3000,
    host: 'localhost',
  },

  imports: {
    dirs: ['composables/**', 'utils/**', 'types/**', 'stores/**'],
  },

  // ✅ ДОБАВЛЕНО: Nuxt 4 experimental features
  experimental: {
    granularCachedData: true, // Детальное управление кешированием
    purgeCachedData: true,     // Автоматический cleanup данных
  },
})
