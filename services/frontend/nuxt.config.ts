// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  ssr: false, // SPA режим для дэшборда

  compatibilityDate: '2025-10-30', // Убираем warning

  devtools: { enabled: true },

  modules: [
    '@pinia/nuxt',
    '@nuxtjs/i18n',
    '@nuxt/icon',
    '@nuxt/image',
    '@nuxtjs/color-mode'
  ],

  i18n: {
    locales: [
      {
        code: 'ru',
        name: 'Русский',
        file: 'ru.json',
        language: 'ru-RU'
      },
      {
        code: 'en',
        name: 'English',
        file: 'en.json',
        language: 'en-US'
      }
    ],
    defaultLocale: 'ru',
    strategy: 'no_prefix',
    langDir: 'locales',
    detectBrowserLanguage: {
      useCookie: true,
      cookieKey: 'i18n_redirected',
      redirectOn: 'root',
      fallbackLocale: 'ru'
    },
    // Для @nuxtjs/i18n v10 - отдельная конфигурация
    vueI18n: './i18n.config.ts'
  },

  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000/api',
      wsBase: process.env.NUXT_PUBLIC_WS_BASE || 'ws://localhost:8000/ws',
      version: '1.0.0'
    }
  },

  css: [
    '~/styles/premium-tokens.css'
  ],

  postcss: {
    plugins: {
      tailwindcss: {},
      autoprefixer: {}
    }
  },

  // Единый корневой путь компонентов, рекурсивное сканирование подпапок
  components: [
    { path: '~/components', global: true, pathPrefix: false },
    { path: '~/components/ui', global: true, pathPrefix: false }
  ],

  app: {
    head: {
      title: 'Гидравлик Диагностик - Промышленные Решения',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: 'Интеллектуальная платформа диагностики гидравлических систем с ИИ-анализом' }
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }
      ]
    }
  },

  // Explicit component transpilation for better compatibility
  build: {
    transpile: []
  },

  // Client-side rendering optimization
  experimental: {
    payloadExtraction: false
  }
})
