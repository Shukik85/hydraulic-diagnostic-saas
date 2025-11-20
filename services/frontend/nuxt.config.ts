export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },

  typescript: {
    strict: true,
    // ✅ Отключаем typeCheck в dev mode (конфликт vite-plugin-checker)
    // Используем npm run typecheck вручную или в CI
    typeCheck: false,
    shim: false,
  },

  modules: [
    '@nuxtjs/tailwindcss',
    '@nuxtjs/i18n',
    '@pinia/nuxt',
    '@nuxt/icon',
    '@vueuse/nuxt',
    'nuxt-security',
  ],

  security: {
    headers: {
      crossOriginResourcePolicy: 'same-origin',
      referrerPolicy: 'strict-origin-when-cross-origin',
      xContentTypeOptions: 'nosniff',
      xFrameOptions: 'SAMEORIGIN',
      strictTransportSecurity: { maxAge: 31536000, includeSubdomains: true, preload: true } as any,
      xXSSProtection: '1; mode=block',
    },
    rateLimiter: {
      tokensPerInterval: 100,
      interval: 'minute'
    },
  },

  css: [
    '~/styles/accessibility.css',
    '~/styles/metallic.css',
    '~/styles/premium-tokens.css',
    '~/styles/components.css',
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
      ],
    },
    ssr: {
      noExternal: ['vue-echarts', 'echarts'],
    },
  },

  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000/api/v1',
      wsBase: process.env.NUXT_PUBLIC_WS_BASE || 'ws://localhost:8000/ws',
      enableMocks: process.env.ENABLE_MOCKS === 'true' || process.env.NODE_ENV === 'development',
      features: {
        enableMockData: true,
        ragInterpretation: true,
      }
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
    compressPublicAssets: {
      gzip: true,
      brotli: true,
    },
    routeRules: {
      '/': {
        swr: 3600,
      },
      '/dashboard': {
        ssr: true,
        swr: 600,
      },
      '/diagnosis/**': {
        ssr: false
      },
      '/api/**': {
        cors: true,
        headers: {
          'cache-control': 'max-age=300'
        }
      },
      '/api-test': process.env.NODE_ENV === 'production' ? { redirect: '/' } : {},
      '/demo': process.env.NODE_ENV === 'production' ? { redirect: '/' } : {},
    },
  },

  build: {
    transpile: ['tslib'],
  },

  devServer: {
    port: 3000,
    host: 'localhost',
  },

  experimental: {
    granularCachedData: true,
    purgeCachedData: true,
  },
})
