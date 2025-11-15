// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },

  // ==========================================
  // MODULES
  // ==========================================
  modules: [
    '@nuxtjs/i18n',
    '@nuxt/eslint',
    '@pinia/nuxt',
    '@vueuse/nuxt',
  ],

  // ==========================================
  // i18n CONFIGURATION
  // ==========================================
  i18n: {
    locales: [
      { code: 'ru', file: 'ru.json', name: 'Русский' },
      { code: 'en', file: 'en.json', name: 'English' },
    ],
    defaultLocale: process.env.NUXT_PUBLIC_DEFAULT_LOCALE || 'ru',
    langDir: 'locales/',
    strategy: 'no_prefix',
    detectBrowserLanguage: {
      useCookie: true,
      cookieKey: 'i18n_redirected',
      redirectOn: 'root',
    },
  },

  // ==========================================
  // TYPESCRIPT CONFIGURATION
  // ==========================================
  typescript: {
    strict: true,
    typeCheck: false,
    shim: false,
  },

  // ==========================================
  // VITE CONFIGURATION
  // ==========================================
  vite: {
    vue: {
      script: {
        defineModel: true,
        propsDestructure: true,
      },
    },

    build: {
      chunkSizeWarningLimit: 1000,
      rollupOptions: {
        output: {
          manualChunks: {
            'echarts': ['echarts', 'vue-echarts'],
            'three': ['three'],
            'ui-components': ['class-variance-authority'],
          },
        },
      },
    },

    optimizeDeps: {
      include: [
        'echarts/core',
        'echarts/charts',
        'echarts/components',
        'echarts/renderers',
        'vue-echarts',
        'three',
        'three/examples/jsm/controls/OrbitControls.js',
        'class-variance-authority',
        '@vueuse/core',
        'pinia',
      ],
    },

    ssr: {
      noExternal: [
        'vue-echarts',
        'echarts',
        'class-variance-authority',
      ],
    },
  },

  // ==========================================
  // BUILD OPTIMIZATION
  // ==========================================
  build: {
    transpile: [
      'vue-echarts',
      'echarts',
      'three',
    ],
  },

  // ==========================================
  // NITRO CONFIGURATION
  // ==========================================
  nitro: {
    compressPublicAssets: true,
    minify: true,
  },

  // ==========================================
  // APP CONFIGURATION
  // ==========================================
  app: {
    head: {
      title: process.env.NUXT_PUBLIC_APP_NAME || 'Hydraulic Diagnostic SaaS',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: 'AI-powered hydraulic systems monitoring and diagnostics' },
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
      ],
    },
  },

  // ==========================================
  // RUNTIME CONFIG
  // ==========================================
  runtimeConfig: {
    // Private (server-only)
    apiSecret: process.env.NUXT_API_SECRET || '',

    // Public (client + server)
    public: {
      // Environment
      environment: process.env.NUXT_PUBLIC_ENVIRONMENT || 'development',
      appName: process.env.NUXT_PUBLIC_APP_NAME || 'Hydraulic Diagnostics',
      appVersion: process.env.NUXT_PUBLIC_APP_VERSION || '1.0.0',

      // API Configuration
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000/api/v1',
      wsBase: process.env.NUXT_PUBLIC_WS_BASE || 'ws://localhost:8000/ws',
      apiTimeout: parseInt(process.env.NUXT_PUBLIC_API_TIMEOUT || '30000'),

      // Service Endpoints
      endpoints: {
        auth: process.env.NUXT_PUBLIC_ENDPOINT_AUTH || '/auth',
        equipment: process.env.NUXT_PUBLIC_ENDPOINT_EQUIPMENT || '/equipment',
        diagnosis: process.env.NUXT_PUBLIC_ENDPOINT_DIAGNOSIS || '/diagnosis',
        gnn: process.env.NUXT_PUBLIC_ENDPOINT_GNN || '/gnn',
        rag: process.env.NUXT_PUBLIC_ENDPOINT_RAG || '/rag',
        admin: process.env.NUXT_PUBLIC_ENDPOINT_ADMIN || '/admin',
      },

      // Feature Flags
      features: {
        ragInterpretation: process.env.NUXT_PUBLIC_ENABLE_RAG === 'true',
        websocket: process.env.NUXT_PUBLIC_ENABLE_WEBSOCKET === 'true',
        charts: process.env.NUXT_PUBLIC_ENABLE_CHARTS === 'true',
        mockData: process.env.NUXT_PUBLIC_ENABLE_MOCK_DATA === 'true',
        debug: process.env.NUXT_PUBLIC_DEBUG === 'true',
      },

      // Localization
      defaultLocale: process.env.NUXT_PUBLIC_DEFAULT_LOCALE || 'ru',
      availableLocales: (process.env.NUXT_PUBLIC_AVAILABLE_LOCALES || 'ru,en').split(','),

      // Security
      tokenStorageKey: process.env.NUXT_PUBLIC_TOKEN_STORAGE_KEY || 'auth_token',
      sessionTimeout: parseInt(process.env.NUXT_PUBLIC_SESSION_TIMEOUT || '30'),
      forceHttps: process.env.NUXT_PUBLIC_FORCE_HTTPS === 'true',

      // Monitoring (optional)
      sentryDsn: process.env.NUXT_PUBLIC_SENTRY_DSN || '',
      gaId: process.env.NUXT_PUBLIC_GA_ID || '',
    },
  },

  // ==========================================
  // CSS CONFIGURATION
  // ==========================================
  css: [
    '~/assets/css/main.css',
  ],

  postcss: {
    plugins: {
      tailwindcss: {},
      autoprefixer: {},
    },
  },

  // ==========================================
  // EXPERIMENTAL FEATURES
  // ==========================================
  experimental: {
    payloadExtraction: true,
    viewTransition: true,
  },

  // ==========================================
  // ROUTER CONFIGURATION
  // ==========================================
  router: {
    options: {
      strict: false,
    },
  },
})
