// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: {
    enabled: true,
    timeline: {
      enabled: true,
    },
  },

  // CSS and styling
  css: ['../assets/css/tailwind.css', '../styles/premium-tokens.css'],

  // Auto-import components from components and components/ui
  components: [
    { path: './components', pathPrefix: false },
    { path: './components/ui', pathPrefix: false },
  ],

  // App configuration
  app: {
    head: {
      charset: 'utf-8',
      viewport: 'width=device-width, initial-scale=1',
      title: 'Hydraulic Diagnostic SaaS - AI-Powered Industrial Monitoring',
      meta: [
        {
          name: 'description',
          content:
            'Revolutionary hydraulic diagnostics platform with predictive maintenance, real-time monitoring, and AI insights. Trusted by 127+ enterprises.',
        },
        { name: 'format-detection', content: 'telephone=no' },
      ],
    },
  },

  // Runtime configuration
  runtimeConfig: {
    // Private keys (only available on server-side)
    apiSecret: '',

    // Public keys (exposed to client-side)
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000/api',
      siteUrl: process.env.NUXT_PUBLIC_SITE_URL || 'http://localhost:3000',
    },
  },

  modules: [
    '@nuxt/content',
    '@nuxt/eslint',
    '@nuxt/image',
    '@nuxt/icon',
    '@nuxt/fonts',
    '@nuxtjs/color-mode',
    '@pinia/nuxt',
    '@nuxtjs/seo',
  ],

  // Color mode configuration
  colorMode: {
    preference: 'system',
    fallback: 'light',
    hid: 'nuxt-color-mode-script',
    globalName: '__NUXT_COLOR_MODE__',
    componentName: 'ColorScheme',
    classPrefix: '',
    classSuffix: '',
    storageKey: 'nuxt-color-mode',
  },

  // Fonts configuration
  fonts: {
    families: [
      { name: 'Inter', provider: 'google', weights: [400, 500, 600, 700, 800, 900] },
      { name: 'JetBrains Mono', provider: 'google', weights: [400, 500, 600] },
    ],
  },

  // Icon configuration
  icon: {
    serverBundle: 'auto',
  },

  // Image optimization
  image: {
    format: ['webp', 'avif'],
    quality: 80,
    screens: {
      xs: 320,
      sm: 640,
      md: 768,
      lg: 1024,
      xl: 1280,
      '2xl': 1536,
    },
  },

  // SEO configuration
  site: {
    url: process.env.NUXT_PUBLIC_SITE_URL || 'http://localhost:3000',
    name: 'Hydraulic Diagnostic SaaS',
    description:
      'AI-powered hydraulic diagnostics platform for industrial monitoring and predictive maintenance',
    defaultLocale: 'ru',
  },

  // Build configuration
  nitro: {
    compressPublicAssets: true,
  },

  // TypeScript configuration
  typescript: {
    strict: false,
    typeCheck: false,
  },
});
