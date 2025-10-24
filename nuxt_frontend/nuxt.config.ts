// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },

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

  modules: ['@nuxt/eslint', '@pinia/nuxt'],

  // CSS and styling
  css: ['~/styles/premium-tokens.css', '~/assets/css/globals.css'],
});
