
// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  ssr: false, // SPA режим для дэшборда
  
  compatibilityDate: '2025-10-30', // Убираем warning
  
  devtools: { enabled: true },
  
  modules: [
    '@pinia/nuxt',
    '@nuxtjs/color-mode',
    '@nuxt/icon',
    '@nuxt/image'
  ],
  
  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000/api',
      wsBase: process.env.NUXT_PUBLIC_WS_BASE || 'ws://localhost:8000/ws'
    }
  },
  
  css: [
    '~/styles/premium-tokens.css'
  ],

  postcss: {
    plugins:{
    '@tailwindcss/postcss': {},
    autoprefixer: {}
  },
},
  
  components: [
    {
      path: '~/components',
      pathPrefix: false,
      // Explicitly enable global registration for UI components
      global: true,
      // Ensure components are scanned properly
      extensions: ['.vue'],
      // Include all subdirectories
      dirs: [
        '~/components/ui',
        '~/components/dashboard'
      ]
    }
  ],
  
  colorMode: {
    classSuffix: '',
    preference: 'system',
    fallback: 'light'
  },
  
  app: {
    head: {
      title: 'Hydraulic Diagnostic SaaS',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: 'Intelligent hydraulic systems diagnostic platform with AI-powered analysis' }
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