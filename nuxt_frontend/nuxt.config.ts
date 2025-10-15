// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-04-03',
  devtools: { enabled: true },

  // Runtime configuration
  runtimeConfig: {
    // Private keys (server-side only)
    apiSecret: process.env.NUXT_API_SECRET || '',
    
    // Public keys (exposed to client)
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000/api',
      appName: 'Hydraulic Diagnostic SaaS',
      appVersion: '1.0.0'
    }
  },

  // App configuration
  app: {
    head: {
      charset: 'utf-8',
      viewport: 'width=device-width, initial-scale=1',
      title: 'Hydraulic Diagnostic SaaS',
      meta: [
        { name: 'description', content: 'Professional hydraulic system diagnostic platform' },
        { name: 'format-detection', content: 'telephone=no' }
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }
      ]
    },
    // Page transitions
    pageTransition: { name: 'page', mode: 'out-in' },
    layoutTransition: { name: 'layout', mode: 'out-in' }
  },

  // Modules
  modules: [
    // Add modules here as needed
    // '@nuxtjs/tailwindcss',
    // '@pinia/nuxt',
    // '@vueuse/nuxt',
  ],

  // CSS configuration
  css: [
    // Add global CSS files here
    // '~/assets/css/main.css'
  ],

  // Build configuration
  build: {
    transpile: [],
  },

  // Vite configuration
  vite: {
    build: {
      // Chunk size warnings
      chunkSizeWarningLimit: 1000,
      // Optimize deps
      rollupOptions: {
        output: {
          manualChunks: {
            'vue-vendor': ['vue', 'vue-router']
          }
        }
      }
    },
    optimizeDeps: {
      include: ['vue', 'vue-router']
    }
  },

  // Nitro configuration (server)
  nitro: {
    compressPublicAssets: true,
    minify: true,
    // Prerender routes for static generation
    prerender: {
      crawlLinks: true,
      routes: ['/']
    },
    // Server handlers
    routeRules: {
      // Add caching rules
      '/**': { 
        headers: {
          'cache-control': 'public, max-age=0, must-revalidate'
        }
      },
      '/api/**': { 
        cors: true,
        headers: {
          'cache-control': 'no-cache'
        }
      }
    }
  },

  // TypeScript configuration
  typescript: {
    strict: true,
    typeCheck: false, // Enable in development if needed
    shim: false
  },

  // Experimental features
  experimental: {
    payloadExtraction: true,
    renderJsonPayloads: true,
    typedPages: true
  },

  // Development configuration
  devServer: {
    port: 3000,
    host: '0.0.0.0'
  },

  // Import auto-imports configuration
  imports: {
    dirs: [
      'composables',
      'composables/**'
    ]
  },

  // Component auto-import configuration
  components: [
    {
      path: '~/components',
      pathPrefix: false
    }
  ]
})
