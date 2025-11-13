// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },
  
  modules: [
    '@pinia/nuxt',
    '@vueuse/nuxt',
    '@nuxt/eslint'
  ],
  
  // Runtime config
  runtimeConfig: {
    public: {
      // API Gateway (Kong) - unified entry point
      apiBase: process.env.API_GATEWAY_URL || 'https://api.hydraulic-diagnostics.com',
      
      // WebSocket endpoint
      wsBase: process.env.WS_URL || 'wss://api.hydraulic-diagnostics.com/ws',
      
      // Service endpoints (за Kong Gateway)
      endpoints: {
        auth: '/api/v1/auth',
        equipment: '/api/v1/equipment',
        diagnosis: '/api/v1/diagnosis',
        gnn: '/api/v1/gnn',
        rag: '/api/v1/rag',
        admin: '/api/v1/admin'
      },
      
      // Feature flags
      features: {
        ragInterpretation: process.env.ENABLE_RAG === 'true',
        realtimeUpdates: process.env.ENABLE_WEBSOCKET === 'true',
        advancedCharts: process.env.ENABLE_CHARTS === 'true'
      }
    }
  },
  
  // TypeScript
  typescript: {
    strict: true,
    typeCheck: true,
    shim: false
  },
  
  // ESLint
  eslint: {
    config: {
      stylistic: true
    }
  },
  
  // Build optimization
  vite: {
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            'api-client': ['./generated/api']
          }
        }
      }
    }
  },
  
  // Ignore generated code from TypeScript checking
  ignore: [
    'generated/**/*'
  ]
})
