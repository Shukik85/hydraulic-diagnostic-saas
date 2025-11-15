<script setup lang="ts">
// Production-ready application root with error boundary
useSeoMeta({
  titleTemplate: '%s | Hydraulic Diagnostic SaaS',
  description:
    'AI-powered hydraulic system monitoring and predictive maintenance platform for industrial operations.',
  ogTitle: 'Hydraulic Diagnostic SaaS - AI-Powered Industrial Monitoring',
  ogDescription:
    'Next-generation predictive maintenance for hydraulic systems. Real-time monitoring, intelligent diagnostics, and proactive maintenance scheduling.',
  ogImage: '/og-image.jpg',
  twitterCard: 'summary_large_image',
});

const config = useRuntimeConfig()

// Error handling for production
const handleError = (error: unknown, instance: any, info: string) => {
  console.error('Application error:', error)
  console.error('Component:', instance)
  console.error('Error info:', info)
  
  // Log to monitoring service (Sentry, etc.) in production
  if (config.public.environment === 'production') {
    // TODO: Send to error tracking service
    // Sentry.captureException(error)
  }
  
  return false // Prevent error propagation
}

onErrorCaptured(handleError)

// Application readiness
onMounted(() => {
  const env = config.public.environment || 'development'
  console.log(`🚀 Hydraulic Diagnostic SaaS - ${env} mode`)
  
  if (env === 'development') {
    console.log('🛠️  Dev Mode:')
    console.log('  API Base:', config.public.apiBase)
    console.log('  WS Base:', config.public.wsBase)
    console.log('  Features:', config.public.features)
  }
})
</script>

<template>
  <div
    id="app"
    class="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white transition-colors duration-200"
  >
    <!-- Error Boundary - перехватывает ошибки и показывает fallback UI -->
    <NuxtErrorBoundary>
      <NuxtLayout>
        <NuxtPage />
      </NuxtLayout>

      <!-- Error Fallback UI -->
      <template #error="{ error, clearError }">
        <div class="min-h-screen flex items-center justify-center p-4 bg-gray-50">
          <div class="max-w-md w-full">
            <div class="u-card p-8 text-center">
              <!-- Error Icon -->
              <div class="w-16 h-16 mx-auto mb-6 rounded-full bg-red-100 flex items-center justify-center">
                <svg class="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>

              <!-- Error Message -->
              <h1 class="u-h3 mb-3 text-gray-900">
                Опа! Что-то пошло не так
              </h1>
              <p class="u-body text-gray-600 mb-6">
                Произошла неожиданная ошибка. Мы уже работаем над этим.
              </p>

              <!-- Error Details (dev only) -->
              <details v-if="config.public.environment === 'development'" class="mb-6 text-left">
                <summary class="cursor-pointer text-sm text-gray-500 hover:text-gray-700 mb-2">
                  Технические детали (только в dev)
                </summary>
                <pre class="text-xs bg-gray-100 p-4 rounded-lg overflow-auto max-h-40">{{ error }}</pre>
              </details>

              <!-- Actions -->
              <div class="flex flex-col sm:flex-row gap-3">
                <button @click="clearError" class="u-btn u-btn-primary flex-1">
                  <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Попробовать снова
                </button>
                <button @click="navigateTo('/')" class="u-btn u-btn-secondary flex-1">
                  <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                  </svg>
                  На главную
                </button>
              </div>

              <!-- Support Link -->
              <p class="u-body-sm text-gray-500 mt-6">
                Проблема повторяется? 
                <a href="mailto:shukik85@ya.ru" class="text-blue-600 hover:text-blue-700 underline">
                  Свяжитесь с поддержкой
                </a>
              </p>
            </div>
          </div>
        </div>
      </template>
    </NuxtErrorBoundary>
  </div>
</template>

<style>
/* Optimized global styles for fast loading */
* {
  box-sizing: border-box;
}

html {
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue',
    Arial, sans-serif;
  scroll-behavior: smooth;
  line-height: 1.6;
}

body {
  margin: 0;
  padding: 0;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

/* Optimized scrollbars */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: rgba(156, 163, 175, 0.3);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(156, 163, 175, 0.5);
}

/* Focus styles */
.premium-focus:focus {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
}

/* Performance-optimized animations */
.premium-fade-in {
  animation: fadeIn 0.4s ease-out;
}

.premium-slide-up {
  animation: slideUp 0.3s ease-out;
}

.premium-scale-in {
  animation: scaleIn 0.2s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(5px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes scaleIn {
  from {
    opacity: 0;
    transform: scale(0.98);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

/* Fast dark mode transitions */
.dark * {
  transition:
    background-color 0.15s ease,
    color 0.15s ease,
    border-color 0.15s ease;
}

/* Fixed print styles (no deprecated properties) */
@media print {
  body {
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  .no-print {
    display: none !important;
  }
}
</style>