<script setup lang="ts">
// Main application root with minimal logic for investor demo
useSeoMeta({
  titleTemplate: '%s | Hydraulic Diagnostic SaaS',
  description: 'AI-powered hydraulic system monitoring and predictive maintenance platform for industrial operations.',
  ogTitle: 'Hydraulic Diagnostic SaaS - AI-Powered Industrial Monitoring',
  ogDescription: 'Next-generation predictive maintenance for hydraulic systems. Real-time monitoring, intelligent diagnostics, and proactive maintenance scheduling.',
  ogImage: '/og-image.jpg',
  twitterCard: 'summary_large_image'
})

// Critical icon preloading for investor demo
if (import.meta.client) {
  const criticalIcons = [
    'heroicons:chart-bar-square',
    'heroicons:shield-check',
    'heroicons:users',
    'heroicons:cog-6-tooth',
    'heroicons:arrow-right',
    'heroicons:sparkles'
  ]

  // Preload icons for smooth demo experience
  criticalIcons.forEach(icon => {
    const link = document.createElement('link')
    link.rel = 'preload'
    link.as = 'image'
    link.href = `/_nuxt/icons/${icon.replace(':', '-')}.svg`
    document.head.appendChild(link)
  })
}

// Global error handling for demo stability
const handleError = (error: unknown, errorInfo?: unknown) => {
  console.error('Application error:', error, errorInfo)
  // Silent error handling for smooth demo experience
}

// Setup error boundaries
onErrorCaptured((error: unknown, instance: unknown, info: unknown) => {
  handleError(error, { instance, info })
  return false // Prevent error propagation
})

// Application lifecycle
onMounted(() => {
  console.log('🚀 Hydraulic Diagnostic SaaS - Ready for investor demo!')
})
</script>

<template>
  <div id="app" class="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white transition-colors duration-200">
    <!-- Main application content -->
    <NuxtLayout>
      <NuxtPage />
    </NuxtLayout>

    <!-- Global loading indicator -->
    <div v-if="$route.meta.loading" class="fixed inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm z-50 flex items-center justify-center">
      <div class="flex items-center space-x-3">
        <div class="animate-spin w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full"></div>
        <span class="text-sm font-medium text-gray-600 dark:text-gray-300">Загрузка...</span>
      </div>
    </div>
  </div>
</template>

<style>
/* Global styles for investor demo */
* {
  box-sizing: border-box;
}

html {
  font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
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

/* Hide scrollbars in demo for cleaner look */
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

/* Focus styles for accessibility */
.premium-focus:focus {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
}

/* Animation utilities for smooth demo */
.premium-fade-in {
  animation: fadeIn 0.6s ease-out;
}

.premium-slide-up {
  animation: slideUp 0.4s ease-out;
}

.premium-scale-in {
  animation: scaleIn 0.3s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes scaleIn {
  from { opacity: 0; transform: scale(0.95); }
  to { opacity: 1; transform: scale(1); }
}

/* Dark mode transitions */
.dark * {
  transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease;
}

/* Print styles for investor materials */
@media print {
  body {
    -webkit-print-color-adjust: exact;
    color-adjust: exact;
  }

  .no-print {
    display: none !important;
  }
}
</style>
