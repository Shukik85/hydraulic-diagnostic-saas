<script setup lang="ts">
// Optimized application root for investor demo performance
useSeoMeta({
  titleTemplate: '%s | Hydraulic Diagnostic SaaS',
  description: 'AI-powered hydraulic system monitoring and predictive maintenance platform for industrial operations.',
  ogTitle: 'Hydraulic Diagnostic SaaS - AI-Powered Industrial Monitoring',
  ogDescription: 'Next-generation predictive maintenance for hydraulic systems. Real-time monitoring, intelligent diagnostics, and proactive maintenance scheduling.',
  ogImage: '/og-image.jpg',
  twitterCard: 'summary_large_image'
})

// Optimized error handling for smooth demo
const handleError = (error: any) => {
  console.error('Application error:', error)
  // Silent handling for investor presentation
  return false
}

onErrorCaptured(handleError)

// Demo readiness indicator
onMounted(() => {
  console.log('🚀 Hydraulic Diagnostic SaaS - Ready for investor demo!')
})
</script>

<template>
  <div id="app" class="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white transition-colors duration-200">
    <NuxtLayout>
      <NuxtPage />
    </NuxtLayout>
  </div>
</template>

<style>
/* Optimized global styles for fast loading */
* {
  box-sizing: border-box;
}

html {
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
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
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes scaleIn {
  from { opacity: 0; transform: scale(0.98); }
  to { opacity: 1; transform: scale(1); }
}

/* Fast dark mode transitions */
.dark * {
  transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease;
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