<script setup lang="ts">
// Enterprise Industrial Theme - Hydraulic Diagnostic Platform
useSeoMeta({
  titleTemplate: '%s | Hydraulic Diagnostics',
  description:
    'AI-powered hydraulic system monitoring and predictive maintenance platform for industrial operations.',
  ogTitle: 'Hydraulic Diagnostics Platform - Enterprise AI Monitoring',
  ogDescription:
    'Next-generation predictive maintenance for hydraulic systems. Real-time monitoring, GNN-powered diagnostics, and proactive maintenance scheduling.',
  ogImage: '/og-image.jpg',
  twitterCard: 'summary_large_image',
});

// Color mode preference (auto-detect system)
const colorMode = useColorMode()

// Global error handling
const handleError = (error: unknown) => {
  console.error('[App] Uncaught error:', error);
  // Error will be handled by useErrorHandler in components
  return false;
};

onErrorCaptured(handleError);

// Application readiness
onMounted(() => {
  console.log('[App] ✅ Hydraulic Diagnostics Platform - Ready');
  console.log('[App] 🎨 Theme:', colorMode.value);
});
</script>

<template>
  <div
    id="app"
    class="min-h-screen bg-industrial-50 dark:bg-industrial-950 text-industrial-900 dark:text-industrial-50 transition-colors duration-200"
  >
    <NuxtLayout>
      <NuxtPage />
    </NuxtLayout>
    
    <!-- Global Toast Notifications -->
    <ToastContainer />
  </div>
</template>

<style>
/* Enterprise Industrial Theme - Global Styles */

* {
  box-sizing: border-box;
}

html {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
  scroll-behavior: smooth;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

body {
  margin: 0;
  padding: 0;
  text-rendering: optimizeLegibility;
}

/* Custom scrollbars - Industrial style */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: rgba(100, 116, 139, 0.3);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(100, 116, 139, 0.5);
}

.dark ::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.3);
}

.dark ::-webkit-scrollbar-thumb:hover {
  background: rgba(148, 163, 184, 0.5);
}

/* Focus styles - Accessible */
*:focus-visible {
  outline: 2px solid #0ea5e9;
  outline-offset: 2px;
}

/* Selection color */
::selection {
  background-color: #0ea5e9;
  color: white;
}

/* Smooth transitions for theme switching */
.theme-transition,
.theme-transition * {
  transition:
    background-color 0.2s ease,
    color 0.2s ease,
    border-color 0.2s ease;
}

/* Utility classes */
.truncate-2-lines {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.truncate-3-lines {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Card hover effects */
.card-interactive {
  @apply transition-all duration-250 hover:shadow-card-hover hover:-translate-y-0.5;
}

/* Status indicators */
.status-pulse {
  animation: status-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes status-pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

/* Loading skeleton */
.skeleton {
  @apply animate-pulse bg-industrial-200 dark:bg-industrial-800 rounded;
}

/* Print styles */
@media print {
  body {
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  .no-print {
    display: none !important;
  }
  
  .print-break {
    page-break-before: always;
  }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
</style>