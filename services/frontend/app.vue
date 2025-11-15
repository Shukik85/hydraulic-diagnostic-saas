<script setup lang="ts">
// Application root with SEO optimization
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

// Global error handling
const handleError = (error: unknown) => {
  console.error('Application error:', error);
  // Return false to prevent error propagation to parent components
  return false;
};

onErrorCaptured(handleError);

// Application initialization
onMounted(() => {
  console.log('🚀 Hydraulic Diagnostic SaaS - Application Ready');
});
</script>

<template>
  <div
    id="app"
    class="min-h-screen bg-background-primary text-text-primary transition-colors duration-200"
  >
    <NuxtLayout>
      <NuxtPage />
    </NuxtLayout>
  </div>
</template>

<style>
/* Import metallic industrial theme */
@import './styles/metallic.css';

/* Optimized global styles */
* {
  box-sizing: border-box;
}

html {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue',
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

/* Focus styles */
.premium-focus:focus {
  outline: 2px solid #4f46e5;
  outline-offset: 2px;
}

/* TODO: Move these animation utilities to tailwind.config.ts for better organization */
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

/* Dark mode transitions */
.dark * {
  transition:
    background-color 0.15s ease,
    color 0.15s ease,
    border-color 0.15s ease;
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
}
</style>
