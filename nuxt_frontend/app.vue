<script setup lang="ts">
useHead({
  htmlAttrs: { lang: 'ru' },
  link: [
    { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
    { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }
  ]
})

const handleError = (error: unknown) => {
  console.error('Application error:', error)
}

// Неблокирующее “прогревание” иконок через скрытый контейнер
const criticalIcons = [
  'heroicons:chart-bar-square',
  'heroicons:shield-check',
  'heroicons:users',
  'heroicons:cog-6-tooth'
]
</script>

<template>
  <div id="app" class="min-h-screen text-gray-900 dark:text-white bg-gray-50 dark:bg-gray-900 transition-colors">
    <NuxtRouteAnnouncer />
    <NuxtLayout>
      <NuxtWelcome v-if="$route.path === '/welcome'" />
      <NuxtPage v-else />
    </NuxtLayout>
    <NuxtErrorBoundary @error="handleError" />

    <!-- Hidden icon pre-render to warm cache -->
    <div aria-hidden="true" class="sr-only">
      <Icon v-for="iconName in criticalIcons" :key="iconName" :name="iconName" />
    </div>
  </div>
</template>


<style>
/* Global styles with premium design system integration */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

* {
  box-sizing: border-box;
}

html {
  font-family: 'Inter', sans-serif;
  scroll-behavior: smooth;
}

body {
  margin: 0;
  padding: 0;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Code font for technical content */
code,
pre,
.font-mono {
  font-family: 'JetBrains Mono', 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
}

/* Ensure consistent focus styles */
:focus {
  outline: 2px solid theme('colors.blue.500');
  outline-offset: 2px;
}

:focus:not(:focus-visible) {
  outline: none;
}

/* Smooth scrolling for internal links */
a[href^="#"] {
  scroll-behavior: smooth;
}

/* Print styles */
@media print {
  .no-print {
    display: none !important;
  }

  .print-break {
    page-break-after: always;
  }
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .premium-card {
    border-width: 2px;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {

  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* Loading states */
.loading {
  opacity: 0.7;
  pointer-events: none;
}

.loading * {
  cursor: wait;
}

/* Selection styles */
::selection {
  background-color: theme('colors.blue.100');
  color: theme('colors.blue.900');
}

::-moz-selection {
  background-color: theme('colors.blue.100');
  color: theme('colors.blue.900');
}

/* Dark mode selection */
@media (prefers-color-scheme: dark) {
  ::selection {
    background-color: theme('colors.blue.800');
    color: theme('colors.blue.100');
  }

  ::-moz-selection {
    background-color: theme('colors.blue.800');
    color: theme('colors.blue.100');
  }
}

/* Scrollbar styling (webkit) */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: theme('colors.gray.100');
}

::-webkit-scrollbar-thumb {
  background: theme('colors.gray.400');
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: theme('colors.gray.500');
}

@media (prefers-color-scheme: dark) {
  ::-webkit-scrollbar-track {
    background: theme('colors.gray.800');
  }

  ::-webkit-scrollbar-thumb {
    background: theme('colors.gray.600');
  }

  ::-webkit-scrollbar-thumb:hover {
    background: theme('colors.gray.500');
  }
}

/* Custom animations for premium effects */
@keyframes shimmer {
  0% {
    background-position: -200% 0;
  }

  100% {
    background-position: 200% 0;
  }
}

.shimmer {
  background: linear-gradient(90deg,
      transparent,
      rgba(255, 255, 255, 0.2),
      transparent);
  background-size: 200% 100%;
  animation: shimmer 2s infinite;
}

/* Gradient animation for premium text */
@keyframes gradient {

  0%,
  100% {
    background-position: 0% 50%;
  }

  50% {
    background-position: 100% 50%;
  }
}

.animate-gradient {
  background-size: 200% 200%;
  animation: gradient 3s ease infinite;
}
</style>
