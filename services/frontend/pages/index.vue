<script setup lang="ts">
definePageMeta({
  layout: 'default',
});

useSeoMeta({
  title: 'Hydraulic Diagnostic SaaS - AI-Powered Diagnostics',
  description: 'Advanced hydraulic system diagnostics powered by Graph Neural Networks',
});
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-gray-50 via-primary-50/30 to-blue-50 dark:from-gray-950 dark:via-gray-900 dark:to-gray-900">
    <!-- Hero Section -->
    <div class="relative overflow-hidden">
      <!-- Animated background -->
      <div class="absolute inset-0 overflow-hidden">
        <div class="absolute -left-1/4 top-0 h-96 w-96 animate-pulse rounded-full bg-primary-400/20 blur-3xl" />
        <div class="absolute right-0 top-1/4 h-96 w-96 animate-pulse rounded-full bg-blue-400/20 blur-3xl animation-delay-2000" />
      </div>

      <div class="container relative mx-auto px-4 py-20 sm:py-32">
        <div class="mx-auto max-w-4xl text-center">
          <!-- Badge -->
          <div class="mb-8 inline-flex items-center gap-2 rounded-full bg-primary-100 px-4 py-2 text-sm font-medium text-primary-700 dark:bg-primary-900/30 dark:text-primary-300">
            <Icon name="heroicons:sparkles" class="h-4 w-4" />
            <span>AI-Powered Predictive Maintenance</span>
          </div>

          <!-- Heading -->
          <h1 class="mb-6 text-5xl font-bold tracking-tight text-gray-900 dark:text-white sm:text-6xl lg:text-7xl">
            Hydraulic System
            <span class="bg-gradient-to-r from-primary-600 to-blue-600 bg-clip-text text-transparent">
              Diagnostics
            </span>
            <br>Reimagined
          </h1>

          <!-- Description -->
          <p class="mb-10 text-lg text-gray-600 dark:text-gray-300 sm:text-xl">
            Real-time monitoring and AI-powered predictive analytics for industrial hydraulic systems.
            Prevent failures before they happen.
          </p>

          <!-- CTA Buttons -->
          <div class="flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Button 
              variant="primary" 
              size="lg" 
              class="!bg-gradient-to-r from-primary-600 to-primary-500 shadow-lg shadow-primary-500/30 hover:shadow-xl hover:shadow-primary-500/40"
              @click="navigateTo('/login')"
            >
              <span class="flex items-center gap-2">
                <span>Get Started</span>
                <Icon name="heroicons:arrow-right" class="h-5 w-5" />
              </span>
            </Button>
            <Button 
              variant="outline" 
              size="lg"
              class="border-2 hover:bg-primary-50 dark:hover:bg-primary-900/20"
              @click="navigateTo('/dashboard')"
            >
              <span class="flex items-center gap-2">
                <Icon name="heroicons:play" class="h-5 w-5" />
                <span>View Demo</span>
              </span>
            </Button>
          </div>

          <!-- Trust Badge -->
          <p class="mt-8 text-sm text-gray-500 dark:text-gray-400">
            Trusted by 1,000+ industrial facilities worldwide
          </p>
        </div>
      </div>
    </div>

    <!-- Stats Section -->
    <div class="container mx-auto px-4 py-16">
      <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <div class="group rounded-2xl bg-white/80 p-8 shadow-lg backdrop-blur-sm transition-all hover:scale-105 hover:shadow-xl dark:bg-gray-900/80">
          <div class="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 shadow-lg">
            <Icon name="heroicons:chart-bar" class="h-6 w-6 text-white" />
          </div>
          <div class="mb-2 flex items-baseline gap-2">
            <span class="text-4xl font-bold text-gray-900 dark:text-white">24/7</span>
            <span class="rounded-full bg-green-100 px-2 py-1 text-xs font-medium text-green-700 dark:bg-green-900/30 dark:text-green-300">
              Live
            </span>
          </div>
          <p class="mb-1 font-semibold text-gray-900 dark:text-white">Real-time Monitoring</p>
          <p class="text-sm text-gray-600 dark:text-gray-400">Continuous sensor data analysis and alerts</p>
        </div>

        <div class="group rounded-2xl bg-white/80 p-8 shadow-lg backdrop-blur-sm transition-all hover:scale-105 hover:shadow-xl dark:bg-gray-900/80">
          <div class="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 shadow-lg">
            <Icon name="heroicons:cpu-chip" class="h-6 w-6 text-white" />
          </div>
          <div class="mb-2 flex items-baseline gap-2">
            <span class="text-4xl font-bold text-gray-900 dark:text-white">95%</span>
            <span class="rounded-full bg-blue-100 px-2 py-1 text-xs font-medium text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
              +5.2%
            </span>
          </div>
          <p class="mb-1 font-semibold text-gray-900 dark:text-white">AI Prediction Accuracy</p>
          <p class="text-sm text-gray-600 dark:text-gray-400">Graph Neural Network anomaly detection</p>
        </div>

        <div class="group rounded-2xl bg-white/80 p-8 shadow-lg backdrop-blur-sm transition-all hover:scale-105 hover:shadow-xl dark:bg-gray-900/80 sm:col-span-2 lg:col-span-1">
          <div class="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-purple-500 to-purple-600 shadow-lg">
            <Icon name="heroicons:server" class="h-6 w-6 text-white" />
          </div>
          <div class="mb-2 flex items-baseline gap-2">
            <span class="text-4xl font-bold text-gray-900 dark:text-white">1,234</span>
            <span class="rounded-full bg-purple-100 px-2 py-1 text-xs font-medium text-purple-700 dark:bg-purple-900/30 dark:text-purple-300">
              +12%
            </span>
          </div>
          <p class="mb-1 font-semibold text-gray-900 dark:text-white">Systems Monitored</p>
          <p class="text-sm text-gray-600 dark:text-gray-400">Active hydraulic systems worldwide</p>
        </div>
      </div>
    </div>

    <!-- Features Section -->
    <div class="container mx-auto px-4 py-16">
      <div class="mx-auto max-w-6xl">
        <div class="mb-12 text-center">
          <h2 class="mb-4 text-3xl font-bold text-gray-900 dark:text-white sm:text-4xl">
            Enterprise-Grade Platform
          </h2>
          <p class="text-lg text-gray-600 dark:text-gray-300">
            Built with modern technologies for reliability and performance
          </p>
        </div>

        <div class="grid gap-8 md:grid-cols-2">
          <!-- Feature 1 -->
          <div class="rounded-2xl bg-white/80 p-8 shadow-lg backdrop-blur-sm dark:bg-gray-900/80">
            <div class="mb-4 flex items-center gap-3">
              <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-100 dark:bg-primary-900/30">
                <Icon name="heroicons:code-bracket" class="h-5 w-5 text-primary-600 dark:text-primary-400" />
              </div>
              <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Modern Stack</h3>
            </div>
            <ul class="space-y-2 text-gray-600 dark:text-gray-300">
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>Nuxt 4 with TypeScript strict mode</span>
              </li>
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>Pinia state management</span>
              </li>
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>Tailwind CSS with design system</span>
              </li>
            </ul>
          </div>

          <!-- Feature 2 -->
          <div class="rounded-2xl bg-white/80 p-8 shadow-lg backdrop-blur-sm dark:bg-gray-900/80">
            <div class="mb-4 flex items-center gap-3">
              <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
                <Icon name="heroicons:shield-check" class="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Quality & Testing</h3>
            </div>
            <ul class="space-y-2 text-gray-600 dark:text-gray-300">
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>Vitest + Cypress (>95% coverage)</span>
              </li>
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>WCAG 2.1 AA accessibility</span>
              </li>
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>ESLint + Prettier enforced</span>
              </li>
            </ul>
          </div>

          <!-- Feature 3 -->
          <div class="rounded-2xl bg-white/80 p-8 shadow-lg backdrop-blur-sm dark:bg-gray-900/80">
            <div class="mb-4 flex items-center gap-3">
              <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/30">
                <Icon name="heroicons:bolt" class="h-5 w-5 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Real-time Features</h3>
            </div>
            <ul class="space-y-2 text-gray-600 dark:text-gray-300">
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>WebSocket real-time updates</span>
              </li>
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>Live sensor monitoring</span>
              </li>
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>Instant anomaly alerts</span>
              </li>
            </ul>
          </div>

          <!-- Feature 4 -->
          <div class="rounded-2xl bg-white/80 p-8 shadow-lg backdrop-blur-sm dark:bg-gray-900/80">
            <div class="mb-4 flex items-center gap-3">
              <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-green-100 dark:bg-green-900/30">
                <Icon name="heroicons:globe-alt" class="h-5 w-5 text-green-600 dark:text-green-400" />
              </div>
              <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Internationalization</h3>
            </div>
            <ul class="space-y-2 text-gray-600 dark:text-gray-300">
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>i18n support (Russian/English)</span>
              </li>
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>RTL layout support</span>
              </li>
              <li class="flex items-start gap-2">
                <Icon name="heroicons:check-circle" class="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
                <span>Locale-aware formatting</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <!-- CTA Section -->
    <div class="container mx-auto px-4 py-16">
      <div class="mx-auto max-w-4xl rounded-2xl bg-gradient-to-r from-primary-600 to-blue-600 p-12 text-center shadow-2xl">
        <h2 class="mb-4 text-3xl font-bold text-white sm:text-4xl">
          Ready to get started?
        </h2>
        <p class="mb-8 text-lg text-primary-100">
          Join thousands of facilities using our platform for predictive maintenance
        </p>
        <Button 
          variant="secondary" 
          size="lg"
          class="!bg-white !text-primary-600 shadow-lg hover:!bg-gray-50"
          @click="navigateTo('/login')"
        >
          <span class="flex items-center gap-2">
            <span>Start Free Trial</span>
            <Icon name="heroicons:arrow-right" class="h-5 w-5" />
          </span>
        </Button>
      </div>
    </div>
  </div>
</template>

<style scoped>
@keyframes pulse {
  0%, 100% {
    opacity: 0.4;
  }
  50% {
    opacity: 0.6;
  }
}

.animation-delay-2000 {
  animation-delay: 2s;
}
</style>
