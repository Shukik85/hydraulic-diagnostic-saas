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
  <div class="min-h-screen bg-gradient-to-br from-primary-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
    <div class="container mx-auto px-4 py-16">
      <!-- Hero Section -->
      <div class="mb-16 text-center">
        <h1 class="mb-4 text-5xl font-bold text-gray-900 dark:text-white">
          Hydraulic Diagnostic SaaS
        </h1>
        <p class="mb-8 text-xl text-gray-600 dark:text-gray-300">
          AI-powered hydraulic system diagnostics with real-time monitoring
        </p>
        <div class="flex justify-center gap-4">
          <Button variant="primary" size="lg" @click="navigateTo('/login')">
            Get Started
          </Button>
          <Button variant="outline" size="lg" @click="navigateTo('/dashboard')">
            View Dashboard
          </Button>
        </div>
      </div>

      <!-- Features Grid -->
      <div class="grid gap-8 md:grid-cols-3">
        <KpiCard
          label="Real-time Monitoring"
          value="24/7"
          icon="heroicons:chart-bar"
          status="success"
          subtext="Continuous sensor data analysis"
        />
        <KpiCard
          label="AI Predictions"
          value="95%"
          :trend="5.2"
          icon="heroicons:cpu-chip"
          status="success"
          subtext="Accuracy in anomaly detection"
        />
        <KpiCard
          label="Systems Monitored"
          value="1,234"
          :trend="12.3"
          icon="heroicons:server"
          status="neutral"
          subtext="Active hydraulic systems"
        />
      </div>

      <!-- Info Section -->
      <div class="mt-16 rounded-lg bg-white p-8 shadow-lg dark:bg-gray-800">
        <h2 class="mb-4 text-3xl font-bold text-gray-900 dark:text-white">
          Production-Ready Frontend
        </h2>
        <div class="space-y-4 text-gray-600 dark:text-gray-300">
          <p>✅ <strong>Nuxt 4</strong> with TypeScript strict mode</p>
          <p>✅ <strong>Pinia</strong> for state management</p>
          <p>✅ <strong>Tailwind CSS</strong> for styling</p>
          <p>✅ <strong>Vitest + Cypress</strong> for testing (>95% coverage target)</p>
          <p>✅ <strong>WCAG 2.1 AA</strong> accessibility compliance</p>
          <p>✅ <strong>Real-time WebSocket</strong> updates</p>
          <p>✅ <strong>i18n</strong> support (Russian/English)</p>
        </div>
      </div>
    </div>
  </div>
</template>
