<script setup lang="ts">
definePageMeta({
  layout: 'default',
  middleware: 'auth',
});

useSeoMeta({
  title: 'Dashboard - Hydraulic Diagnostic SaaS',
  description: 'System monitoring dashboard',
});
</script>

<template>
  <div class="space-y-6">
    <div>
      <h1 class="text-3xl font-bold text-gray-900 dark:text-white">
        Dashboard
      </h1>
      <p class="mt-2 text-gray-600 dark:text-gray-400">
        Welcome to your hydraulic systems monitoring dashboard
      </p>
    </div>

    <!-- Stats Grid -->
    <div class="grid gap-6 md:grid-cols-3">
      <KpiCard
        label="Active Systems"
        value="24"
        icon="heroicons:server"
        status="success"
        subtext="All systems operational"
      />
      <KpiCard
        label="Alerts"
        value="3"
        :trend="-2.1"
        icon="heroicons:exclamation-triangle"
        status="warning"
        subtext="Requires attention"
      />
      <KpiCard
        label="Uptime"
        value="99.8%"
        :trend="0.2"
        icon="heroicons:chart-bar"
        status="success"
        subtext="Last 30 days"
      />
    </div>

    <!-- Quick Actions -->
    <div class="rounded-lg bg-white p-6 shadow dark:bg-gray-800">
      <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
        Quick Actions
      </h2>
      <div class="flex flex-wrap gap-3">
        <Button variant="primary" @click="navigateTo('/diagnosis')">
          <Icon name="heroicons:beaker" class="h-4 w-4" />
          Start Diagnosis
        </Button>
        <Button variant="secondary" @click="navigateTo('/systems')">
          <Icon name="heroicons:cog-6-tooth" class="h-4 w-4" />
          Manage Systems
        </Button>
        <Button variant="outline">
          <Icon name="heroicons:document-text" class="h-4 w-4" />
          View Reports
        </Button>
      </div>
    </div>

    <!-- Recent Activity Placeholder -->
    <div class="rounded-lg bg-white p-6 shadow dark:bg-gray-800">
      <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
        Recent Activity
      </h2>
      <p class="text-gray-600 dark:text-gray-400">
        No recent activity to display
      </p>
    </div>
  </div>
</template>
