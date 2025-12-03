<script setup lang="ts">
definePageMeta({
  layout: 'default',
  middleware: 'auth',
});

useSeoMeta({
  title: 'Dashboard - Hydraulic Diagnostic SaaS',
  description: 'System monitoring dashboard',
});

const { t } = useI18n();
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-gray-50 to-primary-50/20 dark:from-gray-950 dark:to-gray-900">
    <div class="container mx-auto px-4 py-8">
      <!-- Header -->
      <div class="mb-8 rounded-2xl bg-gradient-to-r from-primary-600 to-blue-600 p-8 text-white shadow-xl">
        <div class="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 class="mb-2 text-3xl font-bold sm:text-4xl">
              {{ t('dashboard.title') }}
            </h1>
            <p class="text-primary-100">
              {{ t('dashboard.subtitle') }}
            </p>
          </div>
          <div class="flex gap-3">
            <Button 
              variant="secondary" 
              class="!bg-white/20 !text-white backdrop-blur-sm hover:!bg-white/30"
            >
              <Icon name="heroicons:bell" class="h-5 w-5" />
            </Button>
            <Button 
              variant="secondary" 
              class="!bg-white/20 !text-white backdrop-blur-sm hover:!bg-white/30"
            >
              <Icon name="heroicons:cog-6-tooth" class="h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>

      <!-- Stats Grid -->
      <div class="mb-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <!-- Active Systems Card -->
        <div class="group rounded-2xl bg-white/80 p-6 shadow-lg backdrop-blur-sm transition-all hover:scale-105 hover:shadow-xl dark:bg-gray-900/80">
          <div class="mb-4 flex items-center justify-between">
            <div class="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-green-500 to-green-600 shadow-lg">
              <Icon name="heroicons:server" class="h-6 w-6 text-white" />
            </div>
            <span class="rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-700 dark:bg-green-900/30 dark:text-green-300">
              {{ t('dashboard.stats.allOk') }}
            </span>
          </div>
          <div class="mb-2 text-3xl font-bold text-gray-900 dark:text-white">
            24
          </div>
          <p class="mb-1 font-semibold text-gray-900 dark:text-white">{{ t('dashboard.stats.activeSystems') }}</p>
          <p class="text-sm text-gray-600 dark:text-gray-400">{{ t('dashboard.stats.activeSystemsDesc') }}</p>
        </div>

        <!-- Alerts Card -->
        <div class="group rounded-2xl bg-white/80 p-6 shadow-lg backdrop-blur-sm transition-all hover:scale-105 hover:shadow-xl dark:bg-gray-900/80">
          <div class="mb-4 flex items-center justify-between">
            <div class="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 shadow-lg">
              <Icon name="heroicons:exclamation-triangle" class="h-6 w-6 text-white" />
            </div>
            <span class="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-700 dark:bg-amber-900/30 dark:text-amber-300">
              -2.1%
            </span>
          </div>
          <div class="mb-2 text-3xl font-bold text-gray-900 dark:text-white">
            3
          </div>
          <p class="mb-1 font-semibold text-gray-900 dark:text-white">{{ t('dashboard.stats.activeAlerts') }}</p>
          <p class="text-sm text-gray-600 dark:text-gray-400">{{ t('dashboard.stats.activeAlertsDesc') }}</p>
        </div>

        <!-- Uptime Card -->
        <div class="group rounded-2xl bg-white/80 p-6 shadow-lg backdrop-blur-sm transition-all hover:scale-105 hover:shadow-xl dark:bg-gray-900/80 sm:col-span-2 lg:col-span-1">
          <div class="mb-4 flex items-center justify-between">
            <div class="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 shadow-lg">
              <Icon name="heroicons:chart-bar" class="h-6 w-6 text-white" />
            </div>
            <span class="rounded-full bg-blue-100 px-3 py-1 text-xs font-medium text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
              +0.2%
            </span>
          </div>
          <div class="mb-2 text-3xl font-bold text-gray-900 dark:text-white">
            99.8%
          </div>
          <p class="mb-1 font-semibold text-gray-900 dark:text-white">{{ t('dashboard.stats.systemUptime') }}</p>
          <p class="text-sm text-gray-600 dark:text-gray-400">{{ t('dashboard.stats.systemUptimeDesc') }}</p>
        </div>
      </div>

      <div class="grid gap-6 lg:grid-cols-3">
        <!-- Quick Actions -->
        <div class="lg:col-span-2">
          <div class="rounded-2xl bg-white/80 p-6 shadow-lg backdrop-blur-sm dark:bg-gray-900/80">
            <div class="mb-6 flex items-center justify-between">
              <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
                {{ t('dashboard.quickActions.title') }}
              </h2>
              <Icon name="heroicons:bolt" class="h-5 w-5 text-primary-600 dark:text-primary-400" />
            </div>
            
            <div class="grid gap-4 sm:grid-cols-2">
              <!-- Start Diagnosis -->
              <button 
                class="group flex items-center gap-4 rounded-xl border-2 border-gray-200 bg-gradient-to-br from-primary-50 to-blue-50 p-4 text-left transition-all hover:border-primary-300 hover:shadow-lg dark:border-gray-700 dark:from-primary-900/20 dark:to-blue-900/20 dark:hover:border-primary-600"
                @click="navigateTo('/diagnosis')"
              >
                <div class="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-primary-600 shadow-lg">
                  <Icon name="heroicons:beaker" class="h-6 w-6 text-white" />
                </div>
                <div>
                  <div class="font-semibold text-gray-900 dark:text-white">{{ t('dashboard.quickActions.startDiagnosis') }}</div>
                  <div class="text-sm text-gray-600 dark:text-gray-400">{{ t('dashboard.quickActions.startDiagnosisDesc') }}</div>
                </div>
                <Icon name="heroicons:arrow-right" class="ml-auto h-5 w-5 text-gray-400 transition-transform group-hover:translate-x-1" />
              </button>

              <!-- Manage Systems -->
              <button 
                class="group flex items-center gap-4 rounded-xl border-2 border-gray-200 bg-gradient-to-br from-blue-50 to-indigo-50 p-4 text-left transition-all hover:border-blue-300 hover:shadow-lg dark:border-gray-700 dark:from-blue-900/20 dark:to-indigo-900/20 dark:hover:border-blue-600"
                @click="navigateTo('/systems')"
              >
                <div class="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 shadow-lg">
                  <Icon name="heroicons:cog-6-tooth" class="h-6 w-6 text-white" />
                </div>
                <div>
                  <div class="font-semibold text-gray-900 dark:text-white">{{ t('dashboard.quickActions.manageSystems') }}</div>
                  <div class="text-sm text-gray-600 dark:text-gray-400">{{ t('dashboard.quickActions.manageSystemsDesc') }}</div>
                </div>
                <Icon name="heroicons:arrow-right" class="ml-auto h-5 w-5 text-gray-400 transition-transform group-hover:translate-x-1" />
              </button>

              <!-- View Reports -->
              <button 
                class="group flex items-center gap-4 rounded-xl border-2 border-gray-200 bg-gradient-to-br from-purple-50 to-pink-50 p-4 text-left transition-all hover:border-purple-300 hover:shadow-lg dark:border-gray-700 dark:from-purple-900/20 dark:to-pink-900/20 dark:hover:border-purple-600"
              >
                <div class="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500 to-purple-600 shadow-lg">
                  <Icon name="heroicons:document-text" class="h-6 w-6 text-white" />
                </div>
                <div>
                  <div class="font-semibold text-gray-900 dark:text-white">{{ t('dashboard.quickActions.viewReports') }}</div>
                  <div class="text-sm text-gray-600 dark:text-gray-400">{{ t('dashboard.quickActions.viewReportsDesc') }}</div>
                </div>
                <Icon name="heroicons:arrow-right" class="ml-auto h-5 w-5 text-gray-400 transition-transform group-hover:translate-x-1" />
              </button>

              <!-- Settings -->
              <button 
                class="group flex items-center gap-4 rounded-xl border-2 border-gray-200 bg-gradient-to-br from-gray-50 to-slate-50 p-4 text-left transition-all hover:border-gray-300 hover:shadow-lg dark:border-gray-700 dark:from-gray-800/50 dark:to-slate-800/50 dark:hover:border-gray-600"
              >
                <div class="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-gray-600 to-gray-700 shadow-lg">
                  <Icon name="heroicons:adjustments-horizontal" class="h-6 w-6 text-white" />
                </div>
                <div>
                  <div class="font-semibold text-gray-900 dark:text-white">{{ t('dashboard.quickActions.settings') }}</div>
                  <div class="text-sm text-gray-600 dark:text-gray-400">{{ t('dashboard.quickActions.settingsDesc') }}</div>
                </div>
                <Icon name="heroicons:arrow-right" class="ml-auto h-5 w-5 text-gray-400 transition-transform group-hover:translate-x-1" />
              </button>
            </div>
          </div>
        </div>

        <!-- Recent Activity -->
        <div class="lg:col-span-1">
          <div class="rounded-2xl bg-white/80 p-6 shadow-lg backdrop-blur-sm dark:bg-gray-900/80">
            <div class="mb-6 flex items-center justify-between">
              <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
                {{ t('dashboard.recentActivity.title') }}
              </h2>
              <Icon name="heroicons:clock" class="h-5 w-5 text-gray-400" />
            </div>
            
            <!-- Activity Timeline -->
            <div class="space-y-4">
              <!-- Activity Item 1 -->
              <div class="flex gap-3">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/30">
                  <Icon name="heroicons:check-circle" class="h-4 w-4 text-green-600 dark:text-green-400" />
                </div>
                <div class="flex-1">
                  <p class="text-sm font-medium text-gray-900 dark:text-white">{{ t('dashboard.recentActivity.systemDiagnosed') }}</p>
                  <p class="text-xs text-gray-500 dark:text-gray-400">{{ t('dashboard.recentActivity.minutesAgo', { count: 2 }) }}</p>
                </div>
              </div>

              <!-- Activity Item 2 -->
              <div class="flex gap-3">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-100 dark:bg-amber-900/30">
                  <Icon name="heroicons:exclamation-triangle" class="h-4 w-4 text-amber-600 dark:text-amber-400" />
                </div>
                <div class="flex-1">
                  <p class="text-sm font-medium text-gray-900 dark:text-white">{{ t('dashboard.recentActivity.alertPressure') }}</p>
                  <p class="text-xs text-gray-500 dark:text-gray-400">{{ t('dashboard.recentActivity.minutesAgo', { count: 15 }) }}</p>
                </div>
              </div>

              <!-- Activity Item 3 -->
              <div class="flex gap-3">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/30">
                  <Icon name="heroicons:document-text" class="h-4 w-4 text-blue-600 dark:text-blue-400" />
                </div>
                <div class="flex-1">
                  <p class="text-sm font-medium text-gray-900 dark:text-white">{{ t('dashboard.recentActivity.reportGenerated') }}</p>
                  <p class="text-xs text-gray-500 dark:text-gray-400">{{ t('dashboard.recentActivity.hoursAgo', { count: 1 }) }}</p>
                </div>
              </div>

              <!-- Activity Item 4 -->
              <div class="flex gap-3">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-purple-100 dark:bg-purple-900/30">
                  <Icon name="heroicons:server" class="h-4 w-4 text-purple-600 dark:text-purple-400" />
                </div>
                <div class="flex-1">
                  <p class="text-sm font-medium text-gray-900 dark:text-white">{{ t('dashboard.recentActivity.systemAdded') }}</p>
                  <p class="text-xs text-gray-500 dark:text-gray-400">{{ t('dashboard.recentActivity.hoursAgo', { count: 3 }) }}</p>
                </div>
              </div>
            </div>

            <!-- View All Link -->
            <button class="mt-6 w-full rounded-lg border-2 border-gray-200 py-2 text-sm font-medium text-gray-700 transition-colors hover:border-primary-300 hover:bg-primary-50 hover:text-primary-700 dark:border-gray-700 dark:text-gray-300 dark:hover:border-primary-600 dark:hover:bg-primary-900/20 dark:hover:text-primary-300">
              {{ t('dashboard.recentActivity.viewAll') }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
