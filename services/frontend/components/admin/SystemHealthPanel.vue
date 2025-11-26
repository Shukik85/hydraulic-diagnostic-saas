<script setup lang="ts">
import { computed } from 'vue';
import type { SystemHealth, HealthStatus } from '~/types';

interface Props {
  health: SystemHealth | null;
  loading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
});

const { t } = useI18n();

const statusConfig: Record<HealthStatus, { color: string; icon: string; text: string }> = {
  healthy: {
    color: 'text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400',
    icon: 'heroicons:check-circle',
    text: t('status.healthy'),
  },
  degraded: {
    color: 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 dark:text-yellow-400',
    icon: 'heroicons:exclamation-triangle',
    text: t('status.degraded'),
  },
  critical: {
    color: 'text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400',
    icon: 'heroicons:x-circle',
    text: t('status.critical'),
  },
};

const currentStatus = computed(() => {
  if (!props.health) {
    return statusConfig.degraded;
  }
  return statusConfig[props.health.status];
});

const metrics = computed(() => {
  if (!props.health) {
    return [];
  }

  return [
    {
      label: 'API Latency (P90)',
      value: `${props.health.apiLatencyP90.toFixed(0)}ms`,
      status: props.health.apiLatencyP90 < 200 ? 'good' : props.health.apiLatencyP90 < 500 ? 'warning' : 'critical',
    },
    {
      label: 'DB Latency (P90)',
      value: `${props.health.dbLatencyP90.toFixed(0)}ms`,
      status: props.health.dbLatencyP90 < 100 ? 'good' : props.health.dbLatencyP90 < 300 ? 'warning' : 'critical',
    },
    {
      label: 'ML Latency (P90)',
      value: `${props.health.mlLatencyP90.toFixed(0)}ms`,
      status: props.health.mlLatencyP90 < 1000 ? 'good' : props.health.mlLatencyP90 < 2000 ? 'warning' : 'critical',
    },
  ];
});
</script>

<template>
  <div class="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
    <div class="mb-4 flex items-center justify-between">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
        {{ t('admin.systemHealth') }}
      </h3>
    </div>

    <div v-if="loading" class="space-y-3">
      <div class="h-8 w-full animate-pulse rounded bg-gray-200 dark:bg-gray-700" />
      <div class="h-8 w-full animate-pulse rounded bg-gray-200 dark:bg-gray-700" />
      <div class="h-8 w-full animate-pulse rounded bg-gray-200 dark:bg-gray-700" />
    </div>

    <div v-else class="space-y-4">
      <!-- Status Badge -->
      <div class="flex items-center gap-2">
        <div class="inline-flex items-center gap-2 rounded-lg px-3 py-2" :class="currentStatus.color">
          <Icon :name="currentStatus.icon" class="h-5 w-5" aria-hidden="true" />
          <span class="font-medium">{{ currentStatus.text }}</span>
        </div>
      </div>

      <!-- Metrics -->
      <div class="space-y-3">
        <div v-for="metric in metrics" :key="metric.label" class="flex items-center justify-between">
          <span class="text-sm text-gray-600 dark:text-gray-400">{{ metric.label }}</span>
          <div class="flex items-center gap-2">
            <span class="font-medium text-gray-900 dark:text-white">{{ metric.value }}</span>
            <div
              class="h-2 w-2 rounded-full"
              :class="{
                'bg-green-500': metric.status === 'good',
                'bg-yellow-500': metric.status === 'warning',
                'bg-red-500': metric.status === 'critical',
              }"
              :aria-label="metric.status"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
