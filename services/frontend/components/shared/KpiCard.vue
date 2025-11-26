<script setup lang="ts">
import { computed } from 'vue';

type KpiStatus = 'success' | 'warning' | 'danger' | 'neutral';

interface Props {
  label: string;
  value: string | number;
  trend?: number;
  status?: KpiStatus;
  icon?: string;
  subtext?: string;
  loading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  status: 'neutral',
  loading: false,
});

const statusClasses = computed(() => {
  const classes: Record<KpiStatus, string> = {
    success: 'bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400',
    warning: 'bg-yellow-50 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-400',
    danger: 'bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-400',
    neutral: 'bg-gray-50 text-gray-700 dark:bg-gray-800 dark:text-gray-300',
  };
  return classes[props.status];
});

const trendIcon = computed(() => {
  if (props.trend === undefined) {
    return null;
  }
  return props.trend > 0 ? 'heroicons:arrow-trending-up' : 'heroicons:arrow-trending-down';
});

const trendColor = computed(() => {
  if (props.trend === undefined) {
    return '';
  }
  return props.trend > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
});

const formattedTrend = computed(() => {
  if (props.trend === undefined) {
    return null;
  }
  const sign = props.trend > 0 ? '+' : '';
  return `${sign}${props.trend.toFixed(1)}%`;
});
</script>

<template>
  <div
    class="rounded-lg border border-gray-200 bg-white p-6 shadow-sm transition-shadow hover:shadow-md dark:border-gray-700 dark:bg-gray-900"
    :class="statusClasses"
  >
    <div class="flex items-start justify-between">
      <div class="flex-1">
        <p class="text-sm font-medium text-gray-600 dark:text-gray-400">{{ label }}</p>

        <div v-if="loading" class="mt-2">
          <div class="h-8 w-32 animate-pulse rounded bg-gray-200 dark:bg-gray-700" />
        </div>

        <div v-else class="mt-2 flex items-baseline">
          <p class="text-3xl font-semibold text-gray-900 dark:text-gray-100">{{ value }}</p>

          <div v-if="trend !== undefined" class="ml-3 flex items-center" :class="trendColor">
            <Icon :name="trendIcon!" class="h-5 w-5" aria-hidden="true" />
            <span class="ml-1 text-sm font-medium" :aria-label="`Trend: ${formattedTrend}`">
              {{ formattedTrend }}
            </span>
          </div>
        </div>

        <p v-if="subtext" class="mt-1 text-xs text-gray-500 dark:text-gray-400">{{ subtext }}</p>
      </div>

      <Icon v-if="icon" :name="icon" class="h-8 w-8 text-gray-400" aria-hidden="true" />
    </div>
  </div>
</template>
