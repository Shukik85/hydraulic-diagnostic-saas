<script setup lang="ts">
import { computed } from 'vue';
import type { Alert } from '~/types';

interface Props {
  alerts: Alert[];
  loading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
});

const { t } = useI18n();
const { formatDate } = useFormatting();

const severityConfig = {
  critical: {
    color: 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800',
    iconColor: 'text-red-600 dark:text-red-400',
    icon: 'heroicons:exclamation-circle',
  },
  high: {
    color: 'bg-orange-50 border-orange-200 dark:bg-orange-900/20 dark:border-orange-800',
    iconColor: 'text-orange-600 dark:text-orange-400',
    icon: 'heroicons:exclamation-triangle',
  },
  medium: {
    color: 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800',
    iconColor: 'text-yellow-600 dark:text-yellow-400',
    icon: 'heroicons:information-circle',
  },
  low: {
    color: 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800',
    iconColor: 'text-blue-600 dark:text-blue-400',
    icon: 'heroicons:information-circle',
  },
};

const unresolvedAlerts = computed(() => props.alerts.filter((a) => !a.resolved));
</script>

<template>
  <div class="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
    <div class="mb-4 flex items-center justify-between">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
        {{ t('admin.criticalAlerts') }}
      </h3>
      <span class="rounded-full bg-red-100 px-2.5 py-0.5 text-xs font-medium text-red-800 dark:bg-red-900/20 dark:text-red-400">
        {{ unresolvedAlerts.length }}
      </span>
    </div>

    <div v-if="loading" class="space-y-3">
      <div v-for="i in 3" :key="i" class="h-16 w-full animate-pulse rounded bg-gray-200 dark:bg-gray-700" />
    </div>

    <div v-else-if="unresolvedAlerts.length === 0" class="py-8 text-center">
      <Icon name="heroicons:check-circle" class="mx-auto h-12 w-12 text-green-500" />
      <p class="mt-2 text-sm text-gray-500 dark:text-gray-400">
        {{ t('admin.noAlerts') }}
      </p>
    </div>

    <div v-else class="space-y-3">
      <div
        v-for="alert in unresolvedAlerts.slice(0, 5)"
        :key="alert.id"
        class="flex items-start gap-3 rounded-lg border p-4 transition-colors hover:bg-gray-50 dark:hover:bg-gray-700"
        :class="severityConfig[alert.severity].color"
      >
        <Icon
          :name="severityConfig[alert.severity].icon"
          class="h-5 w-5 flex-shrink-0"
          :class="severityConfig[alert.severity].iconColor"
          aria-hidden="true"
        />
        <div class="flex-1 min-w-0">
          <p class="text-sm font-medium text-gray-900 dark:text-white">
            {{ alert.message }}
          </p>
          <div class="mt-1 flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
            <span>{{ alert.source }}</span>
            <span>•</span>
            <time :datetime="alert.timestamp.toISOString()">
              {{ formatDate(alert.timestamp) }}
            </time>
          </div>
        </div>
        <button
          class="flex-shrink-0 rounded-lg p-1 text-gray-400 hover:bg-white hover:text-gray-600 dark:hover:bg-gray-600 dark:hover:text-gray-200"
          aria-label="Acknowledge alert"
        >
          <Icon name="heroicons:check" class="h-4 w-4" />
        </button>
      </div>
    </div>
  </div>
</template>
