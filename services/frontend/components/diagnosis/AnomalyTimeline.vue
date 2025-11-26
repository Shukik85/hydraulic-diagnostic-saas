<script setup lang="ts">
import { ref, computed } from 'vue';
import type { Anomaly } from '~/types';

interface Props {
  anomalies: Anomaly[];
  loading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
});

interface Emits {
  (e: 'selectAnomaly', anomaly: Anomaly): void;
}

const emit = defineEmits<Emits>();

const { t } = useI18n();
const selectedFilter = ref<'all' | 'critical' | 'high' | 'medium' | 'low'>('all');

const filteredAnomalies = computed(() => {
  if (selectedFilter.value === 'all') {
    return props.anomalies;
  }
  return props.anomalies.filter((a) => a.severity === selectedFilter.value);
});

const severityConfig = {
  critical: {
    color: 'bg-red-500',
    textColor: 'text-red-600 dark:text-red-400',
    icon: 'heroicons:exclamation-circle',
  },
  high: {
    color: 'bg-orange-500',
    textColor: 'text-orange-600 dark:text-orange-400',
    icon: 'heroicons:exclamation-triangle',
  },
  medium: {
    color: 'bg-yellow-500',
    textColor: 'text-yellow-600 dark:text-yellow-400',
    icon: 'heroicons:information-circle',
  },
  low: {
    color: 'bg-blue-500',
    textColor: 'text-blue-600 dark:text-blue-400',
    icon: 'heroicons:information-circle',
  },
};

const handleAnomalyClick = (anomaly: Anomaly): void => {
  emit('selectAnomaly', anomaly);
};
</script>

<template>
  <div class="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
    <div class="mb-4 flex items-center justify-between">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
        {{ t('diagnosis.anomalies.title') }}
      </h3>
      
      <!-- Filter -->
      <div class="flex gap-2">
        <button
          v-for="filter in ['all', 'critical', 'high', 'medium', 'low'] as const"
          :key="filter"
          @click="selectedFilter = filter"
          class="rounded-lg px-3 py-1 text-xs font-medium transition-colors"
          :class="{
            'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300': selectedFilter === filter,
            'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600': selectedFilter !== filter,
          }"
        >
          {{ t(`severity.${filter}`) }}
        </button>
      </div>
    </div>

    <div v-if="loading" class="space-y-3">
      <div v-for="i in 5" :key="i" class="h-16 animate-pulse rounded bg-gray-200 dark:bg-gray-700" />
    </div>

    <div v-else-if="filteredAnomalies.length === 0" class="py-12 text-center">
      <Icon name="heroicons:check-circle" class="mx-auto h-12 w-12 text-green-500" />
      <p class="mt-2 text-sm text-gray-500 dark:text-gray-400">
        {{ t('diagnosis.anomalies.noAnomalies') }}
      </p>
    </div>

    <div v-else class="relative">
      <!-- Timeline line -->
      <div class="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200 dark:bg-gray-700" aria-hidden="true" />

      <!-- Anomaly items -->
      <div class="space-y-4">
        <button
          v-for="anomaly in filteredAnomalies"
          :key="anomaly.id"
          @click="handleAnomalyClick(anomaly)"
          class="group relative flex w-full gap-4 rounded-lg p-3 text-left transition-colors hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          <!-- Severity dot -->
          <div class="relative flex-shrink-0">
            <div class="h-8 w-8 rounded-full border-4 border-white dark:border-gray-800" :class="severityConfig[anomaly.severity].color" />
          </div>

          <!-- Content -->
          <div class="min-w-0 flex-1">
            <div class="flex items-start justify-between">
              <div class="flex items-center gap-2">
                <Icon
                  :name="severityConfig[anomaly.severity].icon"
                  class="h-5 w-5"
                  :class="severityConfig[anomaly.severity].textColor"
                  aria-hidden="true"
                />
                <span class="font-medium text-gray-900 dark:text-white">
                  {{ t(`anomalyType.${anomaly.type}`) }}
                </span>
              </div>
              <span class="text-xs text-gray-500 dark:text-gray-400">
                {{ new Date(anomaly.detectedAt).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) }}
              </span>
            </div>
            
            <p class="mt-1 text-sm text-gray-600 dark:text-gray-300">
              {{ anomaly.description }}
            </p>
            
            <div class="mt-2 flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
              <span>{{ t('diagnosis.anomalies.confidence') }}: {{ (anomaly.confidence * 100).toFixed(1) }}%</span>
              <span v-if="!anomaly.acknowledgedAt" class="text-orange-600 dark:text-orange-400">
                {{ t('diagnosis.anomalies.unacknowledged') }}
              </span>
              <span v-else-if="anomaly.resolvedAt" class="text-green-600 dark:text-green-400">
                {{ t('diagnosis.anomalies.resolved') }}
              </span>
            </div>
          </div>
        </button>
      </div>
    </div>
  </div>
</template>
