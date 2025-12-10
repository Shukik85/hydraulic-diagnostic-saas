<script setup lang="ts">
import { ref, computed } from 'vue';
import type { Sensor } from '~/types';

interface Props {
  sensors: Sensor[];
  loading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
});

const { t } = useI18n();

const sensorTypeIcons: Record<string, string> = {
  pressure: 'heroicons:arrow-trending-up',
  temperature: 'heroicons:fire',
  flow: 'heroicons:arrow-path',
  vibration: 'heroicons:signal',
  position: 'heroicons:map-pin',
};

const statusColors = {
  online: 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-400 dark:border-green-800',
  offline: 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-900/20 dark:text-gray-400 dark:border-gray-800',
  error: 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-400 dark:border-red-800',
  calibrating: 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-400 dark:border-yellow-800',
};

const qualityColors = {
  good: 'text-green-600 dark:text-green-400',
  warning: 'text-yellow-600 dark:text-yellow-400',
  bad: 'text-red-600 dark:text-red-400',
};
</script>

<template>
  <div class="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
    <div class="mb-4 flex items-center justify-between">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
        {{ t('diagnosis.sensors.title') }}
      </h3>
      <span class="text-sm text-gray-500 dark:text-gray-400">
        {{ sensors.length }} {{ t('diagnosis.sensors.count') }}
      </span>
    </div>

    <div v-if="loading" class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <div v-for="i in 6" :key="i" class="h-32 animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700" />
    </div>

    <div v-else class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <div
        v-for="sensor in sensors"
        :key="sensor.id"
        class="group relative overflow-hidden rounded-lg border p-4 transition-all hover:shadow-md"
        :class="statusColors[sensor.status]"
      >
        <!-- Header -->
        <div class="mb-3 flex items-start justify-between">
          <div class="flex items-center gap-2">
            <Icon :name="sensorTypeIcons[sensor.type]" class="h-5 w-5" aria-hidden="true" />
            <span class="font-medium">{{ sensor.name }}</span>
          </div>
          <span
            class="rounded-full px-2 py-0.5 text-xs font-medium uppercase"
            :class="statusColors[sensor.status]"
          >
            {{ t(`status.${sensor.status}`) }}
          </span>
        </div>

        <!-- Reading -->
        <div v-if="sensor.lastReading" class="mb-2">
          <div class="flex items-baseline gap-2">
            <span class="text-2xl font-bold" :class="qualityColors[sensor.lastReading.quality]">
              {{ sensor.lastReading.value.toFixed(2) }}
            </span>
            <span class="text-sm text-gray-600 dark:text-gray-400">
              {{ sensor.unit }}
            </span>
          </div>
          <div class="mt-1 text-xs text-gray-500 dark:text-gray-400">
            {{ new Date(sensor.lastReading.timestamp).toLocaleTimeString('ru-RU') }}
          </div>
        </div>

        <div v-else class="mb-2 text-sm text-gray-500 dark:text-gray-400">
          {{ t('diagnosis.sensors.noData') }}
        </div>

        <!-- Metadata -->
        <div v-if="sensor.metadata?.threshold" class="mt-2 border-t border-current pt-2 text-xs opacity-70">
          <div class="flex items-center justify-between">
            <span>{{ t('diagnosis.sensors.threshold') }}:</span>
            <span class="font-medium">{{ sensor.metadata.threshold }} {{ sensor.unit }}</span>
          </div>
        </div>

        <!-- Live indicator -->
        <div
          v-if="sensor.status === 'online'"
          class="absolute right-2 top-2"
        >
          <span class="relative flex h-2 w-2">
            <span class="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
            <span class="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
