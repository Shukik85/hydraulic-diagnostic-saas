<script setup lang="ts">
import DashboardCharts from '~/components/dashboard/DashboardCharts.client.vue';
import Sparklines from '~/components/dashboard/Sparklines.client.vue';

const source = ref<'btc' | 'eth'>('btc');
const { data, refresh, pending } = await useFetch(
  () => `/api/demo/hydraulic-metrics?source=${source.value}`
);

const temp = computed(() => data.value?.sparklines?.temperature || []);
const pressure = computed(() => data.value?.sparklines?.pressure || []);
const flow = computed(() => data.value?.sparklines?.flow_rate || []);
const vibration = computed(() => data.value?.sparklines?.vibration || []);
const thresholds = computed(() => data.value?.thresholds || {});
const weekStats = computed(() => data.value?.aggregates?.week_stats || {});

watch(source, () => refresh());

const zoneColor = (metric: string, value: number) => {
  const t = thresholds.value?.[metric];
  if (!t) return 'text-gray-600 dark:text-gray-300';
  if (metric === 'pressure') {
    // Для давления зелёная зона выше green (лучше выше), красная ниже red (хуже)
    return value >= t.green
      ? 'text-green-600 dark:text-green-400'
      : value <= t.red
        ? 'text-red-600 dark:text-red-400'
        : 'text-yellow-600 dark:text-yellow-400';
  }
  return value <= t.green
    ? 'text-green-600 dark:text-green-400'
    : value >= t.red
      ? 'text-red-600 dark:text-red-400'
      : 'text-yellow-600 dark:text-yellow-400';
};
</script>
<template>
  <div class="space-y-6">
    <!-- Source Switcher -->
    <div class="premium-card p-4 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <span class="text-sm text-gray-600 dark:text-gray-300">Источник данных:</span>
        <div class="flex rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
          <button
            :class="[
              'px-4 py-2 text-sm font-medium',
              source === 'btc'
                ? 'bg-blue-600 text-white'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300',
            ]"
            @click="source = 'btc'"
          >
            BTC
          </button>
          <button
            :class="[
              'px-4 py-2 text-sm font-medium',
              source === 'eth'
                ? 'bg-blue-600 text-white'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300',
            ]"
            @click="source = 'eth'"
          >
            ETH
          </button>
        </div>
      </div>
      <div v-if="pending" class="text-sm text-gray-500 dark:text-gray-400">Загрузка...</div>
    </div>

    <!-- Sparklines + Zones -->
    <Sparklines :temp="temp" :pressure="pressure" :flow="flow" :vibration="vibration" />

    <!-- Aggregates with zone coloring -->
    <div class="premium-card p-4">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <div class="text-gray-500 dark:text-gray-400 mb-1">Температура (°C)</div>
          <div class="flex items-center space-x-3">
            <span :class="zoneColor('temperature', weekStats.temperature?.avg)"
              >ср: {{ weekStats.temperature?.avg }}</span
            >
            <span class="text-gray-400">/</span>
            <span class="text-gray-600 dark:text-gray-300"
              >мин: {{ weekStats.temperature?.min }}</span
            >
            <span class="text-gray-400">/</span>
            <span class="text-gray-600 dark:text-gray-300"
              >макс: {{ weekStats.temperature?.max }}</span
            >
          </div>
        </div>
        <div>
          <div class="text-gray-500 dark:text-gray-400 mb-1">Давление (бар)</div>
          <div class="flex items-center space-x-3">
            <span :class="zoneColor('pressure', weekStats.pressure?.avg)"
              >ср: {{ weekStats.pressure?.avg }}</span
            >
            <span class="text-gray-400">/</span>
            <span class="text-gray-600 dark:text-gray-300">мин: {{ weekStats.pressure?.min }}</span>
            <span class="text-gray-400">/</span>
            <span class="text-gray-600 dark:text-gray-300"
              >макс: {{ weekStats.pressure?.max }}</span
            >
          </div>
        </div>
        <div>
          <div class="text-gray-500 dark:text-gray-400 mb-1">Расход (л/мин)</div>
          <div class="flex items-center space-x-3">
            <span :class="zoneColor('flow_rate', weekStats.flow_rate?.avg)"
              >ср: {{ weekStats.flow_rate?.avg }}</span
            >
            <span class="text-gray-400">/</span>
            <span class="text-gray-600 dark:text-gray-300"
              >мин: {{ weekStats.flow_rate?.min }}</span
            >
            <span class="text-gray-400">/</span>
            <span class="text-gray-600 dark:text-gray-300"
              >макс: {{ weekStats.flow_rate?.max }}</span
            >
          </div>
        </div>
        <div>
          <div class="text-gray-500 dark:text-gray-400 mb-1">Вибрация (мм/с)</div>
          <div class="flex items-center space-x-3">
            <span :class="zoneColor('vibration', weekStats.vibration?.avg)"
              >ср: {{ weekStats.vibration?.avg }}</span
            >
            <span class="text-gray-400">/</span>
            <span class="text-gray-600 dark:text-gray-300"
              >мин: {{ weekStats.vibration?.min }}</span
            >
            <span class="text-gray-400">/</span>
            <span class="text-gray-600 dark:text-gray-300"
              >макс: {{ weekStats.vibration?.max }}</span
            >
          </div>
        </div>
      </div>
    </div>

    <!-- Main charts (unchanged) -->
    <DashboardCharts />
  </div>
</template>
