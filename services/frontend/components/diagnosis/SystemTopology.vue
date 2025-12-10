<script setup lang="ts">
import { ref, computed } from 'vue';

interface Props {
  systemId: string;
  showAnomalies?: boolean;
  loading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  showAnomalies: true,
  loading: false,
});

const { t } = useI18n();

// Mock topology data - in production, this would come from backend
const components = ref([
  { id: 'pump', type: 'pump', x: 50, y: 100, hasAnomaly: false },
  { id: 'valve1', type: 'valve', x: 150, y: 100, hasAnomaly: true },
  { id: 'sensor1', type: 'sensor', x: 250, y: 100, hasAnomaly: false },
  { id: 'tank', type: 'tank', x: 350, y: 100, hasAnomaly: false },
]);

const connections = ref([
  { from: 'pump', to: 'valve1' },
  { from: 'valve1', to: 'sensor1' },
  { from: 'sensor1', to: 'tank' },
]);
</script>

<template>
  <div class="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
    <div class="mb-4 flex items-center justify-between">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
        {{ t('diagnosis.topology.title') }}
      </h3>
      <div class="flex items-center gap-2">
        <span class="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
          <span class="h-2 w-2 rounded-full bg-green-500" />
          {{ t('diagnosis.topology.normal') }}
        </span>
        <span class="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
          <span class="h-2 w-2 rounded-full bg-red-500" />
          {{ t('diagnosis.topology.anomaly') }}
        </span>
      </div>
    </div>

    <div v-if="loading" class="flex h-80 items-center justify-center">
      <div class="h-8 w-8 animate-spin rounded-full border-4 border-primary-500 border-t-transparent" />
    </div>

    <div v-else class="relative h-80 overflow-hidden rounded-lg bg-gray-50 dark:bg-gray-900">
      <svg class="h-full w-full" viewBox="0 0 400 200">
        <!-- Connections -->
        <g stroke="currentColor" stroke-width="2" class="text-gray-400 dark:text-gray-600">
          <line
            v-for="(conn, index) in connections"
            :key="index"
            :x1="components.find(c => c.id === conn.from)?.x"
            :y1="components.find(c => c.id === conn.from)?.y"
            :x2="components.find(c => c.id === conn.to)?.x"
            :y2="components.find(c => c.id === conn.to)?.y"
            stroke-dasharray="5,5"
          />
        </g>

        <!-- Components -->
        <g v-for="component in components" :key="component.id">
          <!-- Component circle -->
          <circle
            :cx="component.x"
            :cy="component.y"
            r="20"
            :class="{
              'fill-green-100 stroke-green-500 dark:fill-green-900/20': !component.hasAnomaly,
              'fill-red-100 stroke-red-500 dark:fill-red-900/20': component.hasAnomaly,
            }"
            stroke-width="2"
          />
          
          <!-- Icon -->
          <text
            :x="component.x"
            :y="component.y + 5"
            text-anchor="middle"
            class="text-sm font-bold fill-gray-700 dark:fill-gray-300"
          >
            {{ component.type.charAt(0).toUpperCase() }}
          </text>

          <!-- Label -->
          <text
            :x="component.x"
            :y="component.y + 35"
            text-anchor="middle"
            class="text-xs fill-gray-600 dark:fill-gray-400"
          >
            {{ t(`diagnosis.topology.${component.type}`) }}
          </text>

          <!-- Anomaly indicator -->
          <circle
            v-if="component.hasAnomaly && showAnomalies"
            :cx="component.x + 15"
            :cy="component.y - 15"
            r="5"
            class="fill-red-500 animate-pulse"
          />
        </g>
      </svg>

      <!-- Info overlay -->
      <div class="absolute bottom-4 left-4 right-4 rounded-lg bg-white/90 p-3 text-xs backdrop-blur dark:bg-gray-800/90">
        <p class="text-gray-700 dark:text-gray-300">
          <Icon name="heroicons:information-circle" class="mr-1 inline h-4 w-4" />
          {{ t('diagnosis.topology.info') }}
        </p>
      </div>
    </div>
  </div>
</template>
