<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import type { Sensor, Anomaly } from '~/types';

definePageMeta({
  layout: 'default',
  middleware: 'auth',
});

const route = useRoute();
const systemId = route.params.id as string;

const { t } = useI18n();

useSeoMeta({
  title: `${t('diagnosis.title')} #${systemId}`,
  description: t('diagnosis.description'),
});

const isLoading = ref(true);
const systemName = ref(`Hydraulic System #${systemId}`);

// Mock sensor data
const sensors = ref<Sensor[]>([
  {
    id: '1',
    name: 'Pressure 1',
    systemId,
    type: 'pressure',
    unit: 'bar',
    status: 'online',
    lastReading: {
      value: 145.8,
      timestamp: new Date(),
      quality: 'good',
    },
    metadata: {
      threshold: 150,
      minValue: 0,
      maxValue: 200,
    },
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    id: '2',
    name: 'Temperature 1',
    systemId,
    type: 'temperature',
    unit: '°C',
    status: 'online',
    lastReading: {
      value: 65.2,
      timestamp: new Date(),
      quality: 'warning',
    },
    metadata: {
      threshold: 70,
      minValue: 0,
      maxValue: 100,
    },
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    id: '3',
    name: 'Flow 1',
    systemId,
    type: 'flow',
    unit: 'L/min',
    status: 'online',
    lastReading: {
      value: 23.5,
      timestamp: new Date(),
      quality: 'good',
    },
    metadata: {
      threshold: 30,
    },
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    id: '4',
    name: 'Vibration 1',
    systemId,
    type: 'vibration',
    unit: 'mm/s',
    status: 'error',
    lastReading: {
      value: 8.9,
      timestamp: new Date(),
      quality: 'bad',
    },
    metadata: {
      threshold: 5,
    },
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    id: '5',
    name: 'Pressure 2',
    systemId,
    type: 'pressure',
    unit: 'bar',
    status: 'online',
    lastReading: {
      value: 142.3,
      timestamp: new Date(),
      quality: 'good',
    },
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    id: '6',
    name: 'Position 1',
    systemId,
    type: 'position',
    unit: 'mm',
    status: 'offline',
    createdAt: new Date(),
    updatedAt: new Date(),
  },
]);

// Mock anomaly data
const anomalies = ref<Anomaly[]>([
  {
    id: '1',
    sensorId: '4',
    systemId,
    severity: 'critical',
    type: 'vibration_excessive',
    description: 'Vibration levels exceeded safe threshold by 78%',
    detectedAt: new Date(Date.now() - 1800000),
    confidence: 0.95,
    metadata: {
      expectedValue: 5.0,
      actualValue: 8.9,
      deviation: 78,
      impact: 'High risk of mechanical failure',
      recommendedAction: 'Immediate inspection of pump bearings required',
    },
  },
  {
    id: '2',
    sensorId: '2',
    systemId,
    severity: 'high',
    type: 'temperature_high',
    description: 'Operating temperature approaching critical limit',
    detectedAt: new Date(Date.now() - 3600000),
    acknowledgedAt: new Date(Date.now() - 1800000),
    acknowledgedBy: 'user-1',
    confidence: 0.87,
    metadata: {
      expectedValue: 60.0,
      actualValue: 65.2,
      deviation: 8.7,
      recommendedAction: 'Check cooling system',
    },
  },
  {
    id: '3',
    sensorId: '1',
    systemId,
    severity: 'medium',
    type: 'pressure_spike',
    description: 'Pressure fluctuation detected',
    detectedAt: new Date(Date.now() - 7200000),
    acknowledgedAt: new Date(Date.now() - 3600000),
    acknowledgedBy: 'user-1',
    resolvedAt: new Date(Date.now() - 1800000),
    resolvedBy: 'user-1',
    confidence: 0.72,
  },
]);

const selectedAnomaly = ref<Anomaly | null>(null);

const handleAnomalySelect = (anomaly: Anomaly): void => {
  selectedAnomaly.value = anomaly;
  // Scroll to details or open modal
};

onMounted(async () => {
  // Simulate loading
  setTimeout(() => {
    isLoading.value = false;
  }, 500);

  // In production: Subscribe to WebSocket for real-time updates
  // const ws = useWebSocket(`/systems/${systemId}/sensors`);
});

onUnmounted(() => {
  // Cleanup WebSocket
});
</script>

<template>
  <div class="min-h-screen bg-gray-50 p-4 dark:bg-gray-900 sm:p-6 lg:p-8">
    <!-- Header -->
    <div class="mb-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold text-gray-900 dark:text-white">
            {{ systemName }}
          </h1>
          <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
            {{ t('diagnosis.subtitle') }}
          </p>
        </div>
        
        <div class="flex items-center gap-2">
          <span class="flex items-center gap-2 rounded-lg bg-green-50 px-3 py-2 text-sm font-medium text-green-700 dark:bg-green-900/20 dark:text-green-400">
            <span class="relative flex h-2 w-2">
              <span class="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
              <span class="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
            </span>
            {{ t('status.online') }}
          </span>
        </div>
      </div>
    </div>

    <!-- Main Grid -->
    <div class="grid gap-6 lg:grid-cols-3">
      <!-- Left Column - Sensors & Topology -->
      <div class="space-y-6 lg:col-span-2">
        <!-- Sensors Grid -->
        <SensorGrid :sensors="sensors" :loading="isLoading" />

        <!-- System Topology -->
        <SystemTopology :system-id="systemId" :loading="isLoading" />

        <!-- Anomaly Timeline -->
        <AnomalyTimeline
          :anomalies="anomalies"
          :loading="isLoading"
          @select-anomaly="handleAnomalySelect"
        />
      </div>

      <!-- Right Column - RAG Chat -->
      <div class="lg:col-span-1">
        <div class="sticky top-6 h-[calc(100vh-8rem)]">
          <RagChat :system-id="systemId" :system-name="systemName" />
        </div>
      </div>
    </div>

    <!-- Selected Anomaly Details Modal (optional) -->
    <div
      v-if="selectedAnomaly"
      class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      @click="selectedAnomaly = null"
    >
      <div
        class="max-w-2xl rounded-lg bg-white p-6 shadow-xl dark:bg-gray-800"
        @click.stop
      >
        <div class="flex items-start justify-between">
          <h3 class="text-xl font-bold text-gray-900 dark:text-white">
            {{ t('diagnosis.anomalyDetails') }}
          </h3>
          <button
            @click="selectedAnomaly = null"
            class="rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <Icon name="heroicons:x-mark" class="h-5 w-5" />
          </button>
        </div>
        
        <div class="mt-4 space-y-4">
          <div>
            <span class="text-sm font-medium text-gray-500 dark:text-gray-400">{{ t('diagnosis.description') }}</span>
            <p class="mt-1 text-gray-900 dark:text-white">{{ selectedAnomaly.description }}</p>
          </div>
          
          <div v-if="selectedAnomaly.metadata?.recommendedAction">
            <span class="text-sm font-medium text-gray-500 dark:text-gray-400">{{ t('diagnosis.recommendedAction') }}</span>
            <p class="mt-1 text-gray-900 dark:text-white">{{ selectedAnomaly.metadata.recommendedAction }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
