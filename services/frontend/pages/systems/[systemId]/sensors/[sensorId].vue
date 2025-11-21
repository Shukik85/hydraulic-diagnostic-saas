<template>
  <div class="space-y-8">
    <!-- Header -->
    <div>
      <h1 class="u-h2">{{ t('sensors.detail.title') }}</h1>
      <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
        {{ t('breadcrumbs.system') }} #{{ route.params.systemId }} / {{ t('breadcrumbs.sensors') }} #{{
          route.params.sensorId }}
      </p>
    </div>

    <!-- Tabs Navigation -->
    <div class="flex space-x-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg w-fit">
      <button v-for="tab in tabs" :key="tab.id" @click="activeTab = tab.id"
        class="px-4 py-2 text-sm font-medium rounded-md u-transition-fast"
        :class="activeTab === tab.id ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm' : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'">
        <Icon :name="tab.icon" class="w-4 h-4 mr-2 inline" />
        {{ tab.name }}
      </button>
    </div>

    <!-- Overview Tab -->
    <div v-if="activeTab === 'overview'" class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="u-card p-6">
        <h3 class="u-h4 mb-4">{{ t('sensors.detail.metadata') }}</h3>
        <div class="space-y-4">
          <div class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-400">{{ t('sensors.table.name') }}:</span>
            <span class="font-medium">Sensor #{{ route.params.sensorId }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-400">{{ t('sensors.table.type') }}:</span>
            <span class="font-medium">Pressure Sensor</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-400">{{ t('sensors.table.status') }}:</span>
            <span class="u-badge u-badge-success">Active</span>
          </div>
        </div>
      </div>

      <div class="u-card p-6">
        <h3 class="u-h4 mb-4">{{ t('sensors.detail.currentReadings') }}</h3>
        <div class="text-center py-8">
          <Icon name="heroicons:chart-bar" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p class="text-gray-500 dark:text-gray-400">Data stream placeholder</p>
        </div>
      </div>
    </div>

    <!-- Data Stream Tab -->
    <div v-if="activeTab === 'dataStream'" class="u-card p-6">
      <h3 class="u-h4 mb-6">{{ t('sensors.detail.dataStream') }}</h3>
      <div class="text-center py-16">
        <Icon name="heroicons:signal" class="w-20 h-20 text-gray-400 mx-auto mb-6" />
        <p class="text-lg text-gray-500 dark:text-gray-400 mb-2">Real-time data visualization</p>
        <p class="text-gray-400">Connect to API to display live sensor data</p>
      </div>
    </div>

    <!-- Thresholds Tab -->
    <div v-if="activeTab === 'thresholds'" class="u-card p-6">
      <h3 class="u-h4 mb-6">{{ t('sensors.detail.thresholds') }}</h3>
      <div class="text-center py-16">
        <Icon name="heroicons:adjustments-horizontal" class="w-20 h-20 text-gray-400 mx-auto mb-6" />
        <p class="text-lg text-gray-500 dark:text-gray-400 mb-2">Alert thresholds configuration</p>
        <p class="text-gray-400">Set warning and critical limits</p>
      </div>
    </div>

    <!-- Event Log Tab -->
    <div v-if="activeTab === 'eventLog'" class="u-card p-6">
      <h3 class="u-h4 mb-6">{{ t('sensors.detail.eventLog') }}</h3>
      <div class="text-center py-16">
        <Icon name="heroicons:document-text" class="w-20 h-20 text-gray-400 mx-auto mb-6" />
        <p class="text-lg text-gray-500 dark:text-gray-400 mb-2">Sensor event history</p>
        <p class="text-gray-400">View alerts, warnings, and status changes</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

definePageMeta({
  layout: 'dashboard',
  middleware: ['auth']
})

const { t } = useI18n()
const route = useRoute()
const activeTab = ref('overview')

const tabs = [
  { id: 'overview', name: t('sensors.detail.overview'), icon: 'heroicons:information-circle' },
  { id: 'dataStream', name: t('sensors.detail.dataStream'), icon: 'heroicons:signal' },
  { id: 'thresholds', name: t('sensors.detail.thresholds'), icon: 'heroicons:adjustments-horizontal' },
  { id: 'eventLog', name: t('sensors.detail.eventLog'), icon: 'heroicons:document-text' }
]
</script>
