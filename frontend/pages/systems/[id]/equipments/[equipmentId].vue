<template>
  <div class="space-y-8">
    <!-- Header -->
    <div>
      <h1 class="u-h2">{{ t('equipments.detail.title') }}</h1>
      <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
        {{ t('breadcrumbs.system') }} #{{ route.params.id }} / {{ t('breadcrumbs.equipment') }} #{{ route.params.equipmentId }}
      </p>
    </div>

    <!-- Tabs Navigation -->
    <div class="flex space-x-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg w-fit">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        @click="activeTab = tab.id"
        class="px-4 py-2 text-sm font-medium rounded-md u-transition-fast"
        :class="activeTab === tab.id ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm' : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'"
      >
        <Icon :name="tab.icon" class="w-4 h-4 mr-2 inline" />
        {{ tab.name }}
      </button>
    </div>

    <!-- Overview Tab -->
    <div v-if="activeTab === 'overview'" class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="u-card p-6">
        <h3 class="u-h4 mb-4">{{ t('equipments.detail.specifications') }}</h3>
        <div class="space-y-4">
          <div class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-400">Equipment ID:</span>
            <span class="font-medium">#{{ route.params.equipmentId }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-400">Type:</span>
            <span class="font-medium">Hydraulic Pump</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-400">Status:</span>
            <span class="u-badge u-badge-success">Active</span>
          </div>
        </div>
      </div>
      
      <div class="u-card p-6">
        <h3 class="u-h4 mb-4">{{ t('equipments.detail.lastMaintenance') }}</h3>
        <div class="text-center py-8">
          <Icon name="heroicons:wrench-screwdriver" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p class="text-gray-500 dark:text-gray-400">Maintenance history placeholder</p>
        </div>
      </div>
    </div>

    <!-- Sensors Tab -->
    <div v-if="activeTab === 'sensors'" class="u-card p-6">
      <h3 class="u-h4 mb-6">{{ t('equipments.detail.attachedSensors') }}</h3>
      <div class="text-center py-16">
        <Icon name="heroicons:cpu-chip" class="w-20 h-20 text-gray-400 mx-auto mb-6" />
        <p class="text-lg text-gray-500 dark:text-gray-400 mb-2">Connected sensors overview</p>
        <p class="text-gray-400">List of sensors monitoring this equipment</p>
      </div>
    </div>

    <!-- Diagnostics Tab -->
    <div v-if="activeTab === 'diagnostics'" class="u-card p-6">
      <h3 class="u-h4 mb-6">{{ t('equipments.detail.recentDiagnostics') }}</h3>
      <div class="text-center py-16">
        <Icon name="heroicons:chart-pie" class="w-20 h-20 text-gray-400 mx-auto mb-6" />
        <p class="text-lg text-gray-500 dark:text-gray-400 mb-2">Diagnostic history and results</p>
        <p class="text-gray-400">Performance analysis and health metrics</p>
      </div>
    </div>

    <!-- Maintenance Tab -->
    <div v-if="activeTab === 'maintenance'" class="u-card p-6">
      <h3 class="u-h4 mb-6">{{ t('equipments.detail.maintenance') }}</h3>
      <div class="text-center py-16">
        <Icon name="heroicons:calendar-days" class="w-20 h-20 text-gray-400 mx-auto mb-6" />
        <p class="text-lg text-gray-500 dark:text-gray-400 mb-2">Maintenance schedule and history</p>
        <p class="text-gray-400">Planned and completed maintenance tasks</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  layout: 'dashboard',
  middleware: ['auth']
})

const { t } = useI18n()
const route = useRoute()
const activeTab = ref('overview')

const tabs = [
  { id: 'overview', name: t('equipments.detail.overview'), icon: 'heroicons:information-circle' },
  { id: 'sensors', name: t('equipments.detail.sensors'), icon: 'heroicons:cpu-chip' },
  { id: 'diagnostics', name: t('equipments.detail.diagnostics'), icon: 'heroicons:chart-pie' },
  { id: 'maintenance', name: t('equipments.detail.maintenance'), icon: 'heroicons:wrench-screwdriver' }
]
</script>
