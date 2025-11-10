<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="u-h2">{{ t('sensors.title') }}</h1>
        <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
          {{ t('sensors.subtitle') }} - {{ t('breadcrumbs.system') }} #{{ route.params.id }}
        </p>
      </div>
      <button class="u-btn u-btn-primary u-btn-md">
        <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
        {{ t('sensors.addSensor') }}
      </button>
    </div>

    <!-- Sensors Table -->
    <div class="u-card">
      <div class="p-6">
        <div class="overflow-x-auto">
          <table class="u-table">
            <thead>
              <tr>
                <th>{{ t('sensors.table.name') }}</th>
                <th>{{ t('sensors.table.type') }}</th>
                <th>{{ t('sensors.table.status') }}</th>
                <th>{{ t('sensors.table.lastValue') }}</th>
                <th>{{ t('sensors.table.updatedAt') }}</th>
                <th>{{ t('sensors.table.actions') }}</th>
              </tr>
            </thead>
            <tbody>
              <!-- Empty state -->
              <tr v-if="!sensors.length">
                <td colspan="6" class="text-center py-12">
                  <div class="flex flex-col items-center gap-4">
                    <Icon name="heroicons:cpu-chip" class="w-12 h-12 text-gray-400" />
                    <div>
                      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">{{ t('sensors.noSensors') }}</h3>
                      <p class="text-gray-600 dark:text-gray-400 mt-1">{{ t('sensors.noSensorsDesc') }}</p>
                    </div>
                    <button class="u-btn u-btn-primary u-btn-md">
                      <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
                      {{ t('sensors.addSensor') }}
                    </button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
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

// Empty state for now - no API connection
const sensors = ref([])
</script>
