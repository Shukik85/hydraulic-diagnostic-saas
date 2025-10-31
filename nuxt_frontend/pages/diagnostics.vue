<template>
  <div class="space-y-8">
    <!-- ...content omitted for brevity... -->

    <!-- Run Diagnostic Modal -->
    <ClientOnly>
      <URunDiagnosticModal
        v-model="showRunModal"
        :loading="isRunning"
        @submit="startDiagnostic"
        @cancel="showRunModal = false"
      />
    </ClientOnly>

    <!-- Results Modal -->
    <ClientOnly>
      <UModal
        v-model="showResultsModal"
        :title="computedResultsTitle"
        :description="t('diagnostics.results.subtitle')"
        size="xl"
      >
        <div v-if="selectedResult" class="space-y-6">
          <!-- Summary Cards -->
          <div class="grid gap-4 sm:grid-cols-3">
            <div class="u-card p-4 text-center">
              <div class="text-2xl font-bold text-green-600">
                {{ selectedResult.score }}/100
              </div>
              <p class="u-body-sm text-gray-500">{{ t('diagnostics.healthScore') }}</p>
            </div>
            <div class="u-card p-4 text-center">
              <div class="text-2xl font-bold text-gray-900">
                {{ selectedResult.issuesFound }}
              </div>
              <p class="u-body-sm text-gray-500">{{ t('diagnostics.issuesFound') }}</p>
            </div>
            <div class="u-card p-4 text-center">
              <div class="text-2xl font-bold text-gray-900">
                {{ selectedResult.duration }}
              </div>
              <p class="u-body-sm text-gray-500">{{ t('diagnostics.analysisDuration') }}</p>
            </div>
          </div>

          <!-- Recommendations -->
          <div class="u-card p-4 sm:p-6">
            <h4 class="u-h5 mb-4">Рекомендации</h4>
            <div class="space-y-4">
              <div class="p-4 border border-yellow-200 bg-yellow-50 rounded-lg">
                <div class="flex items-start gap-3">
                  <Icon name="heroicons:exclamation-triangle" class="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                  <div class="min-w-0">
                    <p class="font-medium text-yellow-800">
                      {{ t('diagnostics.recommendations.pressureMaintenance') }}
                    </p>
                    <p class="u-body-sm text-yellow-700 mt-1">
                      {{ t('diagnostics.recommendations.pressureMaintenanceDesc') }}
                    </p>
                    <p class="text-xs text-yellow-600 mt-2">
                      {{ t('diagnostics.priority') }}: {{ t('diagnostics.priorityMedium') }}
                    </p>
                  </div>
                </div>
              </div>

              <div class="p-4 border border-green-200 bg-green-50 rounded-lg">
                <div class="flex items-start gap-3">
                  <Icon name="heroicons:check-circle" class="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                  <div class="min-w-0">
                    <p class="font-medium text-green-800">
                      {{ t('diagnostics.recommendations.temperatureMonitoring') }}
                    </p>
                    <p class="u-body-sm text-green-700 mt-1">
                      {{ t('diagnostics.recommendations.temperatureMonitoringDesc') }}
                    </p>
                    <p class="text-xs text-green-600 mt-2">
                      {{ t('diagnostics.statusLabel') }}: {{ t('diagnostics.statusNormal') }}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <template #footer>
          <button @click="showResultsModal = false" class="u-btn u-btn-secondary flex-1">
            {{ t('ui.close') }}
          </button>
          <button class="u-btn u-btn-primary flex-1">
            <Icon name="heroicons:arrow-down-tray" class="w-4 h-4 mr-2" />
            {{ t('diagnostics.exportPDF') }}
          </button>
        </template>
      </UModal>
    </ClientOnly>
  </div>
</template>

<script setup lang="ts">
// ...previous script...

const computedResultsTitle = computed(() => {
  if (!selectedResult.value) return t('diagnostics.results.title')
  return t('diagnostics.results.titleWithName', { title: t('diagnostics.results.title'), name: selectedResult.value.name })
})
</script>
