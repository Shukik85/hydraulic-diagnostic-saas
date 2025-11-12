/**
 * EquipmentOverview - Basic equipment information display
 *
 * Features:
 * - Basic information card
 * - System architecture placeholder
 * - Components list placeholder
 * - Dark mode support
 * - i18n ready
 * - TypeScript strict
 *
 * @example
 * <EquipmentOverview :equipment="equipment" />
 */
<script setup lang="ts">
import type { HydraulicSystem } from '~/types/api'

interface Props {
  equipment: HydraulicSystem
}

const props = defineProps<Props>()
const { t } = useI18n()

/**
 * Map equipment status to display status
 */
function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    active: 'green',
    maintenance: 'yellow',
    inactive: 'gray'
  }
  return colors[status] || 'gray'
}

/**
 * Map equipment status to display label
 */
function getStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    active: t('equipment.status.operational', 'Operational'),
    maintenance: t('equipment.status.maintenance', 'Maintenance'),
    inactive: t('equipment.status.inactive', 'Inactive')
  }
  return labels[status] || status
}
</script>

<template>
  <div class="equipment-overview space-y-6">
    <!-- Basic Information -->
    <UCard>
      <template #header>
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {{ $t('equipment.overview.basicInfo', 'Basic Information') }}
        </h3>
      </template>

      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <p class="text-sm text-gray-600 dark:text-gray-400 mb-1">
            {{ $t('equipment.fields.id', 'ID') }}
          </p>
          <p class="font-medium text-gray-900 dark:text-gray-100">
            {{ equipment.id }}
          </p>
        </div>

        <div>
          <p class="text-sm text-gray-600 dark:text-gray-400 mb-1">
            {{ $t('equipment.fields.name', 'Name') }}
          </p>
          <p class="font-medium text-gray-900 dark:text-gray-100">
            {{ equipment.name }}
          </p>
        </div>

        <div>
          <p class="text-sm text-gray-600 dark:text-gray-400 mb-1">
            {{ $t('equipment.fields.type', 'Type') }}
          </p>
          <p class="font-medium text-gray-900 dark:text-gray-100">
            {{ equipment.type }}
          </p>
        </div>

        <div>
          <p class="text-sm text-gray-600 dark:text-gray-400 mb-1">
            {{ $t('equipment.fields.status', 'Status') }}
          </p>
          <UBadge :color="getStatusColor(equipment.status)" variant="soft">
            {{ getStatusLabel(equipment.status) }}
          </UBadge>
        </div>

        <div v-if="equipment.location">
          <p class="text-sm text-gray-600 dark:text-gray-400 mb-1">
            {{ $t('equipment.fields.location', 'Location') }}
          </p>
          <p class="font-medium text-gray-900 dark:text-gray-100">
            {{ equipment.location }}
          </p>
        </div>

        <div>
          <p class="text-sm text-gray-600 dark:text-gray-400 mb-1">
            {{ $t('equipment.fields.healthScore', 'Health Score') }}
          </p>
          <div class="flex items-center gap-2">
            <UProgress :value="equipment.health_score" :color="equipment.health_score > 80 ? 'green' : equipment.health_score > 60 ? 'yellow' : 'red'" />
            <span class="text-sm font-semibold text-gray-900 dark:text-gray-100">
              {{ equipment.health_score }}%
            </span>
          </div>
        </div>
      </div>
    </UCard>

    <!-- System Architecture Placeholder -->
    <UCard>
      <template #header>
        <div class="flex items-center gap-2">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {{ $t('equipment.overview.architecture', 'System Architecture') }}
          </h3>
          <UBadge color="yellow" variant="soft" size="xs">
            {{ $t('ui.comingSoon', 'Coming Soon') }}
          </UBadge>
        </div>
      </template>

      <div class="flex items-center justify-center py-12 text-center">
        <div>
          <UIcon name="i-heroicons-cube-transparent" class="w-12 h-12 text-gray-400 dark:text-gray-600 mx-auto mb-4" />
          <p class="text-sm text-gray-600 dark:text-gray-400">
            {{ $t('equipment.overview.graphPlaceholder', 'Component graph visualization will be displayed here') }}
          </p>
        </div>
      </div>
    </UCard>

    <!-- Components List Placeholder -->
    <UCard>
      <template #header>
        <div class="flex items-center gap-2">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {{ $t('equipment.overview.components', 'Components') }}
          </h3>
          <UBadge color="yellow" variant="soft" size="xs">
            {{ $t('ui.comingSoon', 'Coming Soon') }}
          </UBadge>
        </div>
      </template>

      <div class="flex items-center justify-center py-12 text-center">
        <div>
          <UIcon name="i-heroicons-cog-6-tooth" class="w-12 h-12 text-gray-400 dark:text-gray-600 mx-auto mb-4" />
          <p class="text-sm text-gray-600 dark:text-gray-400">
            {{ $t('equipment.overview.componentsPlaceholder', 'Components table will be displayed here') }}
          </p>
        </div>
      </div>
    </UCard>
  </div>
</template>
