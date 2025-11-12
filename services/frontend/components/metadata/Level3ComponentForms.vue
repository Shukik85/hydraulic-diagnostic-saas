<!-- components/metadata/Level3ComponentForms.vue -->
<template>
  <div class="level-3 space-y-6">
    <div>
      <h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
        {{ $t('wizard.level3.title') }}
      </h2>
      <p class="text-sm text-gray-600 dark:text-gray-400">
        {{ $t('wizard.level3.description') }}
      </p>
    </div>

    <!-- No components warning -->
    <UAlert
      v-if="store.componentsCount === 0"
      color="yellow"
      icon="i-heroicons-exclamation-triangle"
      :title="$t('wizard.level3.noComponents')"
      :description="$t('wizard.level3.noComponentsWarning')"
    />

    <template v-else>
      <!-- Component selector -->
      <UCard class="p-6">
        <UFormGroup :label="$t('wizard.level3.selectComponent')">
          <USelect
            v-model="selectedComponentId"
            :options="componentOptions"
            placeholder="-- Select --"
            size="lg"
          >
            <template #leading>
              <UIcon :name="getComponentIcon(selectedComponent?.component_type)" class="w-5 h-5" />
            </template>
          </USelect>
        </UFormGroup>
      </UCard>

      <!-- Dynamic form -->
      <Transition name="fade" mode="out-in">
        <component
          v-if="selectedComponentId && currentFormComponent"
          :is="currentFormComponent"
          :component-id="selectedComponentId"
          :key="selectedComponentId"
        />
        
        <UCard v-else class="p-12">
          <div class="text-center">
            <UIcon
              name="i-heroicons-cog-6-tooth"
              class="w-16 h-16 text-gray-400 dark:text-gray-600 mx-auto mb-4"
            />
            <p class="text-gray-500 dark:text-gray-400">
              {{ $t('wizard.level3.emptyState') }}
            </p>
          </div>
        </UCard>
      </Transition>

      <!-- Progress summary -->
      <UCard class="p-6">
        <h3 class="text-base font-semibold text-gray-900 dark:text-gray-100 mb-4">
          {{ $t('wizard.level3.completeness') }}
        </h3>
        
        <div class="space-y-3">
          <div
            v-for="comp in store.wizardState.system.components"
            :key="comp.id"
            class="flex items-center gap-3"
          >
            <UIcon
              :name="getComponentIcon(comp.component_type)"
              class="w-5 h-5 text-gray-600 dark:text-gray-400"
            />
            <span class="text-sm font-medium text-gray-900 dark:text-gray-100 w-32">
              {{ comp.id }}
            </span>
            <UProgress
              :value="getComponentCompleteness(comp) * 100"
              :color="getCompletenessColor(getComponentCompleteness(comp))"
              size="sm"
              class="flex-1"
            />
            <span class="text-xs font-semibold text-gray-600 dark:text-gray-400 w-12 text-right">
              {{ Math.round(getComponentCompleteness(comp) * 100) }}%
            </span>
          </div>
        </div>
      </UCard>
    </template>
  </div>
</template>

<script setup lang="ts">
import { useMetadataStore } from '~/stores/metadata'
import type { ComponentMetadata, ComponentType } from '~/types/metadata'
import PumpForm from '~/components/metadata/Level3ComponentForms/PumpForm.vue'
import MotorForm from '~/components/metadata/Level3ComponentForms/MotorForm.vue'
import CylinderForm from '~/components/metadata/Level3ComponentForms/CylinderForm.vue'
import ValveForm from '~/components/metadata/Level3ComponentForms/ValveForm.vue'
import FilterForm from '~/components/metadata/Level3ComponentForms/FilterForm.vue'
import AccumulatorForm from '~/components/metadata/Level3ComponentForms/AccumulatorForm.vue'

const { t } = useI18n()
const store = useMetadataStore()

const selectedComponentId = ref<string>('')

onMounted(() => {
  const firstComponent = store.wizardState.system.components?.[0]
  if (firstComponent) {
    selectedComponentId.value = firstComponent.id
  }
})

const selectedComponent = computed(() =>
  store.wizardState.system.components?.find(c => c.id === selectedComponentId.value)
)

const componentOptions = computed(() => {
  return store.wizardState.system.components?.map(comp => ({
    value: comp.id,
    label: getComponentLabel(comp)
  })) || []
})

const currentFormComponent = computed(() => {
  if (!selectedComponent.value) return null

  const formComponents: Record<ComponentType, any> = {
    pump: PumpForm,
    motor: MotorForm,
    cylinder: CylinderForm,
    valve: ValveForm,
    filter: FilterForm,
    accumulator: AccumulatorForm,
  }

  return formComponents[selectedComponent.value.component_type]
})

function getComponentLabel(comp: ComponentMetadata): string {
  const typeLabel = t(`wizard.level3.componentTypes.${comp.component_type}`)
  return `${getComponentEmoji(comp.component_type)} ${typeLabel} — ${comp.id}`
}

function getComponentEmoji(type: ComponentType): string {
  const emojis: Record<ComponentType, string> = {
    pump: '⚡',
    motor: '🔄',
    cylinder: '⬌',
    valve: '⬥',
    filter: '◈',
    accumulator: '⬢'
  }
  return emojis[type] || '⚙️'
}

function getComponentIcon(type?: ComponentType): string {
  const icons: Record<ComponentType, string> = {
    pump: 'i-heroicons-bolt',
    motor: 'i-heroicons-arrow-path',
    cylinder: 'i-heroicons-arrows-right-left',
    valve: 'i-heroicons-adjustments-horizontal',
    filter: 'i-heroicons-funnel',
    accumulator: 'i-heroicons-battery-100'
  }
  return icons[type || 'pump'] || 'i-heroicons-cog-6-tooth'
}

function getComponentCompleteness(comp: ComponentMetadata): number {
  let filled = 0
  let total = 5

  if (comp.max_pressure) filled++
  if (comp.normal_ranges.pressure) filled++
  if (comp.normal_ranges.temperature) filled++

  if (comp.component_type === 'pump' && comp.pump_specific?.nominal_flow_rate) filled++
  if (comp.component_type === 'motor' && comp.motor_specific?.displacement) filled++
  if (comp.component_type === 'cylinder' && comp.cylinder_specific?.piston_diameter) filled++

  if (comp.last_maintenance) filled++

  return filled / total
}

function getCompletenessColor(completeness: number): string {
  if (completeness < 0.3) return 'red'
  if (completeness < 0.7) return 'yellow'
  return 'green'
}
</script>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
