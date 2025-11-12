import { defineStore } from 'pinia'
import type { SystemMetadata, SensorMapping, WizardState } from '~/types/metadata'

export const useMetadataStore = defineStore('metadata', () => {
  // State
  const wizardState = ref<WizardState>({
    current_level: 1,
    completed_levels: [],
    system: {
      components: [],
      adjacency_matrix: [],
      observed_problems: [],
      completeness: 0,
      // NEW: equipment_id
      equipment_id: '',
      duty_cycle: undefined
    },
    incompleteness_report: {
      critical_missing: [],
      secondary_missing: [],
      inferred_values: {}
    }
  })
  // NEW: sensor mappings
  const sensor_mappings = ref<SensorMapping[]>([])

  // ... предыдущие геттеры (completeness, currentLevelValid, componentsCount)
  // ... actions: goToLevel, completeLevel, updateBasicInfo, addComponent, ...
  // добавить loadSensorMappings, addSensorMapping, deleteSensorMapping
  async function loadSensorMappings() {
    const equipmentId = wizardState.value.system.equipment_id
    if (!equipmentId) return
    const response = await useApi().get(`/api/sensor-mappings/equipment/${equipmentId}`)
    if (isApiSuccess(response)) {
      sensor_mappings.value = response.data
    }
  }
  async function addSensorMapping(mapping: Partial<SensorMapping>) {
    const response = await useApi().post('/api/sensor-mappings/', mapping)
    if (isApiSuccess(response)) {
      await loadSensorMappings()
    }
  }
  async function deleteSensorMapping(mappingId: string) {
    const response = await useApi().delete(`/api/sensor-mappings/${mappingId}`)
    if (isApiSuccess(response)) {
      await loadSensorMappings()
    }
  }

  // Validation: все основные компоненты должны иметь хотя бы один mapping
  const unmappedComponents = computed(() => {
    const components = wizardState.value.system.components || []
    const mappedIndices = new Set(sensor_mappings.value.map(m => m.component_index))
    return components.filter((_, idx) => !mappedIndices.has(idx))
  })
  const isSensorMappingComplete = computed(() => unmappedComponents.value.length === 0)

  return {
    wizardState,
    sensor_mappings,
    loadSensorMappings,
    addSensorMapping,
    deleteSensorMapping,
    unmappedComponents,
    isSensorMappingComplete,
    // ... все старые методы, геттеры и actions
  }
})
