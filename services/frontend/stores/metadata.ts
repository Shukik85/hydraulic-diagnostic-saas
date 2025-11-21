// stores/metadata.ts - Pinia store для метаданных гидросистемы

import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import type {
  SystemMetadata,
  ComponentMetadata,
  WizardState,
  ValidationResult
} from '~/types/metadata'

export const useMetadataStore = defineStore('metadata', () => {
  // State
  const data = ref<any>(null)
  
  const wizardState = ref<WizardState>({
    current_level: 1,
    completed_levels: [],
    system: {
      components: [],
      adjacency_matrix: [],
      observed_problems: [],
      completeness: 0
    },
    incompleteness_report: {
      critical_missing: [],
      secondary_missing: [],
      inferred_values: {}
    }
  })

  // Getters (Computed)
  const completeness = computed(() => {
    return calculateCompleteness(wizardState.value.system)
  })

  const currentLevelValid = computed(() => {
    const level = wizardState.value.current_level
    return validateLevel(level, wizardState.value.system)
  })

  const componentsCount = computed(() => {
    return wizardState.value.system.components?.length || 0
  })

  // Helper: Calculate completeness percentage
  function calculateCompleteness(system: Partial<SystemMetadata>): number {
    let totalFields = 0
    let filledFields = 0

    // Level 1: Basic info (5 полей)
    const basicFields = ['equipment_type', 'manufacturer', 'model', 'serial_number', 'manufacture_date']
    basicFields.forEach(field => {
      totalFields++
      if (system[field as keyof SystemMetadata]) filledFields++
    })

    // Level 2: Components (минимум 1 компонент)
    totalFields += 1
    if (system.components && system.components.length > 0) filledFields++

    // Level 3: Component details
    const components = system.components || []
    components.forEach(comp => {
      totalFields += 5
      if (comp.component_type) filledFields++
      if (comp.max_pressure) filledFields++
      if (comp.normal_ranges.pressure) filledFields++
      if (comp.normal_ranges.temperature) filledFields++
      if (comp.connected_to.length > 0) filledFields++
    })

    // Level 4: Duty cycle
    totalFields += 1
    if (system.duty_cycle) filledFields++

    // Level 5: Diagnostics
    totalFields += 1
    if (system.observed_problems && system.observed_problems.length > 0) filledFields++

    return totalFields > 0 ? Math.round((filledFields / totalFields) * 100) : 0
  }

  // Helper: Validate specific level
  function validateLevel(level: number, system: Partial<SystemMetadata>): boolean {
    switch (level) {
      case 1:
        return !!(system.equipment_type && system.manufacturer && system.serial_number)
      case 2:
        return !!(system.components && system.components.length > 0)
      case 3:
        return system.components?.every(c =>
          c.component_type && c.max_pressure && c.normal_ranges.pressure
        ) || false
      case 4:
        return !!system.duty_cycle
      case 5:
        return true // Всегда можно перейти к финальной валидации
      default:
        return false
    }
  }

  // Actions
  function processMatrix(matrix: number[][]) {
    for (let i = 0; i < matrix.length; i++) {
      const row = matrix[i]
      if (!row) continue
      
      for (let j = 0; j < row.length; j++) {
        // Enterprise: безопасный доступ с проверкой
        if (row[j] !== undefined) {
          row[j] = 1
        }
      }
    }
  }

  function goToLevel(level: number) {
    if (level >= 1 && level <= 5) {
      wizardState.value.current_level = level
    }
  }

  function completeLevel(level: number) {
    if (!wizardState.value.completed_levels.includes(level)) {
      wizardState.value.completed_levels.push(level)
    }
  }

  function updateBasicInfo(data: Partial<SystemMetadata>) {
    wizardState.value.system = {
      ...wizardState.value.system,
      ...data,
      last_updated: new Date().toISOString()
    }
  }

  function addComponent(component: ComponentMetadata) {
    if (!wizardState.value.system.components) {
      wizardState.value.system.components = []
    }
    wizardState.value.system.components.push(component)
    updateAdjacencyMatrix()
  }

  function updateComponent(id: string, updates: Partial<ComponentMetadata>) {
    const index = wizardState.value.system.components?.findIndex(c => c.id === id)
    if (index !== undefined && index >= 0 && wizardState.value.system.components) {
      wizardState.value.system.components[index] = {
        ...wizardState.value.system.components[index],
        ...updates
      } as ComponentMetadata
    }
  }

  function removeComponent(id: string) {
    if (!wizardState.value.system.components) return

    wizardState.value.system.components = wizardState.value.system.components.filter(
      c => c.id !== id
    )

    // Удаляем связи с этим компонентом
    wizardState.value.system.components.forEach(c => {
      c.connected_to = c.connected_to.filter(connId => connId !== id)
    })

    updateAdjacencyMatrix()
  }

  function addConnection(
    fromId: string,
    toId: string,
    type: 'pressure_line' | 'return_line' | 'pilot_line'
  ) {
    const component = wizardState.value.system.components?.find(c => c.id === fromId)
    if (component) {
      if (!component.connected_to.includes(toId)) {
        component.connected_to.push(toId)
      }
      component.connection_types[toId] = type
      updateAdjacencyMatrix()
    }
  }

  function updateAdjacencyMatrix() {
    const components = wizardState.value.system.components || []
    const n = components.length
    const matrix: number[][] = Array(n).fill(0).map(() => Array(n).fill(0))

    components.forEach((comp, i) => {
      comp.connected_to.forEach(targetId => {
        const j = components.findIndex(c => c.id === targetId)
        // ✅ Optional chaining для безопасности
        if (j >= 0 && matrix[i] && matrix[i][j] !== undefined) {
          matrix[i][j] = 1
        }
      })
    })

    wizardState.value.system.adjacency_matrix = matrix
  }

  function inferMissingValues() {
    const components = wizardState.value.system.components || []
    const inferred: Record<string, any> = {}

    components.forEach(comp => {
      if (!comp) return
      
      // Инфер normal_ranges если известно max_pressure
      if (comp.max_pressure && !comp.normal_ranges.pressure) {
        const inferredRange = {
          min: Math.round(comp.max_pressure * 0.5),
          max: Math.round(comp.max_pressure * 0.85),
          unit: 'bar'
        }
        comp.normal_ranges.pressure = inferredRange

        inferred[`${comp.id}.normal_ranges.pressure`] = {
          value: inferredRange,
          method: 'inferred_from_max_pressure',
          confidence: 0.7
        }
      }

      // Инфер temperature ranges для насосов
      if (comp.component_type === 'pump' && !comp.normal_ranges.temperature) {
        const inferredTemp = {
          min: 40,
          max: 70,
          unit: '°C'
        }
        comp.normal_ranges.temperature = inferredTemp

        inferred[`${comp.id}.normal_ranges.temperature`] = {
          value: inferredTemp,
          method: 'standard_pump_temperature',
          confidence: 0.75
        }
      }

      // Инфер efficiency для аксиально-поршневых насосов
      if (comp.component_type === 'pump' && comp.pump_specific?.pump_type === 'axial_piston') {
        if (!comp.pump_specific.volumetric_efficiency) {
          comp.pump_specific.volumetric_efficiency = 0.96
          inferred[`${comp.id}.volumetric_efficiency`] = {
            value: 0.96,
            method: 'typical_axial_piston',
            confidence: 0.7
          }
        }
        if (!comp.pump_specific.mechanical_efficiency) {
          comp.pump_specific.mechanical_efficiency = 0.94
          inferred[`${comp.id}.mechanical_efficiency`] = {
            value: 0.94,
            method: 'typical_axial_piston',
            confidence: 0.7
          }
        }
      }
    })

    wizardState.value.incompleteness_report.inferred_values = inferred
  }

  function validateConsistency(): ValidationResult[] {
    const errors: ValidationResult[] = []
    const components = wizardState.value.system.components || []

    // Проверка: давление насоса >= давления всех цилиндров
    const pumps = components.filter(c => c.component_type === 'pump')
    const cylinders = components.filter(c => c.component_type === 'cylinder')

    if (pumps.length > 0 && cylinders.length > 0) {
      const maxPumpPressure = Math.max(...pumps.map(p => p.max_pressure || 0))
      const maxCylinderPressure = Math.max(...cylinders.map(c => c.max_pressure || 0))

      if (maxCylinderPressure > maxPumpPressure) {
        errors.push({
          valid: false,
          error: 'Давление цилиндра превышает давление насоса',
          suggestion: `Увеличьте max_pressure насоса до ${maxCylinderPressure} бар или больше`
        })
      }
    }

    // Проверка: расход фильтра >= расхода насоса
    const filters = components.filter(c => c.component_type === 'filter')
    if (pumps.length > 0 && filters.length > 0) {
      const maxPumpFlow = Math.max(...pumps.map(p => p.pump_specific?.nominal_flow_rate || 0))

      filters.forEach(filter => {
        const filterFlow = filter.filter_specific?.flow_capacity || 0
        if (filterFlow < maxPumpFlow) {
          errors.push({
            valid: false,
            error: `Фильтр ${filter.id} имеет недостаточную пропускную способность`,
            suggestion: `Увеличьте flow_capacity до ${maxPumpFlow} л/мин или больше`
          })
        }
      })
    }

    return errors
  }

  async function submitMetadata() {
    // Финальный инфер перед отправкой
    inferMissingValues()

    // Валидация
    const errors = validateConsistency()
    if (errors.length > 0) {
      console.warn('Обнаружены ошибки консистентности:', errors)
    }

    // Отправка на бэкенд
    try {
      const response = await $fetch('/api/ml/graph/metadata/upload', {
        method: 'POST',
        body: {
          system: wizardState.value.system,
          incompleteness_report: wizardState.value.incompleteness_report
        }
      })

      return { success: true, data: response }
    } catch (error) {
      console.error('Ошибка отправки метаданных:', error)
      return { success: false, error }
    }
  }

  // Persistence (LocalStorage)
  function saveToLocalStorage() {
    if (process.client) {
      localStorage.setItem('metadata_wizard_state', JSON.stringify(wizardState.value))
    }
  }

  function loadFromLocalStorage() {
    if (process.client) {
      const saved = localStorage.getItem('metadata_wizard_state')
      if (saved) {
        wizardState.value = JSON.parse(saved)
      }
    }
  }

  // Watch для auto-save
  watch(wizardState, () => {
    saveToLocalStorage()
  }, { deep: true })

  return {
    // State
    data,
    wizardState,

    // Getters (Computed)
    completeness,
    currentLevelValid,
    componentsCount,

    // Actions
    processMatrix,
    goToLevel,
    completeLevel,
    updateBasicInfo,
    addComponent,
    updateComponent,
    removeComponent,
    addConnection,
    updateAdjacencyMatrix,
    inferMissingValues,
    validateConsistency,
    submitMetadata,
    loadFromLocalStorage
  }
})
