<template>
  <div class="level-6-sensor-mapping">
    <h2 class="text-xl font-semibold mb-4">6. Привязка датчиков к компонентам</h2>
    
    <p class="text-industrial-600 dark:text-industrial-400 mb-6">
      Загрузите CSV файл с данными для автоматического определения датчиков,
      или добавьте привязки вручную.
    </p>

    <!-- Auto-detect Section -->
    <BaseCard class="mb-6" title="Автоопределение датчиков">
      <div class="space-y-4">
        <div class="flex items-center gap-4">
          <input
            ref="fileInput"
            type="file"
            accept=".csv"
            class="hidden"
            @change="handleFileUpload"
          />
          <BaseButton
            @click="$refs.fileInput.click()"
            :loading="detecting"
            icon="heroicons:arrow-up-tray"
          >
            Загрузить CSV для автоопределения
          </BaseButton>
          
          <span v-if="uploadedFileName" class="text-sm text-industrial-600">
            {{ uploadedFileName }}
          </span>
        </div>
        
        <div v-if="detectionError" class="text-sm text-red-600">
          ❌ {{ detectionError }}
        </div>
      </div>
    </BaseCard>

    <!-- Suggestions Section -->
    <BaseCard v-if="suggestions.length > 0" class="mb-6" title="Предложения автоопределения">
      <div class="space-y-3">
        <div
          v-for="suggestion in suggestions"
          :key="suggestion.sensor_id"
          class="flex items-center justify-between p-3 bg-industrial-50 dark:bg-industrial-900 rounded-lg"
        >
          <div class="flex items-center gap-3 flex-1">
            <StatusBadge
              :status="suggestion.confidence > 0.7 ? 'operational' : 'degraded'"
              :label="`${(suggestion.confidence * 100).toFixed(0)}%`"
              size="sm"
            />
            
            <div class="flex-1">
              <div class="font-medium text-sm">
                {{ suggestion.sensor_id }}
                <Icon name="heroicons:arrow-right" class="inline mx-2 text-industrial-400" />
                {{ suggestion.component_name }}
              </div>
              <div class="text-xs text-industrial-500">
                {{ suggestion.sensor_type }} ({{ suggestion.expected_range_min }}-{{ suggestion.expected_range_max }} {{ suggestion.unit }})
              </div>
            </div>
          </div>
          
          <div class="flex gap-2">
            <BaseButton
              v-if="!suggestion.accepted"
              size="sm"
              variant="success"
              @click="acceptSuggestion(suggestion)"
            >
              Принять
            </BaseButton>
            <BaseButton
              v-if="!suggestion.accepted"
              size="sm"
              variant="secondary"
              @click="editSuggestion(suggestion)"
            >
              Изменить
            </BaseButton>
            <span v-else class="text-sm text-green-600 flex items-center gap-1">
              <Icon name="heroicons:check-circle" />
              Принято
            </span>
          </div>
        </div>
      </div>
    </BaseCard>

    <!-- Current Mappings -->
    <BaseCard title="Текущие привязки датчиков">
      <template #actions>
        <BaseButton
          size="sm"
          variant="secondary"
          icon="heroicons:plus"
          @click="showManualAddModal = true"
        >
          Добавить вручную
        </BaseButton>
      </template>
      
      <div v-if="mappings.length === 0" class="text-center py-8 text-industrial-500">
        <Icon name="heroicons:inbox" class="w-12 h-12 mx-auto mb-2 opacity-50" />
        <p>Датчики еще не привязаны</p>
        <p class="text-sm mt-1">Загрузите CSV или добавьте вручную</p>
      </div>
      
      <div v-else class="space-y-2">
        <div
          v-for="mapping in mappings"
          :key="mapping.id"
          class="flex items-center justify-between p-3 border border-industrial-200 dark:border-industrial-700 rounded-lg"
        >
          <div class="flex items-center gap-3">
            <StatusBadge
              status="operational"
              :label="mapping.sensor_type"
              size="sm"
            />
            <div>
              <div class="font-medium text-sm">{{ mapping.sensor_id }}</div>
              <div class="text-xs text-industrial-500">{{ mapping.component_name }}</div>
            </div>
          </div>
          
          <BaseButton
            size="sm"
            variant="ghost"
            icon="heroicons:trash"
            @click="deleteMapping(mapping.id)"
          />
        </div>
      </div>
    </BaseCard>

    <!-- Coverage Warning -->
    <div v-if="unmappedComponents.length > 0" class="mt-4">
      <div class="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4">
        <div class="flex items-start gap-3">
          <Icon name="heroicons:exclamation-triangle" class="w-5 h-5 text-orange-500 flex-shrink-0 mt-0.5" />
          <div>
            <h4 class="font-medium text-orange-900 dark:text-orange-200">Неполное покрытие датчиками</h4>
            <p class="text-sm text-orange-700 dark:text-orange-300 mt-1">
              {{ unmappedComponents.length }} компонентов без датчиков:
              {{ unmappedComponents.map(c => c.id).join(', ') }}
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- Manual Add Modal -->
    <Teleport to="body">
      <div
        v-if="showManualAddModal"
        class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
        @click="showManualAddModal = false"
      >
        <BaseCard
          class="w-full max-w-md"
          @click.stop
          title="Добавить датчик вручную"
        >
          <form @submit.prevent="addManualMapping" class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-1">Компонент</label>
              <select
                v-model="manualForm.component_index"
                required
                class="w-full px-3 py-2 border border-industrial-300 dark:border-industrial-600 rounded-lg"
              >
                <option value="">Выберите компонент</option>
                <option
                  v-for="(comp, idx) in store.wizardState.system.components"
                  :key="idx"
                  :value="idx"
                >
                  {{ comp.id }} ({{ comp.component_type }})
                </option>
              </select>
            </div>
            
            <div>
              <label class="block text-sm font-medium mb-1">ID датчика</label>
              <input
                v-model="manualForm.sensor_id"
                type="text"
                required
                placeholder="P1-001"
                class="w-full px-3 py-2 border border-industrial-300 dark:border-industrial-600 rounded-lg"
              />
            </div>
            
            <div>
              <label class="block text-sm font-medium mb-1">Тип датчика</label>
              <select
                v-model="manualForm.sensor_type"
                required
                class="w-full px-3 py-2 border border-industrial-300 dark:border-industrial-600 rounded-lg"
              >
                <option value="pressure">Давление</option>
                <option value="temperature">Температура</option>
                <option value="flow_rate">Расход</option>
                <option value="vibration">Вибрация</option>
              </select>
            </div>
            
            <div class="flex gap-2 justify-end">
              <BaseButton
                type="button"
                variant="secondary"
                @click="showManualAddModal = false"
              >
                Отмена
              </BaseButton>
              <BaseButton type="submit">
                Добавить
              </BaseButton>
            </div>
          </form>
        </BaseCard>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { useMetadataStore } from '~/stores/metadata'
import { useToast } from '~/composables/useToast'
import { useErrorHandler } from '~/composables/useErrorHandler'

const store = useMetadataStore()
const toast = useToast()
const errorHandler = useErrorHandler()
const { get, post, delete: del } = useApi()

const fileInput = ref<HTMLInputElement>()
const detecting = ref(false)
const uploadedFileName = ref('')
const detectionError = ref('')
const suggestions = ref<any[]>([])
const mappings = ref<any[]>([])
const showManualAddModal = ref(false)

const manualForm = reactive({
  component_index: '',
  sensor_id: '',
  sensor_type: 'pressure'
})

const unmappedComponents = computed(() => {
  const components = store.wizardState.system.components || []
  const mappedIndices = new Set(mappings.value.map(m => m.component_index))
  return components.filter((_, idx) => !mappedIndices.has(idx))
})

// Load existing mappings on mount
onMounted(async () => {
  await loadMappings()
})

async function loadMappings() {
  const equipmentId = store.wizardState.system.equipment_id
  if (!equipmentId) return
  
  const response = await get(`/api/sensor-mappings/equipment/${equipmentId}`)
  if (isApiSuccess(response)) {
    mappings.value = response.data
  }
}

async function handleFileUpload(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file) return
  
  uploadedFileName.value = file.name
  detectionError.value = ''
  detecting.value = true
  
  try {
    // Parse CSV columns
    const text = await file.text()
    const lines = text.split('\n')
    const headers = lines[0].split(',')
    const availableSensors = headers.filter(h => h.trim() !== 'timestamp')
    
    // Call auto-detect API
    const equipmentId = store.wizardState.system.equipment_id
    const response = await post(
      `/api/sensor-mappings/equipment/${equipmentId}/auto-detect`,
      { available_sensors: availableSensors }
    )
    
    if (isApiSuccess(response)) {
      suggestions.value = response.data.suggestions.map((s: any) => ({
        ...s,
        accepted: false
      }))
      
      toast.success(
        `Найдено ${response.data.matched_sensors} совпадений из ${response.data.total_sensors} датчиков`
      )
    } else {
      detectionError.value = response.error.message
    }
  } catch (error) {
    errorHandler.handleApiError(error, 'Автоопределение датчиков')
  } finally {
    detecting.value = false
  }
}

async function acceptSuggestion(suggestion: any) {
  const equipmentId = store.wizardState.system.equipment_id
  
  const response = await post('/api/sensor-mappings/', {
    equipment_id: equipmentId,
    component_index: suggestion.component_index,
    sensor_id: suggestion.sensor_id,
    sensor_type: suggestion.sensor_type,
    expected_range_min: suggestion.expected_range_min,
    expected_range_max: suggestion.expected_range_max,
    unit: suggestion.unit
  })
  
  if (isApiSuccess(response)) {
    suggestion.accepted = true
    await loadMappings()
    toast.success(`Датчик ${suggestion.sensor_id} привязан`)
  } else {
    errorHandler.handleApiError(response, 'Создание привязки')
  }
}

function editSuggestion(suggestion: any) {
  manualForm.component_index = suggestion.component_index
  manualForm.sensor_id = suggestion.sensor_id
  manualForm.sensor_type = suggestion.sensor_type
  showManualAddModal.value = true
}

async function addManualMapping() {
  const equipmentId = store.wizardState.system.equipment_id
  const component = store.wizardState.system.components[manualForm.component_index]
  
  const response = await post('/api/sensor-mappings/', {
    equipment_id: equipmentId,
    component_index: parseInt(manualForm.component_index),
    sensor_id: manualForm.sensor_id,
    sensor_type: manualForm.sensor_type,
    expected_range_min: component.normal_ranges[manualForm.sensor_type]?.min || 0,
    expected_range_max: component.normal_ranges[manualForm.sensor_type]?.max || 100,
    unit: component.normal_ranges[manualForm.sensor_type]?.unit || 'bar'
  })
  
  if (isApiSuccess(response)) {
    await loadMappings()
    showManualAddModal.value = false
    toast.success('Датчик добавлен вручную')
    
    // Reset form
    manualForm.component_index = ''
    manualForm.sensor_id = ''
    manualForm.sensor_type = 'pressure'
  } else {
    errorHandler.handleApiError(response, 'Добавление датчика')
  }
}

async function deleteMapping(id: string) {
  const response = await del(`/api/sensor-mappings/${id}`)
  
  if (isApiSuccess(response)) {
    await loadMappings()
    toast.success('Привязка удалена')
  } else {
    errorHandler.handleApiError(response, 'Удаление привязки')
  }
}
</script>

<style scoped>
.level-6-sensor-mapping {
  @apply p-4;
}
</style>