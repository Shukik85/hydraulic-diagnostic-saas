<template>
  <div class="simulator-setup">
    <h2 class="text-xl font-semibold mb-4">Симулятор данных (Demo/Dev)</h2>
    <BaseCard padding="lg">
      <form @submit.prevent="startSimulation" class="space-y-4">
        <div>
          <label class="block text-sm font-medium mb-1">Сценарий</label>
          <select v-model="scenario" class="w-full px-3 py-2 border rounded-lg">
            <option value="normal">Норма</option>
            <option value="degradation">Деградация</option>
            <option value="failure">Отказ</option>
            <option value="cyclic">Цикл</option>
          </select>
        </div>
        <div class="flex gap-4">
          <div class="flex-1">
            <label class="block text-sm font-medium mb-1">Длительность (сек)</label>
            <input type="number" v-model.number="duration" min="10" max="1800" class="w-full px-3 py-2 border rounded-lg" />
          </div>
          <div class="flex-1">
            <label class="block text-sm font-medium mb-1">Уровень шума</label>
            <input type="number" step="0.01" min="0" max="1" v-model.number="noiseLevel" class="w-full px-3 py-2 border rounded-lg" />
          </div>
          <div class="flex-1">
            <label class="block text-sm font-medium mb-1">Частота (Hz)</label>
            <input type="number" v-model.number="samplingRate" min="1" max="100" class="w-full px-3 py-2 border rounded-lg" />
          </div>
        </div>
        <BaseButton type="submit" variant="primary" :loading="simulating">Запустить симуляцию</BaseButton>
      </form>
      <div v-if="simId" class="mt-6">
        <UAlert color="blue">Статус симуляции: {{ simulationStatus }}</UAlert>
        <BaseButton class="mt-4" variant="danger" :loading="stopping" @click="stopSimulation">Остановить</BaseButton>
      </div>
    </BaseCard>
  </div>
</template>

<script setup lang="ts">
import { useMetadataStore } from '~/stores/metadata'
const store = useMetadataStore()
const { post, get } = useApi()
const toast = useToast()
const errorHandler = useErrorHandler()

const scenario = ref('normal')
const duration = ref(300)
const noiseLevel = ref(0.1)
const samplingRate = ref(10)
const simId = ref<string|null>(null)
const simulationStatus = ref('')
const simStatusInterval = ref<NodeJS.Timer>()
const simulating = ref(false)
const stopping = ref(false)

async function startSimulation() {
  simulating.value = true
  try {
    const equipmentId = store.wizardState.system.equipment_id
    const response = await post('/api/simulator/start', {
      equipment_id: equipmentId,
      scenario: scenario.value,
      duration: duration.value,
      noise_level: noiseLevel.value,
      sampling_rate: samplingRate.value
    })
    if (isApiSuccess(response)) {
      simId.value = response.data.simulation_id
      simulationStatus.value = response.data.status
      toast.success('Симуляция запущена')
      simStatusInterval.value = setInterval(checkStatus, 2000)
    } else {
      errorHandler.handleApiError(response, 'Запуск симулятора')
    }
  } finally {
    simulating.value = false
  }
}

async function checkStatus() {
  if (!simId.value) return
  const response = await get(`/api/simulator/status/${simId.value}`)
  if (isApiSuccess(response)) {
    simulationStatus.value = response.data.status
  }
}

async function stopSimulation() {
  stopping.value = true
  try {
    const response = await post(`/api/simulator/stop/${simId.value}`)
    if (isApiSuccess(response)) {
      simulationStatus.value = 'stopped'
      toast.success('Симуляция остановлена')
      clearInterval(simStatusInterval.value)
    } else {
      errorHandler.handleApiError(response, 'Остановка симуляции')
    }
  } finally {
    stopping.value = false
  }
}
</script>

<style scoped>
.simulator-setup {
  @apply p-4;
}
</style>