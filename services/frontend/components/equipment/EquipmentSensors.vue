/**
 * EquipmentSensors.vue - Equipment sensors management tab
 *
 * Features:
 * - Table view of equipment sensors
 * - Status indicators (online/offline)
 * - Last reading display
 * - Mapping status badge
 * - Unified actions (view, edit, delete)
 * - Add sensor modal
 * - Strict TypeScript
 * - i18n, dark mode
 * - Consistent error handling, empty/loading states
 */
<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import type { SensorMapping } from '~/types/api'

interface Props {
  equipmentId: string
}
const props = defineProps<Props>()
const { t } = useI18n()
const api = useApi()
const toast = useToast()

const isLoading = ref(true)
const isSaving = ref(false)
const isAddModalOpen = ref(false)
const sensors = ref<SensorMapping[]>([])
const components = ref<any[]>([])

const newSensor = ref({
  name: '',
  sensor_id: '',
  type: '',
  unit: 'bar',
  component_id: null as string | null
})

const columns = [
  { key: 'name', label: t('equipment.sensors.columns.name', 'Sensor') },
  { key: 'type', label: t('equipment.sensors.columns.type', 'Type') },
  { key: 'status', label: t('equipment.sensors.columns.status', 'Status') },
  { key: 'lastReading', label: t('equipment.sensors.columns.lastReading', 'Last Reading') },
  { key: 'mapping', label: t('equipment.sensors.columns.mapping', 'Mapping') },
  { key: 'actions', label: '' }
]

const sensorTypeOptions = [
  { value: 'pressure', label: t('equipment.sensors.type.pressure', 'Pressure') },
  { value: 'temperature', label: t('equipment.sensors.type.temperature', 'Temperature') },
  { value: 'flow', label: t('equipment.sensors.type.flow', 'Flow Rate') },
  { value: 'level', label: t('equipment.sensors.type.level', 'Level') },
  { value: 'vibration', label: t('equipment.sensors.type.vibration', 'Vibration') },
  { value: 'position', label: t('equipment.sensors.type.position', 'Position') }
]
const componentOptions = computed(() =>
  components.value.map((c: any) => ({ value: c.id, label: c.name }))
)

/**
 * Load equipment sensors and component options
 */
async function loadData() {
  isLoading.value = true
  try {
    // Fetch sensors
    const { data: mappings } = await api.get<{ mappings: SensorMapping[] }>(`/api/sensor-mappings?equipment_id=${props.equipmentId}`)
    sensors.value = mappings || []
    // Fetch component metadata
    const { data: systems } = await api.get<any>(`/api/metadata/systems?equipment_id=${props.equipmentId}`)
    components.value = systems?.system?.components || []
  } catch (err: any) {
    toast.add({
      title: t('equipment.sensors.loadError', 'Failed to load sensors'),
      description: err.message,
      color: 'red'
    })
  } finally {
    isLoading.value = false
  }
}

/**
 * Open modal to add new sensor
 */
function openAddSensorModal() {
  newSensor.value = {
    name: '',
    sensor_id: '',
    type: '',
    unit: 'bar',
    component_id: null
  }
  isAddModalOpen.value = true
}

/**
 * Save newly created sensor
 */
async function saveSensor() {
  if (!newSensor.value.name || !newSensor.value.sensor_id || !newSensor.value.type) {
    toast.add({
      title: t('ui.validationError', 'Validation error'),
      description: t('equipment.sensors.validationRequired', 'Please fill all required fields'),
      color: 'yellow'
    })
    return
  }
  isSaving.value = true
  try {
    await api.post('/api/sensor-mappings', {
      equipment_id: props.equipmentId,
      ...newSensor.value
    })
    toast.add({
      title: t('equipment.sensors.sensorAdded', 'Sensor added'),
      description: t('equipment.sensors.sensorAddedDesc', `${newSensor.value.name} has been added`),
      color: 'green'
    })
    isAddModalOpen.value = false
    await loadData()
  } catch (err: any) {
    toast.add({
      title: t('equipment.sensors.addError', 'Failed to add sensor'),
      description: err.message,
      color: 'red'
    })
  } finally {
    isSaving.value = false
  }
}

function viewSensorData(sensor: SensorMapping) {
  navigateTo(`/sensors/${sensor.id}`)
}
function editSensor(sensor: SensorMapping) {
  // TODO: Open edit modal
  console.log('Edit sensor:', sensor)
}
async function deleteSensor(sensor: SensorMapping) {
  const confirmed = confirm(t('equipment.sensors.deleteConfirm', `Delete sensor "${sensor.name}"?`))
  if (!confirmed) return
  try {
    await api.delete(`/api/sensor-mappings/${sensor.id}`)
    toast.add({
      title: t('equipment.sensors.sensorDeleted', 'Sensor deleted'),
      description: t('equipment.sensors.sensorDeletedDesc', `${sensor.name} has been removed`),
      color: 'green'
    })
    await loadData()
  } catch (err: any) {
    toast.add({
      title: t('equipment.sensors.deleteError', 'Failed to delete sensor'),
      description: err.message,
      color: 'red'
    })
  }
}

// Helpers for icons/colors/labels - unify with style guide
function getSensorIcon(type: string): string {
  const icons: Record<string, string> = {
    pressure: 'i-heroicons-arrow-trending-up',
    temperature: 'i-heroicons-fire',
    flow: 'i-heroicons-beaker',
    level: 'i-heroicons-signal',
    vibration: 'i-heroicons-bolt',
    position: 'i-heroicons-map-pin'
  }
  return icons[type] || 'i-heroicons-cpu-chip'
}
function getSensorIconBg(type: string): string {
  const colors: Record<string, string> = {
    pressure: 'bg-blue-100 dark:bg-blue-900/30',
    temperature: 'bg-red-100 dark:bg-red-900/30',
    flow: 'bg-green-100 dark:bg-green-900/30',
    level: 'bg-purple-100 dark:bg-purple-900/30',
    vibration: 'bg-yellow-100 dark:bg-yellow-900/30',
    position: 'bg-indigo-100 dark:bg-indigo-900/30'
  }
  return colors[type] || 'bg-gray-100 dark:bg-gray-800'
}
function getSensorIconColor(type: string): string {
  const colors: Record<string, string> = {
    pressure: 'text-blue-600 dark:text-blue-400',
    temperature: 'text-red-600 dark:text-red-400',
    flow: 'text-green-600 dark:text-green-400',
    level: 'text-purple-600 dark:text-purple-400',
    vibration: 'text-yellow-600 dark:text-yellow-400',
    position: 'text-indigo-600 dark:text-indigo-400'
  }
  return colors[type] || 'text-gray-600 dark:text-gray-400'
}
function formatSensorType(type: string): string {
  return type.charAt(0).toUpperCase() + type.slice(1)
}
function formatTime(timestamp: string | number): string {
  const date = new Date(timestamp)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  if (diffMins < 1) return t('ui.justNow', 'Just now')
  if (diffMins < 60) return `${diffMins}m ${t('ui.ago', 'ago')}`
  if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ${t('ui.ago', 'ago')}`
  return date.toLocaleDateString()
}
onMounted(loadData)
</script>
