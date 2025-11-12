/**
 * EquipmentSettings.vue - Equipment settings management tab
 *
 * Features:
 * - Edit equipment info
 * - Monitoring, alert, GNN settings
 * - Danger zone actions
 * - Strict TypeScript
 * - i18n integration
 * - English comments
 * - Consistent loading/error patterns
 * - Dark mode support
 */
<script setup lang="ts">
import { ref, onMounted } from 'vue'
import type { HydraulicSystem } from '~/types/api'

interface Props {
  equipmentId: string
}
const props = defineProps<Props>()
const { t } = useI18n()
const api = useApi()
const toast = useToast()
const router = useRouter()

const isLoading = ref(true)
const isSaving = ref(false)
const settings = ref<any>(null)
const originalSettings = ref<any>(null)

const equipmentTypeOptions = [
  { value: 'excavator', label: t('equipment.types.excavator', 'Excavator') },
  { value: 'press', label: t('equipment.types.press', 'Hydraulic Press') },
  { value: 'crane', label: t('equipment.types.crane', 'Crane') },
  { value: 'loader', label: t('equipment.types.loader', 'Loader') },
  { value: 'pump', label: t('equipment.types.pump', 'Hydraulic Pump') },
  { value: 'other', label: t('equipment.types.other', 'Other') }
]
const intervalOptions = [
  { value: '10s', label: t('equipment.settings.interval.10s', 'Every 10 seconds') },
  { value: '30s', label: t('equipment.settings.interval.30s', 'Every 30 seconds') },
  { value: '1m', label: t('equipment.settings.interval.1m', 'Every 1 minute') },
  { value: '5m', label: t('equipment.settings.interval.5m', 'Every 5 minutes') },
  { value: '15m', label: t('equipment.settings.interval.15m', 'Every 15 minutes') }
]
const retentionOptions = [
  { value: '7d', label: t('equipment.settings.retention.7d', '7 days') },
  { value: '30d', label: t('equipment.settings.retention.30d', '30 days') },
  { value: '90d', label: t('equipment.settings.retention.90d', '90 days') },
  { value: '1y', label: t('equipment.settings.retention.1y', '1 year') },
  { value: 'forever', label: t('equipment.settings.retention.forever', 'Forever') }
]
const thresholdOptions = [
  { value: 'low', label: t('equipment.settings.threshold.low', 'Low (more alerts)') },
  { value: 'medium', label: t('equipment.settings.threshold.medium', 'Medium (balanced)') },
  { value: 'high', label: t('equipment.settings.threshold.high', 'High (fewer alerts)') }
]
const gnnFrequencyOptions = [
  { value: '15m', label: t('equipment.settings.gnn.15m', 'Every 15 minutes') },
  { value: '30m', label: t('equipment.settings.gnn.30m', 'Every 30 minutes') },
  { value: '1h', label: t('equipment.settings.gnn.1h', 'Every hour') },
  { value: '6h', label: t('equipment.settings.gnn.6h', 'Every 6 hours') },
  { value: '24h', label: t('equipment.settings.gnn.24h', 'Once per day') }
]

async function loadSettings() {
  isLoading.value = true
  try {
    const { data } = await api.get<{ equipment: any }>(`/api/equipment/${props.equipmentId}`)
    settings.value = { ...data }
    originalSettings.value = { ...data }
  } catch (err: any) {
    toast.add({
      title: t('equipment.settings.loadError', 'Failed to load settings'),
      description: err.message,
      color: 'red'
    })
  } finally {
    isLoading.value = false
  }
}
async function saveSettings() {
  isSaving.value = true
  try {
    await api.patch(`/api/equipment/${props.equipmentId}`, settings.value)
    toast.add({
      title: t('equipment.settings.saved', 'Settings saved'),
      description: t('equipment.settings.savedDesc', 'Equipment settings updated successfully'),
      color: 'green'
    })
    originalSettings.value = { ...settings.value }
  } catch (err: any) {
    toast.add({
      title: t('equipment.settings.saveError', 'Failed to save settings'),
      description: err.message,
      color: 'red'
    })
  } finally {
    isSaving.value = false
  }
}
function resetSettings() {
  if (originalSettings.value) {
    settings.value = { ...originalSettings.value }
    toast.add({
      title: t('equipment.settings.reset', 'Changes reset'),
      description: t('equipment.settings.resetDesc', 'All changes have been discarded'),
      color: 'blue'
    })
  }
}
async function deactivateEquipment() {
  const confirmed = confirm(t('equipment.settings.deactivateConfirm', 'Deactivate this equipment? Monitoring will stop but data will be preserved.'))
  if (!confirmed) return
  try {
    await api.patch(`/api/equipment/${props.equipmentId}`, { status: 'inactive' })
    toast.add({
      title: t('equipment.settings.deactivated', 'Equipment deactivated'),
      description: t('equipment.settings.deactivatedDesc', 'Monitoring has been stopped'),
      color: 'yellow'
    })
    router.push('/equipment')
  } catch (err: any) {
    toast.add({
      title: t('equipment.settings.deactivateError', 'Failed to deactivate'),
      description: err.message,
      color: 'red'
    })
  }
}
async function deleteEquipment() {
  const confirmed = confirm(t('equipment.settings.deleteConfirm', 'DELETE this equipment? This will permanently delete all data including sensors, readings, and history. This action CANNOT be undone!'))
  if (!confirmed) return
  const doubleConfirm = confirm(t('equipment.settings.deleteDouble', 'Are you ABSOLUTELY sure? Type YES in your mind and click OK to proceed.'))
  if (!doubleConfirm) return
  try {
    await api.delete(`/api/equipment/${props.equipmentId}`)
    toast.add({
      title: t('equipment.settings.deleted', 'Equipment deleted'),
      description: t('equipment.settings.deletedDesc', 'All data has been permanently deleted'),
      color: 'red'
    })
    router.push('/equipment')
  } catch (err: any) {
    toast.add({
      title: t('equipment.settings.deleteError', 'Failed to delete'),
      description: err.message,
      color: 'red'
    })
  }
}
onMounted(loadSettings)
</script>
