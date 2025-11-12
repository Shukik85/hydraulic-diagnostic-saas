/**
 * EquipmentDataSources.vue - Equipment data sources tab
 *
 * Features:
 * - List of all data sources by type
 * - Status badge
 * - Last sync time and record count
 * - Add, edit, delete actions
 * - Strict TypeScript
 * - i18n for all text
 * - English comments
 * - Consistent empty/loading/error states
 * - Dark mode
 */
<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

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
const syncingIds = ref<string[]>([])
const dataSources = ref<any[]>([])
const newSource = ref({
  name: '',
  type: '',
  file_path: '',
  endpoint: '',
  polling_interval: '30s',
  config: { gateway_url: '' }
})
const sourceTypeOptions = [
  { value: 'csv', label: t('equipment.dataSources.type.csv', 'CSV File') },
  { value: 'api', label: t('equipment.dataSources.type.api', 'REST API') },
  { value: 'iot', label: t('equipment.dataSources.type.iot', 'IoT Gateway') },
  { value: 'simulator', label: t('equipment.dataSources.type.simulator', 'Data Simulator') }
]
const pollingOptions = [
  { value: '10s', label: t('equipment.dataSources.polling.10s', 'Every 10 seconds') },
  { value: '30s', label: t('equipment.dataSources.polling.30s', 'Every 30 seconds') },
  { value: '1m', label: t('equipment.dataSources.polling.1m', 'Every 1 minute') },
  { value: '5m', label: t('equipment.dataSources.polling.5m', 'Every 5 minutes') },
  { value: 'manual', label: t('equipment.dataSources.polling.manual', 'Manual only') }
]
async function loadData() {
  isLoading.value = true
  try {
    const { data } = await api.get<{ sources: any[] }>(`/api/data-sources?equipment_id=${props.equipmentId}`)
    dataSources.value = data || []
  } catch (err: any) {
    toast.add({
      title: t('equipment.dataSources.loadError', 'Failed to load data sources'),
      description: err.message,
      color: 'red'
    })
  } finally {
    isLoading.value = false
  }
}
function openAddSourceModal() {
  newSource.value = {
    name: '',
    type: '',
    file_path: '',
    endpoint: '',
    polling_interval: '30s',
    config: { gateway_url: '' }
  }
  isAddModalOpen.value = true
}
async function saveSource() {
  if (!newSource.value.name || !newSource.value.type) {
    toast.add({
      title: t('ui.validationError', 'Validation error'),
      description: t('equipment.dataSources.validationRequired', 'Please fill all required fields'),
      color: 'yellow'
    })
    return
  }
  isSaving.value = true
  try {
    await api.post('/api/data-sources', { equipment_id: props.equipmentId, ...newSource.value })
    toast.add({
      title: t('equipment.dataSources.sourceAdded', 'Data source added'),
      description: t('equipment.dataSources.sourceAddedDesc', `${newSource.value.name} has been added`),
      color: 'green'
    })
    isAddModalOpen.value = false
    await loadData()
  } catch (err: any) {
    toast.add({
      title: t('equipment.dataSources.addError', 'Failed to add data source'),
      description: err.message,
      color: 'red'
    })
  } finally {
    isSaving.value = false
  }
}
async function syncSource(source: any) {
  syncingIds.value.push(source.id)
  try {
    await api.post(`/api/data-sources/${source.id}/sync`)
    toast.add({
      title: t('equipment.dataSources.syncStarted', 'Sync started'),
      description: t('equipment.dataSources.syncStartedDesc', `${source.name} is syncing`),
      color: 'blue'
    })
    setTimeout(() => loadData(), 2000)
  } catch (err: any) {
    toast.add({
      title: t('equipment.dataSources.syncError', 'Sync failed'),
      description: err.message,
      color: 'red'
    })
  } finally {
    syncingIds.value = syncingIds.value.filter(id => id !== source.id)
  }
}
async function deleteSource(source: any) {
  const confirmed = confirm(t('equipment.dataSources.deleteConfirm', `Delete data source "${source.name}"?`))
  if (!confirmed) return
  try {
    await api.delete(`/api/data-sources/${source.id}`)
    toast.add({
      title: t('equipment.dataSources.sourceDeleted', 'Data source deleted'),
      description: t('equipment.dataSources.sourceDeletedDesc', `${source.name} has been removed`),
      color: 'green'
    })
    await loadData()
  } catch (err: any) {
    toast.add({
      title: t('equipment.dataSources.deleteError', 'Failed to delete'),
      description: err.message,
      color: 'red'
    })
  }
}
function getSourceIcon(type: string): string {
  const icons: Record<string, string> = {
    csv: 'i-heroicons-document-text',
    api: 'i-heroicons-cloud',
    iot: 'i-heroicons-wifi',
    simulator: 'i-heroicons-beaker'
  }
  return icons[type] || 'i-heroicons-server'
}
function getSourceIconBg(type: string): string {
  const colors: Record<string, string> = {
    csv: 'bg-blue-100 dark:bg-blue-900/30',
    api: 'bg-purple-100 dark:bg-purple-900/30',
    iot: 'bg-green-100 dark:bg-green-900/30',
    simulator: 'bg-yellow-100 dark:bg-yellow-900/30'
  }
  return colors[type] || 'bg-gray-100 dark:bg-gray-800'
}
function getSourceIconColor(type: string): string {
  const colors: Record<string, string> = {
    csv: 'text-blue-600 dark:text-blue-400',
    api: 'text-purple-600 dark:text-purple-400',
    iot: 'text-green-600 dark:text-green-400',
    simulator: 'text-yellow-600 dark:text-yellow-400'
  }
  return colors[type] || 'text-gray-600 dark:text-gray-400'
}
function formatSourceType(type: string): string {
  const labels: Record<string, string> = {
    csv: t('equipment.dataSources.type.csv', 'CSV File'),
    api: t('equipment.dataSources.type.api', 'REST API'),
    iot: t('equipment.dataSources.type.iot', 'IoT Gateway'),
    simulator: t('equipment.dataSources.type.simulator', 'Data Simulator')
  }
  return labels[type] || type
}
function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
  return num.toString()
}
function formatTime(timestamp: string | number | null): string {
  if (!timestamp) return t('ui.never', 'Never')
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
