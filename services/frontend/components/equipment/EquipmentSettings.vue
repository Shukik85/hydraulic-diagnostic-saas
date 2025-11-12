<!--
  EquipmentSettings.vue — Таб с настройками оборудования
  
  Features:
  - Основные настройки (название, тип, производитель)
  - Настройки мониторинга
  - Настройки алертов
  - Настройки GNN
  - Danger zone (deactivate/delete)
-->
<template>
  <div class="equipment-settings space-y-6">
    <!-- Loading state -->
    <div v-if="isLoading" class="space-y-6">
      <USkeleton class="h-64 w-full" />
      <USkeleton class="h-48 w-full" />
      <USkeleton class="h-32 w-full" />
    </div>
    
    <template v-else>
      <!-- Basic Information -->
      <UCard class="p-6">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
          Basic Information
        </h3>
        
        <div class="space-y-4">
          <UFormGroup label="Equipment Name" required>
            <UInput
              v-model="settings.name"
              placeholder="e.g., Excavator XYZ-123"
            />
          </UFormGroup>
          
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <UFormGroup label="Equipment Type" required>
              <USelect
                v-model="settings.equipment_type"
                :options="equipmentTypeOptions"
              />
            </UFormGroup>
            
            <UFormGroup label="Manufacturer">
              <UInput
                v-model="settings.manufacturer"
                placeholder="e.g., Caterpillar"
              />
            </UFormGroup>
          </div>
          
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <UFormGroup label="Model">
              <UInput
                v-model="settings.model"
                placeholder="e.g., 320D"
              />
            </UFormGroup>
            
            <UFormGroup label="Serial Number">
              <UInput
                v-model="settings.serial_number"
                placeholder="e.g., CAT0012345"
              />
            </UFormGroup>
          </div>
          
          <UFormGroup label="Description">
            <UTextarea
              v-model="settings.description"
              :rows="3"
              placeholder="Optional description..."
            />
          </UFormGroup>
        </div>
      </UCard>
      
      <!-- Monitoring Settings -->
      <UCard class="p-6">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
          Monitoring Settings
        </h3>
        
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-gray-100">
                Enable Real-time Monitoring
              </p>
              <p class="text-sm text-gray-600 dark:text-gray-400">
                Continuously monitor sensor data
              </p>
            </div>
            <UToggle v-model="settings.monitoring_enabled" />
          </div>
          
          <UFormGroup label="Data Collection Interval">
            <USelect
              v-model="settings.collection_interval"
              :options="intervalOptions"
              :disabled="!settings.monitoring_enabled"
            />
          </UFormGroup>
          
          <UFormGroup label="Data Retention Period">
            <USelect
              v-model="settings.retention_period"
              :options="retentionOptions"
            />
          </UFormGroup>
        </div>
      </UCard>
      
      <!-- Alert Settings -->
      <UCard class="p-6">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
          Alert Settings
        </h3>
        
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-gray-100">
                Enable Anomaly Alerts
              </p>
              <p class="text-sm text-gray-600 dark:text-gray-400">
                Get notified when anomalies detected
              </p>
            </div>
            <UToggle v-model="settings.alerts_enabled" />
          </div>
          
          <UFormGroup label="Alert Threshold">
            <USelect
              v-model="settings.alert_threshold"
              :options="thresholdOptions"
              :disabled="!settings.alerts_enabled"
            />
          </UFormGroup>
          
          <UFormGroup label="Notification Channels">
            <div class="space-y-2">
              <UCheckbox
                v-model="settings.notification_email"
                label="Email notifications"
                :disabled="!settings.alerts_enabled"
              />
              <UCheckbox
                v-model="settings.notification_sms"
                label="SMS notifications"
                :disabled="!settings.alerts_enabled"
              />
              <UCheckbox
                v-model="settings.notification_webhook"
                label="Webhook notifications"
                :disabled="!settings.alerts_enabled"
              />
            </div>
          </UFormGroup>
        </div>
      </UCard>
      
      <!-- GNN Settings -->
      <UCard class="p-6">
        <div class="flex items-center gap-2 mb-6">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
            GNN Diagnostics
          </h3>
          <UBadge color="blue" variant="soft">
            <UIcon name="i-heroicons-sparkles" class="w-3 h-3" />
            AI-Powered
          </UBadge>
        </div>
        
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-gray-100">
                Auto-run GNN Analysis
              </p>
              <p class="text-sm text-gray-600 dark:text-gray-400">
                Automatically analyze data with Graph Neural Network
              </p>
            </div>
            <UToggle v-model="settings.gnn_auto_run" />
          </div>
          
          <UFormGroup label="Analysis Frequency">
            <USelect
              v-model="settings.gnn_frequency"
              :options="gnnFrequencyOptions"
              :disabled="!settings.gnn_auto_run"
            />
          </UFormGroup>
          
          <UFormGroup label="Confidence Threshold">
            <div class="space-y-2">
              <div class="flex items-center justify-between text-sm">
                <span class="text-gray-700 dark:text-gray-300">{{ settings.gnn_confidence_threshold }}%</span>
                <span class="text-gray-500 dark:text-gray-400">Higher = fewer false positives</span>
              </div>
              <input
                v-model.number="settings.gnn_confidence_threshold"
                type="range"
                min="50"
                max="99"
                class="w-full"
              />
            </div>
          </UFormGroup>
        </div>
      </UCard>
      
      <!-- Danger Zone -->
      <UCard class="p-6 border-red-200 dark:border-red-800">
        <h3 class="text-lg font-semibold text-red-600 dark:text-red-400 mb-6">
          Danger Zone
        </h3>
        
        <div class="space-y-4">
          <div class="flex items-center justify-between p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
            <div>
              <p class="font-medium text-gray-900 dark:text-gray-100">
                Deactivate Equipment
              </p>
              <p class="text-sm text-gray-600 dark:text-gray-400">
                Stop monitoring and data collection
              </p>
            </div>
            <UButton
              color="red"
              variant="outline"
              @click="deactivateEquipment"
            >
              Deactivate
            </UButton>
          </div>
          
          <div class="flex items-center justify-between p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
            <div>
              <p class="font-medium text-gray-900 dark:text-gray-100">
                Delete Equipment
              </p>
              <p class="text-sm text-gray-600 dark:text-gray-400">
                Permanently delete all data (cannot be undone)
              </p>
            </div>
            <UButton
              color="red"
              @click="deleteEquipment"
            >
              Delete Forever
            </UButton>
          </div>
        </div>
      </UCard>
      
      <!-- Save Actions -->
      <div class="flex justify-end gap-3 pt-4">
        <UButton
          color="gray"
          @click="resetSettings"
        >
          Reset Changes
        </UButton>
        <UButton
          color="primary"
          :loading="isSaving"
          @click="saveSettings"
        >
          Save Settings
        </UButton>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

interface Props {
  equipmentId: string
}

const props = defineProps<Props>()

const api = useApiAdvanced()
const toast = useToast()
const router = useRouter()

const isLoading = ref(true)
const isSaving = ref(false)

const settings = ref({
  name: '',
  equipment_type: '',
  manufacturer: '',
  model: '',
  serial_number: '',
  description: '',
  
  monitoring_enabled: true,
  collection_interval: '30s',
  retention_period: '90d',
  
  alerts_enabled: true,
  alert_threshold: 'medium',
  notification_email: true,
  notification_sms: false,
  notification_webhook: false,
  
  gnn_auto_run: true,
  gnn_frequency: '1h',
  gnn_confidence_threshold: 85
})

const originalSettings = ref<typeof settings.value | null>(null)

const equipmentTypeOptions = [
  { value: 'excavator', label: 'Excavator' },
  { value: 'press', label: 'Hydraulic Press' },
  { value: 'crane', label: 'Crane' },
  { value: 'loader', label: 'Loader' },
  { value: 'pump', label: 'Hydraulic Pump' },
  { value: 'other', label: 'Other' }
]

const intervalOptions = [
  { value: '10s', label: 'Every 10 seconds' },
  { value: '30s', label: 'Every 30 seconds' },
  { value: '1m', label: 'Every 1 minute' },
  { value: '5m', label: 'Every 5 minutes' },
  { value: '15m', label: 'Every 15 minutes' }
]

const retentionOptions = [
  { value: '7d', label: '7 days' },
  { value: '30d', label: '30 days' },
  { value: '90d', label: '90 days' },
  { value: '1y', label: '1 year' },
  { value: 'forever', label: 'Forever' }
]

const thresholdOptions = [
  { value: 'low', label: 'Low (more alerts)' },
  { value: 'medium', label: 'Medium (balanced)' },
  { value: 'high', label: 'High (fewer alerts)' }
]

const gnnFrequencyOptions = [
  { value: '15m', label: 'Every 15 minutes' },
  { value: '30m', label: 'Every 30 minutes' },
  { value: '1h', label: 'Every hour' },
  { value: '6h', label: 'Every 6 hours' },
  { value: '24h', label: 'Once per day' }
]

// Load settings
async function loadSettings() {
  isLoading.value = true
  
  try {
    const response = await api.get<any>(
      `/api/equipment/${props.equipmentId}`
    )
    
    const equipment = response.equipment
    settings.value = {
      name: equipment.name || '',
      equipment_type: equipment.equipment_type || '',
      manufacturer: equipment.manufacturer || '',
      model: equipment.model || '',
      serial_number: equipment.serial_number || '',
      description: equipment.description || '',
      
      monitoring_enabled: equipment.monitoring_enabled ?? true,
      collection_interval: equipment.collection_interval || '30s',
      retention_period: equipment.retention_period || '90d',
      
      alerts_enabled: equipment.alerts_enabled ?? true,
      alert_threshold: equipment.alert_threshold || 'medium',
      notification_email: equipment.notification_email ?? true,
      notification_sms: equipment.notification_sms ?? false,
      notification_webhook: equipment.notification_webhook ?? false,
      
      gnn_auto_run: equipment.gnn_auto_run ?? true,
      gnn_frequency: equipment.gnn_frequency || '1h',
      gnn_confidence_threshold: equipment.gnn_confidence_threshold || 85
    }
    
    // Store original for reset
    originalSettings.value = { ...settings.value }
    
  } catch (error: any) {
    toast.add({
      title: 'Failed to load settings',
      description: error.message,
      color: 'red'
    })
  } finally {
    isLoading.value = false
  }
}

// Actions
async function saveSettings() {
  isSaving.value = true
  
  try {
    await api.patch(`/api/equipment/${props.equipmentId}`, settings.value)
    
    toast.add({
      title: 'Settings saved',
      description: 'Equipment settings updated successfully',
      color: 'green'
    })
    
    // Update original
    originalSettings.value = { ...settings.value }
    
  } catch (error: any) {
    toast.add({
      title: 'Failed to save settings',
      description: error.message,
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
      title: 'Changes reset',
      description: 'All changes have been discarded',
      color: 'blue'
    })
  }
}

async function deactivateEquipment() {
  const confirmed = confirm(
    'Deactivate this equipment? Monitoring will stop but data will be preserved.'
  )
  if (!confirmed) return
  
  try {
    await api.patch(`/api/equipment/${props.equipmentId}`, {
      status: 'inactive'
    })
    
    toast.add({
      title: 'Equipment deactivated',
      description: 'Monitoring has been stopped',
      color: 'yellow'
    })
    
    router.push('/equipment')
  } catch (error: any) {
    toast.add({
      title: 'Failed to deactivate',
      description: error.message,
      color: 'red'
    })
  }
}

async function deleteEquipment() {
  const confirmed = confirm(
    'DELETE this equipment? This will permanently delete all data including sensors, readings, and history. This action CANNOT be undone!'
  )
  if (!confirmed) return
  
  // Double confirmation
  const doubleConfirm = confirm(
    'Are you ABSOLUTELY sure? Type YES in your mind and click OK to proceed.'
  )
  if (!doubleConfirm) return
  
  try {
    await api.delete(`/api/equipment/${props.equipmentId}`)
    
    toast.add({
      title: 'Equipment deleted',
      description: 'All data has been permanently deleted',
      color: 'red'
    })
    
    router.push('/equipment')
  } catch (error: any) {
    toast.add({
      title: 'Failed to delete',
      description: error.message,
      color: 'red'
    })
  }
}

// Lifecycle
onMounted(() => {
  loadSettings()
})
</script>
