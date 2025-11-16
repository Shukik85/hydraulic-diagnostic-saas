<template>
  <UModal 
    :model-value="modelValue" 
    @update:model-value="$emit('update:modelValue', $event)" 
    :title="t('diagnostics.runModal.title')" 
    :description="t('diagnostics.runModal.description')" 
    size="md"
  >
    <div class="space-y-5">
      <!-- Equipment Selection -->
      <div class="relative">
        <label class="u-label" for="equipment">{{ t('diagnostics.runModal.equipment') }} *</label>
        <select 
          id="equipment" 
          v-model="form.equipment" 
          class="u-input metallic-select" 
          :disabled="loading"
        >
          <option value="">{{ t('diagnostics.runModal.selectEquipment') }}</option>
          <option value="hyd-001">HYD-001 - {{ t('diagnostics.runModal.pumpStationA') }}</option>
          <option value="hyd-002">HYD-002 - {{ t('diagnostics.runModal.hydraulicMotorB') }}</option>
          <option value="hyd-003">HYD-003 - {{ t('diagnostics.runModal.controlValveC') }}</option>
          <option value="hyd-004">HYD-004 - {{ t('diagnostics.runModal.coolingSystemD') }}</option>
        </select>
        <Icon 
          name="heroicons:chevron-down" 
          class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-steel-light pointer-events-none" 
        />
        <Transition name="fade">
          <p v-if="errors.equipment" class="mt-2 text-sm text-error-500 flex items-center gap-1">
            <Icon name="heroicons:exclamation-circle" class="h-4 w-4 shrink-0" />
            {{ errors.equipment }}
          </p>
        </Transition>
      </div>

      <!-- Diagnostic Type -->
      <div class="relative">
        <label class="u-label" for="diagnostic-type">{{ t('diagnostics.runModal.diagnosticType') }}</label>
        <select 
          id="diagnostic-type" 
          v-model="form.type" 
          class="u-input metallic-select" 
          :disabled="loading"
        >
          <option value="full">{{ t('diagnostics.runModal.fullAnalysis') }}</option>
          <option value="pressure">{{ t('diagnostics.runModal.pressureCheck') }}</option>
          <option value="temperature">{{ t('diagnostics.runModal.temperatureAnalysis') }}</option>
          <option value="vibration">{{ t('diagnostics.runModal.vibrationAnalysis') }}</option>
          <option value="flow">{{ t('diagnostics.runModal.flowAnalysis') }}</option>
        </select>
        <Icon 
          name="heroicons:chevron-down" 
          class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-steel-light pointer-events-none" 
        />
      </div>

      <!-- Options -->
      <div class="space-y-3">
        <div class="flex items-center gap-2">
          <input 
            id="email-notification" 
            v-model="form.emailNotification" 
            type="checkbox" 
            class="w-4 h-4 text-primary-500 bg-steel-dark border-steel-medium rounded focus:ring-primary-500 focus:ring-2" 
          />
          <label for="email-notification" class="text-sm text-text-primary">
            {{ t('diagnostics.runModal.emailNotification') }}
          </label>
        </div>
        <div class="flex items-center gap-2">
          <input 
            id="priority-analysis" 
            v-model="form.priorityAnalysis" 
            type="checkbox" 
            class="w-4 h-4 text-primary-500 bg-steel-dark border-steel-medium rounded focus:ring-primary-500 focus:ring-2" 
          />
          <label for="priority-analysis" class="text-sm text-text-primary">
            {{ t('diagnostics.runModal.priorityAnalysis') }}
          </label>
        </div>
      </div>

      <!-- Estimated Duration -->
      <div class="rounded-lg bg-success-500/5 border border-success-500/30 p-4">
        <div class="flex items-start gap-3">
          <Icon name="heroicons:clock" class="h-5 w-5 text-success-500 mt-0.5 shrink-0" />
          <div>
            <p class="text-sm font-medium text-success-900">{{ t('diagnostics.runModal.estimatedDuration') }}</p>
            <p class="text-sm text-success-700 mt-1">
              {{ getEstimatedDuration() }} - {{ t('diagnostics.runModal.realTimeResults') }}
            </p>
          </div>
        </div>
      </div>
    </div>

    <template #footer>
      <button 
        class="u-btn u-btn-secondary" 
        @click="handleCancel" 
        :disabled="loading" 
        type="button"
      >
        {{ t('ui.cancel') }}
      </button>
      <button 
        class="u-btn u-btn-primary min-w-[140px]" 
        @click="handleSubmit" 
        :disabled="!isValid || loading" 
        type="button"
      >
        <Icon v-if="loading" name="heroicons:arrow-path" class="h-4 w-4 animate-spin mr-2" />
        <Icon v-else name="heroicons:play" class="h-4 w-4 mr-2" />
        {{ loading ? t('diagnostics.runModal.starting') : t('diagnostics.runModal.startDiagnostic') }}
      </button>
    </template>
  </UModal>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface Props {
  modelValue: boolean
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  loading: false
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  submit: [data: typeof form.value]
  cancel: []
}>()

const { t } = useI18n()
const loading = ref(props.loading)

const form = ref({
  equipment: '',
  type: 'full',
  emailNotification: false,
  priorityAnalysis: false
})

const errors = ref<{ equipment?: string }>({})

const isValid = computed(() => {
  return form.value.equipment.trim().length > 0
})

function handleCancel() {
  emit('update:modelValue', false)
  emit('cancel')
}

function handleSubmit() {
  // Validate
  errors.value.equipment = !form.value.equipment 
    ? t('diagnostics.runModal.equipmentRequired') 
    : ''
    
  if (!isValid.value) return
  
  loading.value = true
  emit('submit', form.value)
  
  // Reset form after diagnostic starts
  setTimeout(() => {
    loading.value = false
    form.value = {
      equipment: '',
      type: 'full',
      emailNotification: false,
      priorityAnalysis: false
    }
    errors.value = {}
  }, 1500)
}

function getEstimatedDuration(): string {
  const durations: Record<string, string> = {
    full: '2-5 min',
    pressure: '1-2 min',
    temperature: '1-2 min',
    vibration: '1-3 min',
    flow: '1-2 min'
  }
  return durations[form.value.type] || '2-5 min'
}
</script>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: all 0.15s ease-out;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}

.metallic-select {
  background-color: #191d23 !important;
  color: #edf2fa !important;
  border-color: #4c596f !important;
}
</style>
