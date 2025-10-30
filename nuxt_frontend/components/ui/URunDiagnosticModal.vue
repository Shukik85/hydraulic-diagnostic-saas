<template>
  <UModal
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    title="Run New Diagnostic"
    description="Select equipment and diagnostic parameters"
    size="md"
    :loading="loading"
  >
    <div class="space-y-5">
      <!-- Equipment Selection -->
      <div>
        <label class="u-label" for="equipment">
          Equipment *
        </label>
        <div class="relative">
          <select 
            id="equipment"
            v-model="form.equipment"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="">Select equipment...</option>
            <option value="hyd-001">HYD-001 - Pump Station A</option>
            <option value="hyd-002">HYD-002 - Hydraulic Motor B</option>
            <option value="hyd-003">HYD-003 - Control Valve C</option>
            <option value="hyd-004">HYD-004 - Cooling System D</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
        <Transition name="fade">
          <p v-if="errors.equipment" class="mt-2 text-sm text-red-600 flex items-center gap-1">
            <Icon name="heroicons:exclamation-circle" class="h-4 w-4 flex-shrink-0" />
            {{ errors.equipment }}
          </p>
        </Transition>
      </div>

      <!-- Diagnostic Type -->
      <div>
        <label class="u-label" for="diagnostic-type">
          Diagnostic Type
        </label>
        <div class="relative">
          <select 
            id="diagnostic-type"
            v-model="form.type"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="full">Full System Analysis - Comprehensive check</option>
            <option value="pressure">Pressure System Check - Focus on pressure</option>
            <option value="temperature">Temperature Analysis - Thermal monitoring</option>
            <option value="vibration">Vibration Analysis - Mechanical health</option>
            <option value="flow">Flow Analysis - Fluid dynamics</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <!-- Options -->
      <div class="space-y-3">
        <div class="flex items-center gap-2">
          <input
            id="email-notification"
            v-model="form.emailNotification"
            type="checkbox"
            class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
          />
          <label for="email-notification" class="text-sm text-gray-700">
            Send email notification when complete
          </label>
        </div>
        
        <div class="flex items-center gap-2">
          <input
            id="priority-analysis"
            v-model="form.priorityAnalysis"
            type="checkbox"
            class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
          />
          <label for="priority-analysis" class="text-sm text-gray-700">
            Priority analysis (faster processing)
          </label>
        </div>
      </div>

      <!-- Estimated Duration -->
      <div class="rounded-lg bg-green-50 border border-green-200 p-4">
        <div class="flex items-start gap-3">
          <Icon name="heroicons:clock" class="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
          <div>
            <p class="text-sm font-medium text-green-900">
              Estimated Duration
            </p>
            <p class="text-sm text-green-700 mt-1">
              {{ getEstimatedDuration() }} - Results will be available in real-time
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
        Cancel
      </button>
      <button 
        class="u-btn u-btn-primary min-w-[140px]"
        @click="handleSubmit"
        :disabled="!isValid || loading"
        type="button"
      >
        <Icon 
          v-if="loading" 
          name="heroicons:arrow-path" 
          class="h-4 w-4 animate-spin mr-2" 
        />
        <Icon 
          v-else 
          name="heroicons:play" 
          class="h-4 w-4 mr-2" 
        />
        {{ loading ? 'Starting...' : 'Start Diagnostic' }}
      </button>
    </template>
  </UModal>
</template>

<script setup lang="ts">
interface Props {
  modelValue: boolean
  loading?: boolean
}

interface DiagnosticFormData {
  equipment: string
  type: 'full' | 'pressure' | 'temperature' | 'vibration' | 'flow'
  emailNotification: boolean
  priorityAnalysis: boolean
}

interface FormErrors {
  equipment?: string
}

const props = withDefaults(defineProps<Props>(), {
  loading: false
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  'submit': [data: DiagnosticFormData]
  'cancel': []
}>()

// Form state
const form = reactive<DiagnosticFormData>({
  equipment: '',
  type: 'full',
  emailNotification: true,
  priorityAnalysis: false
})

const errors = reactive<FormErrors>({})

// Helper methods
const getEstimatedDuration = (): string => {
  const durations = {
    'full': '5-8 minutes',
    'pressure': '2-3 minutes',
    'temperature': '2-4 minutes', 
    'vibration': '3-5 minutes',
    'flow': '2-4 minutes'
  }
  return durations[form.type] || '2-8 minutes'
}

// Validation
const validate = (): boolean => {
  // Reset errors
  Object.keys(errors).forEach(key => delete errors[key as keyof FormErrors])
  
  // Equipment validation
  if (!form.equipment) {
    errors.equipment = 'Please select equipment to diagnose'
  }
  
  return !errors.equipment
}

const isValid = computed(() => validate())

// Event handlers
const handleSubmit = async () => {
  if (!validate() || props.loading) return
  
  emit('submit', {
    equipment: form.equipment,
    type: form.type,
    emailNotification: form.emailNotification,
    priorityAnalysis: form.priorityAnalysis
  })
}

const handleCancel = () => {
  if (props.loading) return
  emit('cancel')
  emit('update:modelValue', false)
}

// Reset form when modal closes
watch(() => props.modelValue, (isOpen) => {
  if (!isOpen) {
    // Reset form after transition
    setTimeout(() => {
      form.equipment = ''
      form.type = 'full'
      form.emailNotification = true
      form.priorityAnalysis = false
      Object.keys(errors).forEach(key => delete errors[key as keyof FormErrors])
    }, 300)
  }
})
</script>

<style scoped>
/* Fade transitions for error messages */
.fade-enter-active,
.fade-leave-active {
  transition: all 0.15s ease-out;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>