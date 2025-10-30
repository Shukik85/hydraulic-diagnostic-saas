<template>
  <Teleport to="body">
    <Transition name="modal" appear>
      <div 
        v-if="modelValue" 
        class="fixed inset-0 z-50 overflow-y-auto" 
        aria-modal="true" 
        role="dialog" 
        aria-labelledby="run-diagnostic-title"
        @click="onBackdropClick"
        @keydown.esc="handleEscape"
      >
        <!-- Backdrop -->
        <div class="fixed inset-0 bg-black/50 transition-opacity" />
        
        <!-- Modal Container -->
        <div class="relative flex min-h-screen items-center justify-center p-4">
          <div 
            class="relative w-full max-w-md transform rounded-xl bg-white dark:bg-gray-900 shadow-2xl transition-all"
            @click.stop
            ref="modalRef"
          >
            <!-- Header -->
            <div class="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 px-6 py-4">
              <div>
                <h3 id="run-diagnostic-title" class="text-lg font-semibold text-gray-900 dark:text-white">
                  Run New Diagnostic
                </h3>
                <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  Select equipment and diagnostic parameters
                </p>
              </div>
              <button 
                class="rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-800 dark:hover:text-gray-300 transition-colors"
                @click="handleCancel"
                :disabled="loading"
                aria-label="Close modal"
              >
                <Icon name="i-heroicons-x-mark" class="h-5 w-5" />
              </button>
            </div>

            <!-- Body -->
            <div class="px-6 py-6">
              <div class="space-y-5">
                <!-- Equipment Selection -->
                <div>
                  <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2" for="equipment">
                    Equipment *
                  </label>
                  <div class="relative">
                    <select 
                      id="equipment"
                      v-model="form.equipment"
                      class="block w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 text-gray-900 dark:text-white shadow-sm transition-colors focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:focus:border-blue-400 appearance-none cursor-pointer"
                      :disabled="loading"
                    >
                      <option value="">Select equipment...</option>
                      <option value="hyd-001">HYD-001 - Pump Station A</option>
                      <option value="hyd-002">HYD-002 - Hydraulic Motor B</option>
                      <option value="hyd-003">HYD-003 - Control Valve C</option>
                      <option value="hyd-004">HYD-004 - Cooling System D</option>
                    </select>
                    <Icon name="i-heroicons-chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
                  </div>
                  <Transition name="fade">
                    <p v-if="errors.equipment" class="mt-2 text-sm text-red-600 dark:text-red-400 flex items-center gap-1">
                      <Icon name="i-heroicons-exclamation-circle" class="h-4 w-4 flex-shrink-0" />
                      {{ errors.equipment }}
                    </p>
                  </Transition>
                </div>

                <!-- Diagnostic Type -->
                <div>
                  <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2" for="diagnostic-type">
                    Diagnostic Type
                  </label>
                  <div class="relative">
                    <select 
                      id="diagnostic-type"
                      v-model="form.type"
                      class="block w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 text-gray-900 dark:text-white shadow-sm transition-colors focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:focus:border-blue-400 appearance-none cursor-pointer"
                      :disabled="loading"
                    >
                      <option value="full">Full System Analysis - Comprehensive check</option>
                      <option value="pressure">Pressure System Check - Focus on pressure</option>
                      <option value="temperature">Temperature Analysis - Thermal monitoring</option>
                      <option value="vibration">Vibration Analysis - Mechanical health</option>
                      <option value="flow">Flow Analysis - Fluid dynamics</option>
                    </select>
                    <Icon name="i-heroicons-chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
                  </div>
                </div>

                <!-- Options -->
                <div class="space-y-3">
                  <div class="flex items-center gap-2">
                    <input
                      id="email-notification"
                      v-model="form.emailNotification"
                      type="checkbox"
                      class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                    />
                    <label for="email-notification" class="text-sm text-gray-700 dark:text-gray-300">
                      Send email notification when complete
                    </label>
                  </div>
                  
                  <div class="flex items-center gap-2">
                    <input
                      id="priority-analysis"
                      v-model="form.priorityAnalysis"
                      type="checkbox"
                      class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                    />
                    <label for="priority-analysis" class="text-sm text-gray-700 dark:text-gray-300">
                      Priority analysis (faster processing)
                    </label>
                  </div>
                </div>

                <!-- Estimated Duration -->
                <div class="rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 p-4">
                  <div class="flex items-start gap-3">
                    <Icon name="i-heroicons-clock" class="h-5 w-5 text-green-600 dark:text-green-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <p class="text-sm font-medium text-green-900 dark:text-green-100">
                        Estimated Duration
                      </p>
                      <p class="text-sm text-green-700 dark:text-green-200 mt-1">
                        {{ getEstimatedDuration() }} - Results will be available in real-time
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Footer -->
            <div class="flex items-center justify-end gap-3 border-t border-gray-200 dark:border-gray-700 px-6 py-4">
              <button 
                class="rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-2.5 text-sm font-medium text-gray-700 dark:text-gray-300 shadow-sm hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                @click="handleCancel"
                :disabled="loading"
                type="button"
              >
                Cancel
              </button>
              <button 
                class="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-colors min-w-[140px]"
                @click="handleSubmit"
                :disabled="!isValid || loading"
                type="button"
              >
                <Icon 
                  v-if="loading" 
                  name="i-heroicons-arrow-path" 
                  class="h-4 w-4 animate-spin" 
                />
                <Icon 
                  v-else 
                  name="i-heroicons-play" 
                  class="h-4 w-4" 
                />
                {{ loading ? 'Starting...' : 'Start Diagnostic' }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
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

// Refs
const modalRef = ref<HTMLElement>()

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

const handleEscape = (event: KeyboardEvent) => {
  if (event.key === 'Escape') {
    handleCancel()
  }
}

const onBackdropClick = (event: Event) => {
  if (event.target === event.currentTarget) {
    handleCancel()
  }
}

// Focus management
watch(() => props.modelValue, (isOpen) => {
  if (isOpen) {
    nextTick(() => {
      modalRef.value?.focus()
    })
  }
})

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
/* Modal transitions */
.modal-enter-active,
.modal-leave-active {
  transition: all 0.25s ease-out;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
  transform: scale(0.95);
}

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