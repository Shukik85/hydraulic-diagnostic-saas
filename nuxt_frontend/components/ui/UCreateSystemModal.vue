<template>
  <Teleport to="body">
    <Transition name="modal" appear>
      <div 
        v-if="modelValue" 
        class="fixed inset-0 z-50 overflow-y-auto" 
        aria-modal="true" 
        role="dialog" 
        aria-labelledby="create-system-title"
        @click="onBackdropClick"
        @keydown.esc="handleEscape"
      >
        <!-- Backdrop -->
        <div class="fixed inset-0 bg-black/60 transition-opacity" />
        
        <!-- Modal Container -->
        <div class="relative flex min-h-screen items-center justify-center p-4">
          <div 
            class="relative w-full max-w-md transform rounded-xl bg-white dark:bg-gray-900 shadow-2xl transition-all"
            @click.stop
            ref="modalRef"
          >
            <!-- Header -->
            <div class="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 px-6 py-4">
              <h3 id="create-system-title" class="text-lg font-semibold text-gray-900 dark:text-white">
                Create Hydraulic System
              </h3>
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
                <!-- System Name -->
                <div>
                  <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2" for="system-name">
                    System Name *
                  </label>
                  <input 
                    id="system-name"
                    v-model.trim="form.name"
                    type="text" 
                    class="block w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 shadow-sm transition-colors focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:focus:border-blue-400"
                    placeholder="e.g. Press Line A, Hydraulic Motor #1"
                    :disabled="loading"
                    maxlength="200"
                    ref="nameInputRef"
                  />
                  <Transition name="fade">
                    <p v-if="errors.name" class="mt-2 text-sm text-red-600 dark:text-red-400 flex items-center gap-1">
                      <Icon name="i-heroicons-exclamation-circle" class="h-4 w-4 flex-shrink-0" />
                      {{ errors.name }}
                    </p>
                  </Transition>
                </div>

                <!-- System Status -->
                <div>
                  <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2" for="system-status">
                    Initial Status
                  </label>
                  <div class="relative">
                    <select 
                      id="system-status"
                      v-model="form.status"
                      class="block w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 text-gray-900 dark:text-white shadow-sm transition-colors focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:focus:border-blue-400 appearance-none cursor-pointer"
                      :disabled="loading"
                    >
                      <option value="online">🟢 Online - System operating normally</option>
                      <option value="offline">⚫ Offline - System not operational</option>
                      <option value="warning">🟡 Warning - Requires attention</option>
                      <option value="error">🔴 Error - Critical issue detected</option>
                    </select>
                    <Icon name="i-heroicons-chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
                  </div>
                  <Transition name="fade">
                    <p v-if="errors.status" class="mt-2 text-sm text-red-600 dark:text-red-400 flex items-center gap-1">
                      <Icon name="i-heroicons-exclamation-circle" class="h-4 w-4 flex-shrink-0" />
                      {{ errors.status }}
                    </p>
                  </Transition>
                </div>

                <!-- Helper Text -->
                <div class="rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 p-4">
                  <div class="flex items-start gap-3">
                    <Icon name="i-heroicons-information-circle" class="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <p class="text-sm font-medium text-blue-900 dark:text-blue-100">
                        Quick Setup
                      </p>
                      <p class="text-sm text-blue-700 dark:text-blue-200 mt-1">
                        You can add sensors, components, and detailed configurations after creating the system.
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
                class="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-colors min-w-[100px]"
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
                  name="i-heroicons-plus" 
                  class="h-4 w-4" 
                />
                {{ loading ? 'Creating...' : 'Create System' }}
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

interface SystemFormData {
  name: string
  status: 'online' | 'offline' | 'warning' | 'error'
}

interface FormErrors {
  name?: string
  status?: string
}

const props = withDefaults(defineProps<Props>(), {
  loading: false
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  'submit': [data: SystemFormData]
  'cancel': []
}>()

// Form state
const form = reactive<SystemFormData>({
  name: '',
  status: 'online'
})

const errors = reactive<FormErrors>({})

// Refs
const modalRef = ref<HTMLElement>()
const nameInputRef = ref<HTMLInputElement>()

// Validation
const validate = (): boolean => {
  // Reset errors
  Object.keys(errors).forEach(key => delete errors[key as keyof FormErrors])
  
  // Name validation
  if (!form.name.trim()) {
    errors.name = 'System name is required'
  } else if (form.name.length < 3) {
    errors.name = 'System name must be at least 3 characters'
  } else if (form.name.length > 200) {
    errors.name = 'System name must be less than 200 characters'
  }
  
  // Status validation
  if (!['online', 'offline', 'warning', 'error'].includes(form.status)) {
    errors.status = 'Please select a valid status'
  }
  
  return !errors.name && !errors.status
}

const isValid = computed(() => validate())

// Event handlers
const handleSubmit = async () => {
  if (!validate() || props.loading) return
  
  emit('submit', {
    name: form.name.trim(),
    status: form.status
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
      nameInputRef.value?.focus()
    })
  }
})

// Reset form when modal closes
watch(() => props.modelValue, (isOpen) => {
  if (!isOpen) {
    // Reset form after transition
    setTimeout(() => {
      form.name = ''
      form.status = 'online'
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