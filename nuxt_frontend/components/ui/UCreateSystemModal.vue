<template>
  <UModal
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    title="Add Hydraulic System"
    description="Create and configure new system"
    size="md"
    :loading="loading"
  >
    <div class="space-y-5">
      <!-- System Name -->
      <div>
        <label class="u-label" for="system-name">
          System Name *
        </label>
        <input 
          id="system-name"
          v-model.trim="form.name"
          type="text" 
          class="u-input"
          placeholder="e.g. Pump Station A, Hydraulic Motor #1"
          :disabled="loading"
          maxlength="200"
          ref="nameInputRef"
        />
        <Transition name="fade">
          <p v-if="errors.name" class="mt-2 text-sm text-red-600 flex items-center gap-1">
            <Icon name="heroicons:exclamation-circle" class="h-4 w-4 flex-shrink-0" />
            {{ errors.name }}
          </p>
        </Transition>
      </div>

      <!-- System Type -->
      <div>
        <label class="u-label" for="system-type">
          System Type
        </label>
        <div class="relative">
          <select 
            id="system-type"
            v-model="form.type"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="industrial">Industrial - Factory equipment</option>
            <option value="mobile">Mobile - Vehicles and machinery</option>
            <option value="marine">Marine - Ships and offshore</option>
            <option value="construction">Construction - Heavy machinery</option>
            <option value="mining">Mining - Extraction equipment</option>
            <option value="agricultural">Agricultural - Farm machinery</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <!-- Initial Status -->
      <div>
        <label class="u-label" for="system-status">
          Initial Status
        </label>
        <div class="relative">
          <select 
            id="system-status"
            v-model="form.status"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="active">Active - System operational</option>
            <option value="maintenance">Maintenance - Under service</option>
            <option value="inactive">Inactive - Not operational</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <!-- Description -->
      <div>
        <label class="u-label" for="system-description">
          Description <span class="text-gray-400 font-normal">(optional)</span>
        </label>
        <textarea 
          id="system-description"
          v-model.trim="form.description"
          class="u-input resize-none"
          placeholder="System location, purpose, or additional notes..."
          :disabled="loading"
          rows="3"
          maxlength="500"
        />
      </div>

      <!-- Setup Info -->
      <div class="rounded-lg bg-blue-50 border border-blue-200 p-4">
        <div class="flex items-start gap-3">
          <Icon name="heroicons:information-circle" class="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <div>
            <p class="text-sm font-medium text-blue-900">
              Next Steps After Creation
            </p>
            <p class="text-sm text-blue-700 mt-1">
              Add sensors, components, and configure diagnostic parameters in the system details page.
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
        class="u-btn u-btn-primary min-w-[120px]"
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
          name="heroicons:plus" 
          class="h-4 w-4 mr-2" 
        />
        {{ loading ? 'Creating...' : 'Create System' }}
      </button>
    </template>
  </UModal>
</template>

<script setup lang="ts">
interface Props {
  modelValue: boolean
  loading?: boolean
}

interface SystemFormData {
  name: string
  type: 'industrial' | 'mobile' | 'marine' | 'construction' | 'mining' | 'agricultural'
  status: 'active' | 'maintenance' | 'inactive'
  description: string
}

interface FormErrors {
  name?: string
  type?: string
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
  type: 'industrial',
  status: 'active',
  description: ''
})

const errors = reactive<FormErrors>({})

// Refs
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
  
  return !errors.name && !errors.type && !errors.status
}

const isValid = computed(() => validate())

// Event handlers
const handleSubmit = async () => {
  if (!validate() || props.loading) return
  
  emit('submit', {
    name: form.name.trim(),
    type: form.type,
    status: form.status,
    description: form.description.trim()
  })
}

const handleCancel = () => {
  if (props.loading) return
  emit('cancel')
  emit('update:modelValue', false)
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
      form.type = 'industrial'
      form.status = 'active'
      form.description = ''
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