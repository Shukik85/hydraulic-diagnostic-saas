<template>
  <UModal
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    :title="$t('systems.create.title')"
    :description="$t('systems.create.subtitle')"
    size="md"
    :close-on-backdrop="true"
  >
    <div class="space-y-5">
      <!-- System Name -->
      <div>
        <label class="u-label" for="system-name">
          {{ $t('systems.create.name') }} *
        </label>
        <input 
          id="system-name"
          v-model.trim="form.name"
          type="text" 
          class="u-input"
          :placeholder="$t('systems.create.namePlaceholder')"
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
          {{ $t('systems.create.type') }}
        </label>
        <div class="relative">
          <select 
            id="system-type"
            v-model="form.type"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="industrial">{{ $t('systems.types.industrial') }}</option>
            <option value="mobile">{{ $t('systems.types.mobile') }}</option>
            <option value="marine">{{ $t('systems.types.marine') }}</option>
            <option value="construction">{{ $t('systems.types.construction') }}</option>
            <option value="mining">{{ $t('systems.types.mining') }}</option>
            <option value="agricultural">{{ $t('systems.types.agricultural') }}</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <!-- Initial Status -->
      <div>
        <label class="u-label" for="system-status">
          {{ $t('systems.create.initialStatus') }}
        </label>
        <div class="relative">
          <select 
            id="system-status"
            v-model="form.status"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="active">{{ $t('systems.status.active') }}</option>
            <option value="maintenance">{{ $t('systems.status.maintenance') }}</option>
            <option value="inactive">{{ $t('systems.status.inactive') }}</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <!-- Description -->
      <div>
        <label class="u-label" for="system-description">
          {{ $t('ui.description') }} <span class="text-gray-400 font-normal">({{ $t('ui.optional', 'optional') }})</span>
        </label>
        <textarea 
          id="system-description"
          v-model.trim="form.description"
          class="u-input resize-none"
          :placeholder="$t('systems.create.descriptionPlaceholder')"
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
              {{ $t('systems.create.nextStepsTitle') }}
            </p>
            <p class="text-sm text-blue-700 mt-1">
              {{ $t('systems.create.nextStepsDesc') }}
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
        {{ $t('ui.cancel') }}
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
        {{ loading ? $t('systems.create.creating') : $t('systems.create.createBtn') }}
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
}

const props = withDefaults(defineProps<Props>(), {
  loading: false
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  'submit': [data: SystemFormData]
  'cancel': []
}>()

const { $t } = useI18n()

// Form state
const form = reactive<SystemFormData>({
  name: '',
  type: 'industrial',
  status: 'active',
  description: ''
})

const errors = reactive<FormErrors>({})

const nameInputRef = ref<HTMLInputElement>()

// Validation
const validate = (): boolean => {
  Object.keys(errors).forEach(key => delete errors[key as keyof FormErrors])
  if (!form.name.trim()) {
    errors.name = $t('systems.create.errors.nameRequired')
  } else if (form.name.length < 3) {
    errors.name = $t('systems.create.errors.nameMin')
  } else if (form.name.length > 200) {
    errors.name = $t('systems.create.errors.nameMax')
  }
  return !errors.name
}

const isValid = computed(() => validate())

// Event handlers
const handleSubmit = () => {
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

watch(() => props.modelValue, (isOpen) => {
  if (isOpen) {
    nextTick(() => nameInputRef.value?.focus())
  } else {
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
.fade-enter-active,
.fade-leave-active { transition: all 0.15s ease-out; }
.fade-enter-from,
.fade-leave-to { opacity: 0; transform: translateY(-4px); }
</style>