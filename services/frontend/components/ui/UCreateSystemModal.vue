<template>
  <UDialog
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
  >
    <UDialogContent class="max-w-lg">
      <UDialogHeader>
        <UDialogTitle>{{ t('systems.create.title') }}</UDialogTitle>
        <UDialogDescription>{{ t('systems.create.subtitle') }}</UDialogDescription>
      </UDialogHeader>

      <form @submit.prevent="handleSubmit" class="space-y-6">
        <!-- System Name -->
        <UFormGroup
          :label="t('systems.create.name')"
          helper="Используйте понятное имя для идентификации системы"
          :error="errors.name"
          required
        >
          <UInput 
            v-model="form.name"
            :placeholder="t('systems.create.namePlaceholder')" 
            :disabled="loading" 
            maxlength="200" 
          />
        </UFormGroup>

        <!-- System Type -->
        <UFormGroup
          :label="t('systems.create.type')"
          helper="Тип системы определяет набор доступных параметров"
        >
          <USelect 
            v-model="form.type"
            :disabled="loading"
          >
            <option value="industrial">{{ t('systems.types.industrial') }}</option>
            <option value="mobile">{{ t('systems.types.mobile') }}</option>
            <option value="marine">{{ t('systems.types.marine') }}</option>
            <option value="construction">{{ t('systems.types.construction') }}</option>
            <option value="mining">{{ t('systems.types.mining') }}</option>
            <option value="agricultural">{{ t('systems.types.agricultural') }}</option>
          </USelect>
        </UFormGroup>

        <!-- Initial Status -->
        <UFormGroup
          :label="t('systems.create.initialStatus')"
          helper="Статус можно изменить позже в настройках"
        >
          <USelect 
            v-model="form.status"
            :disabled="loading"
          >
            <option value="active">{{ t('systems.status.active') }}</option>
            <option value="maintenance">{{ t('systems.status.maintenance') }}</option>
            <option value="inactive">{{ t('systems.status.inactive') }}</option>
          </USelect>
        </UFormGroup>

        <!-- Description -->
        <UFormGroup
          :label="t('ui.description')"
          helper="Опишите назначение и особенности системы (опционально)"
        >
          <UTextarea 
            v-model="form.description"
            :placeholder="t('systems.create.descriptionPlaceholder')" 
            :disabled="loading" 
            rows="3" 
            maxlength="500" 
          />
        </UFormGroup>

        <!-- Next Steps Info -->
        <div class="alert-info">
          <Icon name="heroicons:information-circle" class="w-5 h-5" />
          <div>
            <p class="font-medium">
              {{ t('systems.create.nextStepsTitle') }}
            </p>
            <p class="text-sm mt-1">
              {{ t('systems.create.nextStepsDesc') }}
            </p>
          </div>
        </div>
      </form>

      <UDialogFooter>
        <UButton 
          variant="secondary"
          @click="handleCancel" 
          :disabled="loading"
        >
          {{ t('ui.cancel') }}
        </UButton>
        <UButton 
          @click="handleSubmit" 
          :disabled="!isValid || loading"
        >
          <Icon 
            v-if="loading" 
            name="heroicons:arrow-path" 
            class="w-5 h-5 animate-spin mr-2" 
          />
          <Icon 
            v-else 
            name="heroicons:plus" 
            class="w-5 h-5 mr-2" 
          />
          {{ loading ? t('systems.create.creating') : t('systems.create.createBtn') }}
        </UButton>
      </UDialogFooter>
    </UDialogContent>
  </UDialog>
</template>

<script setup lang="ts">
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

const form = ref({
  name: '',
  type: 'industrial',
  status: 'active',
  description: ''
})

const errors = ref<{ name?: string }>({})

const isValid = computed(() => {
  return form.value.name.trim().length > 0
})

function handleCancel() {
  emit('update:modelValue', false)
  emit('cancel')
  errors.value = {}
}

function handleSubmit() {
  errors.value.name = !form.value.name.trim() 
    ? t('systems.create.nameRequired') 
    : undefined
    
  if (!isValid.value) return
  
  emit('submit', form.value)
  
  setTimeout(() => {
    form.value = {
      name: '',
      type: 'industrial',
      status: 'active',
      description: ''
    }
    errors.value = {}
  }, 1500)
}
</script>
