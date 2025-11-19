<template>
  <UDialog 
    :model-value="modelValue" 
    @update:model-value="$emit('update:modelValue', $event)"
  >
    <UDialogContent class="max-w-lg">
      <UDialogHeader>
        <UDialogTitle>{{ t('diagnostics.runModal.title') }}</UDialogTitle>
        <UDialogDescription>{{ t('diagnostics.runModal.description') }}</UDialogDescription>
      </UDialogHeader>

      <form @submit.prevent="handleSubmit" class="space-y-6">
        <!-- Equipment Selection -->
        <UFormGroup
          :label="t('diagnostics.runModal.equipment')"
          helper="Выберите систему для запуска диагностики"
          :error="errors.equipment"
          required
        >
          <USelect 
            v-model="form.equipment"
            :disabled="loading"
          >
            <option value="" disabled>{{ t('diagnostics.runModal.selectEquipment') }}</option>
            <option value="hyd-001">{{ t('diagnostics.runModal.pumpStationA') }}</option>
            <option value="hyd-002">{{ t('diagnostics.runModal.hydraulicMotorB') }}</option>
            <option value="hyd-003">{{ t('diagnostics.runModal.controlValveC') }}</option>
            <option value="hyd-004">{{ t('diagnostics.runModal.coolingSystemD') }}</option>
          </USelect>
        </UFormGroup>

        <!-- Diagnostic Type -->
        <UFormGroup
          :label="t('diagnostics.runModal.diagnosticType')"
          helper="Полный анализ занимает больше времени, но даёт полную картину"
        >
          <USelect 
            v-model="form.type"
            :disabled="loading"
          >
            <option value="full">{{ t('diagnostics.runModal.fullAnalysis') }}</option>
            <option value="pressure">{{ t('diagnostics.runModal.pressureCheck') }}</option>
            <option value="temperature">{{ t('diagnostics.runModal.temperatureAnalysis') }}</option>
            <option value="vibration">{{ t('diagnostics.runModal.vibrationAnalysis') }}</option>
            <option value="flow">{{ t('diagnostics.runModal.flowAnalysis') }}</option>
          </USelect>
        </UFormGroup>

        <!-- Options -->
        <div class="space-y-3">
          <div class="flex items-center gap-3">
            <UCheckbox 
              id="email-notification" 
              v-model:checked="form.emailNotification"
            />
            <ULabel 
              for="email-notification" 
              class="text-sm text-white cursor-pointer"
            >
              {{ t('diagnostics.runModal.emailNotification') }}
            </ULabel>
          </div>
          
          <div class="flex items-center gap-3">
            <UCheckbox 
              id="priority-analysis" 
              v-model:checked="form.priorityAnalysis"
            />
            <ULabel 
              for="priority-analysis" 
              class="text-sm text-white cursor-pointer"
            >
              {{ t('diagnostics.runModal.priorityAnalysis') }}
            </ULabel>
          </div>
        </div>

        <!-- Estimated Duration Info -->
        <div class="alert-success">
          <Icon name="heroicons:clock" class="w-5 h-5" />
          <div>
            <p class="font-medium">
              {{ t('diagnostics.runModal.estimatedDuration') }}: {{ getEstimatedDuration() }}
            </p>
            <p class="text-sm mt-1">
              {{ t('diagnostics.runModal.realTimeResults') }}
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
            name="heroicons:play" 
            class="w-5 h-5 mr-2" 
          />
          {{ loading ? t('diagnostics.runModal.starting') : t('diagnostics.runModal.startDiagnostic') }}
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
  errors.value = {}
}

function handleSubmit() {
  errors.value.equipment = !form.value.equipment 
    ? t('diagnostics.runModal.equipmentRequired') 
    : undefined
    
  if (!isValid.value) return
  
  emit('submit', form.value)
  
  setTimeout(() => {
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
    full: '2-5 мин',
    pressure: '1-2 мин',
    temperature: '1-2 мин',
    vibration: '1-3 мин',
    flow: '1-2 мин'
  }
  return durations[form.value.type] || '2-5 мин'
}
</script>
