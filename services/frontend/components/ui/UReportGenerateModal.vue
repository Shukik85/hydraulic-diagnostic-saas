<template>
  <UDialog :model-value="modelValue" @update:model-value="$emit('update:modelValue', $event)">
    <UDialogContent class="max-w-2xl">
      <UDialogHeader>
        <UDialogTitle>{{ t('reports.generate.title') }}</UDialogTitle>
        <UDialogDescription>{{ t('reports.generate.subtitle') }}</UDialogDescription>
      </UDialogHeader>

      <form @submit.prevent="handleSubmit" class="space-y-6">
        <!-- Report Template -->
        <UFormGroup :label="t('reports.generate.template')" helper="Выберите тип отчёта в зависимости от аудитории"
          required>
          <USelect v-model="form.template" :disabled="loading">
            <option value="executive">{{ t('reports.templates.executive') }}</option>
            <option value="technical">{{ t('reports.templates.technical') }}</option>
            <option value="compliance">{{ t('reports.templates.compliance') }}</option>
            <option value="maintenance">{{ t('reports.templates.maintenance') }}</option>
          </USelect>
        </UFormGroup>

        <!-- Date Range -->
        <UFormGroup :label="t('reports.generate.period')" helper="Временной диапазон для анализа данных" required>
          <USelect v-model="form.range" :disabled="loading">
            <option value="last_24h">{{ t('reports.periods.last_24h') }}</option>
            <option value="last_7d">{{ t('reports.periods.last_7d') }}</option>
            <option value="last_30d">{{ t('reports.periods.last_30d') }}</option>
            <option value="last_90d">{{ t('reports.periods.last_90d') }}</option>
          </USelect>
        </UFormGroup>

        <!-- Report Language -->
        <UFormGroup :label="t('reports.generate.language')" helper="Язык генерируемого отчёта">
          <USelect v-model="form.locale" :disabled="loading">
            <option value="en-US">{{ t('reports.locales.en') }}</option>
            <option value="ru-RU">{{ t('reports.locales.ru') }}</option>
            <option value="de-DE">{{ t('reports.locales.de') }}</option>
          </USelect>
        </UFormGroup>

        <!-- Custom Title -->
        <UFormGroup :label="t('reports.generate.customTitle')" helper="Оставьте пустым для автоматического названия">
          <UInput v-model="form.title" :placeholder="t('reports.generate.customTitlePlaceholder')" :disabled="loading"
            maxlength="255" />
        </UFormGroup>

        <!-- Generation Preview -->
        <div class="alert-success">
          <Icon name="heroicons:document-text" class="w-5 h-5" />
          <div>
            <p class="font-medium">
              {{ t('reports.generate.preview') }}
            </p>
            <p class="text-sm mt-1">
              {{ getPreviewText() }}
            </p>
            <div class="flex items-center gap-4 mt-3 text-xs opacity-75">
              <span class="flex items-center gap-1">
                <Icon name="heroicons:clock" class="w-3 h-3" />
                ~2-5 {{ t('ui.minutes') }}
              </span>
              <span class="flex items-center gap-1">
                <Icon name="heroicons:document-arrow-down" class="w-3 h-3" />
                PDF
              </span>
            </div>
          </div>
        </div>
      </form>

      <UDialogFooter>
        <UButton variant="secondary" @click="handleCancel" :disabled="loading">
          {{ t('ui.cancel') }}
        </UButton>
        <UButton @click="handleSubmit" :disabled="loading">
          <Icon v-if="loading" name="heroicons:arrow-path" class="w-5 h-5 animate-spin mr-2" />
          <Icon v-else name="heroicons:document-plus" class="w-5 h-5 mr-2" />
          {{ loading ? t('reports.generate.generating') : t('reports.generate.generateBtn') }}
        </UButton>
      </UDialogFooter>
    </UDialogContent>
  </UDialog>
</template>

<script setup lang="ts">
import { ref } from 'vue'

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
  template: 'executive',
  range: 'last_7d',
  locale: 'ru-RU',
  title: ''
})

function handleCancel() {
  emit('update:modelValue', false)
  emit('cancel')
}

function handleSubmit() {
  emit('submit', form.value)

  setTimeout(() => {
    form.value = {
      template: 'executive',
      range: 'last_7d',
      locale: 'ru-RU',
      title: ''
    }
  }, 2000)
}

function getPreviewText(): string {
  const templateName = t(`reports.templates.${form.value.template}`)
  const periodName = t(`reports.periods.${form.value.range}`)
  return `${templateName} • ${periodName}`
}
</script>
