<template>
  <UModal
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    :title="t('reports.generate.title')"
    :description="t('reports.generate.subtitle')"
    size="lg"
    :close-on-backdrop="true"
  >
    <div class="space-y-5">
      <!-- Report Template -->
      <div>
        <label class="u-label" for="template">
          {{ t('reports.generate.template') }}
        </label>
        <div class="relative">
          <select 
            id="template"
            v-model="form.template"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="executive">{{ t('reports.templates.executive') }}</option>
            <option value="technical">{{ t('reports.templates.technical') }}</option>
            <option value="compliance">{{ t('reports.templates.compliance') }}</option>
            <option value="maintenance">{{ t('reports.templates.maintenance') }}</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <!-- Date Range -->
      <div>
        <label class="u-label" for="date-range">
          {{ t('reports.generate.period') }}
        </label>
        <div class="relative">
          <select 
            id="date-range"
            v-model="form.range"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="last_24h">{{ t('reports.periods.last_24h') }}</option>
            <option value="last_7d">{{ t('reports.periods.last_7d') }}</option>
            <option value="last_30d">{{ t('reports.periods.last_30d') }}</option>
            <option value="last_90d">{{ t('reports.periods.last_90d') }}</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <!-- Report Language -->
      <div>
        <label class="u-label" for="locale">
          {{ t('reports.generate.language') }}
        </label>
        <div class="relative">
          <select 
            id="locale"
            v-model="form.locale"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="en-US">{{ t('reports.locales.en') }}</option>
            <option value="ru-RU">{{ t('reports.locales.ru') }}</option>
            <option value="de-DE">{{ t('reports.locales.de') }}</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <!-- Custom Title -->
      <div>
        <label class="u-label" for="report-title">
          {{ t('reports.generate.customTitle') }}
          <span class="text-gray-400 font-normal">({{ t('ui.optional') }})</span>
        </label>
        <input 
          id="report-title"
          v-model.trim="form.title"
          type="text" 
          class="u-input"
          :placeholder="t('reports.generate.customTitlePlaceholder')"
          :disabled="loading"
          maxlength="255"
        />
      </div>

      <!-- Generation Preview -->
      <div class="rounded-lg bg-green-50 border border-green-200 p-4">
        <div class="flex items-start gap-3">
          <Icon name="heroicons:document-text" class="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
          <div>
            <p class="text-sm font-medium text-green-900">
              {{ t('reports.generate.preview') }}
            </p>
            <p class="text-sm text-green-700 mt-1">
              {{ getPreviewText() }}
            </p>
            <div class="flex items-center gap-4 mt-3 text-xs text-green-600">
              <span class="flex items-center gap-1">
                <Icon name="heroicons:clock" class="h-3 w-3" />
                ~2-5 {{ t('ui.minutes') }}
              </span>
              <span class="flex items-center gap-1">
                <Icon name="heroicons:document-arrow-down" class="h-3 w-3" />
                PDF
              </span>
            </div>
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
        class="u-btn u-btn-success min-w-[120px]"
        @click="handleSubmit"
        :disabled="loading"
        type="button"
      >
        <Icon 
          v-if="loading" 
          name="heroicons:arrow-path" 
          class="h-4 w-4 animate-spin mr-2" 
        />
        <Icon 
          v-else 
          name="heroicons:document-plus" 
          class="h-4 w-4 mr-2" 
        />
        {{ loading ? t('reports.generate.generating') : t('reports.generate.generateBtn') }}
      </button>
    </template>
  </UModal>
</template>

<script setup lang="ts">
interface Props {
  modelValue: boolean
  loading?: boolean
}

interface ReportFormData {
  template: 'executive' | 'technical' | 'compliance' | 'maintenance'
  range: 'last_24h' | 'last_7d' | 'last_30d' | 'last_90d'
  locale: 'en-US' | 'ru-RU' | 'de-DE'
  title: string
}

const props = withDefaults(defineProps<Props>(), {
  loading: false
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  'submit': [data: ReportFormData]
  'cancel': []
}>()

const { t } = useI18n()

// Form state
const form = reactive<ReportFormData>({
  template: 'executive',
  range: 'last_7d',
  locale: 'en-US',
  title: ''
})

// Helpers
const getTemplateLabel = (template: string): string => {
  const labels: Record<string, string> = {
    executive: t('reports.templates.execShort', 'Executive Summary'),
    technical: t('reports.templates.techShort', 'Technical Analysis'),
    compliance: t('reports.templates.compShort', 'Compliance Report'),
    maintenance: t('reports.templates.maintShort', 'Maintenance Planning')
  }
  return labels[template] || t('reports.generate.title')
}

const getRangeLabel = (range: string): string => {
  const labels: Record<string, string> = {
    last_24h: t('reports.periods.24hShort', '24h'),
    last_7d: t('reports.periods.7dShort', '7d'),
    last_30d: t('reports.periods.30dShort', '30d'),
    last_90d: t('reports.periods.90dShort', '90d')
  }
  return labels[range] || t('reports.generate.period')
}

const getPreviewText = (): string => {
  const template = getTemplateLabel(form.template)
  const period = getRangeLabel(form.range)
  const customTitle = form.title ? `"${form.title}"` : t('reports.generate.autoTitle')
  return `${template} • ${period} • ${customTitle}`
}

// Events
const handleSubmit = () => {
  if (props.loading) return
  emit('submit', { ...form, title: form.title.trim() })
}

const handleCancel = () => {
  if (props.loading) return
  emit('cancel')
  emit('update:modelValue', false)
}

// Reset
watch(() => props.modelValue, (isOpen) => {
  if (!isOpen) {
    setTimeout(() => {
      form.template = 'executive'
      form.range = 'last_7d'
      form.locale = 'en-US'
      form.title = ''
    }, 300)
  }
})
</script>