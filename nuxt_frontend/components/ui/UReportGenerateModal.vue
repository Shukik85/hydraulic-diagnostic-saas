<template>
  <UModal
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    title="Generate Report"
    description="Create comprehensive analysis report"
    size="lg"
    :loading="loading"
  >
    <div class="space-y-5">
      <!-- Report Template -->
      <div>
        <label class="u-label" for="template">
          Report Template
        </label>
        <div class="relative">
          <select 
            id="template"
            v-model="form.template"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="executive">Executive Summary - High-level overview</option>
            <option value="technical">Technical Analysis - Detailed diagnostics</option>
            <option value="compliance">Compliance Report - Regulatory standards</option>
            <option value="maintenance">Maintenance Planning - Action items</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <!-- Date Range -->
      <div>
        <label class="u-label" for="date-range">
          Analysis Period
        </label>
        <div class="relative">
          <select 
            id="date-range"
            v-model="form.range"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="last_24h">Last 24 Hours - Recent activity</option>
            <option value="last_7d">Last 7 Days - Weekly summary</option>
            <option value="last_30d">Last 30 Days - Monthly analysis</option>
            <option value="last_90d">Last 90 Days - Quarterly review</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <!-- Report Language -->
      <div>
        <label class="u-label" for="locale">
          Language & Format
        </label>
        <div class="relative">
          <select 
            id="locale"
            v-model="form.locale"
            class="u-input appearance-none cursor-pointer"
            :disabled="loading"
          >
            <option value="en-US">English (US) - International</option>
            <option value="ru-RU">Русский - Russian Federation</option>
            <option value="de-DE">Deutsch - Germany</option>
          </select>
          <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <!-- Custom Title -->
      <div>
        <label class="u-label" for="report-title">
          Report Title
          <span class="text-gray-400 font-normal">(optional)</span>
        </label>
        <input 
          id="report-title"
          v-model.trim="form.title"
          type="text" 
          class="u-input"
          placeholder="Custom report title (auto-generated if empty)"
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
              Report Preview
            </p>
            <p class="text-sm text-green-700 mt-1">
              {{ getPreviewText() }}
            </p>
            <div class="flex items-center gap-4 mt-3 text-xs text-green-600">
              <span class="flex items-center gap-1">
                <Icon name="heroicons:clock" class="h-3 w-3" />
                ~2-5 min
              </span>
              <span class="flex items-center gap-1">
                <Icon name="heroicons:document-arrow-down" class="h-3 w-3" />
                PDF format
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
        Cancel
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
        {{ loading ? 'Generating...' : 'Generate Report' }}
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

// Form state
const form = reactive<ReportFormData>({
  template: 'executive',
  range: 'last_7d',
  locale: 'en-US',
  title: ''
})

// Helper methods
const getTemplateLabel = (template: string): string => {
  const labels = {
    'executive': 'Executive Summary',
    'technical': 'Technical Analysis', 
    'compliance': 'Compliance Report',
    'maintenance': 'Maintenance Planning'
  }
  return labels[template as keyof typeof labels] || 'Report'
}

const getRangeLabel = (range: string): string => {
  const labels = {
    'last_24h': '24 hours',
    'last_7d': '7 days',
    'last_30d': '30 days',
    'last_90d': '90 days'
  }
  return labels[range as keyof typeof labels] || 'period'
}

const getPreviewText = (): string => {
  const template = getTemplateLabel(form.template)
  const period = getRangeLabel(form.range)
  const customTitle = form.title ? `"${form.title}"` : 'auto-generated title'
  
  return `${template} covering ${period} with ${customTitle}`
}

// Event handlers
const handleSubmit = async () => {
  if (props.loading) return
  
  emit('submit', {
    template: form.template,
    range: form.range,
    locale: form.locale,
    title: form.title.trim()
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
      form.template = 'executive'
      form.range = 'last_7d'
      form.locale = 'en-US'
      form.title = ''
    }, 300)
  }
})
</script>