<template>
  <Teleport to="body">
    <Transition name="modal" appear>
      <div 
        v-if="modelValue" 
        class="fixed inset-0 z-50 overflow-y-auto" 
        aria-modal="true" 
        role="dialog" 
        aria-labelledby="generate-report-title"
        @click="onBackdropClick"
        @keydown.esc="handleEscape"
      >
        <!-- Backdrop -->
        <div class="fixed inset-0 bg-black/60 transition-opacity" />
        
        <!-- Modal Container -->
        <div class="relative flex min-h-screen items-center justify-center p-4">
          <div 
            class="relative w-full max-w-lg transform rounded-xl bg-white dark:bg-gray-900 shadow-2xl transition-all"
            @click.stop
            ref="modalRef"
          >
            <!-- Header -->
            <div class="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 px-6 py-4">
              <div>
                <h3 id="generate-report-title" class="text-lg font-semibold text-gray-900 dark:text-white">
                  Generate Report
                </h3>
                <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  Create comprehensive analysis report
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
                <!-- Report Template -->
                <div>
                  <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2" for="template">
                    Report Template
                  </label>
                  <div class="relative">
                    <select 
                      id="template"
                      v-model="form.template"
                      class="block w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 text-gray-900 dark:text-white shadow-sm transition-colors focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:focus:border-blue-400 appearance-none cursor-pointer"
                      :disabled="loading"
                    >
                      <option value="executive">Executive Summary - High-level overview</option>
                      <option value="technical">Technical Analysis - Detailed diagnostics</option>
                      <option value="compliance">Compliance Report - Regulatory standards</option>
                      <option value="maintenance">Maintenance Planning - Action items</option>
                    </select>
                    <Icon name="i-heroicons-chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
                  </div>
                </div>

                <!-- Date Range -->
                <div>
                  <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2" for="date-range">
                    Analysis Period
                  </label>
                  <div class="relative">
                    <select 
                      id="date-range"
                      v-model="form.range"
                      class="block w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 text-gray-900 dark:text-white shadow-sm transition-colors focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:focus:border-blue-400 appearance-none cursor-pointer"
                      :disabled="loading"
                    >
                      <option value="last_24h">Last 24 Hours - Recent activity</option>
                      <option value="last_7d">Last 7 Days - Weekly summary</option>
                      <option value="last_30d">Last 30 Days - Monthly analysis</option>
                      <option value="last_90d">Last 90 Days - Quarterly review</option>
                    </select>
                    <Icon name="i-heroicons-chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
                  </div>
                </div>

                <!-- Report Language -->
                <div>
                  <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2" for="locale">
                    Language & Format
                  </label>
                  <div class="relative">
                    <select 
                      id="locale"
                      v-model="form.locale"
                      class="block w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 text-gray-900 dark:text-white shadow-sm transition-colors focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:focus:border-blue-400 appearance-none cursor-pointer"
                      :disabled="loading"
                    >
                      <option value="en-US">English (US) - International</option>
                      <option value="ru-RU">Русский - Russian Federation</option>
                      <option value="de-DE">Deutsch - Germany</option>
                    </select>
                    <Icon name="i-heroicons-chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400 pointer-events-none" />
                  </div>
                </div>

                <!-- Custom Title -->
                <div>
                  <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2" for="report-title">
                    Report Title
                    <span class="text-gray-400 font-normal">(optional)</span>
                  </label>
                  <input 
                    id="report-title"
                    v-model.trim="form.title"
                    type="text" 
                    class="block w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 shadow-sm transition-colors focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:focus:border-blue-400"
                    placeholder="Custom report title (auto-generated if empty)"
                    :disabled="loading"
                    maxlength="255"
                  />
                </div>

                <!-- Generation Preview -->
                <div class="rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 p-4">
                  <div class="flex items-start gap-3">
                    <Icon name="i-heroicons-document-text" class="h-5 w-5 text-green-600 dark:text-green-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <p class="text-sm font-medium text-green-900 dark:text-green-100">
                        Report Preview
                      </p>
                      <p class="text-sm text-green-700 dark:text-green-200 mt-1">
                        {{ getPreviewText() }}
                      </p>
                      <div class="flex items-center gap-4 mt-3 text-xs text-green-600 dark:text-green-400">
                        <span class="flex items-center gap-1">
                          <Icon name="i-heroicons-clock" class="h-3 w-3" />
                          ~2-5 min
                        </span>
                        <span class="flex items-center gap-1">
                          <Icon name="i-heroicons-document-arrow-down" class="h-3 w-3" />
                          PDF format
                        </span>
                      </div>
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
                class="inline-flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2.5 text-sm font-medium text-white shadow-sm hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-colors min-w-[120px]"
                @click="handleSubmit"
                :disabled="loading"
                type="button"
              >
                <Icon 
                  v-if="loading" 
                  name="i-heroicons-arrow-path" 
                  class="h-4 w-4 animate-spin" 
                />
                <Icon 
                  v-else 
                  name="i-heroicons-document-plus" 
                  class="h-4 w-4" 
                />
                {{ loading ? 'Generating...' : 'Generate Report' }}
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

// Refs
const modalRef = ref<HTMLElement>()

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
      form.template = 'executive'
      form.range = 'last_7d'
      form.locale = 'en-US'
      form.title = ''
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
</style>