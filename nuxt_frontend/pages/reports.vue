<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
      <div>
        <h1 class="u-h2">{{ t('reports.title') }}</h1>
        <p class="u-body text-gray-600 mt-1">
          {{ t('reports.subtitle') }}
        </p>
      </div>
      <button @click="showGenerateModal = true" class="u-btn u-btn-primary u-btn-md w-full lg:w-auto">
        <Icon name="heroicons:document-plus" class="w-4 h-4 mr-2" />
        {{ t('reports.generate.generateBtn') }}
      </button>
    </div>

    <!-- Reports List -->
    <div class="u-card">
      <div class="p-6 space-y-4">
        <div v-for="report in reports" :key="report.id" class="flex items-center gap-4 p-4 border rounded-lg hover:bg-gray-50 transition-colors">
          <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
            <Icon name="heroicons:document-text" class="w-6 h-6 text-blue-600" />
          </div>
          <div class="flex-1 min-w-0">
            <h3 class="font-semibold text-gray-900 truncate">{{ report.title }}</h3>
            <p class="text-sm text-gray-600">{{ report.description }}</p>
            <div class="flex items-center gap-4 mt-2 text-xs text-gray-500">
              <span>Генерация: {{ report.createdAt }}</span>
              <span>Период: {{ report.period }}</span>
              <span class="flex items-center gap-1">
                <div class="w-2 h-2 rounded-full" :class="getSeverityColor(report.severity)"></div>
                {{ getSeverityText(report.severity) }}
              </span>
            </div>
          </div>
          <div class="flex items-center gap-2">
            <span class="u-badge" :class="getStatusBadgeClass(report.status)">
              <Icon :name="getStatusIcon(report.status)" class="w-3 h-3" />
              {{ getStatusText(report.status) }}
            </span>
            <button class="u-btn u-btn-ghost u-btn-sm">
              <Icon name="heroicons:eye" class="w-4 h-4" />
            </button>
            <button class="u-btn u-btn-ghost u-btn-sm">
              <Icon name="heroicons:arrow-down-tray" class="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Generate Report Modal -->
    <UModal
      v-model="showGenerateModal"
      :title="t('reports.generate.title')"
      :description="t('reports.generate.subtitle')"
      :teleport-to="'#modal-portal'"
      size="lg"
    >
      <form @submit.prevent="generateReport" class="space-y-6">
        <!-- Report Template -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-3">
            {{ t('reports.generate.template') }}
          </label>
          <div class="grid gap-3 sm:grid-cols-2">
            <label v-for="template in reportTemplates" :key="template.key" class="relative">
              <input
                v-model="form.template"
                :value="template.key"
                type="radio"
                class="sr-only peer"
              />
              <div class="p-4 border border-gray-200 rounded-lg cursor-pointer transition-all peer-checked:border-blue-500 peer-checked:bg-blue-50 hover:border-gray-300">
                <div class="font-medium text-gray-900">{{ template.name }}</div>
                <div class="text-sm text-gray-600 mt-1">{{ template.description }}</div>
              </div>
            </label>
          </div>
        </div>

        <!-- Analysis Period -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-3">
            {{ t('reports.generate.period') }}
          </label>
          <select v-model="form.period" class="u-input">
            <option value="last_24h">{{ t('reports.periods.last_24h') }}</option>
            <option value="last_7d">{{ t('reports.periods.last_7d') }}</option>
            <option value="last_30d">{{ t('reports.periods.last_30d') }}</option>
            <option value="last_90d">{{ t('reports.periods.last_90d') }}</option>
          </select>
        </div>

        <!-- Language -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-3">
            {{ t('reports.generate.language') }}
          </label>
          <select v-model="form.locale" class="u-input">
            <option value="ru">{{ t('reports.locales.ru') }}</option>
            <option value="en">{{ t('reports.locales.en') }}</option>
            <option value="de">{{ t('reports.locales.de') }}</option>
          </select>
        </div>

        <!-- Custom Title -->
        <div>
          <label for="customTitle" class="block text-sm font-medium text-gray-700 mb-2">
            {{ t('reports.generate.customTitle') }}
            <span class="text-gray-500 text-xs ml-1">({{ t('ui.optional') }})</span>
          </label>
          <input
            id="customTitle"
            v-model="form.customTitle"
            type="text"
            class="u-input"
            :placeholder="t('reports.generate.customTitlePlaceholder')"
          />
          <p class="text-xs text-gray-500 mt-1">
            {{ t('reports.generate.autoTitle') }}
          </p>
        </div>
      </form>

      <template #footer>
        <button @click="showGenerateModal = false" type="button" class="u-btn u-btn-secondary flex-1">
          {{ t('ui.cancel') }}
        </button>
        <button @click="generateReport" type="submit" class="u-btn u-btn-primary flex-1" :disabled="isGenerating">
          <Icon v-if="isGenerating" name="heroicons:arrow-path" class="w-4 h-4 mr-2 animate-spin" />
          <Icon v-else name="heroicons:document-plus" class="w-4 h-4 mr-2" />
          {{ isGenerating ? t('reports.generate.generating') : t('reports.generate.generateBtn') }}
        </button>
      </template>
    </UModal>
  </div>
</template>

<script setup lang="ts">
definePageMeta({ middleware: ['auth'] })

const { t } = useI18n()

const showGenerateModal = ref(false)
const isGenerating = ref(false)

const form = ref({ template: 'executive', period: 'last_7d', locale: 'ru', customTitle: '' })

const reports = ref([
  { id: 1, title: 'Executive Summary - Weekly Report', description: 'Краткий обзор состояния гидравлических систем', createdAt: '2 часа назад', period: 'Последние 7 дней', severity: 'low' as const, status: 'completed' as const },
  { id: 2, title: 'Technical Analysis - System HYD-001', description: 'Детальный технический анализ насосной станции', createdAt: '1 день назад', period: 'Последние 30 дней', severity: 'medium' as const, status: 'completed' as const }
])

const reportTemplates = [
  { key: 'executive', name: t('reports.templates.execShort'), description: t('reports.templates.executive') },
  { key: 'technical', name: t('reports.templates.techShort'), description: t('reports.templates.technical') },
  { key: 'compliance', name: t('reports.templates.compShort'), description: t('reports.templates.compliance') },
  { key: 'maintenance', name: t('reports.templates.maintShort'), description: t('reports.templates.maintenance') }
]

const getSeverityColor = (severity: string): string => ({ low: 'bg-green-500', medium: 'bg-yellow-500', high: 'bg-orange-500', critical: 'bg-red-500' }[severity] || 'bg-gray-500')
const getSeverityText = (severity: string): string => ({ low: t('reports.severity.low'), medium: t('reports.severity.medium'), high: t('reports.severity.high'), critical: t('reports.severity.critical') }[severity] || severity)
const getStatusBadgeClass = (status: string): string => ({ completed: 'u-badge-success', in_progress: 'u-badge-info', pending: 'u-badge-warning', failed: 'u-badge-error' }[status] || 'u-badge-info')
const getStatusIcon = (status: string): string => ({ completed: 'heroicons:check-circle', in_progress: 'heroicons:arrow-path', pending: 'heroicons:clock', failed: 'heroicons:x-circle' }[status] || 'heroicons:question-mark-circle')
const getStatusText = (status: string): string => ({ completed: t('reports.status.completed'), in_progress: t('reports.status.in_progress'), pending: t('reports.status.pending'), failed: t('reports.status.failed') }[status] || status)

const generateReport = async () => {
  isGenerating.value = true
  setTimeout(() => {
    const templateName = reportTemplates.find(t => t.key === form.value.template)?.name || ''
    const periodText = t(`reports.periods.${form.value.period}`)
    const newReport = { id: Date.now(), title: form.value.customTitle || `${templateName} - ${new Date().toLocaleDateString()}`, description: `Отчёт за ${periodText}`, createdAt: 'Только что', period: periodText, severity: 'low' as const, status: 'completed' as const }
    reports.value.unshift(newReport)
    showGenerateModal.value = false
    isGenerating.value = false
  }, 2000)
}
</script>