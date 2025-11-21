<template>
  <div class="diagnosis-history" role="region" aria-labelledby="history-title">
    <div class="u-card p-6">
      <h2 id="history-title" class="u-h4 mb-6">
        📅 История диагностики
      </h2>

      <!-- Filters -->
      <div class="flex flex-wrap gap-4 mb-6">
        <div>
          <label for="status-filter" class="u-label mb-2 block">Фильтр по статусу</label>
          <select id="status-filter" v-model="selectedStatus" class="u-input w-48" @change="filterHistory">
            <option value="all">Все</option>
            <option value="normal">Нормально</option>
            <option value="warning">Предупреждение</option>
            <option value="critical">Критическое</option>
          </select>
        </div>

        <div>
          <label for="date-range" class="u-label mb-2 block">Период</label>
          <select id="date-range" v-model="selectedDateRange" class="u-input w-48" @change="filterHistory">
            <option value="all">Всё время</option>
            <option value="today">Сегодня</option>
            <option value="week">Неделя</option>
            <option value="month">Месяц</option>
          </select>
        </div>
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="text-center py-12">
        <LoadingSpinner />
        <p class="u-body-sm text-gray-600 mt-4">Загрузка истории...</p>
      </div>

      <!-- Empty State -->
      <div v-else-if="!filteredHistory.length" class="text-center py-12">
        <svg class="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <p class="u-body text-gray-600 mb-2">История пуста</p>
        <p class="u-body-sm text-gray-500">Нет записей, соответствующих выбранным фильтрам</p>
      </div>

      <!-- Timeline -->
      <div v-else class="space-y-4">
        <div v-for="(item, index) in filteredHistory" :key="item.id" class="relative pl-8 pb-8 last:pb-0">
          <!-- Timeline Line -->
          <div v-if="index < filteredHistory.length - 1" class="absolute left-3 top-8 bottom-0 w-0.5 bg-gray-200"
            aria-hidden="true"></div>

          <!-- Timeline Dot -->
          <div class="absolute left-0 top-1 w-6 h-6 rounded-full flex items-center justify-center"
            :class="getTimelineDotClass(item.status)" :aria-label="`Статус: ${getStatusLabel(item.status)}`">
            <div class="w-3 h-3 rounded-full bg-white"></div>
          </div>

          <!-- History Item Card -->
          <div class="u-card p-4 hover:shadow-md transition-shadow cursor-pointer" @click="$emit('select', item)"
            @keypress.enter="$emit('select', item)" tabindex="0" role="button"
            :aria-label="`Диагностика от ${formatDate(item.timestamp)}, статус: ${getStatusLabel(item.status)}`">
            <div class="flex items-start justify-between gap-4 mb-3">
              <div class="flex-1">
                <div class="text-sm text-gray-600 mb-1">
                  {{ formatDate(item.timestamp) }}
                </div>
                <div class="font-semibold text-gray-900">
                  {{ item.equipmentName || 'Неизвестное оборудование' }}
                </div>
              </div>

              <!-- Status Badge -->
              <div class="px-3 py-1 rounded-full text-xs font-medium" :class="getStatusBadgeClass(item.status)">
                {{ getStatusLabel(item.status) }}
              </div>
            </div>

            <!-- Quick Stats -->
            <div class="grid grid-cols-3 gap-4 mb-3 text-sm">
              <div>
                <div class="text-gray-500 text-xs mb-1">Уверенность</div>
                <div class="font-semibold" :class="getConfidenceColor(item.confidence)">
                  {{ Math.round(item.confidence * 100) }}%
                </div>
              </div>
              <div>
                <div class="text-gray-500 text-xs mb-1">Аномалии</div>
                <div class="font-semibold text-gray-900">
                  {{ item.anomalyCount || 0 }}
                </div>
              </div>
              <div>
                <div class="text-gray-500 text-xs mb-1">Модели</div>
                <div class="font-semibold text-gray-900">
                  {{ item.modelCount || 4 }}/4
                </div>
              </div>
            </div>

            <!-- Tags -->
            <div v-if="item.tags?.length" class="flex flex-wrap gap-2">
              <span v-for="tag in item.tags" :key="tag" class="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded">
                {{ tag }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Pagination -->
      <div v-if="totalPages > 1" class="flex items-center justify-center gap-2 mt-6 pt-6 border-t">
        <button class="u-btn u-btn-sm u-btn-secondary" :disabled="currentPage === 1" @click="previousPage">
          ← Предыдущая
        </button>
        <span class="text-sm text-gray-600">
          Страница {{ currentPage }} из {{ totalPages }}
        </span>
        <button class="u-btn u-btn-sm u-btn-secondary" :disabled="currentPage === totalPages" @click="nextPage">
          Следующая →
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from '#imports'
import type { DiagnosticHistoryItem } from '~/types/diagnostics'
import LoadingSpinner from '~/components/Loading/LoadingSpinner.vue'

interface Props {
  history: DiagnosticHistoryItem[]
  loading?: boolean
  itemsPerPage?: number
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
  itemsPerPage: 10,
})

defineEmits<{
  select: [item: DiagnosticHistoryItem]
}>()

const selectedStatus = ref<string>('all')
const selectedDateRange = ref<string>('all')
const currentPage = ref<number>(1)

const filteredHistory = computed<DiagnosticHistoryItem[]>(() => {
  let result = props.history

  // Filter by status
  if (selectedStatus.value !== 'all') {
    result = result.filter(item => item.status === selectedStatus.value)
  }

  // Filter by date range
  if (selectedDateRange.value !== 'all') {
    const now = new Date()
    const filterDate = new Date()

    switch (selectedDateRange.value) {
      case 'today':
        filterDate.setHours(0, 0, 0, 0)
        break
      case 'week':
        filterDate.setDate(now.getDate() - 7)
        break
      case 'month':
        filterDate.setMonth(now.getMonth() - 1)
        break
    }

    result = result.filter(item => new Date(item.timestamp) >= filterDate)
  }

  // Pagination
  const start = (currentPage.value - 1) * props.itemsPerPage
  const end = start + props.itemsPerPage
  return result.slice(start, end)
})

const totalPages = computed<number>(() => {
  return Math.ceil(props.history.length / props.itemsPerPage)
})

const filterHistory = (): void => {
  currentPage.value = 1 // Reset to first page when filtering
}

const previousPage = (): void => {
  if (currentPage.value > 1) {
    currentPage.value--
  }
}

const nextPage = (): void => {
  if (currentPage.value < totalPages.value) {
    currentPage.value++
  }
}

const formatDate = (timestamp: string): string => {
  return new Date(timestamp).toLocaleString('ru-RU', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const getStatusLabel = (status: string): string => {
  const labels: Record<string, string> = {
    normal: 'Нормально',
    warning: 'Предупреждение',
    critical: 'Критическое',
    unknown: 'Неизвестно',
  }
  return labels[status] || status
}

const getTimelineDotClass = (status: string): string => {
  const classes: Record<string, string> = {
    normal: 'bg-green-500',
    warning: 'bg-yellow-500',
    critical: 'bg-red-500',
    unknown: 'bg-gray-500',
  }
  return classes[status] || classes.unknown!
}

const getStatusBadgeClass = (status: string): string => {
  const classes: Record<string, string> = {
    normal: 'bg-green-100 text-green-700 border border-green-300',
    warning: 'bg-yellow-100 text-yellow-700 border border-yellow-300',
    critical: 'bg-red-100 text-red-700 border border-red-300',
    unknown: 'bg-gray-100 text-gray-700 border border-gray-300',
  }
  return classes[status] || classes.unknown!
}

const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.8) return 'text-green-600'
  if (confidence >= 0.5) return 'text-yellow-600'
  return 'text-red-600'
}
</script>

<style scoped>
.diagnosis-history {
  @apply max-w-5xl mx-auto;
}
</style>
