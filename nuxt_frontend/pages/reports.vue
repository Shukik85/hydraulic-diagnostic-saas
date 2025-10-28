<script setup lang="ts">
// Fixed reports page without UiDialog components
definePageMeta({
  middleware: 'auth',
});

useSeoMeta({
  title: 'Отчёты | Hydraulic Diagnostic SaaS',
  description: 'Comprehensive diagnostic reports and analytics for hydraulic systems',
});

interface Report {
  id: number;
  title: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  system_name: string;
  created_at: string;
  completed_at?: string;
  summary?: string;
  recommendations?: string[];
}

// Demo reports data
const reports = ref<Report[]>([
  {
    id: 1,
    title: 'Анализ эффективности HYD-001',
    severity: 'medium',
    status: 'completed',
    system_name: 'Насосная станция A',
    created_at: '2024-10-24T10:30:00Z',
    completed_at: '2024-10-24T10:45:00Z',
    summary:
      'Система работает в пределах нормы. Обнаружены незначительные отклонения в температурном режиме.',
    recommendations: [
      'Проверить систему охлаждения',
      'Заменить фильтр в течение недели',
      'Откалибровать датчик температуры',
    ],
  },
  {
    id: 2,
    title: 'Диагностика давления HYD-002',
    severity: 'high',
    status: 'completed',
    system_name: 'Гидромотор B',
    created_at: '2024-10-24T09:15:00Z',
    completed_at: '2024-10-24T09:30:00Z',
    summary: 'Критические колебания давления. Требуется немедленное вмешательство.',
    recommendations: [
      'Остановить систему для проверки',
      'Проверить состояние уплотнений',
      'Заменить клапан регулировки давления',
    ],
  },
  {
    id: 3,
    title: 'Профилактическая проверка HYD-003',
    severity: 'low',
    status: 'in_progress',
    system_name: 'Клапан управления C',
    created_at: '2024-10-24T08:00:00Z',
  },
]);

// Modal state
const selectedReport = ref<Report | null>(null);
const showReportModal = ref<boolean>(false);

// Filter and sort
const selectedSeverity = ref<string>('all');
const selectedStatus = ref<string>('all');
const searchQuery = ref<string>('');

// Computed filtered reports
const filteredReports = computed(() => {
  return reports.value.filter(report => {
    const matchesSearch =
      !searchQuery.value ||
      report.title.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
      report.system_name.toLowerCase().includes(searchQuery.value.toLowerCase());

    const matchesSeverity =
      selectedSeverity.value === 'all' || report.severity === selectedSeverity.value;
    const matchesStatus = selectedStatus.value === 'all' || report.status === selectedStatus.value;

    return matchesSearch && matchesSeverity && matchesStatus;
  });
});

// Helper functions
const getSeverityColor = (severity: string): string => {
  switch (severity) {
    case 'low':
      return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
    case 'medium':
      return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300';
    case 'high':
      return 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300';
    case 'critical':
      return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300';
  }
};

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'completed':
      return 'text-green-600 dark:text-green-400';
    case 'in_progress':
      return 'text-blue-600 dark:text-blue-400';
    case 'pending':
      return 'text-yellow-600 dark:text-yellow-400';
    case 'failed':
      return 'text-red-600 dark:text-red-400';
    default:
      return 'text-gray-500 dark:text-gray-400';
  }
};

const getStatusIcon = (status: string): string => {
  switch (status) {
    case 'completed':
      return 'heroicons:check-circle';
    case 'in_progress':
      return 'heroicons:clock';
    case 'pending':
      return 'heroicons:pause-circle';
    case 'failed':
      return 'heroicons:x-circle';
    default:
      return 'heroicons:question-mark-circle';
  }
};

const formatDateTime = (dateString: string): string => {
  return new Date(dateString).toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const openReportModal = (report: Report): void => {
  selectedReport.value = report;
  showReportModal.value = true;
};

const closeReportModal = (): void => {
  selectedReport.value = null;
  showReportModal.value = false;
};

// Handle ESC key
onMounted(() => {
  const handleEsc = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      closeReportModal();
    }
  };
  document.addEventListener('keydown', handleEsc);

  onUnmounted(() => {
    document.removeEventListener('keydown', handleEsc);
  });
});
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <div class="container mx-auto px-4 py-8">
      <!-- Header -->
      <div class="mb-8">
        <h1 class="premium-heading-xl text-gray-900 dark:text-white mb-2">📊 Отчёты диагностики</h1>
        <p class="premium-body text-gray-600 dark:text-gray-300">
          Comprehensive analysis and recommendations for hydraulic systems
        </p>
      </div>

      <!-- Filters -->
      <div class="premium-card p-6 mb-8">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <!-- Search -->
          <div class="md:col-span-2">
            <label class="premium-label">Поиск по отчётам</label>
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Название отчёта или система..."
              class="premium-input"
            />
          </div>

          <!-- Severity Filter -->
          <div>
            <label class="premium-label">Критичность</label>
            <select v-model="selectedSeverity" class="premium-input">
              <option value="all">Все уровни</option>
              <option value="low">Низкая</option>
              <option value="medium">Средняя</option>
              <option value="high">Высокая</option>
              <option value="critical">Критическая</option>
            </select>
          </div>

          <!-- Status Filter -->
          <div>
            <label class="premium-label">Статус</label>
            <select v-model="selectedStatus" class="premium-input">
              <option value="all">Все статусы</option>
              <option value="completed">Завершён</option>
              <option value="in_progress">Выполняется</option>
              <option value="pending">Ожидает</option>
              <option value="failed">Ошибка</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Reports List -->
      <div class="space-y-6">
        <div v-if="filteredReports.length === 0" class="premium-card p-12 text-center">
          <Icon name="heroicons:document-text" class="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">Отчёты не найдены</h3>
          <p class="text-gray-500 dark:text-gray-400">Попробуйте изменить фильтры поиска</p>
        </div>

        <div
          v-for="report in filteredReports"
          :key="report.id"
          class="premium-card hover:shadow-xl transition-all duration-300 cursor-pointer"
          @click="openReportModal(report)"
        >
          <div class="p-6">
            <div class="flex items-start justify-between mb-4">
              <div class="flex-1">
                <div class="flex items-center space-x-3 mb-2">
                  <span
                    class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium"
                    :class="getSeverityColor(report.severity)"
                  >
                    {{ report.severity.toUpperCase() }}
                  </span>
                  <div class="flex items-center space-x-2">
                    <Icon
                      :name="getStatusIcon(report.status)"
                      class="w-4 h-4"
                      :class="getStatusColor(report.status)"
                    />
                    <span
                      class="text-sm font-medium capitalize"
                      :class="getStatusColor(report.status)"
                    >
                      {{ report.status.replace('_', ' ') }}
                    </span>
                  </div>
                </div>

                <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  {{ report.title }}
                </h3>

                <p class="text-sm text-gray-600 dark:text-gray-300 mb-3">
                  {{ report.summary || 'Подробная информация доступна в отчёте' }}
                </p>

                <div class="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                  <span class="flex items-center">
                    <Icon name="heroicons:server" class="w-3 h-3 mr-1" />
                    {{ report.system_name }}
                  </span>
                  <span class="flex items-center">
                    <Icon name="heroicons:calendar" class="w-3 h-3 mr-1" />
                    {{ formatDateTime(report.created_at) }}
                  </span>
                  <span v-if="report.completed_at" class="flex items-center">
                    <Icon name="heroicons:check" class="w-3 h-3 mr-1" />
                    Завершён {{ formatDateTime(report.completed_at) }}
                  </span>
                </div>
              </div>

              <div class="text-right">
                <PremiumButton variant="secondary" size="sm"> Подробнее </PremiumButton>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Native HTML Modal (instead of UiDialog) -->
    <div
      v-if="showReportModal && selectedReport"
      class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
      @click="closeReportModal"
    >
      <div class="premium-card max-w-4xl w-full max-h-[90vh] overflow-y-auto" @click.stop>
        <!-- Modal Header -->
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <div class="flex items-start justify-between">
            <div class="flex-1">
              <div class="flex items-center space-x-3 mb-2">
                <span
                  class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium"
                  :class="getSeverityColor(selectedReport.severity)"
                >
                  {{ selectedReport.severity.toUpperCase() }}
                </span>
                <div class="flex items-center space-x-2">
                  <Icon
                    :name="getStatusIcon(selectedReport.status)"
                    class="w-4 h-4"
                    :class="getStatusColor(selectedReport.status)"
                  />
                  <span
                    class="text-sm font-medium capitalize"
                    :class="getStatusColor(selectedReport.status)"
                  >
                    {{ selectedReport.status.replace('_', ' ') }}
                  </span>
                </div>
              </div>

              <h2 class="premium-heading-lg text-gray-900 dark:text-white mb-2">
                {{ selectedReport.title }}
              </h2>

              <p class="text-sm text-gray-500 dark:text-gray-400">
                {{ selectedReport.system_name }} • {{ formatDateTime(selectedReport.created_at) }}
              </p>
            </div>

            <button
              @click="closeReportModal"
              class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
            >
              <Icon name="heroicons:x-mark" class="w-6 h-6" />
            </button>
          </div>
        </div>

        <!-- Modal Content -->
        <div class="p-6">
          <!-- Summary -->
          <div v-if="selectedReport.summary" class="mb-8">
            <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-3">📋 Сводка</h3>
            <div class="p-4 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
              <p class="premium-body text-gray-700 dark:text-gray-300">
                {{ selectedReport.summary }}
              </p>
            </div>
          </div>

          <!-- Recommendations -->
          <div v-if="selectedReport.recommendations?.length" class="mb-8">
            <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-3">💡 Рекомендации</h3>
            <div class="space-y-3">
              <div
                v-for="(recommendation, index) in selectedReport.recommendations"
                :key="index"
                class="flex items-start space-x-3 p-4 bg-green-50 dark:bg-green-900/30 rounded-lg"
              >
                <Icon
                  name="heroicons:light-bulb"
                  class="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5 flex-shrink-0"
                />
                <p class="premium-body text-gray-700 dark:text-gray-300">{{ recommendation }}</p>
              </div>
            </div>
          </div>

          <!-- Technical Details -->
          <div class="mb-8">
            <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-3">
              🔧 Технические данные
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="text-sm text-gray-500 dark:text-gray-400 mb-1">ID отчёта</div>
                <div class="font-mono text-sm text-gray-900 dark:text-white">
                  #{{ selectedReport.id.toString().padStart(4, '0') }}
                </div>
              </div>

              <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="text-sm text-gray-500 dark:text-gray-400 mb-1">Система</div>
                <div class="text-sm text-gray-900 dark:text-white">
                  {{ selectedReport.system_name }}
                </div>
              </div>

              <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="text-sm text-gray-500 dark:text-gray-400 mb-1">Создан</div>
                <div class="text-sm text-gray-900 dark:text-white">
                  {{ formatDateTime(selectedReport.created_at) }}
                </div>
              </div>

              <div
                v-if="selectedReport.completed_at"
                class="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
              >
                <div class="text-sm text-gray-500 dark:text-gray-400 mb-1">Завершён</div>
                <div class="text-sm text-gray-900 dark:text-white">
                  {{ formatDateTime(selectedReport.completed_at) }}
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Modal Footer -->
        <div class="p-6 border-t border-gray-200 dark:border-gray-700">
          <div class="flex items-center justify-end space-x-3">
            <PremiumButton variant="secondary" @click="closeReportModal"> Закрыть </PremiumButton>
            <PremiumButton icon="heroicons:arrow-down-tray" gradient> Скачать PDF </PremiumButton>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Additional styles for modal */
.line-clamp-3 {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
