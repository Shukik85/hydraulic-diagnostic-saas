<script setup lang="ts">
// Enhanced dashboard with proper TypeScript and null safety
definePageMeta({
  middleware: 'auth'
})

useSeoMeta({
  title: 'Дашборд | Hydraulic Diagnostic SaaS',
  description: 'Мониторинг гидравлических систем в реальном времени'
})

const authStore = useAuthStore()

// Demo stats with null safety
const stats = computed(() => {
  const user = authStore.user
  if (!user) {
    return {
      totalSystems: 0,
      activeSystems: 0,
      reportsGenerated: 0,
      criticalAlerts: 0,
      uptime: 0
    }
  }
  
  const totalSystems = user.systems_count || 12
  const activeSystems = Math.floor(totalSystems * 0.9)
  const reportsGenerated = user.reports_generated || 847
  
  // Safe calculation
  const healthyRatio = totalSystems > 0 ? (activeSystems / totalSystems) * 100 : 100
  
  return {
    totalSystems,
    activeSystems,
    reportsGenerated,
    criticalAlerts: Math.max(0, totalSystems - activeSystems),
    uptime: Math.round(healthyRatio)
  }
})

// Async data for systems
const { data: systems, error: systemsError, pending: systemsLoading, refresh: refreshSystems } = await useAsyncData('hydraulic-systems', () => {
  // Simulate API call
  return new Promise<any[]>((resolve) => {
    setTimeout(() => {
      resolve([
        {
          id: 1,
          name: 'HYD-001 - Насосная станция A',
          status: 'active',
          location: 'Цех №1',
          temperature: 45.2,
          pressure: 150.8,
          efficiency_score: 94,
          components_count: 12,
          last_reading_at: new Date().toISOString()
        },
        {
          id: 2,
          name: 'HYD-002 - Гидромотор B',
          status: 'warning',
          location: 'Цех №2',
          temperature: 52.1,
          pressure: 145.2,
          efficiency_score: 78,
          components_count: 8,
          last_reading_at: new Date(Date.now() - 300000).toISOString()
        },
        {
          id: 3,
          name: 'HYD-003 - Клапан управления C',
          status: 'maintenance',
          location: 'Цех №3',
          temperature: 41.8,
          pressure: 140.0,
          efficiency_score: 85,
          components_count: 6,
          last_reading_at: new Date(Date.now() - 600000).toISOString()
        }
      ])
    }, 100)
  })
})

// Async data for reports
const { data: reports, error: reportsError, pending: reportsLoading, refresh: refreshReports } = await useAsyncData('recent-reports', () => {
  return new Promise<any[]>((resolve) => {
    setTimeout(() => {
      resolve([
        {
          id: 1,
          title: 'Еженедельная проверка HYD-001',
          severity: 'low',
          status: 'completed',
          summary: 'Система работает в нормальном режиме. Необходима плановая замена фильтра.',
          created_at: new Date(Date.now() - 3600000).toISOString(),
          system_id: 1
        },
        {
          id: 2,
          title: 'Анализ давления HYD-002',
          severity: 'medium',
          status: 'completed',
          summary: 'Обнаружены колебания давления. Рекомендуется проверка уплотнений.',
          created_at: new Date(Date.now() - 7200000).toISOString(),
          system_id: 2
        }
      ])
    }, 150)
  })
})

// Status helpers
const getSystemStatusColor = (status: string): string => {
  switch (status) {
    case 'active': return 'text-green-600 dark:text-green-400'
    case 'warning': return 'text-yellow-600 dark:text-yellow-400'
    case 'maintenance': return 'text-blue-600 dark:text-blue-400'
    case 'critical': return 'text-red-600 dark:text-red-400'
    default: return 'text-gray-500 dark:text-gray-400'
  }
}

const getSystemStatusIcon = (status: string): string => {
  switch (status) {
    case 'active': return 'heroicons:check-circle'
    case 'warning': return 'heroicons:exclamation-triangle'
    case 'maintenance': return 'heroicons:wrench-screwdriver'
    case 'critical': return 'heroicons:x-circle'
    default: return 'heroicons:question-mark-circle'
  }
}

const getSeverityColor = (severity: string): string => {
  switch (severity) {
    case 'low': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
    case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
    case 'high': return 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300'
    case 'critical': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
    default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300'
  }
}

// Format date with null safety
const formatDateTime = (dateString: string | undefined): string => {
  if (!dateString) return 'Нет данных'
  try {
    return new Date(dateString).toLocaleString('ru-RU', {
      day: '2-digit',
      month: '2-digit', 
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch {
    return 'Неверная дата'
  }
}

// Fixed event handlers - remove opts parameter
const handleRefreshSystems = async (): Promise<void> => {
  await refreshSystems()
}

const handleRefreshReports = async (): Promise<void> => {
  await refreshReports()
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <div class="container mx-auto px-4 py-8">
      <!-- Header -->
      <div class="mb-8">
        <h1 class="premium-heading-xl text-gray-900 dark:text-white mb-2">
          📈 Дашборд
        </h1>
        <p class="premium-body text-gray-600 dark:text-gray-300">
          Мониторинг гидравлических систем в реальном времени
        </p>
      </div>

      <!-- Stats cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <!-- Total Systems -->
        <div class="premium-card p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Всего систем</p>
              <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ stats.totalSystems }}</p>
              <p class="text-xs text-blue-600 dark:text-blue-400 mt-1">
                <Icon name="heroicons:arrow-trending-up" class="w-3 h-3 inline mr-1" />
                +2 за месяц
              </p>
            </div>
            <div class="p-3 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
              <Icon name="heroicons:server-stack" class="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
          </div>
        </div>

        <!-- Active Systems -->
        <div class="premium-card p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Активные</p>
              <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ stats.activeSystems }}</p>
              <p class="text-xs text-green-600 dark:text-green-400 mt-1">
                <Icon name="heroicons:check-circle" class="w-3 h-3 inline mr-1" />
                {{ stats.uptime }}% uptime
              </p>
            </div>
            <div class="p-3 bg-green-50 dark:bg-green-900/30 rounded-lg">
              <Icon name="heroicons:play" class="w-6 h-6 text-green-600 dark:text-green-400" />
            </div>
          </div>
        </div>

        <!-- Reports Generated -->
        <div class="premium-card p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Отчёты</p>
              <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ stats.reportsGenerated }}</p>
              <p class="text-xs text-green-600 dark:text-green-400 mt-1">
                <Icon name="heroicons:arrow-trending-up" class="w-3 h-3 inline mr-1" />
                +{{ Math.round((stats.reportsGenerated || 0) * 0.08) }} за неделю
              </p>
            </div>
            <div class="p-3 bg-purple-50 dark:bg-purple-900/30 rounded-lg">
              <Icon name="heroicons:document-text" class="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
          </div>
        </div>

        <!-- Critical Alerts -->
        <div class="premium-card p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Критические</p>
              <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ stats.criticalAlerts }}</p>
              <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                <Icon name="heroicons:clock" class="w-3 h-3 inline mr-1" />
                Последние 24 часа
              </p>
            </div>
            <div class="p-3 bg-red-50 dark:bg-red-900/30 rounded-lg">
              <Icon name="heroicons:exclamation-triangle" class="w-6 h-6 text-red-600 dark:text-red-400" />
            </div>
          </div>
        </div>
      </div>

      <!-- Systems Grid -->
      <div class="mb-8">
        <div class="flex items-center justify-between mb-6">
          <h2 class="premium-heading-lg text-gray-900 dark:text-white">🔧 Гидравлические системы</h2>
          <div class="flex items-center space-x-3">
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span class="text-sm text-gray-500 dark:text-gray-400">Real-time</span>
              <button 
                @click="handleRefreshSystems"
                :disabled="systemsLoading"
                class="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 disabled:opacity-50 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/30"
                title="Обновить данные"
              >
                <Icon name="heroicons:arrow-path" class="w-4 h-4" :class="{ 'animate-spin': systemsLoading }" />
              </button>
            </div>
          </div>
        </div>

        <!-- Loading state -->
        <div v-if="systemsLoading" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div v-for="i in 3" :key="i" class="premium-card p-6">
            <div class="animate-pulse">
              <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded mb-3"></div>
              <div class="h-8 bg-gray-200 dark:bg-gray-700 rounded mb-2"></div>
              <div class="h-3 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
            </div>
          </div>
        </div>

        <!-- Error state -->
        <div v-else-if="systemsError" class="premium-card p-12 text-center">
          <Icon name="heroicons:exclamation-triangle" class="w-12 h-12 mx-auto text-red-500 mb-4" />
          <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">Ошибка загрузки систем</h3>
          <p class="text-gray-600 dark:text-gray-300 mb-6">Не удалось получить данные о системах</p>
          <button 
            @click="handleRefreshSystems"
            class="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
          >
            Попробовать снова
          </button>
        </div>

        <!-- Systems data -->
        <div v-else-if="systems?.length" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div v-for="system in systems" :key="system.id" class="premium-card hover:shadow-xl transition-shadow">
            <div class="p-6">
              <!-- Status and name -->
              <div class="flex items-start justify-between mb-4">
                <div class="flex-1">
                  <h3 class="font-semibold text-gray-900 dark:text-white mb-1">{{ system.name }}</h3>
                  <div class="flex items-center space-x-2">
                    <Icon :name="getSystemStatusIcon(system.status)" class="w-4 h-4" :class="getSystemStatusColor(system.status)" />
                    <span class="text-sm font-medium capitalize" :class="getSystemStatusColor(system.status)">
                      {{ system.status }}
                    </span>
                  </div>
                </div>
                <div class="text-right">
                  <div class="text-2xl font-bold text-gray-900 dark:text-white">{{ system.efficiency_score }}%</div>
                  <div class="text-xs text-gray-500 dark:text-gray-400">Эффективность</div>
                </div>
              </div>

              <!-- Metrics -->
              <div class="grid grid-cols-2 gap-4 mb-4">
                <div class="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div class="text-lg font-semibold text-gray-900 dark:text-white">{{ system.temperature }}°C</div>
                  <div class="text-xs text-gray-500 dark:text-gray-400">Температура</div>
                </div>
                <div class="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div class="text-lg font-semibold text-gray-900 dark:text-white">{{ system.pressure }} бар</div>
                  <div class="text-xs text-gray-500 dark:text-gray-400">Давление</div>
                </div>
              </div>

              <!-- System info -->
              <div class="space-y-2 text-xs text-gray-500 dark:text-gray-400">
                <div class="flex items-center justify-between">
                  <span class="flex items-center">
                    <Icon name="heroicons:map-pin" class="w-4 h-4 mr-1" />
                    {{ system.location }}
                  </span>
                  <span class="flex items-center">
                    <Icon name="heroicons:cog-6-tooth" class="w-4 h-4 mr-1" />
                    {{ system.components_count }} компонентов
                  </span>
                </div>
                <div class="flex items-center justify-between">
                  <span v-if="system.last_reading_at" class="flex items-center">
                    <Icon name="heroicons:signal" class="w-4 h-4 mr-1" />
                    {{ formatDateTime(system.last_reading_at) }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Recent Reports -->
      <div>
        <div class="flex items-center justify-between mb-6">
          <h2 class="premium-heading-lg text-gray-900 dark:text-white">📊 Последние отчёты</h2>
          <div class="flex items-center space-x-3">
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <span class="text-sm text-gray-500 dark:text-gray-400">Live</span>
              <button 
                @click="handleRefreshReports"
                :disabled="reportsLoading"
                class="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 disabled:opacity-50 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/30"
                title="Обновить отчёты"
              >
                <Icon name="heroicons:arrow-path" class="w-4 h-4" :class="{ 'animate-spin': reportsLoading }" />
              </button>
            </div>
          </div>
        </div>

        <!-- Reports list -->
        <div v-if="reports?.length" class="space-y-4">
          <div v-for="report in reports" :key="report.id" class="premium-card hover:shadow-md transition-shadow">
            <div class="p-6">
              <div class="flex items-start justify-between">
                <div class="flex-1">
                  <div class="flex items-center space-x-3 mb-2">
                    <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium" :class="getSeverityColor(report.severity)">
                      {{ report.severity.toUpperCase() }}
                    </span>
                    <h3 class="font-semibold text-gray-900 dark:text-white">
                      {{ report.title }}
                    </h3>
                  </div>
                  <p class="text-sm text-gray-600 dark:text-gray-300 mb-3 line-clamp-2">
                    {{ report.summary || report.description || 'Описание отсутствует' }}
                  </p>
                  <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-3">
                      <span class="text-xs text-gray-500 dark:text-gray-400">
                        <Icon name="heroicons:calendar" class="w-3 h-3 inline mr-1" />
                        {{ formatDateTime(report.created_at) }}
                      </span>
                      <span class="text-xs text-gray-500 dark:text-gray-400">
                        <Icon name="heroicons:server" class="w-3 h-3 inline mr-1" />
                        Система #{{ report.system_id }}
                      </span>
                    </div>
                    <PremiumButton variant="secondary" size="sm">
                      Подробнее
                    </PremiumButton>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Empty state -->
        <div v-else-if="!reportsLoading" class="premium-card p-12 text-center">
          <Icon name="heroicons:document-text" class="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">Отчёты отсутствуют</h3>
          <p class="text-gray-500 dark:text-gray-400">Новые отчёты появятся по мере выполнения диагностики</p>
        </div>

        <!-- Loading state -->
        <div v-else class="space-y-4">
          <div v-for="i in 2" :key="i" class="premium-card p-6">
            <div class="animate-pulse">
              <div class="flex items-center space-x-3 mb-3">
                <div class="h-5 bg-gray-200 dark:bg-gray-700 rounded w-16"></div>
                <div class="h-5 bg-gray-200 dark:bg-gray-700 rounded flex-1"></div>
              </div>
              <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded mb-2"></div>
              <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>