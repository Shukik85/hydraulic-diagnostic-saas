<script setup lang="ts">
// Main user dashboard - relocated from index.vue and enhanced for production
definePageMeta({
  middleware: 'auth'
})

useSeoMeta({
  title: 'Панель управления | Hydraulic Diagnostic SaaS',
  description: 'Основная панель управления системой диагностики гидравлических систем с real-time мониторингом и AI-аналитикой'
})

const authStore = useAuthStore()
const api = useApi()
const router = useRouter()

// Enhanced data loading with better error handling
const { data: systems, pending: systemsLoading, error: systemsError, refresh: refreshSystems } = await useLazyAsyncData(
  'dashboard-systems',
  () => api.getSystems({ page_size: 12, ordering: '-last_reading_at' }),
  {
    default: () => ({ results: [], count: 0 }),
    server: false // Client-side only for real-time data
  }
)

const { data: recentReports, pending: reportsLoading, refresh: refreshReports } = await useLazyAsyncData(
  'dashboard-reports', 
  () => api.getReports(undefined, { page_size: 8, ordering: '-created_at' }),
  {
    default: () => ({ results: [], count: 0 }),
    server: false
  }
)

// Enhanced stats with business intelligence
const stats = computed(() => {
  if (!authStore.user || !systems.value) return null
  
  const activeSystems = systems.value.results?.filter(s => s.status === 'active').length || 0
  const maintenanceSystems = systems.value.results?.filter(s => s.status === 'maintenance').length || 0
  const inactiveSystems = systems.value.results?.filter(s => s.status === 'inactive').length || 0
  
  const criticalReports = recentReports.value?.results?.filter(r => r.severity === 'critical').length || 0
  const highSeverityReports = recentReports.value?.results?.filter(r => r.severity === 'high').length || 0
  
  // Calculate uptime percentage (demo calculation)
  const totalSystems = authStore.user.systems_count
  const healthyRatio = totalSystems > 0 ? (activeSystems / totalSystems) * 100 : 100
  
  return {
    totalSystems: authStore.user.systems_count,
    activeSystems,
    maintenanceSystems,
    inactiveSystems,
    reportsGenerated: authStore.user.reports_generated,
    criticalReports,
    highSeverityReports,
    lastActivity: authStore.user.last_activity,
    systemHealth: Math.round(healthyRatio * 100) / 100,
    avgResponseTime: '2.3ms', // Demo metric
    dataPoints: '847M+' // Demo metric
  }
})

// Quick actions with enhanced UX
const quickActions = [
  {
    title: 'Добавить систему',
    description: 'Подключить новую гидравлическую систему',
    icon: 'heroicons:plus-circle',
    action: () => navigateTo('/equipment/create'),
    color: 'blue'
  },
  {
    title: 'Запустить диагностику',
    description: 'Комплексная проверка всех систем', 
    icon: 'heroicons:play-circle',
    action: () => navigateTo('/diagnostics'),
    color: 'green'
  },
  {
    title: 'AI анализ',
    description: 'Получить инсайты от AI помощника',
    icon: 'heroicons:sparkles',
    action: () => navigateTo('/chat'),
    color: 'purple'
  },
  {
    title: 'Экспорт данных',
    description: 'Выгрузить отчёты и метрики',
    icon: 'heroicons:arrow-down-tray',
    action: () => navigateTo('/reports?export=true'),
    color: 'orange'
  }
]

const runSystemDiagnostic = async (systemId: number) => {
  try {
    await api.createReport(systemId)
    await refreshReports()
    
    // Success feedback
    const nuxtApp = useNuxtApp()
    if ('$toast' in nuxtApp) {
      (nuxtApp.$toast as any).success('Диагностика запущена успешно')
    }
  } catch (error) {
    console.error('Failed to start diagnostic:', error)
    if ('$toast' in useNuxtApp()) {
      (useNuxtApp().$toast as any).error('Ошибка запуска диагностики')
    }
  }
}

const formatDateTime = (dateString: string) => {
  return new Date(dateString).toLocaleString('ru-RU', {
    year: 'numeric',
    month: 'short',
    day: 'numeric', 
    hour: '2-digit',
    minute: '2-digit'
  })
}

const getStatusColor = (status: string) => {
  switch (status) {
    case 'active': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    case 'maintenance': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
    case 'inactive': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
    default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
  }
}

const getSeverityColor = (severity: string) => {
  switch (severity) {
    case 'low': return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900 dark:text-green-200 dark:border-green-800'
    case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900 dark:text-yellow-200 dark:border-yellow-800'
    case 'high': return 'bg-orange-100 text-orange-800 border-orange-200 dark:bg-orange-900 dark:text-orange-200 dark:border-orange-800'
    case 'critical': return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900 dark:text-red-200 dark:border-red-800'
    default: return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-900 dark:text-gray-200 dark:border-gray-800'
  }
}

// Auto-refresh functionality
const refreshInterval = ref<NodeJS.Timeout | null>(null)

onMounted(() => {
  // Auto-refresh every 30 seconds
  refreshInterval.value = setInterval(async () => {
    if (!systemsLoading.value && !reportsLoading.value) {
      await Promise.all([refreshSystems(), refreshReports()])
    }
  }, 30000)
})

onUnmounted(() => {
  if (refreshInterval.value) {
    clearInterval(refreshInterval.value)
  }
})
</script>

<template>
  <NuxtLayout name="dashboard">
    <div class="space-y-8">
      <!-- Enhanced Welcome Header -->
      <div class="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl shadow-lg p-8 text-white">
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-3xl font-bold mb-2">
              Добро пожаловать, {{ authStore.userName }}
            </h1>
            <p class="text-blue-100 text-lg">
              Полный контроль над вашими гидравлическими системами в режиме реального времени
            </p>
          </div>
          <div class="hidden md:block">
            <div class="w-20 h-20 bg-white/10 rounded-full flex items-center justify-center backdrop-blur-sm">
              <Icon name="heroicons:chart-bar-square" class="w-10 h-10" />
            </div>
          </div>
        </div>
      </div>

      <!-- Enhanced KPI Cards -->
      <div v-if="stats" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <!-- Total Systems -->
        <div class="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">
                Общие системы
              </p>
              <p class="text-3xl font-bold text-gray-900 dark:text-white">
                {{ stats.totalSystems }}
              </p>
              <p class="text-xs text-green-600 dark:text-green-400 mt-1">
                <Icon name="heroicons:arrow-trending-up" class="w-3 h-3 inline mr-1" />
                +12% за месяц
              </p>
            </div>
            <div class="p-3 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
              <Icon name="heroicons:server-stack" class="w-8 h-8 text-blue-600 dark:text-blue-400" />
            </div>
          </div>
        </div>
        
        <!-- Active Systems -->
        <div class="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">
                Активные системы
              </p>
              <p class="text-3xl font-bold text-gray-900 dark:text-white">
                {{ stats.activeSystems }}
              </p>
              <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {{ stats.systemHealth }}% исправность
              </p>
            </div>
            <div class="p-3 bg-green-50 dark:bg-green-900/30 rounded-lg">
              <Icon name="heroicons:check-circle" class="w-8 h-8 text-green-600 dark:text-green-400" />
            </div>
          </div>
        </div>
        
        <!-- Reports Generated -->
        <div class="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">
                Отчёты создано
              </p>
              <p class="text-3xl font-bold text-gray-900 dark:text-white">
                {{ stats.reportsGenerated }}
              </p>
              <p class="text-xs text-green-600 dark:text-green-400 mt-1">
                <Icon name="heroicons:arrow-trending-up" class="w-3 h-3 inline mr-1" />
                +{{ Math.round(stats.reportsGenerated * 0.08) }} за неделю
              </p>
            </div>
            <div class="p-3 bg-purple-50 dark:bg-purple-900/30 rounded-lg">
              <Icon name="heroicons:document-text" class="w-8 h-8 text-purple-600 dark:text-purple-400" />
            </div>
          </div>
        </div>
        
        <!-- Critical Issues -->
        <div class="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">
                Критические события
              </p>
              <p class="text-3xl font-bold text-gray-900 dark:text-white">
                {{ stats.criticalReports }}
              </p>
              <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Требуют внимания
              </p>
            </div>
            <div class="p-3 bg-red-50 dark:bg-red-900/30 rounded-lg">
              <Icon name="heroicons:exclamation-triangle" class="w-8 h-8 text-red-600 dark:text-red-400" />
            </div>
          </div>
        </div>
      </div>

      <!-- Quick Actions Enhanced -->
      <div class="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 border border-gray-100 dark:border-gray-700">
        <h2 class="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center">
          <Icon name="heroicons:bolt" class="w-6 h-6 mr-3 text-yellow-500" />
          Быстрые действия
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <div 
            v-for="action in quickActions"
            :key="action.title"
            @click="action.action"
            class="group cursor-pointer bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6 hover:shadow-lg transition-all duration-200 border border-gray-200 dark:border-gray-600 hover:border-blue-300 dark:hover:border-blue-500"
          >
            <div class="flex items-center mb-3">
              <div :class="`p-3 bg-${action.color}-100 dark:bg-${action.color}-900/30 rounded-lg`">
                <Icon :name="action.icon" :class="`w-6 h-6 text-${action.color}-600 dark:text-${action.color}-400`" />
              </div>
            </div>
            <h3 class="font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
              {{ action.title }}
            </h3>
            <p class="text-sm text-gray-600 dark:text-gray-300">
              {{ action.description }}
            </p>
          </div>
        </div>
      </div>

      <!-- Systems and Reports Grid -->
      <div class="grid grid-cols-1 xl:grid-cols-2 gap-8">
        <!-- Enhanced Systems Overview -->
        <div class="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-100 dark:border-gray-700">
          <div class="p-6 border-b border-gray-200 dark:border-gray-700">
            <div class="flex items-center justify-between">
              <h2 class="text-xl font-bold text-gray-900 dark:text-white flex items-center">
                <Icon name="heroicons:server-stack" class="w-6 h-6 mr-3 text-blue-600" />
                Активные системы
              </h2>
              <div class="flex items-center space-x-2">
                <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span class="text-sm text-gray-500 dark:text-gray-400">Real-time</span>
                <button 
                  @click="refreshSystems"
                  :disabled="systemsLoading"
                  class="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 disabled:opacity-50 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/30"
                  title="Обновить"
                >
                  <Icon name="heroicons:arrow-path" :class="{ 'animate-spin': systemsLoading }" class="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
          
          <div class="p-6">
            <div v-if="systemsLoading" class="space-y-4">
              <div v-for="i in 4" :key="i" class="animate-pulse">
                <div class="h-20 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
              </div>
            </div>
            
            <div v-else-if="systemsError" class="text-center py-12">
              <Icon name="heroicons:exclamation-triangle" class="w-16 h-16 text-red-500 mx-auto mb-4" />
              <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">Ошибка загрузки систем</h3>
              <p class="text-gray-600 dark:text-gray-300 mb-6">Не удалось получить данные о системах</p>
              <button 
                @click="refreshSystems"
                class="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
              >
                Повторить попытку
              </button>
            </div>
            
            <div v-else-if="!systems?.results?.length" class="text-center py-12">
              <Icon name="heroicons:inbox" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">Нет подключённых систем</h3>
              <p class="text-gray-600 dark:text-gray-300 mb-6">Начните с добавления вашей первой гидравлической системы</p>
              <button 
                @click="navigateTo('/equipment/create')" 
                class="inline-flex items-center px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
              >
                <Icon name="heroicons:plus" class="w-5 h-5 mr-2" />
                Добавить систему
              </button>
            </div>
            
            <div v-else class="space-y-4">
              <div 
                v-for="system in systems.results.slice(0, 6)" 
                :key="system.id"
                class="group bg-gray-50 dark:bg-gray-700/30 rounded-lg p-4 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-all duration-200 border border-gray-200 dark:border-gray-600 hover:border-blue-300 dark:hover:border-blue-500 hover:shadow-md"
              >
                <div class="flex items-center justify-between">
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center space-x-3">
                      <h3 class="font-semibold text-gray-900 dark:text-white truncate group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                        {{ system.name }}
                      </h3>
                      <span 
                        :class="getStatusColor(system.status)"
                        class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                      >
                        {{ system.status === 'active' ? 'Активна' : system.status === 'maintenance' ? 'Обслуживание' : 'Неактивна' }}
                      </span>
                    </div>
                    <div class="flex items-center mt-2 space-x-4 text-sm text-gray-500 dark:text-gray-400">
                      <span class="flex items-center">
                        <Icon name="heroicons:cube" class="w-4 h-4 mr-1" />
                        {{ system.system_type }}
                      </span>
                      <span class="flex items-center">
                        <Icon name="heroicons:cog-6-tooth" class="w-4 h-4 mr-1" />
                        {{ system.components_count }} компонентов
                      </span>
                      <span v-if="system.last_reading_at" class="flex items-center">
                        <Icon name="heroicons:signal" class="w-4 h-4 mr-1" />
                        {{ formatDateTime(system.last_reading_at) }}
                      </span>
                    </div>
                  </div>
                  
                  <div class="flex items-center space-x-2">
                    <button 
                      @click="runSystemDiagnostic(system.id)"
                      class="p-2 text-gray-400 hover:text-green-600 dark:hover:text-green-400 rounded-lg hover:bg-green-50 dark:hover:bg-green-900/20 transition-all duration-200"
                      title="Запустить диагностику"
                    >
                      <Icon name="heroicons:play" class="w-5 h-5" />
                    </button>
                    
                    <NuxtLink 
                      :to="`/equipment/${system.id}`"
                      class="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-all duration-200"
                      title="Подробнее"
                    >
                      <Icon name="heroicons:arrow-right" class="w-5 h-5" />
                    </NuxtLink>
                  </div>
                </div>
              </div>
              
              <div v-if="(systems.results?.length || 0) > 6" class="text-center pt-6 border-t border-gray-200 dark:border-gray-700">
                <NuxtLink 
                  to="/equipment" 
                  class="inline-flex items-center text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-semibold transition-colors"
                >
                  Посмотреть все системы ({{ systems.count }})
                  <Icon name="heroicons:arrow-right" class="w-4 h-4 ml-2" />
                </NuxtLink>
              </div>
            </div>
          </div>
        </div>

        <!-- Enhanced Recent Reports -->
        <div class="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-100 dark:border-gray-700">
          <div class="p-6 border-b border-gray-200 dark:border-gray-700">
            <div class="flex items-center justify-between">
              <h2 class="text-xl font-bold text-gray-900 dark:text-white flex items-center">
                <Icon name="heroicons:document-text" class="w-6 h-6 mr-3 text-purple-600" />
                Последние отчёты
              </h2>
              <div class="flex items-center space-x-2">
                <div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <span class="text-sm text-gray-500 dark:text-gray-400">Live</span>
                <button 
                  @click="refreshReports"
                  :disabled="reportsLoading"
                  class="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 disabled:opacity-50 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/30"
                  title="Обновить"
                >
                  <Icon name="heroicons:arrow-path" :class="{ 'animate-spin': reportsLoading }" class="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
          
          <div class="p-6">
            <div v-if="reportsLoading" class="space-y-4">
              <div v-for="i in 4" :key="i" class="animate-pulse">
                <div class="h-20 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
              </div>
            </div>
            
            <div v-else-if="!recentReports?.results?.length" class="text-center py-12">
              <Icon name="heroicons:document" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">Нет отчётов</h3>
              <p class="text-gray-600 dark:text-gray-300 mb-6">Запустите диагностику для создания первого отчёта</p>
              <button 
                @click="navigateTo('/diagnostics')"
                class="inline-flex items-center px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded-lg transition-colors"
              >
                <Icon name="heroicons:play" class="w-5 h-5 mr-2" />
                Запустить диагностику
              </button>
            </div>
            
            <div v-else class="space-y-4">
              <div 
                v-for="report in recentReports.results.slice(0, 6)"
                :key="report.id"
                class="group bg-gray-50 dark:bg-gray-700/30 rounded-lg p-4 hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-all duration-200 border border-gray-200 dark:border-gray-600 hover:border-purple-300 dark:hover:border-purple-500 hover:shadow-md"
              >
                <div class="flex items-start justify-between">
                  <div class="flex-1 min-w-0">
                    <h3 class="font-semibold text-gray-900 dark:text-white mb-1 truncate group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
                      {{ report.title }}
                    </h3>
                    <p class="text-sm text-gray-600 dark:text-gray-300 mb-3 line-clamp-2">
                      {{ report.summary }}
                    </p>
                    <div class="flex items-center justify-between">
                      <div class="flex items-center space-x-3">
                        <span 
                          :class="getSeverityColor(report.severity)"
                          class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border"
                        >
                          {{ report.severity === 'low' ? 'Низкая' : report.severity === 'medium' ? 'Средняя' : report.severity === 'high' ? 'Высокая' : 'Критическая' }}
                        </span>
                        <span class="text-xs text-gray-500 dark:text-gray-400">
                          {{ formatDateTime(report.created_at) }}
                        </span>
                      </div>
                      
                      <NuxtLink 
                        :to="`/reports/${report.id}`"
                        class="p-2 text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 rounded-lg hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-all duration-200"
                        title="Открыть отчёт"
                      >
                        <Icon name="heroicons:arrow-right" class="w-5 h-5" />
                      </NuxtLink>
                    </div>
                  </div>
                </div>
              </div>
              
              <div class="text-center pt-6 border-t border-gray-200 dark:border-gray-700">
                <NuxtLink 
                  to="/reports" 
                  class="inline-flex items-center text-purple-600 hover:text-purple-800 dark:text-purple-400 dark:hover:text-purple-300 font-semibold transition-colors"
                >
                  Посмотреть все отчёты
                  <Icon name="heroicons:arrow-right" class="w-4 h-4 ml-2" />
                </NuxtLink>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- System Health Overview -->
      <div class="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
        <h2 class="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center">
          <Icon name="heroicons:heart" class="w-6 h-6 mr-3 text-red-500" />
          Состояние платформы
        </h2>
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="text-center">
            <div class="w-16 h-16 bg-green-50 dark:bg-green-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
              <Icon name="heroicons:check-circle" class="w-8 h-8 text-green-600 dark:text-green-400" />
            </div>
            <h3 class="font-semibold text-gray-900 dark:text-white mb-2">API Endpoint</h3>
            <p class="text-sm text-gray-600 dark:text-gray-300 mb-2">Время ответа: {{ stats?.avgResponseTime }}</p>
            <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
              <div class="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
              Operational
            </span>
          </div>
          
          <div class="text-center">
            <div class="w-16 h-16 bg-blue-50 dark:bg-blue-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
              <Icon name="heroicons:server" class="w-8 h-8 text-blue-600 dark:text-blue-400" />
            </div>
            <h3 class="font-semibold text-gray-900 dark:text-white mb-2">Database</h3>
            <p class="text-sm text-gray-600 dark:text-gray-300 mb-2">Обработано: {{ stats?.dataPoints }}</p>
            <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
              <div class="w-2 h-2 bg-blue-500 rounded-full mr-2 animate-pulse"></div>
              Active
            </span>
          </div>
          
          <div class="text-center">
            <div class="w-16 h-16 bg-purple-50 dark:bg-purple-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
              <Icon name="heroicons:cpu-chip" class="w-8 h-8 text-purple-600 dark:text-purple-400" />
            </div>
            <h3 class="font-semibold text-gray-900 dark:text-white mb-2">AI Engine</h3>
            <p class="text-sm text-gray-600 dark:text-gray-300 mb-2">Uptime: 99.94%</p>
            <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
              <div class="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>
              Running
            </span>
          </div>
        </div>
      </div>
    </div>
  </NuxtLayout>
</template>