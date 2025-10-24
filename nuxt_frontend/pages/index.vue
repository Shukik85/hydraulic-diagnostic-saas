<script setup lang="ts">
// Main dashboard page - updated to integrate with backend API
definePageMeta({
  middleware: 'auth'
})

useSeoMeta({
  title: 'Панель управления | Hydraulic Diagnostic SaaS',
  description: 'Основная панель управления системой диагностики гидравлических систем'
})

const authStore = useAuthStore()
const api = useApi()

// Data loading with proper error handling
const { data: systems, pending: systemsLoading, error: systemsError, refresh: refreshSystems } = await useLazyAsyncData(
  'dashboard-systems',
  () => api.getSystems({ page_size: 10, ordering: '-last_reading_at' }),
  {
    default: () => ({ results: [], count: 0 })
  }
)

const { data: recentReports, pending: reportsLoading, refresh: refreshReports } = await useLazyAsyncData(
  'dashboard-reports', 
  () => api.getReports(undefined, { page_size: 5, ordering: '-created_at' }),
  {
    default: () => ({ results: [], count: 0 })
  }
)

// Stats computation based on real backend data
const stats = computed(() => {
  if (!authStore.user) return null
  
  const activeSystemsCount = systems.value?.results?.filter(s => s.status === 'active').length || 0
  const maintenanceCount = systems.value?.results?.filter(s => s.status === 'maintenance').length || 0
  const criticalReports = recentReports.value?.results?.filter(r => r.severity === 'critical').length || 0
  
  return {
    totalSystems: authStore.user.systems_count,
    activeSystems: activeSystemsCount,
    maintenanceSystems: maintenanceCount,
    reportsGenerated: authStore.user.reports_generated,
    criticalReports,
    lastActivity: authStore.user.last_activity
  }
})

// Quick actions
const createNewSystem = async () => {
  await navigateTo('/equipment/create')
}

const runDiagnostic = async (systemId: number) => {
  try {
    await api.createReport(systemId)
    await refreshReports()
    
    // Show success notification if toast is available
    const nuxtApp = useNuxtApp()
    if ('$toast' in nuxtApp) {
      (nuxtApp.$toast as any).success('Диагностика запущена')
    }
  } catch (error) {
    console.error('Failed to start diagnostic:', error)
    if ('$toast' in useNuxtApp()) {
      (useNuxtApp().$toast as any).error('Ошибка запуска диагностики')
    }
  }
}

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleDateString('ru-RU', {
    year: 'numeric',
    month: 'short', 
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const getStatusText = (status: string) => {
  switch (status) {
    case 'active': return 'Активна'
    case 'maintenance': return 'Обслуживание'
    case 'inactive': return 'Неактивна'
    default: return status
  }
}

const getSeverityText = (severity: string) => {
  switch (severity) {
    case 'low': return 'Низкая'
    case 'medium': return 'Средняя'
    case 'high': return 'Высокая'
    case 'critical': return 'Критическая'
    default: return severity
  }
}
</script>

<template>
  <NuxtLayout name="dashboard">
    <div class="space-y-6">
      <!-- Welcome header -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
          Добро пожаловать, {{ authStore.userName }}
        </h1>
        <p class="text-gray-600 dark:text-gray-300 mt-2">
          Мониторинг и диагностика гидравлических систем
        </p>
      </div>

      <!-- Stats overview -->
      <div v-if="stats" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div class="flex items-center">
            <div class="p-3 rounded-full bg-blue-100 dark:bg-blue-900">
              <Icon name="heroicons:server-stack" class="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div class="ml-4">
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300">
                Общие системы
              </p>
              <p class="text-2xl font-semibold text-gray-900 dark:text-white">
                {{ stats.totalSystems }}
              </p>
            </div>
          </div>
        </div>
        
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div class="flex items-center">
            <div class="p-3 rounded-full bg-green-100 dark:bg-green-900">
              <Icon name="heroicons:check-circle" class="w-6 h-6 text-green-600 dark:text-green-400" />
            </div>
            <div class="ml-4">
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300">
                Активные
              </p>
              <p class="text-2xl font-semibold text-gray-900 dark:text-white">
                {{ stats.activeSystems }}
              </p>
            </div>
          </div>
        </div>
        
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div class="flex items-center">
            <div class="p-3 rounded-full bg-purple-100 dark:bg-purple-900">
              <Icon name="heroicons:document-text" class="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div class="ml-4">
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300">
                Отчёты
              </p>
              <p class="text-2xl font-semibold text-gray-900 dark:text-white">
                {{ stats.reportsGenerated }}
              </p>
            </div>
          </div>
        </div>
        
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div class="flex items-center">
            <div class="p-3 rounded-full bg-red-100 dark:bg-red-900">
              <Icon name="heroicons:exclamation-triangle" class="w-6 h-6 text-red-600 dark:text-red-400" />
            </div>
            <div class="ml-4">
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300">
                Критические
              </p>
              <p class="text-2xl font-semibold text-gray-900 dark:text-white">
                {{ stats.criticalReports }}
              </p>
            </div>
          </div>
        </div>
      </div>

      <!-- Quick actions -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Быстрые действия
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <button 
            @click="createNewSystem"
            class="flex items-center justify-center px-4 py-3 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg hover:border-blue-500 dark:hover:border-blue-400 transition-colors group"
          >
            <Icon name="heroicons:plus" class="w-5 h-5 mr-2 text-gray-400 group-hover:text-blue-500" />
            <span class="text-sm font-medium text-gray-600 dark:text-gray-300 group-hover:text-blue-600 dark:group-hover:text-blue-400">
              Добавить систему
            </span>
          </button>
          
          <NuxtLink 
            to="/reports"
            class="flex items-center justify-center px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            <Icon name="heroicons:document-text" class="w-5 h-5 mr-2 text-gray-400" />
            <span class="text-sm font-medium text-gray-600 dark:text-gray-300">
              Посмотреть отчёты
            </span>
          </NuxtLink>
          
          <NuxtLink 
            to="/diagnostics"
            class="flex items-center justify-center px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            <Icon name="heroicons:cpu-chip" class="w-5 h-5 mr-2 text-gray-400" />
            <span class="text-sm font-medium text-gray-600 dark:text-gray-300">
              Диагностика
            </span>
          </NuxtLink>
          
          <NuxtLink 
            to="/chat"
            class="flex items-center justify-center px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            <Icon name="heroicons:chat-bubble-left-right" class="w-5 h-5 mr-2 text-gray-400" />
            <span class="text-sm font-medium text-gray-600 dark:text-gray-300">
              AI помощник
            </span>
          </NuxtLink>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Recent systems -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div class="p-6 border-b border-gray-200 dark:border-gray-700">
            <div class="flex items-center justify-between">
              <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
                Последние системы
              </h2>
              <button 
                @click="refreshSystems"
                :disabled="systemsLoading"
                class="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 disabled:opacity-50"
                title="Обновить"
              >
                <Icon :name="systemsLoading ? 'heroicons:arrow-path' : 'heroicons:arrow-path'" :class="{ 'animate-spin': systemsLoading }" class="w-5 h-5" />
              </button>
            </div>
          </div>
          
          <div class="p-6">
            <div v-if="systemsLoading" class="animate-pulse space-y-4">
              <div v-for="i in 3" :key="i" class="h-20 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
            </div>
            
            <div v-else-if="systemsError" class="text-center py-8">
              <Icon name="heroicons:exclamation-triangle" class="w-12 h-12 text-red-500 mx-auto mb-4" />
              <p class="text-gray-600 dark:text-gray-300 mb-4">Ошибка загрузки систем</p>
              <button 
                @click="refreshSystems"
                class="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
              >
                Повторить попытку
              </button>
            </div>
            
            <div v-else-if="!systems?.results?.length" class="text-center py-8">
              <Icon name="heroicons:inbox" class="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p class="text-gray-600 dark:text-gray-300 mb-4">Нет систем</p>
              <button 
                @click="createNewSystem" 
                class="inline-flex items-center px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100 dark:bg-blue-900 dark:text-blue-300 dark:hover:bg-blue-800"
              >
                <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
                Создать первую систему
              </button>
            </div>
            
            <div v-else class="space-y-4">
              <div 
                v-for="system in systems.results.slice(0, 5)" 
                :key="system.id"
                class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow"
              >
                <div class="flex items-start justify-between">
                  <div class="flex-1 min-w-0">
                    <h3 class="font-medium text-gray-900 dark:text-white truncate">
                      {{ system.name }}
                    </h3>
                    <p v-if="system.description" class="text-sm text-gray-600 dark:text-gray-300 mt-1 line-clamp-2">
                      {{ system.description }}
                    </p>
                    <div class="flex items-center mt-2 space-x-4 text-xs text-gray-500 dark:text-gray-400">
                      <span>Тип: {{ system.system_type }}</span>
                      <span>Компоненты: {{ system.components_count }}</span>
                      <span v-if="system.last_reading_at">
                        Последние данные: {{ formatDate(system.last_reading_at) }}
                      </span>
                    </div>
                    <div class="flex items-center mt-2">
                      <span 
                        :class="{
                          'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200': system.status === 'active',
                          'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200': system.status === 'maintenance',
                          'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200': system.status === 'inactive'
                        }"
                        class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                      >
                        {{ getStatusText(system.status) }}
                      </span>
                    </div>
                  </div>
                  
                  <div class="flex flex-col space-y-2 ml-4">
                    <button 
                      @click="runDiagnostic(system.id)"
                      class="p-2 text-gray-400 hover:text-green-600 dark:hover:text-green-400 rounded-lg hover:bg-green-50 dark:hover:bg-green-900/20 transition-colors"
                      title="Запустить диагностику"
                    >
                      <Icon name="heroicons:play" class="w-5 h-5" />
                    </button>
                    
                    <NuxtLink 
                      :to="`/equipment/${system.id}`"
                      class="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
                      title="Подробнее"
                    >
                      <Icon name="heroicons:arrow-right" class="w-5 h-5" />
                    </NuxtLink>
                  </div>
                </div>
              </div>
              
              <div v-if="(systems.results?.length || 0) > 5" class="text-center pt-4 border-t border-gray-200 dark:border-gray-700">
                <NuxtLink 
                  to="/equipment" 
                  class="inline-flex items-center text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
                >
                  Посмотреть все системы ({{ systems.count }})
                  <Icon name="heroicons:arrow-right" class="w-4 h-4 ml-2" />
                </NuxtLink>
              </div>
            </div>
          </div>
        </div>

        <!-- Recent reports -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div class="p-6 border-b border-gray-200 dark:border-gray-700">
            <div class="flex items-center justify-between">
              <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
                Последние отчёты
              </h2>
              <button 
                @click="refreshReports"
                :disabled="reportsLoading"
                class="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 disabled:opacity-50"
                title="Обновить"
              >
                <Icon name="heroicons:arrow-path" :class="{ 'animate-spin': reportsLoading }" class="w-5 h-5" />
              </button>
            </div>
          </div>
          
          <div class="p-6">
            <div v-if="reportsLoading" class="animate-pulse space-y-4">
              <div v-for="i in 3" :key="i" class="h-20 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
            </div>
            
            <div v-else-if="!recentReports?.results?.length" class="text-center py-8">
              <Icon name="heroicons:document" class="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p class="text-gray-600 dark:text-gray-300 mb-4">Нет отчётов</p>
              <p class="text-sm text-gray-500 dark:text-gray-400">
                Запустите диагностику для создания первого отчёта
              </p>
            </div>
            
            <div v-else class="space-y-4">
              <div 
                v-for="report in recentReports.results"
                :key="report.id"
                class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow"
              >
                <div class="flex items-start justify-between">
                  <div class="flex-1 min-w-0">
                    <h3 class="font-medium text-gray-900 dark:text-white truncate">
                      {{ report.title }}
                    </h3>
                    <p class="text-sm text-gray-600 dark:text-gray-300 mt-1 line-clamp-3">
                      {{ report.summary }}
                    </p>
                    <div class="flex items-center mt-2 space-x-4 text-xs text-gray-500 dark:text-gray-400">
                      <span>Статус: {{ 
                        report.status === 'pending' ? 'В обработке' :
                        report.status === 'completed' ? 'Готов' : 
                        report.status === 'failed' ? 'Ошибка' : report.status
                      }}</span>
                      <span>Создан: {{ formatDate(report.created_at) }}</span>
                    </div>
                    <div class="flex items-center mt-2">
                      <span 
                        :class="{
                          'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200': report.severity === 'low',
                          'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200': report.severity === 'medium',
                          'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200': report.severity === 'high',
                          'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200': report.severity === 'critical'
                        }"
                        class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                      >
                        {{ getSeverityText(report.severity) }}
                      </span>
                    </div>
                  </div>
                  
                  <NuxtLink 
                    :to="`/reports/${report.id}`"
                    class="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors ml-4"
                    title="Открыть отчёт"
                  >
                    <Icon name="heroicons:arrow-right" class="w-5 h-5" />
                  </NuxtLink>
                </div>
              </div>
              
              <div class="text-center pt-4 border-t border-gray-200 dark:border-gray-700">
                <NuxtLink 
                  to="/reports" 
                  class="inline-flex items-center text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
                >
                  Посмотреть все отчёты
                  <Icon name="heroicons:arrow-right" class="w-4 h-4 ml-2" />
                </NuxtLink>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </NuxtLayout>
</template>