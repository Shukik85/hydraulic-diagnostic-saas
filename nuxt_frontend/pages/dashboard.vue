<script setup lang="ts">
// Enhanced Russian dashboard with interactive demo elements
definePageMeta({
  middleware: 'auth'
})

useSeoMeta({
  title: 'Дашборд | Гидравлика ИИ',
  description: 'Мониторинг гидравлических систем в реальном времени с ИИ-аналитикой и предиктивными алгоритмами'
})

const authStore = useAuthStore()

// Demo stats with null safety and Russian localization
const stats = computed(() => {
  const user = authStore.user
  if (!user) {
    return {
      totalSystems: 0,
      activeSystems: 0,
      reportsGenerated: 0,
      criticalAlerts: 0,
      uptime: 0,
      efficiency: 0,
      costSavings: 0
    }
  }
  
  const totalSystems = user.systems_count || 12
  const activeSystems = Math.floor(totalSystems * 0.92)
  const reportsGenerated = user.reports_generated || 847
  
  const healthyRatio = totalSystems > 0 ? (activeSystems / totalSystems) * 100 : 100
  
  return {
    totalSystems,
    activeSystems,
    reportsGenerated,
    criticalAlerts: Math.max(0, totalSystems - activeSystems),
    uptime: Math.round(healthyRatio),
    efficiency: 94,
    costSavings: 3200000
  }
})

// Interactive demo elements
const showDemoModal = ref(false)
const demoStep = ref(1)
const isProcessing = ref(false)

const startDemo = () => {
  showDemoModal.value = true
  demoStep.value = 1
}

const nextDemoStep = async () => {
  if (demoStep.value < 3) {
    isProcessing.value = true
    await new Promise(resolve => setTimeout(resolve, 1500))
    demoStep.value++
    isProcessing.value = false
  } else {
    showDemoModal.value = false
    demoStep.value = 1
  }
}

// Async data for systems
const { data: systems, error: systemsError, pending: systemsLoading, refresh: refreshSystems } = await useAsyncData('hydraulic-systems', () => {
  return new Promise<any[]>((resolve) => {
    setTimeout(() => {
      resolve([
        {
          id: 1,
          name: 'ГИД-001 - Насосная станция А',
          status: 'active',
          location: 'Цех №1',
          temperature: 45.2,
          pressure: 150.8,
          efficiency_score: 94,
          components_count: 12,
          last_reading_at: new Date().toISOString(),
          flow_rate: 85.4,
          vibration_level: 0.8
        },
        {
          id: 2,
          name: 'ГИД-002 - Гидромотор Б',
          status: 'warning',
          location: 'Цех №2',
          temperature: 52.1,
          pressure: 145.2,
          efficiency_score: 78,
          components_count: 8,
          last_reading_at: new Date(Date.now() - 300000).toISOString(),
          flow_rate: 72.3,
          vibration_level: 2.1
        },
        {
          id: 3,
          name: 'ГИД-003 - Клапан управления В',
          status: 'maintenance',
          location: 'Цех №3',
          temperature: 41.8,
          pressure: 140.0,
          efficiency_score: 85,
          components_count: 6,
          last_reading_at: new Date(Date.now() - 600000).toISOString(),
          flow_rate: 68.7,
          vibration_level: 1.2
        }
      ])
    }, 100)
  })
})

// Status helpers with Russian localization
const getSystemStatusColor = (status: string): string => {
  switch (status) {
    case 'active': return 'text-green-600 dark:text-green-400'
    case 'warning': return 'text-yellow-600 dark:text-yellow-400'
    case 'maintenance': return 'text-blue-600 dark:text-blue-400'
    case 'critical': return 'text-red-600 dark:text-red-400'
    default: return 'text-gray-500 dark:text-gray-400'
  }
}

const getSystemStatusText = (status: string): string => {
  switch (status) {
    case 'active': return 'Активна'
    case 'warning': return 'Предупреждение'
    case 'maintenance': return 'Обслуживание'
    case 'critical': return 'Критическое'
    default: return 'Неизвестно'
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

// Format helpers
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

const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('ru-RU', {
    style: 'currency',
    currency: 'RUB',
    minimumFractionDigits: 0
  }).format(amount)
}

// Fixed event handlers
const handleRefreshSystems = async (): Promise<void> => {
  await refreshSystems()
}
</script>

<template>
  <div class="container mx-auto px-4 py-6">
    <!-- Header with demo button -->
    <div class="mb-8">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="premium-heading-xl text-gray-900 dark:text-white mb-2">
            📊 Дашборд управления
          </h1>
          <p class="premium-body text-gray-600 dark:text-gray-300">
            Мониторинг гидравлических систем в реальном времени с ИИ-аналитикой
          </p>
        </div>
        
        <div class="flex items-center space-x-3">
          <PremiumButton
            @click="startDemo"
            variant="secondary"
            icon="heroicons:play"
            size="sm"
          >
            Демо-режим
          </PremiumButton>
          <div class="flex items-center space-x-2 px-3 py-2 bg-green-50 dark:bg-green-900/30 rounded-lg">
            <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span class="text-sm font-medium text-green-700 dark:text-green-300">Все системы онлайн</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Stats cards with enhanced Russian content -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <!-- Total Systems -->
      <div class="premium-card p-6 hover:shadow-lg transition-shadow">
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
      <div class="premium-card p-6 hover:shadow-lg transition-shadow">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Активные</p>
            <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ stats.activeSystems }}</p>
            <p class="text-xs text-green-600 dark:text-green-400 mt-1">
              <Icon name="heroicons:check-circle" class="w-3 h-3 inline mr-1" />
              {{ stats.uptime }}% время работы
            </p>
          </div>
          <div class="p-3 bg-green-50 dark:bg-green-900/30 rounded-lg">
            <Icon name="heroicons:play" class="w-6 h-6 text-green-600 dark:text-green-400" />
          </div>
        </div>
      </div>

      <!-- Efficiency Score -->
      <div class="premium-card p-6 hover:shadow-lg transition-shadow">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Эффективность</p>
            <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ stats.efficiency }}%</p>
            <p class="text-xs text-purple-600 dark:text-purple-400 mt-1">
              <Icon name="heroicons:arrow-trending-up" class="w-3 h-3 inline mr-1" />
              +5% за неделю
            </p>
          </div>
          <div class="p-3 bg-purple-50 dark:bg-purple-900/30 rounded-lg">
            <Icon name="heroicons:chart-bar" class="w-6 h-6 text-purple-600 dark:text-purple-400" />
          </div>
        </div>
      </div>

      <!-- Cost Savings -->
      <div class="premium-card p-6 hover:shadow-lg transition-shadow">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Экономия</p>
            <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ formatCurrency(stats.costSavings) }}</p>
            <p class="text-xs text-green-600 dark:text-green-400 mt-1">
              <Icon name="heroicons:banknotes" class="w-3 h-3 inline mr-1" />
              В этом году
            </p>
          </div>
          <div class="p-3 bg-emerald-50 dark:bg-emerald-900/30 rounded-lg">
            <Icon name="heroicons:currency-ruble" class="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
          </div>
        </div>
      </div>
    </div>

    <!-- Systems Grid with enhanced Russian content -->
    <div class="mb-8">
      <div class="flex items-center justify-between mb-6">
        <h2 class="premium-heading-lg text-gray-900 dark:text-white">🔧 Гидравлические системы</h2>
        <div class="flex items-center space-x-3">
          <div class="flex items-center space-x-2">
            <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span class="text-sm text-gray-500 dark:text-gray-400">В реальном времени</span>
            <button 
              @click="handleRefreshSystems"
              :disabled="systemsLoading"
              class="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 disabled:opacity-50 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-colors"
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

      <!-- Systems data with enhanced metrics -->
      <div v-else-if="systems?.length" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div v-for="system in systems" :key="system.id" class="premium-card hover:shadow-xl transition-all duration-300 group">
          <div class="p-6">
            <!-- Header -->
            <div class="flex items-start justify-between mb-4">
              <div class="flex-1">
                <h3 class="font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                  {{ system.name }}
                </h3>
                <div class="flex items-center space-x-2 mb-2">
                  <Icon :name="getSystemStatusIcon(system.status)" class="w-4 h-4" :class="getSystemStatusColor(system.status)" />
                  <span class="text-sm font-medium" :class="getSystemStatusColor(system.status)">
                    {{ getSystemStatusText(system.status) }}
                  </span>
                </div>
                <div class="flex items-center space-x-1 text-xs text-gray-500 dark:text-gray-400">
                  <Icon name="heroicons:map-pin" class="w-3 h-3" />
                  <span>{{ system.location }}</span>
                </div>
              </div>
              
              <!-- Efficiency gauge -->
              <div class="text-right">
                <div class="text-2xl font-bold" :class="[
                  system.efficiency_score >= 90 ? 'text-green-600 dark:text-green-400' :
                  system.efficiency_score >= 80 ? 'text-yellow-600 dark:text-yellow-400' :
                  'text-red-600 dark:text-red-400'
                ]">
                  {{ system.efficiency_score }}%
                </div>
                <div class="text-xs text-gray-500 dark:text-gray-400">Эффективность</div>
              </div>
            </div>

            <!-- Metrics grid -->
            <div class="grid grid-cols-2 gap-3 mb-4">
              <div class="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="text-lg font-semibold text-gray-900 dark:text-white">{{ system.temperature }}°C</div>
                <div class="text-xs text-gray-500 dark:text-gray-400">Температура</div>
              </div>
              <div class="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="text-lg font-semibold text-gray-900 dark:text-white">{{ system.pressure }} бар</div>
                <div class="text-xs text-gray-500 dark:text-gray-400">Давление</div>
              </div>
              <div class="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="text-lg font-semibold text-gray-900 dark:text-white">{{ system.flow_rate }} л/мин</div>
                <div class="text-xs text-gray-500 dark:text-gray-400">Расход</div>
              </div>
              <div class="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="text-lg font-semibold text-gray-900 dark:text-white">{{ system.vibration_level }} мм/с</div>
                <div class="text-xs text-gray-500 dark:text-gray-400">Вибрация</div>
              </div>
            </div>

            <!-- System info -->
            <div class="space-y-2 text-xs text-gray-500 dark:text-gray-400 mb-4">
              <div class="flex items-center justify-between">
                <span class="flex items-center">
                  <Icon name="heroicons:cog-6-tooth" class="w-3 h-3 mr-1" />
                  {{ system.components_count }} компонентов
                </span>
                <span v-if="system.last_reading_at" class="flex items-center">
                  <Icon name="heroicons:signal" class="w-3 h-3 mr-1" />
                  {{ formatDateTime(system.last_reading_at) }}
                </span>
              </div>
            </div>
            
            <!-- Action buttons -->
            <div class="flex space-x-2">
              <PremiumButton size="sm" variant="secondary" class="flex-1">
                Подробнее
              </PremiumButton>
              <PremiumButton size="sm" icon="heroicons:wrench-screwdriver">
                Диагностика
              </PremiumButton>
            </div>
          </div>
        </div>
      </div>

      <!-- Quick Stats Summary -->
      <div class="premium-card p-6 mt-6">
        <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-4">📈 Сводка по производительности</h3>
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div class="text-center p-4 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/30 dark:to-indigo-900/30 rounded-xl">
            <div class="text-2xl font-bold text-blue-600 dark:text-blue-400 mb-1">{{ stats.uptime }}%</div>
            <div class="text-sm text-gray-700 dark:text-gray-300">Время безотказной работы</div>
          </div>
          <div class="text-center p-4 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/30 dark:to-emerald-900/30 rounded-xl">
            <div class="text-2xl font-bold text-green-600 dark:text-green-400 mb-1">{{ stats.efficiency }}%</div>
            <div class="text-sm text-gray-700 dark:text-gray-300">Средняя эффективность</div>
          </div>
          <div class="text-center p-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/30 dark:to-pink-900/30 rounded-xl">
            <div class="text-2xl font-bold text-purple-600 dark:text-purple-400 mb-1">{{ stats.reportsGenerated }}</div>
            <div class="text-sm text-gray-700 dark:text-gray-300">Отчётов сгенерировано</div>
          </div>
          <div class="text-center p-4 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/30 dark:to-red-900/30 rounded-xl">
            <div class="text-2xl font-bold text-orange-600 dark:text-orange-400 mb-1">{{ stats.criticalAlerts }}</div>
            <div class="text-sm text-gray-700 dark:text-gray-300">Критических предупреждений</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Interactive Demo Modal -->
    <div 
      v-if="showDemoModal"
      class="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
      @click="showDemoModal = false"
    >
      <div 
        class="premium-card max-w-2xl w-full"
        @click.stop
      >
        <!-- Demo Header -->
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-3">
              <div class="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <Icon name="heroicons:play" class="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Интерактивное демо</h3>
                <p class="text-sm text-gray-500 dark:text-gray-400">Шаг {{ demoStep }} из 3</p>
              </div>
            </div>
            <button
              @click="showDemoModal = false"
              class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <Icon name="heroicons:x-mark" class="w-6 h-6" />
            </button>
          </div>
          
          <!-- Progress bar -->
          <div class="mt-4">
            <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div 
                class="bg-gradient-to-r from-blue-600 to-purple-600 h-2 rounded-full transition-all duration-500"
                :style="`width: ${(demoStep / 3) * 100}%`"
              ></div>
            </div>
          </div>
        </div>
        
        <!-- Demo Content -->
        <div class="p-6">
          <div v-if="demoStep === 1" class="text-center">
            <Icon name="heroicons:eye" class="w-16 h-16 mx-auto text-blue-500 mb-4" />
            <h4 class="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Мониторинг в реальном времени
            </h4>
            <p class="text-gray-600 dark:text-gray-300 mb-6">
              Наша платформа непрерывно отслеживает ключевые параметры всех гидравлических систем:
              температуру, давление, расход и уровень вибрации.
            </p>
            <div class="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/30 dark:to-indigo-900/30 p-4 rounded-lg">
              <div class="flex items-center justify-center space-x-6 text-sm">
                <div class="text-center">
                  <div class="text-lg font-bold text-blue-600 dark:text-blue-400">< 1.2с</div>
                  <div class="text-gray-600 dark:text-gray-300">Отклик</div>
                </div>
                <div class="text-center">
                  <div class="text-lg font-bold text-green-600 dark:text-green-400">99.94%</div>
                  <div class="text-gray-600 dark:text-gray-300">Uptime</div>
                </div>
                <div class="text-center">
                  <div class="text-lg font-bold text-purple-600 dark:text-purple-400">24/7</div>
                  <div class="text-gray-600 dark:text-gray-300">Контроль</div>
                </div>
              </div>
            </div>
          </div>
          
          <div v-else-if="demoStep === 2" class="text-center">
            <Icon name="heroicons:cpu-chip" class="w-16 h-16 mx-auto text-purple-500 mb-4" />
            <h4 class="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              ИИ Предиктивная аналитика
            </h4>
            <p class="text-gray-600 dark:text-gray-300 mb-6">
              Искусственный интеллект анализирует паттерны данных и предсказывает потенциальные неисправности 
              за 30 дней до их возникновения с точностью 94.8%.
            </p>
            <div class="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/30 dark:to-pink-900/30 p-4 rounded-lg">
              <div class="flex items-center justify-center space-x-6 text-sm">
                <div class="text-center">
                  <div class="text-lg font-bold text-purple-600 dark:text-purple-400">94.8%</div>
                  <div class="text-gray-600 dark:text-gray-300">Точность</div>
                </div>
                <div class="text-center">
                  <div class="text-lg font-bold text-orange-600 dark:text-orange-400">30 дней</div>
                  <div class="text-gray-600 dark:text-gray-300">Прогноз</div>
                </div>
                <div class="text-center">
                  <div class="text-lg font-bold text-green-600 dark:text-green-400">89%</div>
                  <div class="text-gray-600 dark:text-gray-300">Экономия</div>
                </div>
              </div>
            </div>
          </div>
          
          <div v-else class="text-center">
            <Icon name="heroicons:check-circle" class="w-16 h-16 mx-auto text-green-500 mb-4" />
            <h4 class="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Готово к работе!
            </h4>
            <p class="text-gray-600 dark:text-gray-300 mb-6">
              Платформа полностью настроена для мониторинга ваших гидравлических систем. 
              Начните использовать все возможности ИИ-диагностики прямо сейчас!
            </p>
            <div class="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/30 dark:to-emerald-900/30 p-4 rounded-lg">
              <p class="text-sm font-medium text-green-700 dark:text-green-300">
                🎉 Добро пожаловать в будущее промышленного мониторинга!
              </p>
            </div>
          </div>
        </div>
        
        <!-- Demo Footer -->
        <div class="p-6 border-t border-gray-200 dark:border-gray-700">
          <div class="flex items-center justify-between">
            <button
              v-if="demoStep > 1"
              @click="demoStep--"
              class="px-4 py-2 text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-white transition-colors"
            >
              ← Назад
            </button>
            <div v-else></div>
            
            <PremiumButton
              @click="nextDemoStep"
              :loading="isProcessing"
              icon="demoStep === 3 ? 'heroicons:check' : 'heroicons:arrow-right'"
              gradient
            >
              {{ demoStep === 3 ? 'Завершить демо' : 'Далее' }}
            </PremiumButton>
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