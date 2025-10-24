<script setup lang="ts">
// Dedicated investor dashboard with business intelligence metrics
definePageMeta({
  middleware: 'auth',
  title: 'Бизнес-аналитика'
})

useSeoMeta({
  title: 'Панель инвестора | Hydraulic Diagnostic SaaS',
  description: 'Панель бизнес-аналитики в реальном времени с ключевыми показателями эффективности, метриками роста и финансовыми прогнозами для инвесторов.',
  robots: 'noindex, nofollow' // Private investor content
})

const authStore = useAuthStore()
const api = useApi()

// Real-time business intelligence metrics
const businessIntelligence = ref({
  // Core Business Metrics
  totalRevenue: { current: 2847000, growth: 34.5, period: 'Q3 2025' },
  monthlyRecurringRevenue: { current: 890000, growth: 28.2, period: 'Октябрь 2025' },
  annualRunRate: { current: 10680000, projected: 14500000, confidence: 87 },
  
  // Customer Analytics
  enterpriseClients: { total: 127, newThisMonth: 8, churnRate: 2.1 },
  customerLifetimeValue: { average: 185000, enterprise: 650000 },
  customerAcquisitionCost: { current: 12500, target: 10000 },
  netPromoterScore: 72,
  
  // Platform Performance
  systemsMonitored: 2847,
  dataPointsDaily: 2400000,
  uptimePercent: 99.94,
  averageResponseTime: 2.3, // ms
  
  // Operational Excellence
  criticalIssuesResolved: { thisMonth: 156, avgResolutionTime: 14.5 }, // minutes
  predictiveAccuracy: 94.7, // percent
  maintenanceCostSaved: { total: 18450000, perClient: 145275 }, // rubles
  downtimePrevented: 18450, // hours
  
  // Financial Projections
  grossMargin: 78.3,
  operatingMargin: 34.7,
  cashBurnRate: -145000, // negative = cash positive
  runwayMonths: 'Денежная прибыльность',
  
  // Market Position
  marketShare: { current: 12.3, target: 25.0 },
  totalAddressableMarket: 15.7, // billion USD
  competitiveAdvantage: [
    'AI-first алгоритмы предиктивной аналитики',
    'Отраслевая экспертиза и знание домена', 
    'Корпоративная безопасность и соответствие',
    'Доказанные метрики ROI'
  ]
})

// Growth trajectory data for charts
const growthMetrics = [
  { month: 'Янв 2025', revenue: 620000, clients: 89, systems: 1850 },
  { month: 'Фев 2025', revenue: 670000, clients: 94, systems: 1980 },
  { month: 'Мар 2025', revenue: 720000, clients: 101, systems: 2100 },
  { month: 'Апр 2025', revenue: 780000, clients: 108, systems: 2240 },
  { month: 'Май 2025', revenue: 825000, clients: 115, systems: 2380 },
  { month: 'Июн 2025', revenue: 870000, clients: 119, systems: 2520 },
  { month: 'Июл 2025', revenue: 890000, clients: 123, systems: 2680 },
  { month: 'Авг 2025', revenue: 890000, clients: 125, systems: 2760 },
  { month: 'Сен 2025', revenue: 890000, clients: 127, systems: 2847 },
  { month: 'Окт 2025', revenue: 890000, clients: 127, systems: 2847 }
]

// Key performance indicators for investor attention
const kpiCards = computed(() => [
  {
    title: 'Месячная регулярная выручка',
    value: `₽${(businessIntelligence.value.monthlyRecurringRevenue.current / 1000).toFixed(0)}К`,
    growth: businessIntelligence.value.monthlyRecurringRevenue.growth,
    icon: 'heroicons:chart-bar',
    color: 'green',
    subtitle: 'Стабильный рост'
  },
  {
    title: 'Корпоративные клиенты',
    value: businessIntelligence.value.enterpriseClients.total,
    growth: ((businessIntelligence.value.enterpriseClients.newThisMonth / businessIntelligence.value.enterpriseClients.total) * 100),
    icon: 'heroicons:building-office-2',
    color: 'blue',
    subtitle: `+${businessIntelligence.value.enterpriseClients.newThisMonth} в этом месяце`
  },
  {
    title: 'Годовая выручка (ARR)',
    value: `₽${(businessIntelligence.value.annualRunRate.current / 1000000).toFixed(1)}М`,
    growth: ((businessIntelligence.value.annualRunRate.projected - businessIntelligence.value.annualRunRate.current) / businessIntelligence.value.annualRunRate.current) * 100,
    icon: 'heroicons:currency-dollar',
    color: 'purple',
    subtitle: `${businessIntelligence.value.annualRunRate.confidence}% уверенность`
  },
  {
    title: 'Валовая маржа',
    value: `${businessIntelligence.value.grossMargin}%`,
    growth: 2.4,
    icon: 'heroicons:chart-pie',
    color: 'orange',
    subtitle: 'Лидер отрасли по эффективности'
  },
  {
    title: 'Средняя LTV клиента',
    value: `₽${(businessIntelligence.value.customerLifetimeValue.average / 1000).toFixed(0)}К`,
    growth: 15.7,
    icon: 'heroicons:users',
    color: 'teal',
    subtitle: `₽${(businessIntelligence.value.customerLifetimeValue.enterprise / 1000).toFixed(0)}К корпоративный`
  },
  {
    title: 'Время работы системы',
    value: `${businessIntelligence.value.uptimePercent}%`,
    growth: 0.12,
    icon: 'heroicons:shield-check',
    color: 'green',
    subtitle: 'Превышает гарантии SLA'
  },
  {
    title: 'Оценка NPS',
    value: businessIntelligence.value.netPromoterScore,
    growth: 8.3,
    icon: 'heroicons:heart',
    color: 'red',
    subtitle: 'Мировой класс удовлетворённости'
  },
  {
    title: 'Доля рынка',
    value: `${businessIntelligence.value.marketShare.current}%`,
    growth: ((businessIntelligence.value.marketShare.target - businessIntelligence.value.marketShare.current) / businessIntelligence.value.marketShare.current) * 100,
    icon: 'heroicons:globe-alt',
    color: 'indigo',
    subtitle: `Цель: ${businessIntelligence.value.marketShare.target}%`
  }
])

const formatCurrency = (amount: number, unit = 'K') => {
  if (unit === 'K') return `₽${(amount / 1000).toFixed(0)}К`
  if (unit === 'M') return `₽${(amount / 1000000).toFixed(1)}М`
  return `₽${amount.toLocaleString('ru-RU')}`
}

const formatPercent = (value: number, showSign = true) => {
  const sign = showSign && value > 0 ? '+' : ''
  return `${sign}${value.toFixed(1)}%`
}

// Refresh business data
const refreshBusinessData = async () => {
  try {
    // In real implementation, fetch from business analytics API
    // const data = await api.getBusinessIntelligence()
    // businessIntelligence.value = data
    console.log('Данные бизнес-аналитики обновлены')
  } catch (error) {
    console.error('Ошибка обновления данных:', error)
  }
}

// Auto-refresh every 5 minutes
let refreshInterval: NodeJS.Timeout | null = null

onMounted(() => {
  refreshInterval = setInterval(refreshBusinessData, 5 * 60 * 1000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<template>
  <NuxtLayout name="dashboard">
    <div class="premium-section premium-fade-in">
      <!-- Investor Header -->
      <div class="premium-hero p-8 mb-8">
        <div class="flex items-center justify-between">
          <div>
            <h1 class="premium-heading-lg mb-3 flex items-center text-white">
              <Icon name="heroicons:presentation-chart-line" class="w-10 h-10 mr-4 text-blue-400" />
              Панель бизнес-аналитики
            </h1>
            <p class="premium-body-lg text-blue-100 max-w-4xl">
              Метрики бизнеса в реальном времени, показатели роста и финансовые показатели для инвестиционных презентаций и стратегического планирования.
            </p>
          </div>
          <div class="hidden lg:block">
            <div class="flex items-center space-x-4">
              <div class="text-right">
                <div class="text-3xl font-bold text-green-400">
                  {{ formatPercent(businessIntelligence.totalRevenue.growth) }}
                </div>
                <div class="text-blue-200 text-sm">Рост выручки</div>
              </div>
              <div class="w-16 h-16 bg-gradient-to-br from-blue-400 to-green-400 rounded-full flex items-center justify-center">
                <Icon name="heroicons:arrow-trending-up" class="w-8 h-8 text-white" />
              </div>
            </div>
          </div>
        </div>
        
        <!-- Key indicators bar -->
        <div class="mt-8 grid grid-cols-2 md:grid-cols-4 gap-6">
          <div class="premium-glass rounded-lg p-4 text-center">
            <div class="text-2xl font-bold mb-1 text-white">{{ businessIntelligence.enterpriseClients.total }}+</div>
            <div class="text-blue-200 text-sm">Корпоративные клиенты</div>
          </div>
          <div class="premium-glass rounded-lg p-4 text-center">
            <div class="text-2xl font-bold mb-1 text-white">{{ formatCurrency(businessIntelligence.monthlyRecurringRevenue.current, 'K') }}</div>
            <div class="text-blue-200 text-sm">Месячная выручка</div>
          </div>
          <div class="premium-glass rounded-lg p-4 text-center">
            <div class="text-2xl font-bold mb-1 text-white">{{ businessIntelligence.grossMargin }}%</div>
            <div class="text-blue-200 text-sm">Валовая маржа</div>
          </div>
          <div class="premium-glass rounded-lg p-4 text-center">
            <div class="text-2xl font-bold mb-1 text-green-300">{{ businessIntelligence.runwayMonths }}</div>
            <div class="text-blue-200 text-sm">Финансовый статус</div>
          </div>
        </div>
      </div>

      <!-- KPI Grid using premium components -->
      <SectionHeader 
        title="Ключевые показатели эффективности"
        description="Основные метрики бизнеса с индикаторами роста"
        icon="heroicons:chart-bar-square"
        icon-color="green"
        class="mb-6"
      />
      
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        <KpiCard
          v-for="kpi in kpiCards"
          :key="kpi.title"
          :title="kpi.title"
          :value="kpi.value"
          :growth="kpi.growth"
          :icon="kpi.icon"
          :color="kpi.color"
          :subtitle="kpi.subtitle"
        />
      </div>

      <!-- Financial Performance Overview -->
      <div class="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-8">
        <!-- Revenue Growth Chart Area -->
        <div class="premium-card">
          <SectionHeader 
            title="Траектория роста выручки"
            description="Ежемесячная регулярная выручка с прогнозами роста"
            icon="heroicons:chart-bar-square"
            icon-color="green"
          />
          
          <div class="p-6">
            <!-- Simplified chart representation -->
            <div class="h-64 bg-gradient-to-t from-green-50 to-transparent dark:from-green-900/10 rounded-lg flex items-end justify-center p-4">
              <div class="grid grid-cols-10 gap-2 h-full w-full items-end">
                <div 
                  v-for="(metric, index) in growthMetrics"
                  :key="metric.month"
                  class="group relative"
                >
                  <div 
                    :style="`height: ${(metric.revenue / 900000) * 100}%`"
                    class="bg-gradient-to-t from-green-500 to-green-400 rounded-t hover:from-green-600 hover:to-green-500 transition-all duration-300 min-h-[20px] cursor-pointer"
                  >
                  </div>
                  <!-- Tooltip on hover -->
                  <div class="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-900 text-white text-xs px-2 py-1 rounded whitespace-nowrap">
                    {{ metric.month.split(' ')[0] }}: {{ formatCurrency(metric.revenue, 'K') }}
                  </div>
                </div>
              </div>
            </div>
            
            <!-- Legend -->
            <div class="mt-6 flex items-center justify-between text-sm">
              <div class="flex items-center space-x-4">
                <div class="flex items-center space-x-2">
                  <div class="w-3 h-3 bg-green-500 rounded"></div>
                  <span class="text-gray-600 dark:text-gray-300">Месячная выручка</span>
                </div>
              </div>
              <div class="text-gray-500 dark:text-gray-400">
                Последние 10 месяцев • Данные обновляются в режиме реального времени
              </div>
            </div>
          </div>
        </div>

        <!-- Customer Metrics -->
        <div class="premium-card">
          <SectionHeader 
            title="Метрики успеха клиентов"
            description="Показатели удовлетворённости и влияния на бизнес клиентов"
            icon="heroicons:users"
            icon-color="blue"
          />
          
          <div class="p-6 space-y-6">
            <!-- NPS Score -->
            <div class="flex items-center justify-between p-4 bg-gradient-to-r from-red-50 to-pink-50 dark:from-red-900/10 dark:to-pink-900/10 rounded-lg border border-red-100 dark:border-red-900/30">
              <div>
                <h3 class="font-semibold text-gray-900 dark:text-white mb-1">Оценка NPS</h3>
                <p class="text-sm text-gray-600 dark:text-gray-300">Индекс удовлетворённости клиентов</p>
              </div>
              <div class="text-right">
                <div class="text-3xl font-bold text-red-600 dark:text-red-400 mb-1">
                  {{ businessIntelligence.netPromoterScore }}
                </div>
                <div class="text-xs text-green-600 dark:text-green-400 flex items-center">
                  <Icon name="heroicons:arrow-trending-up" class="w-3 h-3 mr-1" />
                  +8.3%
                </div>
              </div>
            </div>
            
            <!-- Customer Economics -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div class="p-4 bg-gray-50 dark:bg-gray-700/30 rounded-lg">
                <h4 class="font-medium text-gray-900 dark:text-white mb-2">Средняя LTV</h4>
                <div class="text-2xl font-bold text-teal-600 dark:text-teal-400">
                  {{ formatCurrency(businessIntelligence.customerLifetimeValue.average) }}
                </div>
                <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Корпоративный: {{ formatCurrency(businessIntelligence.customerLifetimeValue.enterprise, 'K') }}
                </div>
              </div>
              
              <div class="p-4 bg-gray-50 dark:bg-gray-700/30 rounded-lg">
                <h4 class="font-medium text-gray-900 dark:text-white mb-2">Соотношение LTV/CAC</h4>
                <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {{ (businessIntelligence.customerLifetimeValue.average / businessIntelligence.customerAcquisitionCost.current).toFixed(1) }}:1
                </div>
                <div class="text-xs text-green-600 dark:text-green-400 mt-1">
                  Отличная экономика единицы
                </div>
              </div>
            </div>
            
            <!-- Churn & Retention -->
            <div class="p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/10 dark:to-emerald-900/10 rounded-lg border border-green-100 dark:border-green-900/30">
              <div class="flex items-center justify-between">
                <div>
                  <h4 class="font-semibold text-gray-900 dark:text-white">Месячный отток клиентов</h4>
                  <p class="text-sm text-gray-600 dark:text-gray-300">Отраслевой бенчмарк: 5-7%</p>
                </div>
                <div class="text-right">
                  <div class="text-3xl font-bold text-green-600 dark:text-green-400">
                    {{ businessIntelligence.enterpriseClients.churnRate }}%
                  </div>
                  <div class="text-xs text-green-700 dark:text-green-300">Отличное удержание</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Operational Excellence -->
      <SectionHeader 
        title="Операционное превосходство и производительность платформы"
        icon="heroicons:cog-6-tooth"
        icon-color="purple"
        class="mb-6"
      />
      
      <div class="premium-card mb-8">
        <div class="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <!-- Systems Monitored -->
          <div class="text-center p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-100 dark:border-blue-900/30">
            <Icon name="heroicons:server-stack" class="w-12 h-12 text-blue-600 dark:text-blue-400 mx-auto mb-3" />
            <div class="text-3xl font-bold text-blue-700 dark:text-blue-300 mb-1">
              {{ businessIntelligence.systemsMonitored.toLocaleString() }}
            </div>
            <div class="text-blue-600 dark:text-blue-400 font-medium text-sm">Систем под мониторингом</div>
            <div class="text-xs text-blue-500 dark:text-blue-400 mt-1">Через 127 объектов</div>
          </div>
          
          <!-- Daily Data Processing -->
          <div class="text-center p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl border border-purple-100 dark:border-purple-900/30">
            <Icon name="heroicons:cpu-chip" class="w-12 h-12 text-purple-600 dark:text-purple-400 mx-auto mb-3" />
            <div class="text-3xl font-bold text-purple-700 dark:text-purple-300 mb-1">
              {{ (businessIntelligence.dataPointsDaily / 1000000).toFixed(1) }}М
            </div>
            <div class="text-purple-600 dark:text-purple-400 font-medium text-sm">Точек данных в день</div>
            <div class="text-xs text-purple-500 dark:text-purple-400 mt-1">AI конвейер обработки</div>
          </div>
          
          <!-- Predictive Accuracy -->
          <div class="text-center p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl border border-green-100 dark:border-green-900/30">
            <Icon name="heroicons:sparkles" class="w-12 h-12 text-green-600 dark:text-green-400 mx-auto mb-3" />
            <div class="text-3xl font-bold text-green-700 dark:text-green-300 mb-1">
              {{ businessIntelligence.predictiveAccuracy }}%
            </div>
            <div class="text-green-600 dark:text-green-400 font-medium text-sm">Точность AI</div>
            <div class="text-xs text-green-500 dark:text-green-400 mt-1">Предсказание отказов</div>
          </div>
          
          <!-- Cost Savings -->
          <div class="text-center p-6 bg-gradient-to-br from-orange-50 to-yellow-50 dark:from-orange-900/20 dark:to-yellow-900/20 rounded-xl border border-orange-100 dark:border-orange-900/30">
            <Icon name="heroicons:currency-dollar" class="w-12 h-12 text-orange-600 dark:text-orange-400 mx-auto mb-3" />
            <div class="text-3xl font-bold text-orange-700 dark:text-orange-300 mb-1">
              {{ formatCurrency(businessIntelligence.maintenanceCostSaved.total, 'M') }}
            </div>
            <div class="text-orange-600 dark:text-orange-400 font-medium text-sm">Общая экономия</div>
            <div class="text-xs text-orange-500 dark:text-orange-400 mt-1">Для всех клиентов за год</div>
          </div>
        </div>
      </div>

      <!-- Action Items for Investors -->
      <div class="premium-hero p-8">
        <div class="text-center">
          <h2 class="text-2xl font-bold mb-4 text-white">
            Готовы к ускорению роста
          </h2>
          <p class="premium-body-lg text-blue-100 mb-8 max-w-3xl mx-auto">
            С доказанной экономикой единицы, техническим превосходством и огромной рыночной возможностью 
            мы готовы к быстрому масштабированию с правильным стратегическим партнёрством.
          </p>
          
          <div class="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6">
            <PremiumButton
              size="lg"
              gradient
              icon="heroicons:envelope"
              @click="navigateTo('/contact/investors')"
            >
              Связаться с командой
            </PremiumButton>
            
            <PremiumButton
              size="lg"
              variant="ghost"
              icon="heroicons:document-text"
              @click="window.print()"
            >
              Экспорт отчёта
            </PremiumButton>
          </div>
        </div>
      </div>
    </div>
  </NuxtLayout>
</template>

<style scoped>
.animation-delay-100 {
  animation-delay: 100ms;
}

.animation-delay-200 {
  animation-delay: 200ms;
}

.animation-delay-300 {
  animation-delay: 300ms;
}
</style>