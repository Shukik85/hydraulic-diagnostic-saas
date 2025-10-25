<script setup lang="ts">
// Investor dashboard with proper TypeScript and browser API access
import type { ButtonColor } from '~/types/api'

definePageMeta({
  title: 'Инвесторы | Hydraulic Diagnostic SaaS',
  middleware: 'auth'
})

useSeoMeta({
  title: 'Инвесторы - Бизнес Аналитика | Hydraulic Diagnostic SaaS',
  description: 'Комплексная аналитика для инвесторов и руководства'
})

interface KPI {
  id: string
  title: string
  value: string | number
  growth: number
  icon: string
  color: ButtonColor
  subtitle: string
}

const authStore = useAuthStore()

// Business KPIs with proper typing
const kpis = ref<KPI[]>([
  {
    id: 'revenue',
    title: 'Месячная выручка',
    value: '2.4M ₽',
    growth: 23.5,
    icon: 'heroicons:currency-dollar',
    color: 'blue' as ButtonColor,
    subtitle: '+23.5% к прошлому месяцу'
  },
  {
    id: 'customers',
    title: 'Клиенты',
    value: 127,
    growth: 18.2,
    icon: 'heroicons:users',
    color: 'green' as ButtonColor,
    subtitle: '+23 новых клиента'
  },
  {
    id: 'retention',
    title: 'Удержание',
    value: '94.3%',
    growth: 5.1,
    icon: 'heroicons:heart',
    color: 'purple' as ButtonColor,
    subtitle: 'Высокая лояльность'
  },
  {
    id: 'systems',
    title: 'Мониторинг систем',
    value: '1,847',
    growth: 31.8,
    icon: 'heroicons:server-stack',
    color: 'orange' as ButtonColor,
    subtitle: 'Активных систем'
  }
])

// Market data
const marketData = ref({
  businessIntelligence: {
    totalRevenue: { current: 28800000, growth: 23.5, period: 'Месяц' },
    monthlyRecurringRevenue: { current: 2400000, growth: 18.2, period: 'Месяц' },
    annualRunRate: { current: 34560000, projected: 45000000, confidence: 85 },
    
    customerMetrics: {
      totalCustomers: 127,
      newCustomers: 23,
      churnRate: 2.1,
      netRevenueRetention: 118.3,
      averageContractValue: 186000,
      customerSatisfactionScore: 4.7
    },
    
    operationalMetrics: {
      systemsMonitored: 1847,
      uptimePercentage: 99.94,
      alertsProcessed: 12840,
      avgResponseTime: 1.2,
      predictiveAccuracy: 94.8,
      maintenanceCostSavings: 3200000
    }
  }
})

// Fixed browser API access
const handlePrintReport = (): void => {
  if (typeof window !== 'undefined') {
    window.print()
  }
}

// Date formatting helper
const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('ru-RU', {
    style: 'currency',
    currency: 'RUB',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(amount)
}

const formatNumber = (number: number): string => {
  return new Intl.NumberFormat('ru-RU').format(number)
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <div class="container mx-auto px-4 py-8">
      <!-- Header -->
      <div class="mb-8">
        <div class="flex items-center justify-between">
          <div>
            <h1 class="premium-heading-xl text-gray-900 dark:text-white mb-2">
              📈 Бизнес-Аналитика
            </h1>
            <p class="premium-body text-gray-600 dark:text-gray-300">
              Ключевые показатели и метрики для стейкхолдеров
            </p>
          </div>
          <div class="flex items-center space-x-3">
            <PremiumButton
              variant="secondary"
              icon="heroicons:document-text"
              @click="handlePrintReport"
            >
              Экспорт отчёта
            </PremiumButton>
            <PremiumButton
              gradient
              icon="heroicons:presentation-chart-line"
            >
              Презентация
            </PremiumButton>
          </div>
        </div>
      </div>

      <!-- Key Metrics Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div
          v-for="kpi in kpis"
          :key="kpi.id"
          class="premium-card p-6 hover:shadow-lg transition-all"
        >
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-sm font-medium text-gray-600 dark:text-gray-400">{{ kpi.title }}</h3>
            <div :class="[
              'p-2 rounded-lg',
              kpi.color === 'blue' ? 'bg-blue-50 dark:bg-blue-900/30' :
              kpi.color === 'green' ? 'bg-green-50 dark:bg-green-900/30' :
              kpi.color === 'purple' ? 'bg-purple-50 dark:bg-purple-900/30' :
              'bg-orange-50 dark:bg-orange-900/30'
            ]">
              <Icon :name="kpi.icon" :class="[
                'w-5 h-5',
                kpi.color === 'blue' ? 'text-blue-600 dark:text-blue-400' :
                kpi.color === 'green' ? 'text-green-600 dark:text-green-400' :
                kpi.color === 'purple' ? 'text-purple-600 dark:text-purple-400' :
                'text-orange-600 dark:text-orange-400'
              ]" />
            </div>
          </div>
          
          <div class="mb-2">
            <div class="text-2xl font-bold text-gray-900 dark:text-white mb-1">{{ kpi.value }}</div>
            <div class="flex items-center">
              <Icon name="heroicons:arrow-trending-up" class="w-3 h-3 text-green-500 mr-1" />
              <span class="text-xs font-medium text-green-600 dark:text-green-400">+{{ kpi.growth }}%</span>
            </div>
          </div>
          
          <p class="text-xs text-gray-500 dark:text-gray-400">{{ kpi.subtitle }}</p>
        </div>
      </div>

      <!-- Detailed Analytics -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <!-- Revenue Analytics -->
        <div class="premium-card">
          <div class="p-6 border-b border-gray-200 dark:border-gray-700">
            <h3 class="premium-heading-sm text-gray-900 dark:text-white">💰 Финансовые показатели</h3>
          </div>
          <div class="p-6 space-y-6">
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600 dark:text-gray-400">Месячная выручка</span>
              <span class="font-semibold text-gray-900 dark:text-white">{{ formatCurrency(marketData.businessIntelligence.totalRevenue.current) }}</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600 dark:text-gray-400">MRR</span>
              <span class="font-semibold text-gray-900 dark:text-white">{{ formatCurrency(marketData.businessIntelligence.monthlyRecurringRevenue.current) }}</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600 dark:text-gray-400">ARR (прогноз)</span>
              <span class="font-semibold text-gray-900 dark:text-white">{{ formatCurrency(marketData.businessIntelligence.annualRunRate.projected) }}</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600 dark:text-gray-400">Уверенность прогноза</span>
              <span class="font-semibold text-green-600 dark:text-green-400">{{ marketData.businessIntelligence.annualRunRate.confidence }}%</span>
            </div>
          </div>
        </div>

        <!-- Customer Analytics -->
        <div class="premium-card">
          <div class="p-6 border-b border-gray-200 dark:border-gray-700">
            <h3 class="premium-heading-sm text-gray-900 dark:text-white">👥 Клиентская база</h3>
          </div>
          <div class="p-6 space-y-6">
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600 dark:text-gray-400">Общий клиенты</span>
              <span class="font-semibold text-gray-900 dark:text-white">{{ formatNumber(marketData.businessIntelligence.customerMetrics.totalCustomers) }}</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600 dark:text-gray-400">Новые за месяц</span>
              <span class="font-semibold text-green-600 dark:text-green-400">+{{ marketData.businessIntelligence.customerMetrics.newCustomers }}</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600 dark:text-gray-400">Churn Rate</span>
              <span class="font-semibold text-gray-900 dark:text-white">{{ marketData.businessIntelligence.customerMetrics.churnRate }}%</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600 dark:text-gray-400">NRR</span>
              <span class="font-semibold text-green-600 dark:text-green-400">{{ marketData.businessIntelligence.customerMetrics.netRevenueRetention }}%</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600 dark:text-gray-400">Средняя стоимость контракта</span>
              <span class="font-semibold text-gray-900 dark:text-white">{{ formatCurrency(marketData.businessIntelligence.customerMetrics.averageContractValue) }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Operational Metrics -->
      <div class="premium-card mb-8">
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 class="premium-heading-sm text-gray-900 dark:text-white">⚙️ Операционные показатели</h3>
        </div>
        <div class="p-6">
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div class="text-center p-4 bg-blue-50 dark:bg-blue-900/30 rounded-xl">
              <div class="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                {{ formatNumber(marketData.businessIntelligence.operationalMetrics.systemsMonitored) }}
              </div>
              <div class="text-sm font-medium text-gray-700 dark:text-gray-300">Мониторинг систем</div>
            </div>
            
            <div class="text-center p-4 bg-green-50 dark:bg-green-900/30 rounded-xl">
              <div class="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">
                {{ marketData.businessIntelligence.operationalMetrics.uptimePercentage }}%
              </div>
              <div class="text-sm font-medium text-gray-700 dark:text-gray-300">Uptime SLA</div>
            </div>
            
            <div class="text-center p-4 bg-purple-50 dark:bg-purple-900/30 rounded-xl">
              <div class="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-2">
                {{ marketData.businessIntelligence.operationalMetrics.predictiveAccuracy }}%
              </div>
              <div class="text-sm font-medium text-gray-700 dark:text-gray-300">Точность прогнозов</div>
            </div>
            
            <div class="text-center p-4 bg-orange-50 dark:bg-orange-900/30 rounded-xl">
              <div class="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-2">
                {{ formatNumber(marketData.businessIntelligence.operationalMetrics.alertsProcessed) }}
              </div>
              <div class="text-sm font-medium text-gray-700 dark:text-gray-300">Обработанные алерты</div>
            </div>
            
            <div class="text-center p-4 bg-teal-50 dark:bg-teal-900/30 rounded-xl">
              <div class="text-3xl font-bold text-teal-600 dark:text-teal-400 mb-2">
                {{ marketData.businessIntelligence.operationalMetrics.avgResponseTime }}s
              </div>
              <div class="text-sm font-medium text-gray-700 dark:text-gray-300">Среднее время отклика</div>
            </div>
            
            <div class="text-center p-4 bg-indigo-50 dark:bg-indigo-900/30 rounded-xl">
              <div class="text-3xl font-bold text-indigo-600 dark:text-indigo-400 mb-2">
                {{ formatCurrency(marketData.businessIntelligence.operationalMetrics.maintenanceCostSavings) }}
              </div>
              <div class="text-sm font-medium text-gray-700 dark:text-gray-300">Экономия на обслуживании</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Market Position -->
      <div class="premium-card">
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 class="premium-heading-sm text-gray-900 dark:text-white">🎯 Маркетинговые инсайты</h3>
        </div>
        <div class="p-6">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <!-- Market Traction -->
            <div>
              <h4 class="font-semibold text-gray-900 dark:text-white mb-4">Тракшен на рынке</h4>
              <div class="space-y-3">
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-600 dark:text-gray-400">Общий рынок (TAM)</span>
                  <span class="font-medium text-gray-900 dark:text-white">$12.4B</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-600 dark:text-gray-400">Доступный рынок (SAM)</span>
                  <span class="font-medium text-gray-900 dark:text-white">$2.1B</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-600 dark:text-gray-400">Обслуживаемый рынок (SOM)</span>
                  <span class="font-medium text-gray-900 dark:text-white">$180M</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-600 dark:text-gray-400">Наша доля</span>
                  <span class="font-medium text-blue-600 dark:text-blue-400">0.19%</span>
                </div>
              </div>
            </div>
            
            <!-- Growth Projections -->
            <div>
              <h4 class="font-semibold text-gray-900 dark:text-white mb-4">Прогнозы роста</h4>
              <div class="space-y-3">
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-600 dark:text-gray-400">2025 ARR</span>
                  <span class="font-medium text-gray-900 dark:text-white">{{ formatCurrency(45000000) }}</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-600 dark:text-gray-400">2026 ARR</span>
                  <span class="font-medium text-gray-900 dark:text-white">{{ formatCurrency(78000000) }}</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-600 dark:text-gray-400">2027 ARR</span>
                  <span class="font-medium text-gray-900 dark:text-white">{{ formatCurrency(125000000) }}</span>
                </div>
                <div class="flex items-center justify-between border-t border-gray-200 dark:border-gray-700 pt-3">
                  <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Годовой рост</span>
                  <span class="font-semibold text-green-600 dark:text-green-400">+73% CAGR</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Investment Highlights -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
        <!-- Competitive Advantages -->
        <div class="premium-card">
          <div class="p-6 border-b border-gray-200 dark:border-gray-700">
            <h3 class="premium-heading-sm text-gray-900 dark:text-white">✨ Конкурентные преимущества</h3>
          </div>
          <div class="p-6">
            <div class="space-y-4">
              <div class="flex items-start space-x-3">
                <Icon name="heroicons:cpu-chip" class="w-5 h-5 text-blue-500 mt-0.5" />
                <div>
                  <h4 class="font-medium text-gray-900 dark:text-white">AI-поведенческая диагностика</h4>
                  <p class="text-sm text-gray-600 dark:text-gray-400">Предсказание неисправностей за 30 дней с 94.8% точностью</p>
                </div>
              </div>
              
              <div class="flex items-start space-x-3">
                <Icon name="heroicons:shield-check" class="w-5 h-5 text-green-500 mt-0.5" />
                <div>
                  <h4 class="font-medium text-gray-900 dark:text-white">Enterprise безопасность</h4>
                  <p class="text-sm text-gray-600 dark:text-gray-400">SOC 2 Type II, ISO 27001, шифрование класса AES-256</p>
                </div>
              </div>
              
              <div class="flex items-start space-x-3">
                <Icon name="heroicons:bolt" class="w-5 h-5 text-yellow-500 mt-0.5" />
                <div>
                  <h4 class="font-medium text-gray-900 dark:text-white">Реальное время мониторинг</h4>
                  <p class="text-sm text-gray-600 dark:text-gray-400">Обработка данных с задержкой менее 1.2 секунд</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- ROI Calculator -->
        <div class="premium-card">
          <div class="p-6 border-b border-gray-200 dark:border-gray-700">
            <h3 class="premium-heading-sm text-gray-900 dark:text-white">📉 ROI для клиентов</h3>
          </div>
          <div class="p-6">
            <div class="space-y-4">
              <div class="text-center p-6 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/30 dark:to-blue-900/30 rounded-xl">
                <div class="text-4xl font-bold text-green-600 dark:text-green-400 mb-2">
                  {{ formatCurrency(marketData.businessIntelligence.operationalMetrics.maintenanceCostSavings) }}
                </div>
                <div class="text-sm font-medium text-gray-700 dark:text-gray-300">Средняя экономия на обслуживании в год</div>
                <div class="text-xs text-gray-500 dark:text-gray-400 mt-2">За счёт предикативного обслуживания</div>
              </div>
              
              <div class="grid grid-cols-2 gap-4">
                <div class="text-center p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                  <div class="text-2xl font-bold text-gray-900 dark:text-white mb-1">18</div>
                  <div class="text-xs text-gray-500 dark:text-gray-400">месяцев окупаемости</div>
                </div>
                <div class="text-center p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                  <div class="text-2xl font-bold text-gray-900 dark:text-white mb-1">340%</div>
                  <div class="text-xs text-gray-500 dark:text-gray-400">ROI через 3 года</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>