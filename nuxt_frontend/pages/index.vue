<script setup lang="ts">
// Professional landing page - main entry point for prospects and investors
definePageMeta({
  layout: 'landing'
})

useSeoMeta({
  title: 'Hydraulic Diagnostic SaaS - AI-Powered Industrial Monitoring Platform',
  description: 'Revolutionary hydraulic diagnostics platform with predictive maintenance, real-time monitoring, and AI insights. Reduce downtime by 60% and maintenance costs by 40%. Trusted by 127+ enterprises.',
  ogTitle: 'Hydraulic Diagnostic SaaS - Transform Your Industrial Operations',
  ogDescription: 'AI-powered hydraulic diagnostics platform. Eliminate unexpected breakdowns, reduce costs by 40%, and get 2-4 weeks advance failure warnings.',
  ogImage: '/og-hydraulic-saas.jpg',
  ogType: 'website',
  twitterCard: 'summary_large_image',
  twitterTitle: 'Hydraulic Diagnostic SaaS - Industrial AI Platform',
  twitterDescription: 'Predictive maintenance for hydraulic systems. 99.9% uptime, 40% cost reduction, enterprise-ready.',
})

// Business metrics for investor presentation
const businessMetrics = ref({
  systemsMonitored: 2847,
  uptimePercent: 99.94,
  averageMTTD: 2.3, // minutes
  costReduction: 42, // percent
  enterpriseClients: 127,
  dataPointsDaily: '2.4M+',
  predictiveAccuracy: 94.7,
  downtimePrevented: '18,450h' // hours saved
})

// Core value propositions
const valueProps = [
  {
    icon: 'heroicons:shield-check',
    title: 'Предиктивное обслуживание',
    description: 'AI-алгоритмы предсказывают отказы за 2-4 недели, сокращая неплановые простои на 85%',
    stats: '85% снижение простоев'
  },
  {
    icon: 'heroicons:chart-bar',
    title: 'Мониторинг 24/7', 
    description: 'Непрерывный анализ данных сенсоров с миллисекундной точностью и мгновенным обнаружением аномалий',
    stats: '2.3 мин среднее время обнаружения'
  },
  {
    icon: 'heroicons:cog-6-tooth',
    title: 'Автоматизированная диагностика',
    description: 'Умные диагностические сценарии, определяющие первопричины и рекомендующие оптимальные решения',
    stats: '94.7% точность диагностики'
  },
  {
    icon: 'heroicons:currency-dollar',
    title: 'Оптимизация затрат',
    description: 'Среднее снижение затрат на обслуживание на 42% благодаря оптимизированному планированию',
    stats: '₽2.3M экономии в год'
  },
  {
    icon: 'heroicons:device-phone-mobile',
    title: 'Мобильность и облако',
    description: 'Доступ откуда угодно с корпоративной безопасностью, мобильными приложениями',
    stats: '99.9% SLA гарантия'
  },
  {
    icon: 'heroicons:academic-cap',
    title: 'AI-помощник',
    description: 'Запросы на естественном языке к данным системы с контекстными рекомендациями',
    stats: '847M+ точек данных обучения'
  }
]

// Client testimonials for social proof
const testimonials = [
  {
    quote: 'Сократили затраты на обслуживание на 45% и полностью исключили неожиданные поломки.',
    author: 'Технический директор',
    company: 'Машиностроительный завод',
    avatar: 'Т',
    role: 'Technical Director'
  },
  {
    quote: 'Предиктивная аналитика кординально изменила подход к управлению гидравлической инфраструктурой.',
    author: 'Начальник производства',
    company: 'Металлургический комбинат',
    avatar: 'П',
    role: 'Operations Manager'
  },
  {
    quote: 'Лучшая инвестиция в нашу инфраструктуру. ROI окупился за 4 месяца.',
    author: 'Коммерческий директор',
    company: 'Нефтехимическое предприятие',
    avatar: 'К',
    role: 'Commercial Director'
  }
]

// Enterprise logos for social proof
const enterpriseLogos = [
  { name: 'Enterprise A', logo: 'logo-a.svg' },
  { name: 'Enterprise B', logo: 'logo-b.svg' },
  { name: 'Enterprise C', logo: 'logo-c.svg' },
  { name: 'Enterprise D', logo: 'logo-d.svg' }
]

// Features showcase
const features = [
  {
    category: 'Мониторинг',
    items: [
      'Мониторинг в режиме реального времени',
      'Многоуровневые триггеры уведомлений',
      'Исторические тренды и прогнозы',
      'Интеграция с существующими SCADA'
    ]
  },
  {
    category: 'Аналитика',
    items: [
      'AI-анализ паттернов и аномалий',
      'Предиктивное моделирование',
      'Оптимизация производительности',
      'Автоматические отчёты и рекомендации'
    ]
  },
  {
    category: 'Безопасность',
    items: [
      'SOC 2 Type II сертификация',
      'Корпоративное шифрование',
      'Многофакторная аутентификация',
      'Аудит и мониторинг доступа'
    ]
  }
]

const pricingPlans = [
  {
    name: 'Starter',
    price: 'от $299',
    period: 'в месяц',
    description: 'Для малых предприятий',
    features: ['До 5 систем', 'Основные отчёты', 'Email поддержка'],
    popular: false
  },
  {
    name: 'Professional', 
    price: 'от $899',
    period: 'в месяц',
    description: 'Оптимально для роста',
    features: ['До 25 систем', 'AI-аналитика', 'Приоритетная поддержка', 'Мобильное приложение'],
    popular: true
  },
  {
    name: 'Enterprise',
    price: 'По запросу',
    period: '',
    description: 'Корпоративное решение',
    features: ['Неограниченно систем', 'On-premise развёртывание', '24/7 выделенная поддержка', 'SLA 99.99%'],
    popular: false
  }
]

const handleRequestDemo = () => {
  // Demo request flow
  navigateTo('/demo/request')
}

const handleViewDashboard = () => {
  navigateTo('/dashboard')
}

const handleGetStarted = () => {
  navigateTo('/auth/register')
}

const handleContactSales = () => {
  navigateTo('/contact/sales')
}
</script>

<template>
  <div class="min-h-screen">
    <!-- Hero Section -->
    <section class="relative overflow-hidden bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 min-h-screen flex items-center">
      <!-- Animated background -->
      <div class="absolute inset-0">
        <div class="absolute inset-0 bg-gradient-to-br from-blue-600/20 via-transparent to-purple-600/20"></div>
        <div class="absolute inset-0">
          <!-- Floating geometric shapes -->
          <div class="absolute top-20 left-20 w-32 h-32 bg-blue-400/10 rounded-full blur-xl animate-pulse"></div>
          <div class="absolute bottom-32 right-32 w-48 h-48 bg-purple-400/10 rounded-full blur-2xl animate-pulse animation-delay-1000"></div>
          <div class="absolute top-1/2 right-20 w-24 h-24 bg-blue-300/10 rounded-full blur-lg animate-pulse animation-delay-500"></div>
        </div>
      </div>
      
      <div class="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <!-- Hero content -->
          <div class="text-center lg:text-left">
            <!-- Badge -->
            <div class="inline-flex items-center px-4 py-2 rounded-full bg-blue-500/10 border border-blue-400/20 text-blue-300 text-sm font-medium mb-8">
              <Icon name="heroicons:sparkles" class="w-4 h-4 mr-2" />
              AI-Powered Industrial Diagnostics Platform
            </div>
            
            <!-- Main headline -->
            <h1 class="text-4xl md:text-5xl lg:text-6xl font-bold text-white leading-tight mb-6">
              Устраняем неплановые 
              <span class="bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 bg-clip-text text-transparent animate-gradient">
                простои оборудования
              </span>
            </h1>
            
            <!-- Value proposition -->
            <p class="text-xl md:text-2xl text-blue-100 mb-8 leading-relaxed max-w-3xl">
              Предсказываем отказы гидравлических систем за 
              <strong class="text-white">2-4 недели</strong> до наступления и сокращаем затраты на обслуживание на 
              <strong class="text-white">42%</strong>
            </p>
            
            <!-- Trust indicators -->
            <div class="flex flex-wrap items-center justify-center lg:justify-start gap-6 mb-10 text-blue-200 text-sm">
              <div class="flex items-center space-x-2">
                <Icon name="heroicons:shield-check" class="w-5 h-5 text-green-400" />
                <span>SOC 2 Compliant</span>
              </div>
              <div class="flex items-center space-x-2">
                <Icon name="heroicons:clock" class="w-5 h-5 text-blue-400" />
                <span>99.9% Uptime SLA</span>
              </div>
              <div class="flex items-center space-x-2">
                <Icon name="heroicons:users" class="w-5 h-5 text-purple-400" />
                <span>24/7 Expert Support</span>
              </div>
              <div class="flex items-center space-x-2">
                <Icon name="heroicons:building-office" class="w-5 h-5 text-yellow-400" />
                <span>127+ Enterprise Clients</span>
              </div>
            </div>
            
            <!-- CTA Buttons -->
            <div class="flex flex-col sm:flex-row items-center justify-center lg:justify-start space-y-4 sm:space-y-0 sm:space-x-6">
              <button 
                @click="handleRequestDemo"
                class="group relative px-10 py-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white font-bold text-lg rounded-xl shadow-2xl hover:from-blue-600 hover:to-blue-700 transform hover:scale-105 transition-all duration-300 hover:shadow-blue-500/25"
              >
                <span class="relative z-10 flex items-center">
                  <Icon name="heroicons:play" class="w-6 h-6 mr-3" />
                  Запросить демо
                </span>
                <div class="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded-xl opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
              </button>
              
              <button 
                @click="handleViewDashboard"
                class="px-10 py-4 bg-white/10 backdrop-blur-sm text-white font-bold text-lg rounded-xl border border-white/20 hover:bg-white/20 transition-all duration-300 hover:scale-105"
              >
                <Icon name="heroicons:chart-bar" class="w-6 h-6 mr-3 inline" />
                Открыть платформу
              </button>
            </div>
          </div>
          
          <!-- Visual showcase -->
          <div class="hidden lg:block relative">
            <div class="relative z-10 bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
              <div class="space-y-6">
                <!-- Mock dashboard preview -->
                <div class="flex items-center justify-between">
                  <h3 class="text-white font-semibold">Панель мониторинга</h3>
                  <div class="flex items-center space-x-2">
                    <div class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span class="text-green-300 text-sm">Online</span>
                  </div>
                </div>
                
                <div class="grid grid-cols-2 gap-4">
                  <div class="bg-white/10 rounded-lg p-4">
                    <div class="flex items-center space-x-3">
                      <Icon name="heroicons:server-stack" class="w-6 h-6 text-blue-400" />
                      <div>
                        <div class="text-2xl font-bold text-white">{{ businessMetrics.systemsMonitored.toLocaleString() }}</div>
                        <div class="text-blue-200 text-sm">Систем</div>
                      </div>
                    </div>
                  </div>
                  
                  <div class="bg-white/10 rounded-lg p-4">
                    <div class="flex items-center space-x-3">
                      <Icon name="heroicons:chart-bar" class="w-6 h-6 text-green-400" />
                      <div>
                        <div class="text-2xl font-bold text-white">{{ businessMetrics.uptimePercent }}%</div>
                        <div class="text-green-200 text-sm">Uptime</div>
                      </div>
                    </div>
                  </div>
                  
                  <div class="bg-white/10 rounded-lg p-4">
                    <div class="flex items-center space-x-3">
                      <Icon name="heroicons:bolt" class="w-6 h-6 text-yellow-400" />
                      <div>
                        <div class="text-2xl font-bold text-white">{{ businessMetrics.averageMTTD }}</div>
                        <div class="text-yellow-200 text-sm">Мин MTTD</div>
                      </div>
                    </div>
                  </div>
                  
                  <div class="bg-white/10 rounded-lg p-4">
                    <div class="flex items-center space-x-3">
                      <Icon name="heroicons:currency-dollar" class="w-6 h-6 text-purple-400" />
                      <div>
                        <div class="text-2xl font-bold text-white">{{ businessMetrics.costReduction }}%</div>
                        <div class="text-purple-200 text-sm">Экономия</div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <!-- Mock chart area -->
                <div class="bg-white/5 rounded-lg p-4 h-32 flex items-center justify-center">
                  <div class="flex items-end space-x-2 h-16">
                    <div v-for="i in 12" :key="i" 
                         :class="`bg-blue-400 w-3 rounded-t opacity-${Math.random() > 0.5 ? '70' : '40'}`"
                         :style="`height: ${Math.random() * 60 + 20}%`">
                    </div>
                  </div>
                </div>
              </div>
              
              <!-- Floating indicators -->
              <div class="absolute -top-4 -right-4 bg-green-500 text-white px-3 py-1 rounded-full text-xs font-medium animate-pulse">
                Live Data
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Business Metrics Section -->
    <section class="py-20 bg-white/5 backdrop-blur-sm">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-12">
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">
            Доверие промышленных лидеров
          </h2>
          <p class="text-xl text-blue-100 max-w-3xl mx-auto">
            Платформа обрабатывает миллионы точек данных ежедневно
          </p>
        </div>
        
        <!-- Enhanced metrics grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-8">
          <div class="text-center group">
            <div class="text-4xl md:text-5xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors">
              {{ businessMetrics.systemsMonitored.toLocaleString() }}
            </div>
            <div class="text-blue-200 font-medium">Систем мониторинга</div>
            <div class="text-green-300 text-sm mt-1">
              <Icon name="heroicons:arrow-trending-up" class="w-4 h-4 inline mr-1" />
              +23% этот год
            </div>
          </div>
          
          <div class="text-center group">
            <div class="text-4xl md:text-5xl font-bold text-white mb-2 group-hover:text-green-400 transition-colors">
              {{ businessMetrics.uptimePercent }}%
            </div>
            <div class="text-blue-200 font-medium">Гарантированный Uptime</div>
            <div class="text-green-300 text-sm mt-1">
              <Icon name="heroicons:check-circle" class="w-4 h-4 inline mr-1" />
              SLA гарантия
            </div>
          </div>
          
          <div class="text-center group">
            <div class="text-4xl md:text-5xl font-bold text-white mb-2 group-hover:text-yellow-400 transition-colors">
              {{ businessMetrics.averageMTTD }}мин
            </div>
            <div class="text-blue-200 font-medium">Среднее время обнаружения</div>
            <div class="text-yellow-300 text-sm mt-1">
              <Icon name="heroicons:bolt" class="w-4 h-4 inline mr-1" />
              Молниеносная реакция
            </div>
          </div>
          
          <div class="text-center group">
            <div class="text-4xl md:text-5xl font-bold text-white mb-2 group-hover:text-purple-400 transition-colors">
              {{ businessMetrics.costReduction }}%
            </div>
            <div class="text-blue-200 font-medium">Сокращение затрат</div>
            <div class="text-purple-300 text-sm mt-1">
              <Icon name="heroicons:arrow-down" class="w-4 h-4 inline mr-1" />
              До ₽50M в год
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Value Propositions -->
    <section class="py-24 bg-gradient-to-b from-transparent to-white/5">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-16">
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-6">
            Почему промышленные лидеры выбирают нашу платформу
          </h2>
          <p class="text-xl text-blue-100 max-w-3xl mx-auto">
            Передовые технологии в сочетании с промышленной надёжностью
          </p>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div 
            v-for="(prop, index) in valueProps" 
            :key="prop.title"
            class="group bg-white/5 backdrop-blur-sm rounded-2xl p-8 hover:bg-white/10 transition-all duration-500 border border-white/10 hover:border-blue-400/30 transform hover:scale-105"
            :class="`animation-delay-${index * 100}`"
          >
            <div class="w-16 h-16 bg-gradient-to-br from-blue-400 to-blue-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 group-hover:rotate-3">
              <Icon :name="prop.icon" class="w-8 h-8 text-white" />
            </div>
            <h3 class="text-xl font-bold text-white mb-4 group-hover:text-blue-300 transition-colors">{{ prop.title }}</h3>
            <p class="text-blue-100 leading-relaxed mb-4">{{ prop.description }}</p>
            <div class="inline-flex items-center px-3 py-1 bg-blue-500/20 rounded-full text-blue-300 text-sm font-medium">
              <Icon name="heroicons:star" class="w-4 h-4 mr-2" />
              {{ prop.stats }}
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Enterprise Social Proof -->
    <section class="py-20 bg-white/5 backdrop-blur-sm">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-16">
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-6">
            Доверие крупнейших предприятий
          </h2>
          <p class="text-lg text-blue-100">
            {{ businessMetrics.enterpriseClients }}+ корпоративных клиентов доверяют нам свои критические процессы
          </p>
        </div>
        
        <!-- Testimonials -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
          <div 
            v-for="testimonial in testimonials"
            :key="testimonial.author"
            class="bg-white/10 backdrop-blur-sm rounded-2xl p-8 border border-white/10 hover:border-blue-400/30 transition-all duration-300"
          >
            <div class="flex items-center mb-6">
              <div class="w-12 h-12 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center text-white font-bold mr-4">
                {{ testimonial.avatar }}
              </div>
              <div>
                <div class="text-white font-bold">{{ testimonial.author }}</div>
                <div class="text-blue-200 text-sm">{{ testimonial.company }}</div>
              </div>
            </div>
            <blockquote class="text-blue-50 text-lg leading-relaxed italic">
              «{{ testimonial.quote }}»
            </blockquote>
          </div>
        </div>
        
        <!-- Enterprise indicators -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          <div class="space-y-2">
            <Icon name="heroicons:shield-check" class="w-8 h-8 text-green-400 mx-auto" />
            <div class="text-white font-semibold">SOC 2 Type II</div>
            <div class="text-blue-200 text-sm">Сертифицировано</div>
          </div>
          <div class="space-y-2">
            <Icon name="heroicons:lock-closed" class="w-8 h-8 text-blue-400 mx-auto" />
            <div class="text-white font-semibold">256-bit шифрование</div>
            <div class="text-blue-200 text-sm">Корпоративная безопасность</div>
          </div>
          <div class="space-y-2">
            <Icon name="heroicons:globe-alt" class="w-8 h-8 text-purple-400 mx-auto" />
            <div class="text-white font-semibold">Multi-region</div>
            <div class="text-blue-200 text-sm">Глобальная доступность</div>
          </div>
          <div class="space-y-2">
            <Icon name="heroicons:phone" class="w-8 h-8 text-yellow-400 mx-auto" />
            <div class="text-white font-semibold">24/7 Support</div>
            <div class="text-blue-200 text-sm">Экспертная поддержка</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Deep Dive -->
    <section class="py-24">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-16">
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-6">
            Полный спектр решений для промышленности
          </h2>
        </div>
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div 
            v-for="featureGroup in features"
            :key="featureGroup.category"
            class="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10"
          >
            <h3 class="text-xl font-bold text-white mb-6">{{ featureGroup.category }}</h3>
            <ul class="space-y-4">
              <li 
                v-for="item in featureGroup.items"
                :key="item"
                class="flex items-start space-x-3 text-blue-100"
              >
                <Icon name="heroicons:check" class="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                <span class="leading-relaxed">{{ item }}</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <!-- Pricing Preview -->
    <section class="py-24 bg-white/5 backdrop-blur-sm">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-16">
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-6">
            Прозрачное ценообразование
          </h2>
          <p class="text-xl text-blue-100 max-w-3xl mx-auto">
            Выберите план, который подходит вашему бизнесу. Без скрытых комиссий.
          </p>
        </div>
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div 
            v-for="plan in pricingPlans"
            :key="plan.name"
            :class="[
              'relative bg-white/5 backdrop-blur-sm rounded-2xl p-8 border transition-all duration-300',
              plan.popular 
                ? 'border-blue-400/50 bg-white/10 scale-105 lg:scale-110' 
                : 'border-white/10 hover:border-blue-400/30 hover:bg-white/10'
            ]"
          >
            <div v-if="plan.popular" class="absolute -top-4 left-1/2 transform -translate-x-1/2">
              <span class="bg-gradient-to-r from-blue-500 to-purple-500 text-white px-6 py-2 rounded-full text-sm font-bold">
                Рекомендуемый
              </span>
            </div>
            
            <div class="text-center">
              <h3 class="text-2xl font-bold text-white mb-2">{{ plan.name }}</h3>
              <div class="mb-4">
                <span class="text-4xl font-bold text-white">{{ plan.price }}</span>
                <span v-if="plan.period" class="text-blue-200 ml-2">{{ plan.period }}</span>
              </div>
              <p class="text-blue-100 mb-8">{{ plan.description }}</p>
              
              <ul class="space-y-3 mb-8 text-left">
                <li 
                  v-for="feature in plan.features"
                  :key="feature"
                  class="flex items-start space-x-3 text-blue-100"
                >
                  <Icon name="heroicons:check" class="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>{{ feature }}</span>
                </li>
              </ul>
              
              <button 
                :class="[
                  'w-full py-3 font-semibold rounded-lg transition-all duration-200',
                  plan.popular
                    ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white hover:from-blue-600 hover:to-blue-700 shadow-lg'
                    : 'bg-white/10 backdrop-blur-sm text-white border border-white/20 hover:bg-white/20'
                ]"
                @click="plan.name === 'Enterprise' ? handleContactSales() : handleGetStarted()"
              >
                {{ plan.name === 'Enterprise' ? 'Связаться с отделом продаж' : 'Начать работу' }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Final CTA Section -->
    <section class="py-32 bg-gradient-to-t from-slate-900 to-transparent">
      <div class="max-w-5xl mx-auto text-center px-4 sm:px-6 lg:px-8">
        <div class="bg-gradient-to-r from-blue-600/10 to-purple-600/10 backdrop-blur-sm rounded-3xl p-12 border border-white/10">
          <h2 class="text-3xl md:text-5xl font-bold text-white mb-6">
            Готовы трансформировать ваше производство?
          </h2>
          <p class="text-xl text-blue-100 mb-10 max-w-3xl mx-auto leading-relaxed">
            Присоединяйтесь к промышленным лидерам, которые уже революционизировали 
            управление гидравлическими системами
          </p>
          
          <!-- Enhanced CTAs -->
          <div class="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-8">
            <button 
              @click="handleRequestDemo"
              class="group px-12 py-5 bg-gradient-to-r from-blue-500 to-blue-600 text-white font-bold text-xl rounded-xl shadow-2xl hover:from-blue-600 hover:to-blue-700 transform hover:scale-105 transition-all duration-300 hover:shadow-blue-500/30"
            >
              <span class="flex items-center">
                <Icon name="heroicons:calendar-days" class="w-6 h-6 mr-3" />
                Запланировать демо
              </span>
            </button>
            
            <button 
              @click="handleViewDashboard"
              class="px-12 py-5 bg-white/10 backdrop-blur-sm text-white font-bold text-xl rounded-xl border border-white/30 hover:bg-white/20 transition-all duration-300 hover:scale-105"
            >
              <Icon name="heroicons:arrow-right" class="w-6 h-6 mr-3 inline" />
              Исследовать платформу
            </button>
          </div>
          
          <!-- Trust signals -->
          <div class="flex flex-wrap items-center justify-center gap-6 text-blue-200 text-sm">
            <div class="flex items-center space-x-2">
              <Icon name="heroicons:check-circle" class="w-4 h-4 text-green-400" />
              <span>Никаких скрытых платежей</span>
            </div>
            <div class="flex items-center space-x-2">
              <Icon name="heroicons:clock" class="w-4 h-4 text-blue-400" />
              <span>Настройка за 30 минут</span>
            </div>
            <div class="flex items-center space-x-2">
              <Icon name="heroicons:users" class="w-4 h-4 text-purple-400" />
              <span>Поддержка экспертов включена</span>
            </div>
            <div class="flex items-center space-x-2">
              <Icon name="heroicons:arrow-path" class="w-4 h-4 text-green-400" />
              <span>14 дней бесплатно</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>