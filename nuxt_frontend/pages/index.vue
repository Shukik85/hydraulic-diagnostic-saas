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
  dataPointsDaily: '2.4М+',
  predictiveAccuracy: 94.7,
  downtimePrevented: '18,450ч' // hours saved
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
    stats: '₽2.3М экономии в год'
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
    stats: '847М+ точек данных обучения'
  }
]

// Client testimonials for social proof
const testimonials = [
  {
    quote: 'Сократили затраты на обслуживание на 45% и полностью исключили неожиданные поломки.',
    author: 'Технический директор',
    company: 'Машиностроительный завод',
    avatar: 'Т',
    role: 'Технический директор'
  },
  {
    quote: 'Предиктивная аналитика кардинально изменила подход к управлению гидравлической инфраструктурой.',
    author: 'Начальник производства',
    company: 'Металлургический комбинат',
    avatar: 'П',
    role: 'Начальник производства'
  },
  {
    quote: 'Лучшая инвестиция в нашу инфраструктуру. ROI окупился за 4 месяца.',
    author: 'Коммерческий директор',
    company: 'Нефтехимическое предприятие',
    avatar: 'К',
    role: 'Коммерческий директор'
  }
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
    price: 'от ₽89К',
    period: 'в месяц',
    description: 'Для малых предприятий',
    features: ['До 5 систем', 'Основные отчёты', 'Email поддержка'],
    popular: false
  },
  {
    name: 'Professional', 
    price: 'от ₽269К',
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
    features: ['Неограничено систем', 'On-premise развёртывание', '24/7 выделенная поддержка', 'SLA 99.99%'],
    popular: false
  }
]

const handleRequestDemo = () => {
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
    <section class="relative overflow-hidden premium-hero min-h-screen flex items-center">
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
      
      <div class="premium-container relative py-20 lg:py-32">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <!-- Hero content -->
          <div class="text-center lg:text-left premium-fade-in">
            <!-- Badge -->
            <div class="inline-flex items-center px-4 py-2 rounded-full bg-blue-500/10 border border-blue-400/20 text-blue-300 text-sm font-medium mb-8">
              <Icon name="heroicons:sparkles" class="w-4 h-4 mr-2" />
              AI-Powered Industrial Diagnostics Platform
            </div>
            
            <!-- Main headline -->
            <h1 class="premium-heading-xl text-white mb-6">
              Устраняем неплановые 
              <span class="premium-gradient-text">
                простои оборудования
              </span>
            </h1>
            
            <!-- Value proposition -->
            <p class="premium-body-lg text-blue-100 mb-8 max-w-3xl">
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
              <PremiumButton 
                size="xl"
                gradient
                icon="heroicons:play"
                @click="handleRequestDemo"
              >
                Запросить демо
              </PremiumButton>
              
              <PremiumButton 
                size="xl"
                variant="ghost"
                icon="heroicons:chart-bar"
                @click="handleViewDashboard"
              >
                Открыть платформу
              </PremiumButton>
            </div>
          </div>
          
          <!-- Visual showcase -->
          <div class="hidden lg:block relative premium-scale-in">
            <div class="relative z-10 premium-glass rounded-2xl p-8">
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
                  <KpiCard
                    title="Систем"
                    :value="businessMetrics.systemsMonitored.toLocaleString()"
                    icon="heroicons:server-stack"
                    color="blue"
                    :growth="8.3"
                    loading-state="false"
                  />
                  
                  <KpiCard
                    title="Uptime"
                    :value="`${businessMetrics.uptimePercent}%`"
                    icon="heroicons:chart-bar"
                    color="green"
                    :growth="0.1"
                    loading-state="false"
                  />
                  
                  <KpiCard
                    title="MTTD"
                    :value="`${businessMetrics.averageMTTD} мин`"
                    icon="heroicons:bolt"
                    color="orange"
                    :growth="-12.5"
                    loading-state="false"
                  />
                  
                  <KpiCard
                    title="Экономия"
                    :value="`${businessMetrics.costReduction}%`"
                    icon="heroicons:currency-dollar"
                    color="purple"
                    :growth="3.8"
                    loading-state="false"
                  />
                </div>
                
                <!-- Mock chart area -->
                <div class="premium-glass rounded-lg p-4 h-32 flex items-center justify-center">
                  <div class="flex items-end space-x-2 h-16">
                    <div v-for="i in 12" :key="i" 
                         :class="`bg-blue-400 w-3 rounded-t opacity-${Math.random() > 0.5 ? '70' : '40'}`"
                         :style="`height: ${Math.random() * 60 + 20}%`">
                    </div>
                  </div>
                </div>
              </div>
              
              <!-- Floating indicators -->
              <div class="absolute -top-4 -right-4 premium-badge-success animate-pulse">
                Live Data
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Business Metrics Section -->
    <section class="py-20 premium-glass">
      <div class="premium-container">
        <SectionHeader 
          title="Доверие промышленных лидеров"
          description="Платформа обрабатывает миллионы точек данных ежедневно"
          icon="heroicons:chart-bar-square"
          icon-color="green"
          class="mb-12"
        />
        
        <!-- Enhanced metrics grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-8">
          <div class="text-center group premium-slide-up">
            <div class="text-4xl md:text-5xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors">
              {{ businessMetrics.systemsMonitored.toLocaleString() }}
            </div>
            <div class="text-blue-200 font-medium">Систем мониторинга</div>
            <div class="text-green-300 text-sm mt-1 premium-badge-success">
              <Icon name="heroicons:arrow-trending-up" class="w-4 h-4 inline mr-1" />
              +23% этот год
            </div>
          </div>
          
          <div class="text-center group premium-slide-up animation-delay-200">
            <div class="text-4xl md:text-5xl font-bold text-white mb-2 group-hover:text-green-400 transition-colors">
              {{ businessMetrics.uptimePercent }}%
            </div>
            <div class="text-blue-200 font-medium">Гарантированный Uptime</div>
            <div class="text-green-300 text-sm mt-1 premium-badge-success">
              <Icon name="heroicons:check-circle" class="w-4 h-4 inline mr-1" />
              SLA гарантия
            </div>
          </div>
          
          <div class="text-center group premium-slide-up animation-delay-300">
            <div class="text-4xl md:text-5xl font-bold text-white mb-2 group-hover:text-yellow-400 transition-colors">
              {{ businessMetrics.averageMTTD }}мин
            </div>
            <div class="text-blue-200 font-medium">Среднее время обнаружения</div>
            <div class="text-yellow-300 text-sm mt-1 premium-badge-warning">
              <Icon name="heroicons:bolt" class="w-4 h-4 inline mr-1" />
              Молниеносная реакция
            </div>
          </div>
          
          <div class="text-center group premium-slide-up animation-delay-400">
            <div class="text-4xl md:text-5xl font-bold text-white mb-2 group-hover:text-purple-400 transition-colors">
              {{ businessMetrics.costReduction }}%
            </div>
            <div class="text-blue-200 font-medium">Сокращение затрат</div>
            <div class="text-purple-300 text-sm mt-1 premium-badge-info">
              <Icon name="heroicons:arrow-down" class="w-4 h-4 inline mr-1" />
              До ₽50М в год
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Value Propositions -->
    <section class="py-24 bg-gradient-to-b from-transparent to-white/5">
      <div class="premium-container">
        <SectionHeader 
          title="Почему промышленные лидеры выбирают нашу платформу"
          description="Передовые технологии в сочетании с промышленной надёжностью"
          icon="heroicons:sparkles"
          icon-color="purple"
          class="mb-16"
        />
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div 
            v-for="(prop, index) in valueProps" 
            :key="prop.title"
            class="premium-card-hover p-8"
            :class="`premium-slide-up animation-delay-${index * 100}`"
          >
            <div class="w-16 h-16 bg-gradient-to-br from-blue-400 to-blue-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 group-hover:rotate-3">
              <Icon :name="prop.icon" class="w-8 h-8 text-white" />
            </div>
            <h3 class="premium-heading-md text-white mb-4 group-hover:text-blue-300 transition-colors">{{ prop.title }}</h3>
            <p class="premium-body text-blue-100 mb-4">{{ prop.description }}</p>
            <div class="premium-badge-info">
              <Icon name="heroicons:star" class="w-4 h-4 mr-2" />
              {{ prop.stats }}
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Enterprise Social Proof -->
    <section class="py-20 premium-glass">
      <div class="premium-container">
        <SectionHeader 
          title="Доверие крупнейших предприятий"
          :description="`${businessMetrics.enterpriseClients}+ корпоративных клиентов доверяют нам свои критические процессы`"
          icon="heroicons:building-office"
          icon-color="green"
          class="mb-16"
        />
        
        <!-- Testimonials -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
          <div 
            v-for="testimonial in testimonials"
            :key="testimonial.author"
            class="premium-card p-8 premium-slide-up"
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
            <blockquote class="premium-body text-blue-50 italic">
              «{{ testimonial.quote }}»
            </blockquote>
          </div>
        </div>
        
        <!-- Enterprise indicators -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          <div class="space-y-2 premium-slide-up">
            <Icon name="heroicons:shield-check" class="w-8 h-8 text-green-400 mx-auto" />
            <div class="text-white font-semibold">SOC 2 Type II</div>
            <div class="text-blue-200 text-sm">Сертифицировано</div>
          </div>
          <div class="space-y-2 premium-slide-up animation-delay-100">
            <Icon name="heroicons:lock-closed" class="w-8 h-8 text-blue-400 mx-auto" />
            <div class="text-white font-semibold">256-bit шифрование</div>
            <div class="text-blue-200 text-sm">Корпоративная безопасность</div>
          </div>
          <div class="space-y-2 premium-slide-up animation-delay-200">
            <Icon name="heroicons:globe-alt" class="w-8 h-8 text-purple-400 mx-auto" />
            <div class="text-white font-semibold">Multi-region</div>
            <div class="text-blue-200 text-sm">Глобальная доступность</div>
          </div>
          <div class="space-y-2 premium-slide-up animation-delay-300">
            <Icon name="heroicons:phone" class="w-8 h-8 text-yellow-400 mx-auto" />
            <div class="text-white font-semibold">24/7 Support</div>
            <div class="text-blue-200 text-sm">Экспертная поддержка</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Deep Dive -->
    <section class="py-24">
      <div class="premium-container">
        <SectionHeader 
          title="Полный спектр решений для промышленности"
          icon="heroicons:squares-plus"
          icon-color="indigo"
          class="mb-16"
        />
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div 
            v-for="featureGroup in features"
            :key="featureGroup.category"
            class="premium-card p-8 premium-slide-up"
          >
            <h3 class="premium-heading-md text-white mb-6">{{ featureGroup.category }}</h3>
            <ul class="space-y-4">
              <li 
                v-for="item in featureGroup.items"
                :key="item"
                class="flex items-start space-x-3 premium-body text-blue-100"
              >
                <Icon name="heroicons:check" class="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                <span>{{ item }}</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <!-- Pricing Preview -->
    <section class="py-24 premium-glass">
      <div class="premium-container">
        <SectionHeader 
          title="Прозрачное ценообразование"
          description="Выберите план, который подходит вашему бизнесу. Никаких скрытых комиссий."
          icon="heroicons:banknotes"
          icon-color="green"
          class="mb-16"
        />
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div 
            v-for="plan in pricingPlans"
            :key="plan.name"
            :class="[
              'premium-card p-8 text-center premium-slide-up',
              plan.popular ? 'premium-card-hover scale-105 lg:scale-110 border-blue-400/50' : ''
            ]"
          >
            <div v-if="plan.popular" class="absolute -top-4 left-1/2 transform -translate-x-1/2">
              <span class="premium-badge premium-badge-info">
                Рекомендуемый
              </span>
            </div>
            
            <h3 class="premium-heading-md text-white mb-2">{{ plan.name }}</h3>
            <div class="mb-4">
              <span class="text-4xl font-bold text-white">{{ plan.price }}</span>
              <span v-if="plan.period" class="text-blue-200 ml-2">{{ plan.period }}</span>
            </div>
            <p class="premium-body text-blue-100 mb-8">{{ plan.description }}</p>
            
            <ul class="space-y-3 mb-8 text-left">
              <li 
                v-for="feature in plan.features"
                :key="feature"
                class="flex items-start space-x-3 premium-body text-blue-100"
              >
                <Icon name="heroicons:check" class="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                <span>{{ feature }}</span>
              </li>
            </ul>
            
            <PremiumButton
              full-width
              size="lg"
              :variant="plan.popular ? 'primary' : 'secondary'"
              :gradient="plan.popular"
              @click="plan.name === 'Enterprise' ? handleContactSales() : handleGetStarted()"
            >
              {{ plan.name === 'Enterprise' ? 'Связаться с отделом продаж' : 'Начать работу' }}
            </PremiumButton>
          </div>
        </div>
      </div>
    </section>

    <!-- Final CTA Section -->
    <section class="py-32 bg-gradient-to-t from-slate-900 to-transparent">
      <div class="max-w-5xl mx-auto text-center premium-container">
        <div class="premium-hero p-12">
          <h2 class="premium-heading-xl text-white mb-6">
            Готовы трансформировать ваше производство?
          </h2>
          <p class="premium-body-lg text-blue-100 mb-10 max-w-3xl mx-auto">
            Присоединяйтесь к промышленным лидерам, которые уже революционизировали 
            управление гидравлическими системами
          </p>
          
          <!-- Enhanced CTAs -->
          <div class="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-8">
            <PremiumButton 
              size="xl"
              gradient
              icon="heroicons:calendar-days"
              @click="handleRequestDemo"
            >
              Запланировать демо
            </PremiumButton>
            
            <PremiumButton 
              size="xl"
              variant="ghost"
              icon="heroicons:arrow-right"
              @click="handleViewDashboard"
            >
              Исследовать платформу
            </PremiumButton>
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

.animation-delay-400 {
  animation-delay: 400ms;
}

.animation-delay-500 {
  animation-delay: 500ms;
}

.animation-delay-1000 {
  animation-delay: 1000ms;
}
</style>