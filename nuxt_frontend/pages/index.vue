<script setup lang="ts">
// Russian-localized landing page with improved UX
definePageMeta({
  layout: 'landing'
})

useSeoMeta({
  title: 'Гидравлика ИИ - Платформа мониторинга промышленного оборудования',
  description: 'Enterprise-платформа мониторинга гидравлических систем с ИИ-аналитикой, предиктивным обслуживанием и интеллектуальным планированием ремонтов.'
})

// SSR-safe demo data with fixed initial values
const demoMetrics = ref({
  systems: { value: 0, target: 127, label: 'Активных систем', suffix: '' },
  uptime: { value: 0, target: 99.94, label: 'Время работы', suffix: '%' },
  alerts: { value: 0, target: 23, label: 'Предотвращено аварий', suffix: '' },
  savings: { value: 0, target: 89, label: 'Снижение затрат', suffix: '%' }
})

const isAnimated = ref(false)

// Client-side animation after mount (prevents hydration mismatch)
onMounted(() => {
  nextTick(() => {
    isAnimated.value = true
    animateCounters()
  })
})

const animateCounters = () => {
  Object.keys(demoMetrics.value).forEach((key, index) => {
    setTimeout(() => {
      const metric = demoMetrics.value[key as keyof typeof demoMetrics.value]
      const duration = 2000
      const steps = 60
      const increment = metric.target / steps
      
      let current = 0
      const timer = setInterval(() => {
        current += increment
        if (current >= metric.target) {
          current = metric.target
          clearInterval(timer)
        }
        metric.value = Math.round(current * 100) / 100
      }, duration / steps)
    }, index * 200)
  })
}
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-900 dark:via-blue-900 dark:to-indigo-900">
    <!-- Hero Section -->
    <section class="relative overflow-hidden">
      <!-- Background Pattern -->
      <div class="absolute inset-0 opacity-10">
        <div class="absolute top-0 left-0 w-72 h-72 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
        <div class="absolute top-0 right-0 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-1000"></div>
        <div class="absolute bottom-0 left-1/2 w-72 h-72 bg-indigo-500 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-500"></div>
      </div>
      
      <div class="container mx-auto px-4 py-20 relative z-10">
        <div class="text-center max-w-4xl mx-auto">
          <!-- Main Headline -->
          <div class="mb-8 premium-fade-in">
            <h1 class="text-4xl md:text-6xl font-bold text-gray-900 dark:text-white mb-6">
              <span class="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                ИИ-Платформа
              </span><br>
              Гидравлической Диагностики
            </h1>
            <p class="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-8 leading-relaxed">
              Предиктивное обслуживание промышленных гидравлических систем. 
              <strong class="text-blue-600 dark:text-blue-400">Снижение простоев на 89%</strong> 
              с помощью интеллектуального мониторинга.
            </p>
          </div>
          
          <!-- CTA Buttons -->
          <div class="flex flex-col sm:flex-row gap-4 justify-center mb-16 premium-slide-up">
            <PremiumButton 
              to="/auth/register" 
              size="lg" 
              gradient 
              icon="heroicons:rocket-launch"
              class="text-lg px-8 py-4"
            >
              Начать бесплатно
            </PremiumButton>
            <PremiumButton 
              to="/investors" 
              variant="secondary" 
              size="lg" 
              icon="heroicons:presentation-chart-line"
              class="text-lg px-8 py-4"
            >
              Смотреть демо
            </PremiumButton>
          </div>
          
          <!-- Live Metrics Dashboard -->
          <div class="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
            <div 
              v-for="(metric, key) in demoMetrics" 
              :key="key"
              class="premium-card p-6 text-center premium-scale-in"
            >
              <div class="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                <span 
                  :class="{ 'transition-all duration-300': isAnimated }"
                >
                  {{ 
                    key === 'uptime' || key === 'savings' 
                      ? metric.value.toFixed(2) 
                      : Math.round(metric.value) 
                  }}{{ metric.suffix }}
                </span>
              </div>
              <div class="text-sm text-gray-600 dark:text-gray-300">{{ metric.label }}</div>
            </div>
          </div>
        </div>
      </div>
    </section>
    
    <!-- Features Section -->
    <section class="py-20 bg-white dark:bg-gray-800">
      <div class="container mx-auto px-4">
        <div class="text-center mb-16">
          <h2 class="premium-heading-xl text-gray-900 dark:text-white mb-4">
            🚀 Возможности платформы
          </h2>
          <p class="premium-body-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Передовой мониторинг гидравлических систем с ИИ-аналитикой и предиктивным обслуживанием
          </p>
        </div>
        
        <div class="grid md:grid-cols-3 gap-8">
          <!-- Feature 1 -->
          <div class="premium-card p-8 text-center hover:shadow-xl transition-all duration-300">
            <div class="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mx-auto mb-6">
              <Icon name="heroicons:cpu-chip" class="w-8 h-8 text-white" />
            </div>
            <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-4">
              ИИ Предиктивная Аналитика
            </h3>
            <p class="premium-body text-gray-600 dark:text-gray-300 mb-4">
              Алгоритмы машинного обучения анализируют данные сенсоров для прогнозирования поломок за 30 дней с точностью 94.8%
            </p>
            <div class="text-sm text-blue-600 dark:text-blue-400 font-medium">
              ✓ Обнаружение аномалий в реальном времени
            </div>
          </div>
          
          <!-- Feature 2 -->
          <div class="premium-card p-8 text-center hover:shadow-xl transition-all duration-300">
            <div class="w-16 h-16 bg-gradient-to-br from-green-500 to-green-600 rounded-xl flex items-center justify-center mx-auto mb-6">
              <Icon name="heroicons:chart-bar-square" class="w-8 h-8 text-white" />
            </div>
            <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-4">
              Мониторинг в реальном времени
            </h3>
            <p class="premium-body text-gray-600 dark:text-gray-300 mb-4">
              Непрерывный контроль давления, температуры, расхода и вибрации с мгновенными уведомлениями
            </p>
            <div class="text-sm text-green-600 dark:text-green-400 font-medium">
              ✓ Время отклика 1.2 секунды
            </div>
          </div>
          
          <!-- Feature 3 -->
          <div class="premium-card p-8 text-center hover:shadow-xl transition-all duration-300">
            <div class="w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center mx-auto mb-6">
              <Icon name="heroicons:wrench-screwdriver" class="w-8 h-8 text-white" />
            </div>
            <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-4">
              Умное обслуживание
            </h3>
            <p class="premium-body text-gray-600 dark:text-gray-300 mb-4">
              Автоматическое планирование технического обслуживания на основе фактического состояния и режима эксплуатации
            </p>
            <div class="text-sm text-purple-600 dark:text-purple-400 font-medium">
              ✓ Снижение затрат на 89%
            </div>
          </div>
        </div>
      </div>
    </section>
    
    <!-- Stats Section -->
    <section class="py-20 bg-gradient-to-r from-blue-600 to-purple-600">
      <div class="container mx-auto px-4">
        <div class="text-center text-white mb-16">
          <h2 class="premium-heading-xl mb-4">Нам доверяют лидеры отрасли</h2>
          <p class="premium-body-lg opacity-90">
            Присоединяйтесь к 127+ предприятиям, оптимизирующим свои гидравлические системы
          </p>
        </div>
        
        <div class="grid grid-cols-2 md:grid-cols-4 gap-8 text-center text-white">
          <div>
            <div class="text-4xl font-bold mb-2">1,847</div>
            <div class="text-sm opacity-80">Систем под мониторингом</div>
          </div>
          <div>
            <div class="text-4xl font-bold mb-2">99.94%</div>
            <div class="text-sm opacity-80">Достигнутое время работы</div>
          </div>
          <div>
            <div class="text-4xl font-bold mb-2">₽3.2М</div>
            <div class="text-sm opacity-80">Экономия средств</div>
          </div>
          <div>
            <div class="text-4xl font-bold mb-2">24/7</div>
            <div class="text-sm opacity-80">Экспертная поддержка</div>
          </div>
        </div>
      </div>
    </section>
    
    <!-- CTA Section -->
    <section class="py-20 bg-gray-50 dark:bg-gray-900">
      <div class="container mx-auto px-4 text-center">
        <div class="max-w-3xl mx-auto">
          <h2 class="premium-heading-xl text-gray-900 dark:text-white mb-6">
            Готовы трансформировать свои операции?
          </h2>
          <p class="premium-body-lg text-gray-600 dark:text-gray-300 mb-8">
            Начните путь к предиктивному обслуживанию и операционному совершенству. 
            Доступна корпоративная пробная версия.
          </p>
          <div class="flex flex-col sm:flex-row gap-4 justify-center">
            <PremiumButton 
              to="/auth/register" 
              size="lg" 
              gradient 
              icon="heroicons:arrow-right"
            >
              Начать сейчас
            </PremiumButton>
            <PremiumButton 
              to="/investors" 
              variant="secondary" 
              size="lg" 
              icon="heroicons:phone"
            >
              Запланировать демо
            </PremiumButton>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.animation-delay-500 {
  animation-delay: 500ms;
}

.animation-delay-1000 {
  animation-delay: 1000ms;
}
</style>