<script setup lang="ts">
// Fixed landing page with SSR-safe random values
definePageMeta({
  layout: 'landing'
})

useSeoMeta({
  title: 'Hydraulic Diagnostic SaaS - AI-Powered Industrial Monitoring',
  description: 'Enterprise-grade hydraulic systems monitoring with predictive analytics, real-time diagnostics, and intelligent maintenance scheduling.'
})

// SSR-safe demo data with fixed initial values
const demoMetrics = ref({
  systems: { value: 0, target: 127, label: 'Active Systems' },
  uptime: { value: 0, target: 99.94, label: 'Uptime %' },
  alerts: { value: 0, target: 23, label: 'Prevented Failures' },
  savings: { value: 0, target: 89, label: 'Cost Reduction %' }
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
                AI-Powered
              </span><br>
              Hydraulic Diagnostics
            </h1>
            <p class="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-8 leading-relaxed">
              Predictive maintenance for industrial hydraulic systems. 
              <strong class="text-blue-600 dark:text-blue-400">Reduce downtime by 89%</strong> 
              with intelligent monitoring.
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
              Start Free Trial
            </PremiumButton>
            <PremiumButton 
              to="/investors" 
              variant="secondary" 
              size="lg" 
              icon="heroicons:presentation-chart-line"
              class="text-lg px-8 py-4"
            >
              View Demo
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
                  v-if="key === 'uptime' || key === 'savings'"
                  :class="{ 'transition-all duration-300': isAnimated }"
                >
                  {{ metric.value.toFixed(key === 'systems' || key === 'alerts' ? 0 : 2) }}{{ key === 'uptime' || key === 'savings' ? '%' : '' }}
                </span>
                <span 
                  v-else
                  :class="{ 'transition-all duration-300': isAnimated }"
                >
                  {{ Math.round(metric.value) }}
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
            🚀 Enterprise Features
          </h2>
          <p class="premium-body-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Advanced hydraulic system monitoring with AI-driven insights and predictive maintenance capabilities
          </p>
        </div>
        
        <div class="grid md:grid-cols-3 gap-8">
          <!-- Feature 1 -->
          <div class="premium-card p-8 text-center hover:shadow-xl transition-all duration-300">
            <div class="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mx-auto mb-6">
              <Icon name="heroicons:cpu-chip" class="w-8 h-8 text-white" />
            </div>
            <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-4">
              AI Predictive Analytics
            </h3>
            <p class="premium-body text-gray-600 dark:text-gray-300 mb-4">
              Machine learning algorithms analyze sensor data to predict failures 30 days in advance with 94.8% accuracy
            </p>
            <div class="text-sm text-blue-600 dark:text-blue-400 font-medium">
              ✓ Real-time anomaly detection
            </div>
          </div>
          
          <!-- Feature 2 -->
          <div class="premium-card p-8 text-center hover:shadow-xl transition-all duration-300">
            <div class="w-16 h-16 bg-gradient-to-br from-green-500 to-green-600 rounded-xl flex items-center justify-center mx-auto mb-6">
              <Icon name="heroicons:chart-bar-square" class="w-8 h-8 text-white" />
            </div>
            <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-4">
              Real-Time Monitoring
            </h3>
            <p class="premium-body text-gray-600 dark:text-gray-300 mb-4">
              Continuous monitoring of pressure, temperature, flow rate, and vibration with instant alerts
            </p>
            <div class="text-sm text-green-600 dark:text-green-400 font-medium">
              ✓ 1.2s response time
            </div>
          </div>
          
          <!-- Feature 3 -->
          <div class="premium-card p-8 text-center hover:shadow-xl transition-all duration-300">
            <div class="w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center mx-auto mb-6">
              <Icon name="heroicons:wrench-screwdriver" class="w-8 h-8 text-white" />
            </div>
            <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-4">
              Smart Maintenance
            </h3>
            <p class="premium-body text-gray-600 dark:text-gray-300 mb-4">
              Automated maintenance scheduling based on actual system condition and usage patterns
            </p>
            <div class="text-sm text-purple-600 dark:text-purple-400 font-medium">
              ✓ 89% cost reduction
            </div>
          </div>
        </div>
      </div>
    </section>
    
    <!-- Stats Section -->
    <section class="py-20 bg-gradient-to-r from-blue-600 to-purple-600">
      <div class="container mx-auto px-4">
        <div class="text-center text-white mb-16">
          <h2 class="premium-heading-xl mb-4">Trusted by Industry Leaders</h2>
          <p class="premium-body-lg opacity-90">
            Join 127+ enterprises optimizing their hydraulic systems
          </p>
        </div>
        
        <div class="grid grid-cols-2 md:grid-cols-4 gap-8 text-center text-white">
          <div>
            <div class="text-4xl font-bold mb-2">1,847</div>
            <div class="text-sm opacity-80">Systems Monitored</div>
          </div>
          <div>
            <div class="text-4xl font-bold mb-2">99.94%</div>
            <div class="text-sm opacity-80">Uptime Achieved</div>
          </div>
          <div>
            <div class="text-4xl font-bold mb-2">₽3.2M</div>
            <div class="text-sm opacity-80">Cost Savings</div>
          </div>
          <div>
            <div class="text-4xl font-bold mb-2">24/7</div>
            <div class="text-sm opacity-80">Expert Support</div>
          </div>
        </div>
      </div>
    </section>
    
    <!-- CTA Section -->
    <section class="py-20 bg-gray-50 dark:bg-gray-900">
      <div class="container mx-auto px-4 text-center">
        <div class="max-w-3xl mx-auto">
          <h2 class="premium-heading-xl text-gray-900 dark:text-white mb-6">
            Ready to Transform Your Operations?
          </h2>
          <p class="premium-body-lg text-gray-600 dark:text-gray-300 mb-8">
            Start your journey towards predictive maintenance and operational excellence. 
            Enterprise trial available.
          </p>
          <div class="flex flex-col sm:flex-row gap-4 justify-center">
            <PremiumButton 
              to="/auth/register" 
              size="lg" 
              gradient 
              icon="heroicons:arrow-right"
            >
              Get Started Now
            </PremiumButton>
            <PremiumButton 
              to="/investors" 
              variant="secondary" 
              size="lg" 
              icon="heroicons:phone"
            >
              Schedule Demo
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