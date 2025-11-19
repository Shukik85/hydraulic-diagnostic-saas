<script setup lang="ts">
import { ref } from 'vue'

// Professional landing page for investors and prospects
definePageMeta({
  layout: 'landing',
});

useSeoMeta({
  title: 'Hydraulic Diagnostic SaaS - Industrial-Grade System Monitoring',
  description:
    'Professional hydraulic system diagnostics platform with AI-powered insights, predictive maintenance, and real-time monitoring. Reduce downtime by 60% and maintenance costs by 40%.',
  ogTitle: 'Hydraulic Diagnostic SaaS - Revolutionary Industrial Monitoring',
  ogDescription:
    'AI-powered hydraulic diagnostics platform trusted by industrial leaders. Real-time monitoring, predictive insights, 99.9% uptime guarantee.',
  ogImage: '/og-image.jpg',
  twitterCard: 'summary_large_image',
});

const stats = ref({
  systemsMonitored: 2847,
  uptimePercent: 99.94,
  averageMTTD: 2.3, // minutes
  costSavings: 42, // percent
  clientsServed: 127,
  dataPointsProcessed: '847M+',
});

const benefits = [
  {
    icon: 'heroicons:shield-check',
    title: 'Predictive Maintenance',
    description:
      'AI algorithms predict failures 2-4 weeks in advance, reducing emergency downtime by 85%',
  },
  {
    icon: 'heroicons:chart-bar',
    title: 'Real-time Monitoring',
    description:
      'Continuous sensor data analysis with millisecond precision and instant anomaly detection',
  },
  {
    icon: 'heroicons:cog-6-tooth',
    title: 'Automated Diagnostics',
    description:
      'Smart diagnostic workflows that identify root causes and recommend optimal solutions',
  },
  {
    icon: 'heroicons:currency-dollar',
    title: 'Cost Optimization',
    description:
      'Average 40% reduction in maintenance costs through optimized scheduling and parts management',
  },
  {
    icon: 'heroicons:device-phone-mobile',
    title: 'Mobile & Cloud Ready',
    description:
      'Access from anywhere with enterprise security, mobile apps, and cloud-native architecture',
  },
  {
    icon: 'heroicons:academic-cap',
    title: 'AI-Powered Insights',
    description:
      'Natural language queries to your system data with contextual recommendations and reports',
  },
];

const testimonials = [
  {
    quote: 'Reduced our maintenance costs by 45% and eliminated unexpected breakdowns entirely.',
    author: 'Technical Director',
    company: 'Major Industrial Facility',
    avatar: 'T',
  },
  {
    quote: 'The predictive analytics have transformed how we manage our hydraulic infrastructure.',
    author: 'Operations Manager',
    company: 'Manufacturing Plant',
    avatar: 'O',
  },
];

const handleRequestDemo = () => {
  // Navigate to contact or demo booking
  navigateTo('/demo/request');
};

const handleViewDashboard = () => {
  navigateTo('/dashboard');
};
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
    <!-- Hero Section -->
    <section class="relative overflow-hidden">
      <!-- Background decoration -->
      <div class="absolute inset-0">
        <div class="absolute inset-0 bg-gradient-to-br from-blue-600/20 via-transparent to-purple-600/20"></div>
        <div
          class="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-blue-400/50 to-transparent">
        </div>
      </div>

      <div class="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 lg:py-32">
        <div class="text-center">
          <!-- Logo -->
          <div class="flex justify-center mb-8">
            <div class="flex items-center space-x-3">
              <div
                class="w-12 h-12 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center">
                <Icon name="heroicons:wrench-screwdriver" class="w-7 h-7 text-white" />
              </div>
              <span class="text-2xl font-bold text-white">Hydraulic SaaS</span>
            </div>
          </div>

          <!-- Main headline -->
          <h1 class="text-4xl md:text-6xl lg:text-7xl font-bold text-white leading-tight mb-6">
            Industrial-Grade
            <span class="bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
              Hydraulic Diagnostics
            </span>
          </h1>

          <!-- Subheadline -->
          <p class="text-xl md:text-2xl text-blue-100 max-w-4xl mx-auto mb-8 leading-relaxed">
            AI-powered predictive maintenance platform that eliminates unexpected breakdowns and
            reduces maintenance costs by <strong class="text-white">40%+</strong>
          </p>

          <!-- Trust indicators -->
          <div class="flex items-center justify-center space-x-8 mb-12 text-blue-200 text-sm">
            <div class="flex items-center space-x-2">
              <Icon name="heroicons:shield-check" class="w-5 h-5" />
              <span>Enterprise Security</span>
            </div>
            <div class="flex items-center space-x-2">
              <Icon name="heroicons:clock" class="w-5 h-5" />
              <span>99.9% Uptime SLA</span>
            </div>
            <div class="flex items-center space-x-2">
              <Icon name="heroicons:users" class="w-5 h-5" />
              <span>24/7 Expert Support</span>
            </div>
          </div>

          <!-- CTA Buttons -->
          <div class="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6">
            <button @click="handleRequestDemo"
              class="group relative px-8 py-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white font-semibold rounded-lg shadow-lg hover:from-blue-600 hover:to-blue-700 transform hover:scale-105 transition-all duration-200 hover:shadow-blue-500/25">
              <span class="relative z-10 flex items-center">
                <Icon name="heroicons:play" class="w-5 h-5 mr-2" />
                Request Live Demo
              </span>
              <div
                class="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              </div>
            </button>

            <button @click="handleViewDashboard"
              class="px-8 py-4 bg-white/10 backdrop-blur-sm text-white font-semibold rounded-lg border border-white/20 hover:bg-white/20 transition-all duration-200">
              <Icon name="heroicons:chart-bar" class="w-5 h-5 mr-2 inline" />
              View Live Dashboard
            </button>
          </div>
        </div>
      </div>
    </section>

    <!-- Stats Section -->
    <section class="py-16 bg-white/5 backdrop-blur-sm">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8">
          <div class="text-center">
            <div class="text-3xl md:text-4xl font-bold text-white mb-2">
              {{ stats.systemsMonitored.toLocaleString() }}
            </div>
            <div class="text-blue-200 text-sm font-medium">Systems Monitored</div>
          </div>
          <div class="text-center">
            <div class="text-3xl md:text-4xl font-bold text-white mb-2">
              {{ stats.uptimePercent }}%
            </div>
            <div class="text-blue-200 text-sm font-medium">Platform Uptime</div>
          </div>
          <div class="text-center">
            <div class="text-3xl md:text-4xl font-bold text-white mb-2">
              {{ stats.averageMTTD }}min
            </div>
            <div class="text-blue-200 text-sm font-medium">Avg Detection Time</div>
          </div>
          <div class="text-center">
            <div class="text-3xl md:text-4xl font-bold text-white mb-2">
              {{ stats.costSavings }}%
            </div>
            <div class="text-blue-200 text-sm font-medium">Cost Reduction</div>
          </div>
          <div class="text-center">
            <div class="text-3xl md:text-4xl font-bold text-white mb-2">
              {{ stats.clientsServed }}+
            </div>
            <div class="text-blue-200 text-sm font-medium">Enterprise Clients</div>
          </div>
          <div class="text-center">
            <div class="text-3xl md:text-4xl font-bold text-white mb-2">
              {{ stats.dataPointsProcessed }}
            </div>
            <div class="text-blue-200 text-sm font-medium">Data Points</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Benefits Section -->
    <section class="py-24 bg-gradient-to-b from-transparent to-white/5">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-16">
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">
            Why Industry Leaders Choose Our Platform
          </h2>
          <p class="text-xl text-blue-100 max-w-3xl mx-auto">
            Cutting-edge technology meets industrial reliability in a platform designed for
            mission-critical operations
          </p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div v-for="benefit in benefits" :key="benefit.title"
            class="group bg-white/5 backdrop-blur-sm rounded-xl p-6 hover:bg-white/10 transition-all duration-300 border border-white/10 hover:border-blue-400/30">
            <div
              class="w-12 h-12 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
              <Icon :name="benefit.icon" class="w-6 h-6 text-white" />
            </div>
            <h3 class="text-xl font-semibold text-white mb-3">{{ benefit.title }}</h3>
            <p class="text-blue-100 leading-relaxed">{{ benefit.description }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- Social Proof Section -->
    <section class="py-24 bg-white/5 backdrop-blur-sm">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-16">
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">
            Trusted by Industry Leaders
          </h2>
        </div>

        <!-- Testimonials -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          <div v-for="testimonial in testimonials" :key="testimonial.author"
            class="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/10">
            <blockquote class="text-blue-50 text-lg mb-6 leading-relaxed">
              "{{ testimonial.quote }}"
            </blockquote>
            <div class="flex items-center">
              <div
                class="w-12 h-12 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center text-white font-bold mr-4">
                {{ testimonial.avatar }}
              </div>
              <div>
                <div class="text-white font-semibold">{{ testimonial.author }}</div>
                <div class="text-blue-200 text-sm">{{ testimonial.company }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Final CTA Section -->
    <section class="py-24">
      <div class="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
        <h2 class="text-3xl md:text-4xl font-bold text-white mb-6">
          Ready to Transform Your Operations?
        </h2>
        <p class="text-xl text-blue-100 mb-8">
          Join industry leaders who've already revolutionized their hydraulic system management
        </p>

        <div class="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6">
          <button @click="handleRequestDemo"
            class="group px-10 py-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white font-bold text-lg rounded-lg shadow-xl hover:from-blue-600 hover:to-blue-700 transform hover:scale-105 transition-all duration-200">
            <span class="flex items-center">
              <Icon name="heroicons:calendar" class="w-6 h-6 mr-3" />
              Schedule Your Demo
            </span>
          </button>

          <button @click="handleViewDashboard"
            class="px-10 py-4 bg-white/10 backdrop-blur-sm text-white font-bold text-lg rounded-lg border border-white/30 hover:bg-white/20 transition-all duration-200">
            <Icon name="heroicons:arrow-right" class="w-6 h-6 mr-3 inline" />
            Explore Platform
          </button>
        </div>

        <div class="mt-8 flex items-center justify-center space-x-6 text-blue-200 text-sm">
          <div class="flex items-center space-x-2">
            <Icon name="heroicons:check" class="w-4 h-4 text-green-400" />
            <span>No credit card required</span>
          </div>
          <div class="flex items-center space-x-2">
            <Icon name="heroicons:check" class="w-4 h-4 text-green-400" />
            <span>Setup in under 30 minutes</span>
          </div>
          <div class="flex items-center space-x-2">
            <Icon name="heroicons:check" class="w-4 h-4 text-green-400" />
            <span>Enterprise support included</span>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
/* Custom animations and effects */
@keyframes float {

  0%,
  100% {
    transform: translateY(0px);
  }

  50% {
    transform: translateY(-10px);
  }
}

.animate-float {
  animation: float 6s ease-in-out infinite;
}

/* Gradient text animation */
@keyframes gradient {

  0%,
  100% {
    background-position: 0% 50%;
  }

  50% {
    background-position: 100% 50%;
  }
}

.animate-gradient {
  background-size: 200% 200%;
  animation: gradient 3s ease infinite;
}
</style>
