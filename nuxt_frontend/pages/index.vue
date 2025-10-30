<template>
  <div class="min-h-screen bg-white dark:bg-gray-900">
    <!-- Navigation -->
    <nav class="sticky top-0 z-50 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
      <div class="u-container u-flex-between py-4">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-600 to-blue-400 flex items-center justify-center">
            <Icon name="heroicons:chart-bar-square" class="w-6 h-6 text-white" />
          </div>
          <span class="text-lg font-bold text-gray-900 dark:text-white">Hydraulic Diagnostic</span>
        </div>
        <div class="flex items-center gap-4">
          <NuxtLink 
            to="/auth/login" 
            class="u-btn u-btn-ghost u-btn-md"
          >
            <Icon name="heroicons:arrow-right-on-rectangle" class="w-4 h-4 mr-1" />
            Sign In
          </NuxtLink>
          <NuxtLink 
            to="/auth/register" 
            class="u-btn u-btn-primary u-btn-md"
          >
            <Icon name="heroicons:plus" class="w-4 h-4 mr-1" />
            Get Started
          </NuxtLink>
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <section class="py-20 bg-gradient-to-b from-blue-50 dark:from-gray-800 to-white dark:to-gray-900">
      <div class="u-container">
        <div class="max-w-3xl mx-auto text-center">
          <h1 class="u-h1 mb-6">
            Advanced Hydraulic Systems Diagnostics
          </h1>
          <p class="u-body-lg text-gray-600 dark:text-gray-300 mb-8">
            Real-time monitoring, AI-powered analysis, and predictive maintenance for hydraulic systems. Keep your operations running at peak efficiency.
          </p>
          <div class="flex items-center justify-center gap-4">
            <NuxtLink 
              to="/auth/register"
              class="u-btn u-btn-primary u-btn-lg"
            >
              <Icon name="heroicons:rocket-launch" class="w-5 h-5 mr-2" />
              Start Free Trial
            </NuxtLink>
            <button class="u-btn u-btn-secondary u-btn-lg">
              <Icon name="heroicons:play" class="w-5 h-5 mr-2" />
              Watch Demo
            </button>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="u-section">
      <div class="u-container">
        <div class="text-center mb-12">
          <h2 class="u-h2 mb-4">Powerful Features</h2>
          <p class="u-body-lg text-gray-600 dark:text-gray-300">Everything you need to monitor and optimize hydraulic systems</p>
        </div>

        <div class="u-grid-responsive">
          <!-- Feature 1 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                <Icon name="heroicons:chart-line" class="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 class="u-h5">Real-Time Monitoring</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              Live data streaming from all connected systems with instant alerts for anomalies.
            </p>
          </div>

          <!-- Feature 2 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                <Icon name="heroicons:sparkles" class="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <h3 class="u-h5">AI-Powered Analytics</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              Machine learning algorithms detect patterns and predict maintenance needs before failures occur.
            </p>
          </div>

          <!-- Feature 3 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                <Icon name="heroicons:bell-alert" class="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 class="u-h5">Smart Alerts</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              Customizable notifications via email, SMS, or in-app alerts for critical system events.
            </p>
          </div>

          <!-- Feature 4 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-yellow-100 dark:bg-yellow-900/30 flex items-center justify-center">
                <Icon name="heroicons:document-chart-bar" class="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
              </div>
              <h3 class="u-h5">Detailed Reports</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              Comprehensive performance reports with historical trends and recommendations.
            </p>
          </div>

          <!-- Feature 5 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
                <Icon name="heroicons:lock-closed" class="w-6 h-6 text-red-600 dark:text-red-400" />
              </div>
              <h3 class="u-h5">Enterprise Security</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              Bank-level encryption and compliance with industrial security standards.
            </p>
          </div>

          <!-- Feature 6 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
                <Icon name="heroicons:cube-transparent" class="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
              </div>
              <h3 class="u-h5">API Integration</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              RESTful API and webhooks for seamless integration with your existing systems.
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- Stats Section -->
    <section class="u-section bg-gray-50 dark:bg-gray-800">
      <div class="u-container">
        <div class="grid grid-cols-3 gap-8">
          <div class="text-center">
            <div class="u-h2 text-blue-600 mb-2">99.9%</div>
            <p class="u-body text-gray-600 dark:text-gray-400">System Uptime</p>
          </div>
          <div class="text-center">
            <div class="u-h2 text-green-600 mb-2">50K+</div>
            <p class="u-body text-gray-600 dark:text-gray-400">Active Users</p>
          </div>
          <div class="text-center">
            <div class="u-h2 text-blue-600 mb-2">24/7</div>
            <p class="u-body text-gray-600 dark:text-gray-400">Support</p>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="u-section">
      <div class="u-container">
        <div class="u-card u-gradient-primary p-12 text-center text-white">
          <h2 class="text-3xl font-bold mb-4">Ready to optimize your hydraulic systems?</h2>
          <p class="text-lg mb-8 opacity-90">Join hundreds of companies using Hydraulic Diagnostic to improve efficiency and reduce downtime.</p>
          <NuxtLink 
            to="/auth/register"
            class="u-btn u-btn-lg bg-white hover:bg-gray-100 text-blue-600 font-medium"
          >
            Start Your Free Trial Now
          </NuxtLink>
        </div>
      </div>
    </section>

    <!-- Footer -->
    <footer class="bg-gray-900 text-gray-400 py-12">
      <div class="u-container">
        <div class="grid grid-cols-4 gap-8 mb-8">
          <div>
            <h4 class="font-semibold text-white mb-4">Product</h4>
            <ul class="space-y-2">
              <li><a href="#" class="hover:text-white transition">Features</a></li>
              <li><a href="#" class="hover:text-white transition">Pricing</a></li>
              <li><a href="#" class="hover:text-white transition">Documentation</a></li>
            </ul>
          </div>
          <div>
            <h4 class="font-semibold text-white mb-4">Company</h4>
            <ul class="space-y-2">
              <li><a href="#" class="hover:text-white transition">About</a></li>
              <li><a href="#" class="hover:text-white transition">Blog</a></li>
              <li><a href="#" class="hover:text-white transition">Careers</a></li>
            </ul>
          </div>
          <div>
            <h4 class="font-semibold text-white mb-4">Legal</h4>
            <ul class="space-y-2">
              <li><a href="#" class="hover:text-white transition">Privacy</a></li>
              <li><a href="#" class="hover:text-white transition">Terms</a></li>
              <li><a href="#" class="hover:text-white transition">Security</a></li>
            </ul>
          </div>
          <div>
            <h4 class="font-semibold text-white mb-4">Contact</h4>
            <ul class="space-y-2">
              <li>Email: contact@hydraulic-diagnostic.com</li>
              <li>Phone: +1 (555) 123-4567</li>
              <li>Chat: Available 24/7</li>
            </ul>
          </div>
        </div>
        <div class="border-t border-gray-800 pt-8 text-center">
          <p>&copy; 2025 Hydraulic Diagnostic. All rights reserved.</p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  layout: 'blank',
  middleware: []
})
</script>