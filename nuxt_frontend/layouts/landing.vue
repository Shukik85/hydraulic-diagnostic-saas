<script setup lang="ts">
// Landing page layout - clean and focused
const route = useRoute()

// Navigation for landing
const navigation = [
  { name: 'Platform', href: '/dashboard' },
  { name: 'Features', href: '#features' },
  { name: 'Pricing', href: '#pricing' },
  { name: 'Contact', href: '#contact' }
]

const mobileMenuOpen = ref(false)

const handleSignIn = () => {
  navigateTo('/auth/login')
}

const handleRequestDemo = () => {
  navigateTo('/demo/request')
}
</script>

<template>
  <div class="min-h-screen">
    <!-- Navigation -->
    <nav class="absolute top-0 left-0 right-0 z-50">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex items-center justify-between py-6">
          <!-- Logo -->
          <div class="flex items-center space-x-3">
            <div class="w-10 h-10 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center">
              <Icon name="heroicons:wrench-screwdriver" class="w-6 h-6 text-white" />
            </div>
            <span class="text-xl font-bold text-white">Hydraulic SaaS</span>
          </div>
          
          <!-- Desktop Navigation -->
          <div class="hidden md:flex items-center space-x-8">
            <a 
              v-for="item in navigation" 
              :key="item.name"
              :href="item.href"
              class="text-blue-100 hover:text-white font-medium transition-colors duration-200"
            >
              {{ item.name }}
            </a>
          </div>
          
          <!-- CTA Buttons -->
          <div class="hidden md:flex items-center space-x-4">
            <button 
              @click="handleSignIn"
              class="text-blue-100 hover:text-white font-medium transition-colors duration-200"
            >
              Sign In
            </button>
            <button 
              @click="handleRequestDemo"
              class="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-all duration-200 hover:scale-105"
            >
              Request Demo
            </button>
          </div>
          
          <!-- Mobile menu button -->
          <button 
            @click="mobileMenuOpen = !mobileMenuOpen"
            class="md:hidden p-2 rounded-lg text-blue-100 hover:text-white hover:bg-white/10 transition-colors"
          >
            <Icon :name="mobileMenuOpen ? 'heroicons:x-mark' : 'heroicons:bars-3'" class="w-6 h-6" />
          </button>
        </div>
        
        <!-- Mobile Navigation -->
        <div 
          v-if="mobileMenuOpen"
          class="md:hidden bg-slate-800/95 backdrop-blur-sm rounded-lg mt-2 p-4 border border-white/10"
        >
          <div class="space-y-3">
            <a 
              v-for="item in navigation" 
              :key="item.name"
              :href="item.href"
              @click="mobileMenuOpen = false"
              class="block text-blue-100 hover:text-white font-medium py-2"
            >
              {{ item.name }}
            </a>
            <hr class="border-white/10 my-4">
            <button 
              @click="handleSignIn; mobileMenuOpen = false"
              class="block w-full text-left text-blue-100 hover:text-white font-medium py-2"
            >
              Sign In
            </button>
            <button 
              @click="handleRequestDemo; mobileMenuOpen = false"
              class="block w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-all duration-200"
            >
              Request Demo
            </button>
          </div>
        </div>
      </div>
    </nav>

    <!-- Main content -->
    <main>
      <slot />
    </main>

    <!-- Footer -->
    <footer class="bg-slate-900/90 backdrop-blur-sm border-t border-white/10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-8">
          <!-- Company info -->
          <div class="col-span-1 md:col-span-2">
            <div class="flex items-center space-x-3 mb-4">
              <div class="w-8 h-8 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center">
                <Icon name="heroicons:wrench-screwdriver" class="w-5 h-5 text-white" />
              </div>
              <span class="text-xl font-bold text-white">Hydraulic SaaS</span>
            </div>
            <p class="text-blue-100 max-w-md leading-relaxed">
              Industrial-grade hydraulic system diagnostics with AI-powered predictive maintenance. 
              Trusted by enterprises worldwide.
            </p>
            <div class="flex items-center space-x-4 mt-6">
              <div class="flex items-center space-x-2 text-blue-200 text-sm">
                <Icon name="heroicons:shield-check" class="w-4 h-4 text-green-400" />
                <span>SOC 2 Compliant</span>
              </div>
              <div class="flex items-center space-x-2 text-blue-200 text-sm">
                <Icon name="heroicons:lock-closed" class="w-4 h-4 text-green-400" />
                <span>Enterprise Security</span>
              </div>
            </div>
          </div>
          
          <!-- Quick Links -->
          <div>
            <h3 class="text-white font-semibold mb-4">Platform</h3>
            <ul class="space-y-2">
              <li><a href="/dashboard" class="text-blue-100 hover:text-white transition-colors">Dashboard</a></li>
              <li><a href="/diagnostics" class="text-blue-100 hover:text-white transition-colors">Diagnostics</a></li>
              <li><a href="/reports" class="text-blue-100 hover:text-white transition-colors">Reports</a></li>
              <li><a href="/chat" class="text-blue-100 hover:text-white transition-colors">AI Assistant</a></li>
            </ul>
          </div>
          
          <!-- Contact -->
          <div>
            <h3 class="text-white font-semibold mb-4">Contact</h3>
            <ul class="space-y-2">
              <li><a href="#" class="text-blue-100 hover:text-white transition-colors">Support</a></li>
              <li><a href="#" class="text-blue-100 hover:text-white transition-colors">Documentation</a></li>
              <li><a href="#" class="text-blue-100 hover:text-white transition-colors">API Reference</a></li>
              <li><a href="#" class="text-blue-100 hover:text-white transition-colors">Status Page</a></li>
            </ul>
          </div>
        </div>
        
        <!-- Bottom bar -->
        <div class="border-t border-white/10 mt-12 pt-8 flex flex-col sm:flex-row items-center justify-between">
          <p class="text-blue-200 text-sm">
            © 2025 Hydraulic Diagnostic SaaS. All rights reserved.
          </p>
          <div class="flex items-center space-x-6 mt-4 sm:mt-0">
            <a href="#" class="text-blue-200 hover:text-white text-sm transition-colors">Privacy Policy</a>
            <a href="#" class="text-blue-200 hover:text-white text-sm transition-colors">Terms of Service</a>
            <a href="#" class="text-blue-200 hover:text-white text-sm transition-colors">Security</a>
          </div>
        </div>
      </div>
    </footer>
  </div>
</template>