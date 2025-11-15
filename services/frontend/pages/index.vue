<template>
  <div class="min-h-screen bg-metal-dark">
    <!-- Navigation with Metallic theme -->
    <nav class="sticky top-0 z-50 backdrop-blur-sm" style="background: linear-gradient(to bottom, rgba(26, 29, 35, 0.95), rgba(26, 29, 35, 0.85));">
      <div class="container mx-auto px-4 py-4">
        <div class="flex items-center justify-between">
          <!-- Logo -->
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg gradient-primary flex items-center justify-center shadow-lg">
              <Icon name="heroicons:chart-bar-square" class="w-6 h-6 text-white" />
            </div>
            <span class="text-lg font-bold text-white">{{ t('app.title') }}</span>
          </div>

          <!-- Right actions -->
          <div class="flex items-center gap-3">
            <!-- Language selector -->
            <div class="relative">
              <button 
                @click="showLanguageDropdown = !showLanguageDropdown"
                class="px-3 py-2 rounded-lg text-steel-shine hover:text-white transition-colors flex items-center gap-2 hover:bg-white/5"
                :aria-label="t('ui.language.switch')"
              >
                <Icon name="heroicons:language" class="w-5 h-5" />
                <span class="text-sm font-medium uppercase">{{ currentLocale?.code }}</span>
                <Icon name="heroicons:chevron-down" class="w-4 h-4 transition-transform" :class="{ 'rotate-180': showLanguageDropdown }" />
              </button>

              <!-- Dropdown -->
              <Transition
                enter-active-class="transition ease-out duration-200"
                enter-from-class="opacity-0 scale-95"
                enter-to-class="opacity-100 scale-100"
                leave-active-class="transition ease-in duration-150"
                leave-from-class="opacity-100 scale-100"
                leave-to-class="opacity-0 scale-95"
              >
                <div v-show="showLanguageDropdown" class="absolute right-0 mt-2 w-40 card-metal py-1 shadow-lg">
                  <button
                    v-for="lang in availableLocales"
                    :key="lang.code"
                    @click="switchLanguage(lang.code)"
                    class="w-full px-4 py-2 text-left text-sm transition-colors flex items-center justify-between"
                    :class="currentLocale?.code === lang.code 
                      ? 'bg-primary-600/20 text-primary-400' 
                      : 'text-steel-shine hover:bg-white/5 hover:text-white'"
                  >
                    <span>{{ lang.name }}</span>
                    <span class="text-xs opacity-50 font-mono">{{ lang.code.toUpperCase() }}</span>
                  </button>
                </div>
              </Transition>
            </div>

            <!-- Auth buttons -->
            <NuxtLink to="/auth/login" class="btn-metal px-4 py-2">
              <Icon name="heroicons:arrow-right-on-rectangle" class="w-4 h-4 mr-1" />
              {{ t('auth.signIn') }}
            </NuxtLink>
            <NuxtLink to="/auth/register" class="btn-primary px-4 py-2">
              <Icon name="heroicons:rocket-launch" class="w-4 h-4 mr-1" />
              {{ t('auth.getStarted') }}
            </NuxtLink>
          </div>
        </div>
      </div>
    </nav>

    <!-- Hero Section with metallic gradient -->
    <section class="py-20 relative overflow-hidden">
      <!-- Metallic background effect -->
      <div class="absolute inset-0 gradient-metal opacity-50"></div>
      <div class="absolute inset-0 gradient-steel opacity-30"></div>
      
      <div class="container mx-auto px-4 relative z-10">
        <div class="max-w-4xl mx-auto text-center">
          <h1 class="text-5xl md:text-6xl font-bold text-white mb-6 header-shine">
            {{ t('landing.hero.title') }}
          </h1>
          <p class="text-xl text-steel-shine mb-10 leading-relaxed">
            {{ t('landing.hero.subtitle') }}
          </p>
          <div class="flex flex-col sm:flex-row items-center justify-center gap-4">
            <NuxtLink to="/auth/register" class="btn-primary px-8 py-4 text-lg">
              <Icon name="heroicons:rocket-launch" class="w-5 h-5 mr-2" />
              {{ t('auth.startFreeTrial') }}
            </NuxtLink>
            <button class="btn-metal px-8 py-4 text-lg">
              <Icon name="heroicons:play-circle" class="w-5 h-5 mr-2" />
              {{ t('auth.watchDemo') }}
            </button>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Grid with metallic cards -->
    <section class="py-16 container mx-auto px-4">
      <div class="text-center mb-12">
        <h2 class="text-4xl font-bold text-white mb-4">{{ t('landing.features.title') }}</h2>
        <p class="text-lg text-steel-shine">{{ t('landing.features.subtitle') }}</p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <!-- Feature cards with icons -->
        <div v-for="feature in features" :key="feature.key" class="card-metal">
          <div class="flex items-start gap-4">
            <div class="w-12 h-12 rounded-lg flex items-center justify-center shrink-0" :class="feature.iconBg">
              <Icon :name="feature.icon" class="w-6 h-6" :class="feature.iconColor" />
            </div>
            <div class="flex-1">
              <h3 class="text-lg font-semibold text-white mb-2">{{ t(feature.title) }}</h3>
              <p class="text-sm text-steel-shine leading-relaxed">{{ t(feature.description) }}</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Stats Section -->
    <section class="py-16 container mx-auto px-4">
      <div class="card-metal p-12">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div class="text-center">
            <div class="text-5xl font-bold gradient-primary bg-clip-text text-transparent mb-3">99.9%</div>
            <p class="text-steel-shine font-medium">{{ t('landing.stats.uptime') }}</p>
          </div>
          <div class="text-center">
            <div class="text-5xl font-bold text-green-400 mb-3">50K+</div>
            <p class="text-steel-shine font-medium">{{ t('landing.stats.users') }}</p>
          </div>
          <div class="text-center">
            <div class="text-5xl font-bold gradient-primary bg-clip-text text-transparent mb-3">24/7</div>
            <p class="text-steel-shine font-medium">{{ t('landing.stats.support') }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="py-16 container mx-auto px-4">
      <div class="card-metal p-12 text-center relative overflow-hidden">
        <div class="absolute inset-0 gradient-primary opacity-10"></div>
        <div class="relative z-10">
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">{{ t('landing.cta.title') }}</h2>
          <p class="text-lg text-steel-shine mb-8 max-w-2xl mx-auto">{{ t('landing.cta.subtitle') }}</p>
          <NuxtLink to="/auth/register" class="btn-primary px-8 py-4 text-lg inline-flex items-center">
            <Icon name="heroicons:arrow-right" class="w-5 h-5 mr-2" />
            {{ t('auth.startTrialNow') }}
          </NuxtLink>
        </div>
      </div>
    </section>

    <!-- Footer -->
    <footer class="mt-16 py-12 border-t border-white/10">
      <div class="container mx-auto px-4">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          <div>
            <h4 class="font-semibold text-white mb-4">{{ t('landing.footer.product.title') }}</h4>
            <ul class="space-y-2">
              <li><a href="#" class="text-steel-shine hover:text-white transition">{{ t('landing.footer.product.features') }}</a></li>
              <li><a href="#" class="text-steel-shine hover:text-white transition">{{ t('landing.footer.product.pricing') }}</a></li>
              <li><a href="#" class="text-steel-shine hover:text-white transition">{{ t('landing.footer.product.documentation') }}</a></li>
            </ul>
          </div>
          <div>
            <h4 class="font-semibold text-white mb-4">{{ t('landing.footer.company.title') }}</h4>
            <ul class="space-y-2">
              <li><a href="#" class="text-steel-shine hover:text-white transition">{{ t('landing.footer.company.about') }}</a></li>
              <li><a href="#" class="text-steel-shine hover:text-white transition">{{ t('landing.footer.company.blog') }}</a></li>
              <li><a href="#" class="text-steel-shine hover:text-white transition">{{ t('landing.footer.company.careers') }}</a></li>
            </ul>
          </div>
          <div>
            <h4 class="font-semibold text-white mb-4">{{ t('landing.footer.legal.title') }}</h4>
            <ul class="space-y-2">
              <li><a href="#" class="text-steel-shine hover:text-white transition">{{ t('landing.footer.legal.privacy') }}</a></li>
              <li><a href="#" class="text-steel-shine hover:text-white transition">{{ t('landing.footer.legal.terms') }}</a></li>
              <li><a href="#" class="text-steel-shine hover:text-white transition">{{ t('landing.footer.legal.security') }}</a></li>
            </ul>
          </div>
          <div>
            <h4 class="font-semibold text-white mb-4">{{ t('landing.footer.contact.title') }}</h4>
            <ul class="space-y-2 text-steel-shine">
              <li class="flex items-center gap-2">
                <Icon name="heroicons:envelope" class="w-4 h-4" />
                {{ t('landing.footer.contact.email') }}
              </li>
              <li class="flex items-center gap-2">
                <Icon name="heroicons:phone" class="w-4 h-4" />
                {{ t('landing.footer.contact.phone') }}
              </li>
              <li class="flex items-center gap-2">
                <Icon name="heroicons:chat-bubble-left-right" class="w-4 h-4" />
                {{ t('landing.footer.contact.chat') }}
              </li>
            </ul>
          </div>
        </div>
        <div class="border-t border-white/10 pt-8 text-center text-steel-shine">
          <p>{{ t('landing.footer.copyright') }}</p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { definePageMeta } from '#imports'
import { computed, ref } from 'vue'

type AppLocale = 'ru' | 'en'

definePageMeta({
  layout: 'blank' as const,
  middleware: []
})

const { locale, setLocale, t } = useI18n()
const showLanguageDropdown = ref(false)

const availableLocales = [
  { code: 'ru' as AppLocale, name: 'Русский' },
  { code: 'en' as AppLocale, name: 'English' }
]

const currentLocale = computed(() =>
  availableLocales.find(l => l.code === (locale.value as AppLocale)) ?? availableLocales[0]
)

const switchLanguage = async (code: string) => {
  await setLocale(code as AppLocale)
  showLanguageDropdown.value = false
}

// Features data
const features = [
  {
    key: 'monitoring',
    icon: 'heroicons:chart-bar',
    iconBg: 'bg-blue-500/20',
    iconColor: 'text-blue-400',
    title: 'landing.features.monitoring.title',
    description: 'landing.features.monitoring.description'
  },
  {
    key: 'analytics',
    icon: 'heroicons:sparkles',
    iconBg: 'bg-green-500/20',
    iconColor: 'text-green-400',
    title: 'landing.features.analytics.title',
    description: 'landing.features.analytics.description'
  },
  {
    key: 'alerts',
    icon: 'heroicons:bell-alert',
    iconBg: 'bg-purple-500/20',
    iconColor: 'text-purple-400',
    title: 'landing.features.alerts.title',
    description: 'landing.features.alerts.description'
  },
  {
    key: 'reports',
    icon: 'heroicons:document-chart-bar',
    iconBg: 'bg-yellow-500/20',
    iconColor: 'text-yellow-400',
    title: 'landing.features.reports.title',
    description: 'landing.features.reports.description'
  },
  {
    key: 'security',
    icon: 'heroicons:lock-closed',
    iconBg: 'bg-red-500/20',
    iconColor: 'text-red-400',
    title: 'landing.features.security.title',
    description: 'landing.features.security.description'
  },
  {
    key: 'integration',
    icon: 'heroicons:cube-transparent',
    iconBg: 'bg-indigo-500/20',
    iconColor: 'text-indigo-400',
    title: 'landing.features.integration.title',
    description: 'landing.features.integration.description'
  }
]
</script>

<style scoped>
/* Additional metallic animations */
.bg-clip-text {
  -webkit-background-clip: text;
  background-clip: text;
}
</style>