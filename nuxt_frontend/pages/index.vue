<template>
  <div class="min-h-screen bg-white dark:bg-gray-900">
    <!-- Navigation -->
    <nav class="sticky top-0 z-50 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
      <div class="u-container u-flex-between py-4">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-linear-to-br from-blue-600 to-blue-400 flex items-center justify-center">
            <Icon name="heroicons:chart-bar-square" class="w-6 h-6 text-white" />
          </div>
          <span class="text-lg font-bold text-gray-900 dark:text-white">{{ t('app.title') }}</span>
        </div>
        <div class="flex items-center gap-4">
          <!-- Language Toggle -->
          <div class="relative language-dropdown">
            <button @click="showLanguageDropdown = !showLanguageDropdown"
              class="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors flex items-center gap-1"
              :aria-label="t('ui.language.switch')">
              <Icon name="heroicons:language" class="w-5 h-5" />
              <span class="text-sm font-medium">{{ currentLocale?.code?.toUpperCase() }}</span>
              <Icon name="heroicons:chevron-down" class="w-3 h-3 transition-transform"
                :class="{ 'rotate-180': showLanguageDropdown }" />
            </button>
            <transition enter-active-class="transition ease-out duration-200"
              enter-from-class="transform opacity-0 scale-95" enter-to-class="transform opacity-100 scale-100"
              leave-active-class="transition ease-in duration-150" leave-from-class="transform opacity-100 scale-100"
              leave-to-class="opacity-0 scale-95">
              <div v-show="showLanguageDropdown"
                class="absolute right-0 mt-2 w-48 bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-50">
                <button v-for="langOption in availableLocales" :key="langOption.code"
                  @click="switchLanguage(langOption.code)"
                  class="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-3"
                  :class="{ 'bg-blue-50 text-blue-600': currentLocale?.code === langOption.code }">
                  <span class="text-base">{{ langOption.flag }}</span>
                  <span>{{ langOption.name }}</span>
                  <Icon v-if="currentLocale?.code === langOption.code" name="heroicons:check"
                    class="w-4 h-4 ml-auto text-blue-600" />
                </button>
              </div>
            </transition>
          </div>
          <NuxtLink to="/auth/login" class="u-btn u-btn-ghost u-btn-md">
            <Icon name="heroicons:arrow-right-on-rectangle" class="w-4 h-4 mr-1" />
            {{ t('auth.signIn') }}
          </NuxtLink>
          <NuxtLink to="/auth/register" class="u-btn u-btn-primary u-btn-md">
            <Icon name="heroicons:plus" class="w-4 h-4 mr-1" />
            {{ t('auth.getStarted') }}
          </NuxtLink>
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <section class="py-20 bg-linear-to-b from-blue-50 dark:from-gray-800 to-white dark:to-gray-900">
      <div class="u-container">
        <div class="max-w-3xl mx-auto text-center">
          <h1 class="u-h1 mb-6">
            {{ t('landing.hero.title') }}
          </h1>
          <p class="u-body-lg text-gray-600 dark:text-gray-300 mb-8">
            {{ t('landing.hero.subtitle') }}
          </p>
          <div class="flex items-center justify-center gap-4">
            <NuxtLink to="/auth/register" class="u-btn u-btn-primary u-btn-lg">
              <Icon name="heroicons:rocket-launch" class="w-5 h-5 mr-2" />
              {{ t('auth.startFreeTrial') }}
            </NuxtLink>
            <button class="u-btn u-btn-secondary u-btn-lg">
              <Icon name="heroicons:play" class="w-5 h-5 mr-2" />
              {{ t('auth.watchDemo') }}
            </button>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="u-section">
      <div class="u-container">
        <div class="text-center mb-12">
          <h2 class="u-h2 mb-4">{{ t('landing.features.title') }}</h2>
          <p class="u-body-lg text-gray-600 dark:text-gray-300">{{ t('landing.features.subtitle') }}</p>
        </div>

        <div class="u-grid-responsive">
          <!-- Feature 1 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                <Icon name="heroicons:chart-line" class="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 class="u-h5">{{ t('landing.features.monitoring.title') }}</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              {{ t('landing.features.monitoring.description') }}
            </p>
          </div>

          <!-- Feature 2 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                <Icon name="heroicons:sparkles" class="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <h3 class="u-h5">{{ t('landing.features.analytics.title') }}</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              {{ t('landing.features.analytics.description') }}
            </p>
          </div>

          <!-- Feature 3 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                <Icon name="heroicons:bell-alert" class="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 class="u-h5">{{ t('landing.features.alerts.title') }}</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              {{ t('landing.features.alerts.description') }}
            </p>
          </div>

          <!-- Feature 4 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-yellow-100 dark:bg-yellow-900/30 flex items-center justify-center">
                <Icon name="heroicons:document-chart-bar" class="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
              </div>
              <h3 class="u-h5">{{ t('landing.features.reports.title') }}</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              {{ t('landing.features.reports.description') }}
            </p>
          </div>

          <!-- Feature 5 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
                <Icon name="heroicons:lock-closed" class="w-6 h-6 text-red-600 dark:text-red-400" />
              </div>
              <h3 class="u-h5">{{ t('landing.features.security.title') }}</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              {{ t('landing.features.security.description') }}
            </p>
          </div>

          <!-- Feature 6 -->
          <div class="u-card p-6">
            <div class="flex items-center gap-4 mb-4">
              <div class="w-12 h-12 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
                <Icon name="heroicons:cube-transparent" class="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
              </div>
              <h3 class="u-h5">{{ t('landing.features.integration.title') }}</h3>
            </div>
            <p class="u-body text-gray-600 dark:text-gray-400">
              {{ t('landing.features.integration.description') }}
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
            <p class="u-body text-gray-600 dark:text-gray-400">{{ t('landing.stats.uptime') }}</p>
          </div>
          <div class="text-center">
            <div class="u-h2 text-green-600 mb-2">50K+</div>
            <p class="u-body text-gray-600 dark:text-gray-400">{{ t('landing.stats.users') }}</p>
          </div>
          <div class="text-center">
            <div class="u-h2 text-blue-600 mb-2">24/7</div>
            <p class="u-body text-gray-600 dark:text-gray-400">{{ t('landing.stats.support') }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="u-section">
      <div class="u-container">
        <div class="u-card u-gradient-primary p-12 text-center text-white">
          <h2 class="text-3xl font-bold mb-4">{{ t('landing.cta.title') }}</h2>
          <p class="text-lg mb-8 opacity-90">{{ t('landing.cta.subtitle') }}</p>
          <NuxtLink to="/auth/register" class="u-btn u-btn-lg bg-white hover:bg-gray-100 text-blue-600 font-medium">
            {{ t('auth.startTrialNow') }}
          </NuxtLink>
        </div>
      </div>
    </section>

    <!-- Footer -->
    <footer class="bg-gray-900 text-gray-400 py-12">
      <div class="u-container">
        <div class="grid grid-cols-4 gap-8 mb-8">
          <div>
            <h4 class="font-semibold text-white mb-4">{{ t('landing.footer.product.title') }}</h4>
            <ul class="space-y-2">
              <li><a href="#" class="hover:text-white transition">{{ t('landing.footer.product.features') }}</a></li>
              <li><a href="#" class="hover:text-white transition">{{ t('landing.footer.product.pricing') }}</a></li>
              <li><a href="#" class="hover:text-white transition">{{ t('landing.footer.product.documentation') }}</a>
              </li>
            </ul>
          </div>
          <div>
            <h4 class="font-semibold text-white mb-4">{{ t('landing.footer.company.title') }}</h4>
            <ul class="space-y-2">
              <li><a href="#" class="hover:text-white transition">{{ t('landing.footer.company.about') }}</a></li>
              <li><a href="#" class="hover:text-white transition">{{ t('landing.footer.company.blog') }}</a></li>
              <li><a href="#" class="hover:text-white transition">{{ t('landing.footer.company.careers') }}</a></li>
            </ul>
          </div>
          <div>
            <h4 class="font-semibold text-white mb-4">{{ t('landing.footer.legal.title') }}</h4>
            <ul class="space-y-2">
              <li><a href="#" class="hover:text-white transition">{{ t('landing.footer.legal.privacy') }}</a></li>
              <li><a href="#" class="hover:text-white transition">{{ t('landing.footer.legal.terms') }}</a></li>
              <li><a href="#" class="hover:text-white transition">{{ t('landing.footer.legal.security') }}</a></li>
            </ul>
          </div>
          <div>
            <h4 class="font-semibold text-white mb-4">{{ t('landing.footer.contact.title') }}</h4>
            <ul class="space-y-2">
              <li>{{ t('landing.footer.contact.email') }}</li>
              <li>{{ t('landing.footer.contact.phone') }}</li>
              <li>{{ t('landing.footer.contact.chat') }}</li>
            </ul>
          </div>
        </div>
        <div class="border-t border-gray-800 pt-8 text-center">
          <p>{{ t('landing.footer.copyright') }}</p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

type AppLocale = 'ru' | 'en'

definePageMeta({
  layout: 'blank',
  middleware: []
})

const { locale, setLocale, t } = useI18n()
const showLanguageDropdown = ref(false)

const availableLocales = [
  { code: 'ru' as AppLocale, name: 'Русский', flag: '🇷🇺' },
  { code: 'en' as AppLocale, name: 'English', flag: '🇺🇸' }
]

const currentLocale = computed(() =>
  availableLocales.find(l => l.code === (locale.value as AppLocale)) ?? availableLocales[0]
)

const switchLanguage = async (code: string) => {
  await setLocale(code as AppLocale)
  showLanguageDropdown.value = false
}
</script>