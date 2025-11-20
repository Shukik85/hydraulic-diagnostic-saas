<script setup lang="ts">
import { ref, computed } from '#imports'

type AppLocale = 'ru' | 'en'

interface LocaleOption {
  code: AppLocale
  name: string
}

const { locale, setLocale } = useI18n()
const showUserMenu = ref(false)
const showLanguageDropdown = ref(false)

const availableLocales: LocaleOption[] = [
  { code: 'ru', name: 'Русский' },
  { code: 'en', name: 'English' }
]

// Enterprise: безопасный дефолт вместо non-null assertion
const DEFAULT_LOCALE: LocaleOption = { code: 'ru', name: 'Русский' }

const currentLocale = computed<LocaleOption>(() =>
  availableLocales.find(l => l.code === (locale.value as AppLocale)) ?? DEFAULT_LOCALE
)

const switchLanguage = async (code: AppLocale) => {
  await setLocale(code)
  showLanguageDropdown.value = false
}
</script>

<template>
  <div class="dashboard-layout">
    <!-- Header, Sidebar, Content -->
    <slot />
  </div>
</template>
