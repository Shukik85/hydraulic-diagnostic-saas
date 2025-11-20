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

// Добавлен ! для non-null assertion
const currentLocale = computed<LocaleOption>(() =>
  availableLocales.find(l => l.code === (locale.value as AppLocale)) ?? availableLocales[0]!
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
