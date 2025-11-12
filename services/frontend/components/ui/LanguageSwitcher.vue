<!-- components/ui/LanguageSwitcher.vue -->
<template>
  <UDropdown :items="languageItems">
    <UButton
      color="gray"
      variant="ghost"
      size="sm"
      class="gap-2"
    >
      <UIcon :name="currentFlag" class="w-5 h-5" />
      <span class="hidden sm:inline text-sm">{{ currentLocaleName }}</span>
    </UButton>
  </UDropdown>
</template>

<script setup lang="ts">
const { locale, locales, setLocale } = useI18n()
const toast = useToast()

const currentLocaleName = computed(() => {
  const current = (locales.value as any[]).find(l => l.code === locale.value)
  return current?.name || 'Русский'
})

const currentFlag = computed(() => {
  const flags: Record<string, string> = {
    ru: 'i-twemoji-flag-russia',
    en: 'i-twemoji-flag-united-states'
  }
  return flags[locale.value] || 'i-heroicons-language'
})

const languageItems = computed(() => [[
  ...((locales.value as any[]) || []).map(l => ({
    label: l.name,
    icon: l.code === 'ru' ? 'i-twemoji-flag-russia' : 'i-twemoji-flag-united-states',
    click: async () => {
      await setLocale(l.code)
      toast.add({
        title: l.code === 'ru' ? 'Язык изменён' : 'Language changed',
        description: l.code === 'ru' ? `Текущий язык: ${l.name}` : `Current language: ${l.name}`,
        color: 'blue',
        icon: 'i-heroicons-language'
      })
    }
  }))
]])
</script>
