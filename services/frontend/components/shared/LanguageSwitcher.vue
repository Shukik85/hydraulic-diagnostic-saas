<script setup lang="ts">
const { locale, setLocale } = useI18n();

const languages = [
  { code: 'ru', label: 'RU', name: 'Русский' },
  { code: 'en', label: 'EN', name: 'English' },
];

const switchLanguage = (code: string) => {
  setLocale(code);
};
</script>

<template>
  <div class="flex items-center gap-1 rounded-lg border border-border bg-background p-1">
    <button
      v-for="lang in languages"
      :key="lang.code"
      type="button"
      :aria-label="`Switch to ${lang.name}`"
      :aria-current="locale === lang.code ? 'true' : 'false'"
      class="px-3 py-1 text-sm font-medium transition-all rounded-md"
      :class="[
        locale === lang.code
          ? 'bg-primary text-primary-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground hover:bg-secondary',
      ]"
      @click="switchLanguage(lang.code)"
    >
      {{ lang.label }}
    </button>
  </div>
</template>
