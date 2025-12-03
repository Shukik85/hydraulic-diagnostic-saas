<script setup lang="ts">
import { computed } from 'vue';

const languages = [
  { code: 'ru', label: 'RU', name: 'Русский' },
  { code: 'en', label: 'EN', name: 'English' },
];

// Check if i18n is available
let locale: any;
let setLocale: any;

try {
  const i18n = useI18n();
  locale = i18n.locale;
  setLocale = i18n.setLocale;
} catch (error) {
  console.error('i18n not available:', error);
  // Fallback to ref
  locale = ref('ru');
  setLocale = (code: string) => {
    locale.value = code;
    console.log('i18n not available, locale set to:', code);
  };
}

const switchLanguage = (code: string) => {
  console.log('Language switching:', locale.value, '->', code);
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
