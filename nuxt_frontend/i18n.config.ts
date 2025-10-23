import { defineI18nConfig } from '@nuxtjs/i18n'

export default defineI18nConfig({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      welcome: 'Welcome to HydraulicsTell',
      // Add more English translations here
    },
    ru: {
      welcome: 'Добро пожаловать в HydraulicsTell',
      // Add more Russian translations here
    }
  }
})
