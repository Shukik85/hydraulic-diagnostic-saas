import { defineNuxtConfig } from 'nuxt/config';

export default defineNuxtConfig({
  // Укажите базовые настройки вашего приложения
  app: {
    head: {
      title: 'My Nuxt App',
      meta: [{ name: 'description', content: 'Описание вашего приложения' }],
    },
  },
  // Настройки для серверной части
  // Настройки для плагинов
  plugins: [
    // Укажите ваши плагины здесь
  ],
  // Настройки для модулей
  modules: [
    // Укажите ваши модули здесь
  ],
  // Если нужны настройки сервера/статики, настроите через Nitro / public dir
});
