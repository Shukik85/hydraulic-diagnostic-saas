<script setup lang="ts">
definePageMeta({ layout: 'dashboard' })
</script>

<template>
  <div class="container mx-auto px-4 py-6">
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">Настройки</h1>
      <p class="text-gray-600 dark:text-gray-400">Управление аккаунтом и системой</p>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <NuxtLink to="/settings/profile" class="p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Профиль</h3>
        <p class="text-gray-600 dark:text-gray-400 text-sm">Личные данные, аватар, язык</p>
      </NuxtLink>
      <NuxtLink to="/settings/notifications" class="p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Уведомления</h3>
        <p class="text-gray-600 dark:text-gray-400 text-sm">Пороги, email, push</p>
      </NuxtLink>
      <NuxtLink to="/settings/security" class="p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Безопасность</h3>
        <p class="text-gray-600 dark:text-gray-400 text-sm">Пароль, 2FA, сессии</p>
      </NuxtLink>
      <NuxtLink to="/settings/integrations" class="p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Интеграции</h3>
        <p class="text-gray-600 dark:text-gray-400 text-sm">API ключи, Webhooks</p>
      </NuxtLink>
      <NuxtLink to="/settings/billing" class="p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Биллинг</h3>
        <p class="text-gray-600 dark:text-gray-400 text-sm">Планы, платежи, счета</p>
      </NuxtLink>
    </div>
  </div>
</template>