<script setup lang="ts">
definePageMeta({
  layout: 'default',
  middleware: 'auth',
});

const { t } = useI18n();

useSeoMeta({
  title: t('wizard.pageTitle'),
  description: t('wizard.pageDescription'),
});
</script>

<template>
  <div class="min-h-screen bg-gray-50 p-4 dark:bg-gray-900 sm:p-6 lg:p-8">
    <div class="mx-auto max-w-5xl">
      <!-- Header -->
      <div class="mb-8 text-center">
        <h1 class="text-4xl font-bold text-gray-900 dark:text-white">
          {{ $t('wizard.pageTitle') }}
        </h1>
        <p class="mt-3 text-lg text-gray-600 dark:text-gray-400">
          {{ $t('wizard.pageDescription') }}
        </p>
      </div>

      <!-- Info Cards -->
      <div class="mb-8 grid gap-4 sm:grid-cols-3">
        <div class="rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
          <div class="mb-2 flex items-center gap-2">
            <Icon name="heroicons:document-text" class="h-5 w-5 text-blue-600 dark:text-blue-400" />
            <h3 class="font-semibold text-blue-900 dark:text-blue-300">{{ $t('wizard.info.step1') }}</h3>
          </div>
          <p class="text-sm text-blue-700 dark:text-blue-400">{{ $t('wizard.info.step1Desc') }}</p>
        </div>

        <div class="rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-800 dark:bg-green-900/20">
          <div class="mb-2 flex items-center gap-2">
            <Icon name="heroicons:cpu-chip" class="h-5 w-5 text-green-600 dark:text-green-400" />
            <h3 class="font-semibold text-green-900 dark:text-green-300">{{ $t('wizard.info.step2') }}</h3>
          </div>
          <p class="text-sm text-green-700 dark:text-green-400">{{ $t('wizard.info.step2Desc') }}</p>
        </div>

        <div class="rounded-lg border border-purple-200 bg-purple-50 p-4 dark:border-purple-800 dark:bg-purple-900/20">
          <div class="mb-2 flex items-center gap-2">
            <Icon name="heroicons:sparkles" class="h-5 w-5 text-purple-600 dark:text-purple-400" />
            <h3 class="font-semibold text-purple-900 dark:text-purple-300">{{ $t('wizard.info.step3') }}</h3>
          </div>
          <p class="text-sm text-purple-700 dark:text-purple-400">{{ $t('wizard.info.step3Desc') }}</p>
        </div>
      </div>

      <!-- Wizard Component -->
      <MetadataWizard />
    </div>
  </div>
</template>
