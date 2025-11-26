<script setup lang="ts">
interface Props {
  title: string;
  description: string;
  level: number;
}

defineProps<Props>();
</script>

<template>
  <div class="h-full">
    <!-- Header -->
    <div class="mb-6">
      <div class="mb-2 flex items-center gap-2">
        <span class="rounded-lg bg-primary-100 px-2.5 py-1 text-xs font-semibold text-primary-700 dark:bg-primary-900/20 dark:text-primary-400">
          {{ $t('wizard.level') }} {{ level }}
        </span>
      </div>
      <h3 class="text-2xl font-bold text-gray-900 dark:text-white">
        {{ title }}
      </h3>
      <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">
        {{ description }}
      </p>
    </div>

    <!-- Content Slot -->
    <div class="flex-1">
      <slot />
    </div>
  </div>
</template>