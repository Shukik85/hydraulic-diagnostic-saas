<script setup lang="ts">
import type { TenantUsage } from '~/types';

interface Props {
  tenants: TenantUsage[];
  loading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
});

const { t } = useI18n();
const { formatNumber, formatBytes } = useFormatting();

const planBadgeColors = {
  starter: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300',
  professional: 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400',
  enterprise: 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400',
};
</script>

<template>
  <div class="rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
    <div class="border-b border-gray-200 p-6 dark:border-gray-700">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
        {{ t('admin.topTenants') }}
      </h3>
    </div>

    <div v-if="loading" class="p-6">
      <div v-for="i in 5" :key="i" class="mb-3 h-12 w-full animate-pulse rounded bg-gray-200 dark:bg-gray-700" />
    </div>

    <div v-else class="overflow-x-auto">
      <table class="w-full">
        <thead class="border-b border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900">
          <tr>
            <th class="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
              {{ t('admin.tenantName') }}
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
              {{ t('admin.plan') }}
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
              {{ t('admin.sensors') }}
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
              {{ t('admin.apiCalls') }}
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
              {{ t('admin.storage') }}
            </th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-200 dark:divide-gray-700">
          <tr v-for="tenant in tenants" :key="tenant.tenantId" class="hover:bg-gray-50 dark:hover:bg-gray-700">
            <td class="whitespace-nowrap px-6 py-4 text-sm font-medium text-gray-900 dark:text-white">
              {{ tenant.tenantName }}
            </td>
            <td class="whitespace-nowrap px-6 py-4 text-sm">
              <span class="inline-flex rounded-full px-2 py-1 text-xs font-semibold" :class="planBadgeColors[tenant.plan]">
                {{ tenant.plan }}
              </span>
            </td>
            <td class="whitespace-nowrap px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
              {{ formatNumber(tenant.sensors) }}
            </td>
            <td class="whitespace-nowrap px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
              {{ formatNumber(tenant.apiCalls) }}
            </td>
            <td class="whitespace-nowrap px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
              {{ formatBytes(tenant.storage) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
