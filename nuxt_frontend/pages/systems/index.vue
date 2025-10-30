<template>
  <div class="space-y-6">
    <!-- Unified header - remove green button -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="u-h2">Hydraulic Systems</h1>
        <p class="u-body text-gray-600 dark:text-gray-400 mt-1">Manage and monitor your hydraulic systems</p>
      </div>
      <div class="flex items-center gap-3">
        <button class="u-btn u-btn-secondary u-btn-md">
          <Icon name="heroicons:arrow-path" class="w-4 h-4 mr-2" />
          Refresh
        </button>
        <!-- Removed green button per requirements -->
      </div>
    </div>

    <!-- Systems Table -->
    <div class="u-card overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h3 class="u-h4 text-gray-900 dark:text-white">All Systems</h3>
      </div>

      <div class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead class="bg-gray-50 dark:bg-gray-800">
            <tr>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">System</th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Pressure</th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Temperature</th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
            <tr v-for="system in systemsStore.systems" :key="system.id" class="hover:bg-gray-50 dark:hover:bg-gray-800">
              <td class="px-6 py-4 whitespace-nowrap">
                <NuxtLink 
                  :to="`/systems/${system.id}`" 
                  class="text-blue-600 dark:text-blue-400 hover:text-blue-900 dark:hover:text-blue-300 font-medium"
                >
                  {{ system.name }}
                </NuxtLink>
              </td>
              <td class="px-6 py-4 whitespace-nowrap">
                <span 
                  :class="getStatusClass(system.status)"
                  class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                >
                  {{ system.status }}
                </span>
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                {{ system.pressure }} MPa
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                {{ system.temperature }}°C
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                <NuxtLink 
                  :to="`/systems/${system.id}`" 
                  class="text-blue-600 dark:text-blue-400 hover:text-blue-900 dark:hover:text-blue-300 mr-4"
                >
                  View
                </NuxtLink>
                <button class="text-red-600 dark:text-red-400 hover:text-red-900 dark:hover:text-red-300">
                  Delete
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Use unified dashboard layout
definePageMeta({
  layout: 'dashboard',
  title: 'Hydraulic Systems',
  middleware: ['auth']
})

const systemsStore = useSystemsStore()

onMounted(() => {
  systemsStore.fetchSystems()
})

const getStatusClass = (status: string) => {
  const classes = {
    'online': 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    'warning': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    'error': 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    'offline': 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
  }
  return classes[status] || classes['offline']
}
</script>
