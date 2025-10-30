<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="u-h2">Hydraulic Systems</h1>
        <p class="u-body text-gray-600 dark:text-gray-400 mt-1">Manage and monitor your hydraulic systems</p>
      </div>
      <div class="flex items-center gap-3">
        <button class="u-btn u-btn-secondary u-btn-md" @click="onRefresh" :disabled="loading">
          <Icon name="i-heroicons-arrow-path" class="w-4 h-4 mr-2" :class="{ 'animate-spin': loading }" />
          Refresh
        </button>
        <button class="u-btn u-btn-primary u-btn-md" @click="openCreateModal = true">
          <Icon name="i-heroicons-plus" class="w-4 h-4 mr-2" />
          Add System
        </button>
      </div>
    </div>

    <!-- Systems Table -->
    <div class="u-card overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h3 class="u-h4 text-gray-900 dark:text-white">All Systems</h3>
      </div>

      <div v-if="loading && systemsStore.systems.length === 0" class="p-12 text-center">
        <Icon name="i-heroicons-arrow-path" class="w-8 h-8 mx-auto text-gray-400 animate-spin mb-4" />
        <p class="text-gray-500 dark:text-gray-400">Loading systems...</p>
      </div>

      <div v-else-if="!loading && systemsStore.systems.length === 0" class="p-12 text-center">
        <Icon name="i-heroicons-cpu-chip" class="w-16 h-16 mx-auto text-gray-400 mb-4" />
        <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">No systems yet</h3>
        <p class="text-gray-500 dark:text-gray-400 mb-6">Create your first hydraulic system to start monitoring</p>
        <button class="u-btn u-btn-primary u-btn-md" @click="openCreateModal = true">
          <Icon name="i-heroicons-plus" class="w-4 h-4 mr-2" />
          Add System
        </button>
      </div>

      <div v-else class="overflow-x-auto">
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

    <!-- Create System Modal Component -->
    <UCreateSystemModal 
      v-model="openCreateModal" 
      :loading="createLoading"
      @submit="onCreate"
      @cancel="onCancelCreate"
    />
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  layout: 'dashboard',
  title: 'Hydraulic Systems',
  middleware: ['auth']
})

const systemsStore = useSystemsStore()
const loading = ref(false)
const createLoading = ref(false)
const openCreateModal = ref(false)

const getStatusClass = (status: string) => {
  const classes = {
    'active': 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    'maintenance': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    'emergency': 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    'inactive': 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
  }
  return classes[status] || classes['inactive']
}

const onRefresh = async () => {
  if (loading.value) return
  loading.value = true
  try {
    await systemsStore.fetchSystems()
  } catch (error) {
    console.error('Failed to refresh systems:', error)
    // TODO: Show toast notification
  } finally {
    loading.value = false
  }
}

// Updated to match new form structure with type and description
const onCreate = async (data: { name: string; type: string; status: string; description: string }) => {
  createLoading.value = true
  try {
    // Check if store has createSystem method
    if (typeof systemsStore.createSystem !== 'function') {
      // Graceful fallback for MVP - show friendly message
      alert('Создание систем будет реализовано на этапе 2-3. Пока используйте Admin Panel: http://localhost:8000/admin/')
      openCreateModal.value = false
      return
    }
    
    // Call real API through store
    await systemsStore.createSystem(data)
    
    // Refresh the systems list
    await systemsStore.fetchSystems()
    
    // Close modal and show success
    openCreateModal.value = false
    alert(`System "${data.name}" created successfully!`)
    
  } catch (error: any) {
    console.error('Failed to create system:', error)
    alert(`Failed to create system: ${error?.message || 'Unknown error'}`)
  } finally {
    createLoading.value = false
  }
}

const onCancelCreate = () => {
  openCreateModal.value = false
}

// Load systems on mount
onMounted(() => {
  systemsStore.fetchSystems()
})
</script>