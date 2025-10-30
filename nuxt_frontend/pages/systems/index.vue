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
          <Icon name="i-heroicons-arrow-path" class="w-4 h-4 mr-2" />
          Refresh
        </button>
        <button class="u-btn u-btn-secondary u-btn-md" @click="openCreateModal = true">
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
            <tr v-if="!loading && systemsStore.systems.length === 0">
              <td colspan="5" class="px-6 py-10">
                <div class="u-empty">
                  <Icon name="i-heroicons-cpu-chip" class="u-empty-icon" />
                  <div class="u-empty-title">No systems yet</div>
                  <div class="u-empty-desc">Create your first hydraulic system to start monitoring</div>
                  <button class="u-btn u-btn-secondary u-btn-md" @click="openCreateModal = true">
                    <Icon name="i-heroicons-plus" class="w-4 h-4 mr-2" />
                    Add System
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Create System Modal -->
    <Transition name="fade">
      <div v-if="openCreateModal" class="fixed inset-0 z-50" aria-modal="true" role="dialog" aria-labelledby="create-system-title">
        <div class="fixed inset-0 bg-black/60" @click="closeModal" />
        <div class="relative z-10 mx-auto mt-24 w-full max-w-lg px-4">
          <div class="u-card">
            <div class="u-card-header">
              <h3 id="create-system-title" class="u-h4">Create System</h3>
              <button class="u-icon-btn" @click="closeModal" aria-label="Close">
                <Icon name="i-heroicons-x-mark" class="w-5 h-5" />
              </button>
            </div>
            <div class="u-card-body space-y-4">
              <div>
                <label class="u-label" for="name">Name</label>
                <input id="name" v-model.trim="form.name" type="text" class="u-input w-full" placeholder="e.g. Press Line A" />
                <p v-if="errors.name" class="u-error mt-1">{{ errors.name }}</p>
              </div>
              <div>
                <label class="u-label" for="status">Status</label>
                <select id="status" v-model="form.status" class="u-input w-full">
                  <option value="online">online</option>
                  <option value="offline">offline</option>
                  <option value="warning">warning</option>
                  <option value="error">error</option>
                </select>
                <p v-if="errors.status" class="u-error mt-1">{{ errors.status }}</p>
              </div>
            </div>
            <div class="u-card-footer flex justify-end gap-3">
              <button class="u-btn u-btn-secondary" @click="closeModal" :disabled="loading">Cancel</button>
              <button class="u-btn u-btn-primary" @click="onCreate" :disabled="!isValid || loading">
                <Icon v-if="loading" name="i-heroicons-arrow-path" class="w-4 h-4 mr-2 animate-spin" />
                Create
              </button>
            </div>
          </div>
        </div>
      </div>
    </Transition>
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
const openCreateModal = ref(false)

const form = reactive({
  name: '',
  status: 'online' as 'online' | 'offline' | 'warning' | 'error'
})

const errors = reactive<{ name?: string; status?: string }>({})

const validate = () => {
  errors.name = !form.name ? 'Name is required' : undefined
  errors.status = !['online','offline','warning','error'].includes(form.status) ? 'Invalid status' : undefined
  return !errors.name && !errors.status
}

const isValid = computed(() => validate())

const getStatusClass = (status: string) => {
  const classes = {
    'online': 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    'warning': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    'error': 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    'offline': 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
  }
  return classes[status] || classes['offline']
}

const closeModal = () => {
  if (loading.value) return
  openCreateModal.value = false
}

const onRefresh = async () => {
  loading.value = true
  try { await systemsStore.fetchSystems() } finally { loading.value = false }
}

const onCreate = async () => {
  if (!validate()) return
  loading.value = true
  try {
    if (typeof systemsStore.createSystem !== 'function') {
      // Graceful fallback if API not implemented yet
      alert('Create System API is not available yet (will be implemented in Stage 2/3).')
      return
    }
    await systemsStore.createSystem({ name: form.name, status: form.status })
    await systemsStore.fetchSystems()
    openCreateModal.value = false
    form.name = ''
    form.status = 'online'
  } catch (e: any) {
    alert(e?.message || 'Failed to create system')
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  systemsStore.fetchSystems()
})
</script>
