<template>
  <div class="flex h-screen">
    <!-- Sidebar -->
    <aside
      :class="[
        'bg-steel-darker border-r border-steel-medium transition-all duration-300 ease-in-out flex flex-col',
        'shadow-lg',
        isCollapsed ? 'w-16' : 'w-64',
      ]"
    >
      <!-- Header -->
      <div class="p-4 border-b border-steel-medium flex items-center justify-between">
        <div
          :class="[
            'flex items-center gap-3 transition-opacity duration-200',
            isCollapsed ? 'opacity-0' : 'opacity-100',
          ]"
        >
          <div class="h-8 w-8 rounded-lg bg-gradient-to-br from-primary-600 to-primary-700 flex items-center justify-center shadow-md shadow-primary-500/20">
            <Icon name="lucide:gauge" class="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 class="font-bold text-sm text-text-primary">HydraulicsTell</h2>
            <p class="text-xs text-text-secondary">Диагностика</p>
          </div>
        </div>
        <UiButton variant="ghost" size="icon" @click="toggleSidebar" class="h-8 w-8 shrink-0 hover:bg-steel-dark">
          <Icon
            :name="isCollapsed ? 'lucide:chevron-right' : 'lucide:chevron-left'"
            class="h-4 w-4 text-text-secondary"
          />
        </UiButton>
      </div>

      <!-- Navigation -->
      <nav class="flex-1 p-2 space-y-1">
        <button
          v-for="item in navigationItems"
          :key="item.key"
          :class="[
            'w-full flex items-center gap-3 h-10 rounded-lg transition-all duration-200',
            isCollapsed ? 'px-2 justify-center' : 'px-3 justify-start',
            activeItem === item.key
              ? 'bg-gradient-to-r from-primary-600 to-primary-700 text-white shadow-md shadow-primary-500/30'
              : 'text-text-secondary hover:text-primary-300 hover:bg-steel-dark',
          ]"
          @click="setActiveItem(item.key)"
        >
          <Icon :name="item.icon" class="h-4 w-4 shrink-0" />
          <span
            :class="[
              'text-sm font-medium transition-opacity duration-200 truncate',
              isCollapsed ? 'opacity-0 w-0' : 'opacity-100',
            ]"
          >
            {{ item.label }}
          </span>
        </button>
      </nav>

      <!-- Footer -->
      <div class="p-2 border-t border-steel-medium">
        <button
          :class="[
            'w-full flex items-center gap-3 h-10 rounded-lg transition-all duration-200',
            'text-error-500 hover:bg-error-500/10',
            isCollapsed ? 'px-2 justify-center' : 'px-3 justify-start',
          ]"
          @click="handleLogout"
        >
          <Icon name="lucide:log-out" class="h-4 w-4 shrink-0" />
          <span
            :class="[
              'text-sm font-medium transition-opacity duration-200 truncate',
              isCollapsed ? 'opacity-0 w-0' : 'opacity-100',
            ]"
          >
            Выход
          </span>
        </button>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 overflow-hidden bg-background-primary">
      <slot />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

interface NavigationItem {
  key: string
  label: string
  icon: string
}

const props = defineProps<{
  defaultCollapsed?: boolean
}>()

const emit = defineEmits<{
  'item-click': [key: string]
  logout: []
}>()

const isCollapsed = ref(props.defaultCollapsed || false)
const activeItem = ref('dashboard')

const navigationItems: NavigationItem[] = [
  { key: 'dashboard', label: 'Dashboard', icon: 'lucide:layout-dashboard' },
  { key: 'equipment', label: 'Equipment', icon: 'lucide:cog' },
  { key: 'sensors', label: 'Sensors', icon: 'lucide:thermometer' },
  { key: 'diagnostics', label: 'Diagnostics', icon: 'lucide:activity' },
  { key: 'chat', label: 'AI Chat', icon: 'lucide:message-square' },
  { key: 'reports', label: 'Reports', icon: 'lucide:file-text' },
  { key: 'settings', label: 'Settings', icon: 'lucide:settings' },
]

const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value
}

const setActiveItem = (key: string) => {
  activeItem.value = key
  emit('item-click', key)
}

const handleLogout = () => {
  emit('logout')
}

// Expose methods for parent components
defineExpose({
  toggle: toggleSidebar,
  collapse: () => (isCollapsed.value = true),
  expand: () => (isCollapsed.value = false),
})
</script>